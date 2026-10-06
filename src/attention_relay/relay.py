"""Relaying the LLM's reading weights into an embedder's pooling.

In mean pooling, the embedding averages the embedder's final token states, and the LLM's weights
become the averaging weights. In CLS or last-token pooling, the embedding is the final state of one
pooling token, and the relay acts on that token's attention row in every head of every layer, or of
the chosen layers. Each head keeps its attention outside the text, spends its attention on the text
as the LLM's weights direct, and recomputes its output at the pooling token from the embedder's own
values. Every other row of the attention is the model's own, and the embedder never sees the
instruction.
"""

import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AttentionInterface, AutoTokenizer, PreTrainedModel
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, eager_mask, sdpa_mask
from transformers.utils import cached_file

from attention_relay import gte
from attention_relay._attention import (
    allowed_keys,
    attention_row,
    model_eager_attention,
    weighted_values,
)
from attention_relay._batching import batches, pad
from attention_relay._loading import load_model

POOLINGS = ("mean", "cls", "last")
BIASES = ("mass_preserving", "plain")

FLOOR = 1e-8
"""Carried weights are raised to at least this value before the power, so that no text token's
attention becomes exactly zero."""

# The attention implementations that relay at the pooling token, one per underlying implementation.
RELAY_SDPA = "attention_relay_relay_sdpa"
RELAY_EAGER = "attention_relay_relay_eager"

_POOLING_MODES = {
    "pooling_mode_mean_tokens": "mean",
    "pooling_mode_cls_token": "cls",
    "pooling_mode_lasttoken": "last",
}


def relay_row(
    attention: torch.Tensor,
    text_mask: torch.Tensor,
    weights: torch.Tensor,
    beta: float = 1.0,
    bias: str = "mass_preserving",
) -> torch.Tensor:
    """A pooling token's attention row after relaying, for every head.

    Args:
        attention: The head's own attention row a, of shape [..., positions].
        text_mask: The text tokens T, broadcastable to `attention`.
        weights: The LLM's carried weights w, zero outside T, broadcastable to `attention`.
        beta: The strength; 0 returns the row unchanged.
        bias: "mass_preserving" keeps the head's attention on the text at its own total m and
            moves it within the text to r' proportional to r w^beta. "plain"
            multiplies each text token's attention by (|T| w)^beta and renormalizes the whole row.

    Returns:
        The relayed row, of the same shape as `attention`.
    """
    if bias not in BIASES:
        raise ValueError(f"bias must be one of {BIASES}")
    zero = torch.zeros_like(attention)
    on_text = torch.where(text_mask, attention, zero)
    mass = on_text.sum(dim=-1, keepdim=True)
    if bias == "plain" and beta != 0:
        n_text = text_mask.sum(dim=-1, keepdim=True).float()
        factor = (n_text * weights).clamp_min(FLOOR) ** beta
        scaled = torch.where(text_mask, attention * factor, attention)
        return scaled / scaled.sum(dim=-1, keepdim=True)

    share = on_text / mass.clamp_min(1e-30)  # r: the head's own distribution over the text
    if beta != 0:
        share = torch.where(text_mask, share * weights.clamp_min(FLOOR) ** beta, zero)
        share = share / share.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    return torch.where(text_mask, mass * share, attention)


@dataclass(frozen=True)
class Encoding:
    """A text in the embedder's tokens."""

    ids: list[int]
    offsets: list[tuple[int, int]]
    """The character span of each token in `prefix + text`."""
    special: list[bool]
    text_mask: list[bool]
    """The tokens the relay acts on: the text's tokens, without special tokens and the prefix,
    and in last-token pooling without the first token, which serves as the attention sink."""


class Embedder:
    """An instruction-agnostic embedding model whose pooling can take the LLM's weights."""

    def __init__(
        self,
        model: str | PreTrainedModel,
        tokenizer=None,
        *,
        pooling: str | None = None,
        revision: str | None = None,
        prefix: str = "",
        max_tokens: int | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = False,
        code_revision: str | None = None,
    ):
        """Load the embedder and, for CLS or last-token pooling, install the relay in its attention.

        Args:
            model: A Hugging Face repository id, a local directory or a loaded model. For CLS and
                last-token pooling, a loaded model's attention is switched in place through its
                config, so other models sharing that config object switch too.
            tokenizer: The tokenizer, when `model` is a loaded model.
            pooling: "mean", "cls" or "last"; read from the model's sentence-transformers
                configuration when not given.
            revision: The model's revision.
            prefix: A fixed string the embedder reads before every text, such as e5's "query: ".
            max_tokens: Texts are truncated to this many embedder tokens. Defaults to the
                sentence-transformers max_seq_length, else the tokenizer's limit.
            device: Defaults to CUDA when available, otherwise the CPU.
            dtype: Defaults to float32 for mean pooling, and for CLS and last-token pooling to
                bfloat16 on CUDA and float32 elsewhere.
            trust_remote_code: Allow custom modeling code, as gte-large-en-v1.5 needs.
            code_revision: The revision of that code.
        """
        settings = (
            {} if isinstance(model, PreTrainedModel) else _sentence_transformers(model, revision)
        )
        self.pooling = pooling or settings.get("pooling")
        if self.pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}; pass it explicitly")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            on_cuda = torch.device(device).type == "cuda"
            dtype = torch.bfloat16 if on_cuda and self.pooling != "mean" else torch.float32

        if isinstance(model, str):
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model, revision=revision)
            options = {"revision": revision, "dtype": dtype, "trust_remote_code": trust_remote_code}
            if code_revision is not None:
                options["code_revision"] = code_revision
            model = load_model(model, **options)
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        if gte.is_gte(self.model):
            gte.repair(self.model)

        self.prefix = prefix
        self.max_tokens = max_tokens or settings.get("max_seq_length")
        if self.max_tokens is None and tokenizer is not None:
            self.max_tokens = tokenizer.model_max_length
        self.first_token_is_sink = self.pooling == "last"
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        self.pad_token_id = pad_token_id if pad_token_id is not None else 0

        if self.pooling != "mean":
            if gte.is_gte(self.model):
                gte.install_relay(self.model)
            elif self.model._supports_attention_backend:
                use_relay_attention(self.model)
            else:
                raise ValueError(
                    "CLS and last-token pooling need a model whose attention goes through "
                    "transformers' attention interface"
                )

    @property
    def n_layers(self) -> int:
        return self.model.config.num_hidden_layers

    def tokenize(self, texts: Sequence[str]) -> list[Encoding]:
        """Tokenize texts, each after the prefix, and mark the tokens the relay acts on."""
        encodings = []
        for text in texts:
            encoded = self.tokenizer(
                self.prefix + text,
                truncation=True,
                max_length=self.max_tokens,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
            )
            special = [bool(flag) for flag in encoded["special_tokens_mask"]]
            offsets = [tuple(span) for span in encoded["offset_mapping"]]
            text_mask = [
                not is_special and end > len(self.prefix)
                for is_special, (_, end) in zip(special, offsets, strict=True)
            ]
            if self.first_token_is_sink and text_mask:
                text_mask[0] = False
            encodings.append(Encoding(list(encoded["input_ids"]), offsets, special, text_mask))
        return encodings

    def embed(
        self,
        encodings: Sequence[Encoding],
        weights: Sequence[np.ndarray] | None = None,
        *,
        beta: float = 1.0,
        bias: str = "mass_preserving",
        layers: Collection[int] | None = None,
        max_batch_tokens: int = 16384,
    ) -> np.ndarray:
        """Embed texts, relaying weights on the embedder's tokens into its pooling.

        Args:
            encodings: The texts, from `tokenize`.
            weights: For each text, weights over its embedder tokens, such as the LLM's weights from
                `carry_weights`. None gives the embedder as released.
            beta: The relay's strength.
            bias: "mass_preserving" or "plain"; used by CLS and last-token pooling.
            layers: The layers the relay acts in, for CLS and last-token pooling; None means all.
            max_batch_tokens: The most tokens in one batch, padding included.

        Returns:
            L2-normalized embeddings of shape [len(encodings), dimension].
        """
        if weights is not None and len(weights) != len(encodings):
            raise ValueError("pass one weight vector per text")
        if self.pooling == "mean":
            if layers is not None:
                raise ValueError("a layer range needs CLS or last-token pooling")
            states = self.token_states(encodings, max_batch_tokens=max_batch_tokens)
            return mean_pool(states, encodings, weights, beta)
        return self._pooled_states(encodings, weights, beta, bias, layers, max_batch_tokens)

    def token_states(
        self, encodings: Sequence[Encoding], *, max_batch_tokens: int = 16384
    ) -> list[np.ndarray]:
        """The final state of every token of every text, in float32, as the embedder computes it."""
        states: list[np.ndarray | None] = [None] * len(encodings)
        for batch, ids, attention_mask in self._batches(encodings, max_batch_tokens):
            with torch.no_grad():
                hidden = self.model(input_ids=ids, attention_mask=attention_mask)[0]
            hidden = hidden.float().cpu().numpy()
            for row, index in enumerate(batch):
                states[index] = hidden[row, : len(encodings[index].ids)]
        return states

    def pooling_attention(
        self, encodings: Sequence[Encoding], *, max_batch_tokens: int = 16384
    ) -> list[np.ndarray]:
        """The pooling token's own attention in every layer, as the relay sees it before acting.

        Returns:
            For each text, an array of shape [layers, heads, tokens].
        """
        if self.pooling == "mean":
            raise ValueError("mean pooling has no pooling token")
        rows: list[np.ndarray | None] = [None] * len(encodings)
        for batch, ids, attention_mask in self._batches(encodings, max_batch_tokens):
            relay = self._relay(batch, encodings, attention_mask, None, 0.0, BIASES[0], None)
            relay.record = {}
            self._forward(ids, attention_mask, relay)
            recorded = torch.stack([relay.record[layer] for layer in sorted(relay.record)], dim=1)
            for row, index in enumerate(batch):
                length = len(encodings[index].ids)
                rows[index] = recorded[row, :, :, :length].cpu().numpy()
        return rows

    def _pooled_states(self, encodings, weights, beta, bias, layers, max_batch_tokens):
        embeddings = np.zeros((len(encodings), self.model.config.hidden_size), np.float32)
        for batch, ids, attention_mask in self._batches(encodings, max_batch_tokens):
            relay = None
            if weights is not None:
                relay = self._relay(batch, encodings, attention_mask, weights, beta, bias, layers)
            hidden = self._forward(ids, attention_mask, relay)
            positions = self._pooling_positions(batch, encodings).to(hidden.device)
            pooled = hidden[torch.arange(len(batch), device=hidden.device), positions].float()
            embeddings[batch] = (pooled / pooled.norm(dim=-1, keepdim=True)).cpu().numpy()
        return embeddings

    def _batches(self, encodings, max_batch_tokens):
        """Batches of texts, longest first, padded on the right."""
        lengths = [len(encoding.ids) for encoding in encodings]
        for batch in batches(lengths, max_batch_tokens, max_size=len(lengths)):
            ids = pad([encodings[index].ids for index in batch], self.pad_token_id, "right")
            attention_mask = pad([[1] * lengths[index] for index in batch], 0, "right")
            yield batch, ids.to(self.model.device), attention_mask.to(self.model.device)

    def _pooling_positions(self, batch, encodings) -> torch.Tensor:
        if self.pooling == "cls":
            return torch.zeros(len(batch), dtype=torch.long)
        return torch.tensor([len(encodings[index].ids) - 1 for index in batch])

    def _relay(self, batch, encodings, attention_mask, weights, beta, bias, layers):
        device = self.model.device
        text_mask = pad([encodings[index].text_mask for index in batch], False, "right", torch.bool)
        carried = None
        if weights is not None:
            carried = pad([weights[index] for index in batch], 0.0, "right", torch.float32)
            carried = carried.to(device)
        return Relay(
            positions=self._pooling_positions(batch, encodings).to(device),
            text_mask=text_mask.to(device),
            weights=carried,
            key_mask=attention_mask.bool(),
            beta=float(beta),
            bias=bias,
            layers=None if layers is None else frozenset(layers),
        )

    def _forward(self, ids, attention_mask, relay):
        with torch.no_grad():
            if gte.is_gte(self.model):
                with gte.relaying(self.model, relay):
                    return self.model(input_ids=ids, attention_mask=attention_mask)[0]
            return self.model(input_ids=ids, attention_mask=attention_mask, relay=relay)[0]


def mean_pool(
    states: Sequence[np.ndarray],
    encodings: Sequence[Encoding],
    weights: Sequence[np.ndarray] | None = None,
    beta: float = 1.0,
) -> np.ndarray:
    """Mean-pooled, L2-normalized embeddings.

    Without weights, every token counts equally, as in the released model. With weights, the
    pooling weights are proportional to weight ** beta on the text's tokens and zero elsewhere.
    """
    pooled = []
    for index, token_states in enumerate(states):
        if weights is None:
            pooling_weights = np.full(len(token_states), 1.0 / len(token_states))
        else:
            text_mask = np.array(encodings[index].text_mask)
            lifted = np.where(text_mask, np.asarray(weights[index], np.float64) ** beta, 0.0)
            pooling_weights = lifted / lifted.sum()
        pooled.append(pooling_weights @ token_states)
    pooled = np.stack(pooled)
    return (pooled / np.linalg.norm(pooled, axis=1, keepdims=True)).astype(np.float32)


@dataclass
class Relay:
    """What one batch's forward pass relays; the attention reads it at every layer."""

    positions: torch.Tensor
    """The pooling token of each sequence, shape [batch]."""
    text_mask: torch.Tensor
    """The tokens the relay acts on, shape [batch, tokens]."""
    weights: torch.Tensor | None
    """The weights to relay, shape [batch, tokens]; None only records the pooling rows."""
    key_mask: torch.Tensor
    """The non-padding tokens, shape [batch, tokens]."""
    beta: float
    bias: str
    layers: frozenset[int] | None
    """The layers the relay acts in; None means all."""
    record: dict[int, torch.Tensor] | None = None
    """When a dict, the pooling token's own attention rows, by layer."""
    calls: int = field(default=0)

    def layer_of(self, module) -> int:
        """The layer of an attention call: the module's own index, else the order of the calls."""
        index = getattr(module, "layer_idx", None)
        index = self.calls if index is None else index
        self.calls += 1
        return index

    def acts_in(self, layer: int) -> bool:
        return self.layers is None or layer in self.layers

    def pooling_rows(self, query, key, value, allowed, scaling, softcap, layer):
        """The attention outputs at the pooling token after relaying, [batch, heads, head_dim];
        None when the relay only records."""
        attention = attention_row(query, key, self.positions, allowed, scaling, softcap)
        if self.record is not None:
            self.record[layer] = attention
        if self.weights is None:
            return None
        relayed = relay_row(
            attention, self.text_mask[:, None, :], self.weights[:, None, :], self.beta, self.bias
        )
        return weighted_values(relayed, value)


def use_relay_attention(model: PreTrainedModel) -> None:
    """Switch a loaded model to the attention implementation that relays at the pooling token."""
    model.set_attn_implementation(RELAY_SDPA if model._supports_sdpa else RELAY_EAGER)


def _relaying_attention(underlying):
    """An attention function that runs `underlying`, then relays at the pooling token."""

    def attention(module, query, key, value, attention_mask, **options):
        relay = options.pop("relay", None)
        output, attention_weights = underlying(module)(
            module, query, key, value, attention_mask, **options
        )
        if relay is None:
            return output, attention_weights
        layer = relay.layer_of(module)
        if not relay.acts_in(layer):
            return output, attention_weights
        if options.get("s_aux") is not None:
            raise NotImplementedError("relaying into a model with attention sinks")
        allowed = allowed_keys(attention_mask, relay.positions)
        rows = relay.pooling_rows(
            query, key, value, allowed, options.get("scaling"), options.get("softcap"), layer
        )
        if rows is None:
            return output, attention_weights
        output = output.clone()  # [batch, queries, heads, head_dim]
        batch = torch.arange(len(relay.positions), device=output.device)
        output[batch, relay.positions] = rows.to(output.dtype)
        return output, attention_weights

    return attention


AttentionInterface.register(RELAY_SDPA, _relaying_attention(lambda module: sdpa_attention_forward))
AttentionMaskInterface.register(RELAY_SDPA, sdpa_mask)
AttentionInterface.register(RELAY_EAGER, _relaying_attention(model_eager_attention))
AttentionMaskInterface.register(RELAY_EAGER, eager_mask)


def _sentence_transformers(model: str, revision: str | None) -> dict:
    """The pooling mode and maximum length from a sentence-transformers configuration, if any."""

    def read(filename):
        path = cached_file(
            model, filename, revision=revision, _raise_exceptions_for_missing_entries=False
        )
        if path is None:
            return None
        with open(path) as handle:
            return json.load(handle)

    settings = {}
    modules = read("modules.json") or []
    kinds = [module["type"].rsplit(".", 1)[-1] for module in modules]
    unsupported = [kind for kind in kinds if kind not in ("Transformer", "Pooling", "Normalize")]
    if unsupported:
        raise ValueError(f"unsupported sentence-transformers modules: {unsupported}")
    for module in modules:
        if module["type"].endswith("Pooling"):
            config = read(f"{module['path']}/config.json") or {}
            modes = [_POOLING_MODES[key] for key in _POOLING_MODES if config.get(key)]
            if len(modes) == 1:
                settings["pooling"] = modes[0]
    config = read("sentence_bert_config.json") or {}
    if "max_seq_length" in config:
        settings["max_seq_length"] = config["max_seq_length"]
    return settings
