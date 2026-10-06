"""The LLM's reading weights: where it looks in a text when it is about to answer.

The LLM reads the text followed by the instruction in its chat template, and the last position of
the prompt, where the model is about to answer, is the answer position. At block b, the reading
weights are the answer position's attention on each text token, summed over the block's heads and
renormalized over the text. Because the text precedes the instruction, the causal mask keeps the
text's states identical under every instruction; the instruction changes only where the answer
position looks.

The weights are read inside the model's own attention call, through transformers' attention
interface: the queries and keys arrive there after the model has applied its projections, norms
and rotary embeddings, so the same code serves every architecture that routes attention through
the interface.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AttentionInterface, AutoTokenizer, PreTrainedModel
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, eager_mask, sdpa_mask

from attention_relay._attention import allowed_keys, attention_row, model_eager_attention
from attention_relay._batching import batches, pad
from attention_relay._loading import load_model

# The user turn of the prompt: the text first, then the instruction.
TEXT_LABEL = "Text:\n"
INSTRUCTION_SEPARATOR = "\n\n"
CLOSING_SENTENCE = "\n\nAnswer in one word."

# The attention implementations that record reading weights, one per underlying implementation.
READING_SDPA = "attention_relay_reading_sdpa"
READING_EAGER = "attention_relay_reading_eager"


@dataclass(frozen=True)
class Prompt:
    """A prompt in the LLM's tokens, with the text's tokens marked."""

    ids: list[int]
    text_mask: list[bool]


@dataclass(frozen=True)
class Reading:
    """Where the LLM looked in one text when it was about to answer one instruction."""

    text: str
    """The text as the LLM read it, after truncation."""
    offsets: list[tuple[int, int]]
    """The character span of each of the LLM's text tokens in `text`."""
    blocks: tuple[int, ...]
    """The blocks the weights were read at."""
    weights: np.ndarray
    """Reading weights of shape [len(blocks), len(offsets)]; each row sums to one over the text.
    When read with normalize=False, each row holds the answer position's attention on each text
    token, summed over heads but not renormalized."""

    def at(self, block: int) -> np.ndarray:
        """The reading weights at one block."""
        return self.weights[self.blocks.index(block)]


class LLMReader:
    """An instruction-tuned LLM that reads texts under instructions and reports where it looks."""

    def __init__(
        self,
        model: str | PreTrainedModel,
        tokenizer=None,
        *,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        max_text_tokens: int | None = None,
    ):
        """Load the LLM and switch its attention to the recording implementation.

        Args:
            model: A Hugging Face repository id or a loaded model. A loaded model's attention is
                switched in place through its config, so other models sharing that config object
                switch too.
            tokenizer: A repository id or a loaded tokenizer whose chat template frames the prompt.
                Defaults to the model's own. A base checkpoint is read in its instruction-tuned
                sibling's chat template by passing the sibling's repository here.
            revision: The model's revision, when `model` is a repository id.
            tokenizer_revision: The tokenizer's revision, when `tokenizer` is a repository id.
            device: Defaults to CUDA when available, otherwise the CPU.
            dtype: Defaults to bfloat16 on CUDA and float32 elsewhere.
            max_text_tokens: Truncate texts to this many LLM tokens; None (the default) keeps them
                whole.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
        if isinstance(model, str):
            if tokenizer is None:
                tokenizer, tokenizer_revision = model, revision
            model = load_model(model, revision=revision, dtype=dtype)
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, revision=tokenizer_revision)
        if tokenizer is None:
            raise ValueError("pass the tokenizer that goes with a loaded model")
        self.name = model.config.name_or_path
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens
        use_reading_attention(self.model)

    @property
    def n_blocks(self) -> int:
        return self.model.config.num_hidden_layers

    def read(
        self,
        texts: Sequence[str],
        instructions: str | Sequence[str],
        blocks: int | Sequence[int],
        *,
        normalize: bool = True,
        max_batch_tokens: int = 16000,
        max_batch_size: int = 48,
    ) -> list[Reading]:
        """Read each text under its instruction and return the reading weights at the given blocks.

        Args:
            texts: The texts.
            instructions: One instruction for every text, or one per text.
            blocks: A block or a sequence of blocks, counted from 0.
            normalize: Renormalize each row over the text. False keeps the head-summed attention
                on the text tokens as it is, for storage before renormalizing.
            max_batch_tokens: The most tokens in one batch, padding included.
            max_batch_size: The most sequences in one batch.
        """
        if isinstance(instructions, str):
            instructions = [instructions] * len(texts)
        if len(instructions) != len(texts):
            raise ValueError("pass one instruction, or one per text")
        blocks = (blocks,) if isinstance(blocks, int) else tuple(sorted(set(blocks)))

        read_texts = [truncate_text(self.tokenizer, text, self.max_text_tokens) for text in texts]
        prompts = [
            build_prompt(self.tokenizer, text, instruction)
            for text, instruction in zip(read_texts, instructions, strict=True)
        ]
        weights = read_sequences(
            self.model,
            [(prompt.ids, prompt.text_mask) for prompt in prompts],
            blocks,
            pad_token_id=_pad_token_id(self.tokenizer),
            normalize=normalize,
            max_batch_tokens=max_batch_tokens,
            max_batch_size=max_batch_size,
        )

        readings = []
        for text, text_weights in zip(read_texts, weights, strict=True):
            encoding = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = [tuple(span) for span in encoding["offset_mapping"]]
            if len(offsets) != text_weights.shape[1]:
                raise RuntimeError(
                    "the text's tokens in the prompt differ from its own tokenization"
                )
            readings.append(Reading(text, offsets, blocks, text_weights))
        return readings


def truncate_text(tokenizer, text: str, max_tokens: int | None) -> str:
    """The text as the LLM reads it: its first `max_tokens` tokens, decoded back to a string."""
    if max_tokens is None:
        return text
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:max_tokens]
    return tokenizer.decode(ids)


def build_prompt(tokenizer, text: str, instruction: str) -> Prompt:
    """The LLM's prompt for one text and one instruction.

    The model's own chat template frames the user turn "Text: <text> <instruction> Answer in one
    word.", with the template's default system message and thinking turned off where the template
    offers it. The prompt is tokenized piece by piece, so that no token crosses the text's
    boundaries and the text's tokens are exactly its own tokenization. The answer position is the
    prompt's last token.
    """
    message = TEXT_LABEL + text + INSTRUCTION_SEPARATOR + instruction + CLOSING_SENTENCE
    rendered = _render_user_turn(tokenizer, message)
    if rendered.count(message) != 1:
        raise ValueError("the chat template does not contain the user message verbatim")
    before, after = rendered.split(message)
    pieces = [
        (before + TEXT_LABEL, False),
        (text, True),
        (INSTRUCTION_SEPARATOR + instruction, False),
        (CLOSING_SENTENCE, False),
        (after, False),
    ]
    ids, text_mask = [], []
    for piece, is_text in pieces:
        piece_ids = tokenizer(piece, add_special_tokens=False)["input_ids"]
        ids += piece_ids
        text_mask += [is_text] * len(piece_ids)
    return Prompt(ids, text_mask)


def read_sequences(
    model: PreTrainedModel,
    sequences: Sequence[tuple[Sequence[int], Sequence[bool]]],
    blocks: Sequence[int],
    *,
    pad_token_id: int,
    normalize: bool = True,
    max_batch_tokens: int = 16000,
    max_batch_size: int = 48,
) -> list[np.ndarray]:
    """The reading weights of token sequences whose last position is the answer position.

    Args:
        model: A model switched to the recording attention by `use_reading_attention`.
        sequences: Pairs of token ids and a mask of the text's tokens.
        blocks: The blocks to read at.
        pad_token_id: The token that pads shorter sequences on the left.
        normalize: Renormalize over the text; False keeps the head-summed attention as it is.
        max_batch_tokens: The most tokens in one batch, padding included.
        max_batch_size: The most sequences in one batch.

    Returns:
        For each sequence, an array of shape [len(blocks), number of text tokens] whose rows sum
        to one.
    """
    if model.config._attn_implementation not in (READING_SDPA, READING_EAGER):
        raise ValueError("switch the model to the recording attention with use_reading_attention")
    blocks = sorted(set(blocks))
    if not blocks or blocks[0] < 0 or blocks[-1] >= model.config.num_hidden_layers:
        raise ValueError(f"blocks must lie in 0..{model.config.num_hidden_layers - 1}")

    results: list[np.ndarray | None] = [None] * len(sequences)
    lengths = [len(ids) for ids, _ in sequences]
    for batch in batches(lengths, max_batch_tokens, max_batch_size):
        ids = pad([sequences[i][0] for i in batch], pad_token_id, "left").to(model.device)
        attention_mask = pad([[1] * lengths[i] for i in batch], 0, "left").to(model.device)
        text_mask = pad([sequences[i][1] for i in batch], False, "left", torch.bool)
        text_mask = text_mask.to(model.device)
        recorder = _Recorder(blocks, text_mask)
        try:
            with torch.no_grad():
                # the recorder travels with the call, through every layer, to the attention
                model(
                    input_ids=ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    reading_recorder=recorder,
                )
        except _ReadingDone:
            pass
        missing = [block for block in blocks if block not in recorder.text_attention]
        if missing:
            raise RuntimeError(
                f"blocks {missing} were never read; the model's attention must go through "
                "transformers' attention interface"
            )
        for row, index in enumerate(batch):
            per_block = torch.stack(
                [recorder.text_attention[block][row][text_mask[row]] for block in blocks]
            )
            per_block = per_block.cpu().double()  # float64 on the CPU; some devices lack it
            if normalize:
                per_block = per_block / per_block.sum(dim=-1, keepdim=True)
            results[index] = per_block.numpy()
    return results


def use_reading_attention(model: PreTrainedModel) -> None:
    """Switch a loaded model to the attention implementation that records reading weights."""
    model.set_attn_implementation(READING_SDPA if model._supports_sdpa else READING_EAGER)


class _ReadingDone(Exception):
    """Raised after the last requested block, to skip the rest of the forward pass."""


class _Recorder:
    """Records the answer position's attention on the text at the requested blocks of one batch."""

    def __init__(self, blocks: Sequence[int], text_mask: torch.Tensor):
        self.blocks = set(blocks)
        self.last_block = max(blocks)
        self.text_mask = text_mask
        self.text_attention: dict[int, torch.Tensor] = {}

    def observe(self, module, query, key, attention_mask, options) -> None:
        block = module.layer_idx
        if block not in self.blocks:
            return
        last = torch.full((query.shape[0],), query.shape[2] - 1, device=query.device)
        weights = attention_row(
            query,
            key,
            last,
            allowed_keys(attention_mask, last),
            options.get("scaling"),
            options.get("softcap"),
            options.get("s_aux"),
        )
        self.text_attention[block] = (weights * self.text_mask[:, None, :]).sum(dim=1)
        if block == self.last_block:
            raise _ReadingDone


def _recording_attention(underlying):
    """An attention function that lets a recorder observe the call, then runs `underlying`."""

    def attention(module, query, key, value, attention_mask, **options):
        recorder = options.pop("reading_recorder", None)
        if recorder is not None:
            recorder.observe(module, query, key, attention_mask, options)
        return underlying(module)(module, query, key, value, attention_mask, **options)

    return attention


AttentionInterface.register(
    READING_SDPA, _recording_attention(lambda module: sdpa_attention_forward)
)
AttentionMaskInterface.register(READING_SDPA, sdpa_mask)
AttentionInterface.register(READING_EAGER, _recording_attention(model_eager_attention))
AttentionMaskInterface.register(READING_EAGER, eager_mask)


def _render_user_turn(tokenizer, message: str) -> str:
    conversation = [{"role": "user", "content": message}]
    options = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(conversation, enable_thinking=False, **options)
    except TypeError:
        return tokenizer.apply_chat_template(conversation, **options)


def _pad_token_id(tokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    return tokenizer.eos_token_id
