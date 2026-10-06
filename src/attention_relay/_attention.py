"""One row of a model's attention, recomputed from the queries and keys its attention gets."""

import sys

import torch


def attention_row(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    allowed_keys: torch.Tensor | None,
    scaling: float | None,
    softcap: float | None = None,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    """The attention weights of one query position per sequence over all keys, in float32.

    Args:
        query: Queries of shape [batch, heads, queries, head_dim], as the attention call gets them.
        key: Keys of shape [batch, kv_heads, keys, head_dim]; each key-value head serves
            heads // kv_heads consecutive query heads, as in transformers' repeat_kv.
        positions: The query position of each sequence, shape [batch].
        allowed_keys: The keys each position may attend to, shape [batch, keys]; None allows all.
        scaling: The factor on the logits; None means head_dim ** -0.5.
        softcap: The model's logit soft-cap, if it has one.
        sinks: The model's per-head attention-sink logits, if it has them; their weight is left out.

    Returns:
        Attention weights of shape [batch, heads, keys].
    """
    batch, heads, _, head_dim = query.shape
    kv_heads = key.shape[1]
    group = heads // kv_heads
    rows = query[torch.arange(batch, device=query.device), :, positions]  # [batch, heads, head_dim]
    rows = rows.float().reshape(batch, kv_heads, group, head_dim)
    logits = torch.einsum("bkgd,bkld->bkgl", rows, key.float()).reshape(batch, heads, -1)
    logits = logits * (head_dim**-0.5 if scaling is None else scaling)

    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    if allowed_keys is not None:
        logits = logits.masked_fill(~allowed_keys[:, None, :], float("-inf"))

    if sinks is None:
        return logits.softmax(dim=-1)
    sink_logits = sinks.float().reshape(1, heads, 1).expand(batch, heads, 1)
    return torch.cat([logits, sink_logits], dim=-1).softmax(dim=-1)[..., :-1]


def allowed_keys(attention_mask: torch.Tensor | None, positions: torch.Tensor):
    """The keys each query position may attend to, from the 4-D mask a model passes its attention.

    The mask has shape [batch, 1, queries, keys] and is either boolean (True where attention is
    allowed) or additive (0 where it is allowed). None means every key is allowed.
    """
    if attention_mask is None:
        return None
    rows = attention_mask[torch.arange(len(positions), device=attention_mask.device), 0, positions]
    return rows if rows.dtype == torch.bool else rows == 0


def weighted_values(weights: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Attention outputs from per-head weights [batch, heads, keys] and values [batch, kv_heads,
    keys, head_dim], in float32: shape [batch, heads, head_dim]."""
    batch, heads, n_keys = weights.shape
    kv_heads, head_dim = value.shape[1], value.shape[-1]
    grouped = weights.reshape(batch, kv_heads, heads // kv_heads, n_keys)
    outputs = torch.einsum("bkgl,bkld->bkgd", grouped, value.float())
    return outputs.reshape(batch, heads, head_dim)


def model_eager_attention(module):
    """The eager attention function defined next to an attention module's model."""
    return sys.modules[type(module).__module__].eager_attention_forward
