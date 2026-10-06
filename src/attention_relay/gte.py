"""Support for gte-large-en-v1.5 and its siblings.

These models run custom modeling code (Alibaba-NLP/new-impl) written for transformers 4. Under
transformers 5 that code needs two repairs: its non-persistent buffers (position ids and the rotary
tables) come out of loading uninitialized, and a mask helper it calls no longer exists. Its
attention also bypasses transformers' attention interface, so the relay is installed by wrapping
each layer's attention, and a relay reaches it through `relaying` rather than as an argument of
the forward call.
"""

import math
import types
from contextlib import contextmanager

import torch


def is_gte(model) -> bool:
    """Whether a loaded model runs gte's custom code."""
    return type(model).__name__ == "NewModel" and hasattr(model, "encoder")


def repair(model) -> None:
    """Rebuild the buffers transformers 5 leaves uninitialized, and restore the removed mask helper.

    The buffers are computed exactly as the model's own constructor computes them.
    """
    model.get_extended_attention_mask = types.MethodType(_extended_attention_mask, model)
    embeddings, config = model.embeddings, model.config
    device = embeddings.word_embeddings.weight.device
    embeddings.position_ids = torch.arange(config.max_position_embeddings, device=device)

    rotary = embeddings.rotary_emb
    options = {
        "dim": int(config.hidden_size / config.num_attention_heads),
        "max_position_embeddings": config.max_position_embeddings,
        "base": config.rope_theta,
    }
    if config.rope_scaling is not None:
        options["scaling_factor"] = config.rope_scaling["factor"]
        options["mixed_b"] = config.rope_scaling.get("mixed_b", None)
    with torch.device("cpu"):
        fresh = type(rotary)(**options)
    rotary.inv_freq = fresh.inv_freq.to(device)
    rotary.cos_cached = fresh.cos_cached.to(device)
    rotary.sin_cached = fresh.sin_cached.to(device)
    rotary.max_seq_len_cached = fresh.max_seq_len_cached


def install_relay(model) -> None:
    """Wrap each layer's attention so that a relay given with `relaying` acts at pooling."""
    for index, layer in enumerate(model.encoder.layer):
        module = layer.attention
        scaling = 1.0 / math.sqrt(module.attention_head_size)
        module._attention = _relaying(model, module._attention, index, scaling)


@contextmanager
def relaying(model, relay):
    """Make `relay` the relay of every forward call of `model` inside the block."""
    model._attention_relay = relay
    try:
        yield
    finally:
        model._attention_relay = None


def _relaying(model, original, index, scaling):
    def attention(query, key, value, attention_bias, head_mask):
        # gte's query, key and value are [batch, tokens, heads, head_dim], after its rotary
        context, probabilities = original(query, key, value, attention_bias, head_mask)
        relay = getattr(model, "_attention_relay", None)
        if relay is None or not relay.acts_in(index):
            return context, probabilities
        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
        rows = relay.pooling_rows(query, key, value, relay.key_mask, scaling, None, index)
        if rows is None:
            return context, probabilities
        context = context.clone()
        batch = torch.arange(len(relay.positions), device=context.device)
        context[batch, relay.positions] = rows.to(context.dtype)
        return context, probabilities

    return attention


def _extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
    """transformers 4's get_extended_attention_mask for an encoder, which gte's code calls."""
    dtype = dtype or next(self.parameters()).dtype
    if attention_mask.dim() == 3:
        extended = attention_mask[:, None, :, :]
    else:
        extended = attention_mask[:, None, None, :]
    extended = extended.to(dtype=dtype)
    return (1.0 - extended) * torch.finfo(dtype).min
