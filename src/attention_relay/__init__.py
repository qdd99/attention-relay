"""Attention Relay: instruction-aware text embeddings from instruction-agnostic embedding models."""

from attention_relay.align import alignment_matrix, carry_weights
from attention_relay.blocks import (
    KNOWN_BLOCKS,
    choose_block,
    default_block,
    question_sensitivity,
    select_block,
)
from attention_relay.model import AttentionRelay
from attention_relay.reading import LLMReader, Reading
from attention_relay.relay import Embedder, Encoding, relay_row

__all__ = [
    "KNOWN_BLOCKS",
    "AttentionRelay",
    "Embedder",
    "Encoding",
    "LLMReader",
    "Reading",
    "alignment_matrix",
    "carry_weights",
    "choose_block",
    "default_block",
    "question_sensitivity",
    "relay_row",
    "select_block",
]
__version__ = "0.1.0.dev0"
