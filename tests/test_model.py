"""Tests for AttentionRelay and the choice of block."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import transformers

from attention_relay import (
    AttentionRelay,
    Embedder,
    LLMReader,
    choose_block,
    default_block,
    question_sensitivity,
    select_block,
)
from attention_relay.reading import truncate_text

REFERENCE = json.loads((Path(__file__).parent / "data" / "reference_reading.json").read_text())
TEXTS = [
    "It's really confusing, how exactly are you determining your exchange rates?",
    "WASHINGTON - The Senate voted on Tuesday to confirm the nominee, ending a long standoff.",
    "Great service but the app crashed twice, and nobody answered my emails.",
]
QUESTIONS = ("How does the customer feel?", "What does the customer need?")


def cached_tokenizer(repository, revision):
    try:
        return transformers.AutoTokenizer.from_pretrained(
            repository, revision=revision, local_files_only=True
        )
    except OSError:
        pytest.skip(f"the {repository} tokenizer is not in the local cache")


@pytest.fixture(scope="module")
def small_relay():
    """Random-weight models with real tokenizers: a small Qwen3 LLM and a small BERT embedder."""
    llm_tokenizer = cached_tokenizer("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca")
    embedder_tokenizer = cached_tokenizer(
        "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    )
    torch.manual_seed(0)
    llm = transformers.AutoModel.from_config(
        transformers.Qwen3Config(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=len(llm_tokenizer),
        )
    )
    embedder = transformers.AutoModel.from_config(
        transformers.BertConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=3,
            num_attention_heads=4,
            vocab_size=len(embedder_tokenizer),
        )
    )
    reader = LLMReader(llm, llm_tokenizer, device="cpu", max_text_tokens=12)
    return AttentionRelay(
        reader, Embedder(embedder, embedder_tokenizer, pooling="cls", device="cpu"), block=2
    )


def test_embeddings_are_unit_vectors_that_follow_the_question(small_relay):
    first = small_relay.encode(TEXTS, QUESTIONS[0])
    second = small_relay.encode(TEXTS, QUESTIONS[1])
    assert first.shape == (len(TEXTS), 64)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, rtol=1e-5)
    assert np.linalg.norm(first - second, axis=1).min() > 1e-4


def test_without_an_instruction_the_embedder_reads_the_same_truncated_text(small_relay):
    reader, embedder = small_relay.reader, small_relay.embedder
    truncated = [truncate_text(reader.tokenizer, text, reader.max_text_tokens) for text in TEXTS]
    expected = embedder.embed(embedder.tokenize(truncated))
    np.testing.assert_allclose(small_relay.encode(TEXTS), expected, atol=1e-6)
    assert all(text.startswith(short) for text, short in zip(TEXTS, truncated, strict=True))


def test_one_instruction_per_text(small_relay):
    per_text = small_relay.encode(TEXTS, [QUESTIONS[0]] * len(TEXTS))
    np.testing.assert_allclose(per_text, small_relay.encode(TEXTS, QUESTIONS[0]), atol=1e-6)


def test_an_llm_without_a_known_block_is_read_about_70_percent_of_the_way_up(small_relay):
    relay = AttentionRelay(small_relay.reader, small_relay.embedder)
    assert relay.block == default_block(small_relay.reader.n_blocks)


def test_the_default_block_is_about_70_percent_of_the_way_up():
    assert [default_block(n) for n in (28, 32, 36)] == [20, 22, 25]
    assert default_block(1) == 0


def test_the_block_rule_picks_the_most_sensitive_block_of_the_second_half():
    sensitivity = np.zeros(28)
    sensitivity[5] = 0.9  # first half: never chosen
    sensitivity[27] = 0.8  # the last block: never chosen
    sensitivity[[17, 24]] = 0.5  # a tie goes to the earlier block
    assert choose_block(sensitivity) == 17
    sensitivity[24] = 0.6
    assert choose_block(sensitivity) == 24
    assert choose_block(np.arange(7)) == 5  # 7 blocks: candidates 4 and 5


def test_question_sensitivity_is_zero_for_one_question_and_positive_for_two(small_relay):
    reader = small_relay.reader
    same = question_sensitivity(reader, TEXTS, (QUESTIONS[0], QUESTIONS[0]))
    different = question_sensitivity(reader, TEXTS, QUESTIONS)
    assert same.shape == different.shape == (reader.n_blocks,)
    np.testing.assert_allclose(same, 0.0, atol=1e-12)
    assert (different > 0).all() and (different <= 1).all()
    by_hand = choose_block((different + different) / 2)
    assert select_block(reader, (TEXTS, QUESTIONS), (TEXTS, QUESTIONS)) == by_hand


def test_a_stored_reading_is_reproduced():
    """Qwen3-1.7B's reading weights for one customer message match a stored reference."""
    try:
        reader = LLMReader(REFERENCE["model"], revision=REFERENCE["revision"], device="cpu")
    except OSError:
        pytest.skip("Qwen3-1.7B is not in the local cache")
    readings = [
        reader.read([REFERENCE["text"]], q, blocks=REFERENCE["block"])[0] for q in QUESTIONS
    ]
    assert list(QUESTIONS) == REFERENCE["questions"]
    for reading, stored in zip(readings, REFERENCE["weights"], strict=True):
        assert reading.text == REFERENCE["text"]
        assert [list(span) for span in reading.offsets] == REFERENCE["offsets"]
        assert [reading.text[start:end] for start, end in reading.offsets] == REFERENCE["tokens"]
        # the reference was computed in bfloat16 on a GPU; the reader here runs in float32
        np.testing.assert_allclose(reading.at(REFERENCE["block"]), stored, atol=0.01)
