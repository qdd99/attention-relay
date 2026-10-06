"""Tests for the character alignment."""

import json
from pathlib import Path

import numpy as np
import pytest

from attention_relay import alignment_matrix, carry_weights

# Real tokenizations of seven texts (with an emoji, accents, Japanese, extra whitespace and a URL)
# by Qwen3-1.7B and five embedders; the expected values come from an earlier, independent
# implementation.
FIXTURE = json.loads((Path(__file__).parent / "data" / "alignment_cases.json").read_text())


@pytest.mark.parametrize(
    "case", FIXTURE["cases"], ids=lambda c: f"{c['embedder']}-{c['text'][:16]}"
)
def test_matches_the_code_behind_the_paper(case):
    matrix = alignment_matrix(
        case["text"],
        case["llm_offsets"],
        case["embedder_offsets"],
        case["embedder_special"],
        case["prefix_length"],
    )
    np.testing.assert_array_equal(matrix, np.array(case["expected_matrix"]))

    weights = carry_weights(matrix, np.array(case["llm_weights"]), np.array(case["text_mask"]))
    np.testing.assert_array_equal(weights, np.array(case["expected_weights"]))


def test_identical_tokenizations_give_the_identity():
    text = "Hello world"
    offsets = [(0, 5), (5, 11)]  # " world" carries its leading space, as in byte-level tokenizers
    matrix = alignment_matrix(text, offsets, offsets, [False, False])
    np.testing.assert_array_equal(matrix, np.eye(2))


def test_special_tokens_and_the_prefix_get_no_weight():
    prefix = "query: "
    # [CLS] query : hello world [SEP], with spans in prefix + text
    embedder_offsets = [(0, 0), (0, 5), (5, 6), (7, 12), (13, 18), (0, 0)]
    special = [True, False, False, False, False, True]
    matrix = alignment_matrix(
        "Hello world", [(0, 5), (5, 11)], embedder_offsets, special, len(prefix)
    )
    expected = np.zeros((6, 2))
    expected[3, 0] = 1.0
    expected[4, 1] = 1.0
    np.testing.assert_array_equal(matrix, expected)


def test_a_token_split_in_two_shares_its_weight_by_characters():
    matrix = alignment_matrix("unhappy", [(0, 7)], [(0, 2), (2, 7)], [False, False])
    np.testing.assert_allclose(matrix, [[2 / 7], [5 / 7]])


def test_a_whitespace_token_carries_no_weight():
    llm_offsets = [(0, 1), (1, 4), (4, 5)]  # "a", " \n\n", "b"
    matrix = alignment_matrix("a \n\nb", llm_offsets, [(0, 1), (4, 5)], [False, False])
    np.testing.assert_array_equal(matrix, [[1, 0, 0], [0, 0, 1]])


def test_an_empty_text_gives_an_empty_matrix():
    matrix = alignment_matrix("", [], [(0, 0), (0, 0)], [True, True])
    assert matrix.shape == (2, 0)


def test_carried_weights_sum_to_one_on_the_text():
    matrix = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    weights = carry_weights(matrix, np.array([0.2, 0.6]), np.array([False, True, True, False]))
    np.testing.assert_allclose(weights, [0.0, 0.25, 0.75, 0.0])


def test_carried_weights_fall_back_to_uniform_when_nothing_lands():
    weights = carry_weights(np.zeros((3, 1)), np.array([1.0]), np.array([False, True, True]))
    np.testing.assert_allclose(weights, [0.0, 0.5, 0.5])


def test_an_empty_text_mask_is_an_error():
    with pytest.raises(ValueError, match="selects no embedder token"):
        carry_weights(np.zeros((2, 1)), np.array([1.0]), np.array([False, False]))
