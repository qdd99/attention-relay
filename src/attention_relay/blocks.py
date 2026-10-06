"""The block the reading weights come from.

The block is set once per LLM and is the same for every embedder and dataset. `select_block` sets
it with a rule that reads no labels: the block, in the LLM's second half and before its last block,
where the reading weights move most when the question about the same texts changes. For an LLM
outside `KNOWN_BLOCKS`, `default_block` reads about 70% of the way up.
"""

import math
from collections.abc import Sequence

import numpy as np

from attention_relay.reading import LLMReader

KNOWN_BLOCKS = {
    "Qwen/Qwen3-0.6B": 24,
    "Qwen/Qwen3-1.7B": 24,
    "Qwen/Qwen3-4B": 29,
    "Qwen/Qwen3-8B": 29,
    "meta-llama/Llama-3.1-8B-Instruct": 24,
    "allenai/Olmo-3-7B-Instruct": 23,
}
"""Blocks, counted from 0, already chosen with `select_block` for some instruction-tuned LLMs."""


def default_block(n_blocks: int) -> int:
    """A block about 70% of the way up the LLM: the default for an LLM outside `KNOWN_BLOCKS`. Any
    block in the LLM's upper half carries the instruction; `select_block` finds the most
    question-sensitive one."""
    return min(round(0.7 * n_blocks), n_blocks - 1)


def question_sensitivity(
    reader: LLMReader, texts: Sequence[str], questions: tuple[str, str], **read_options
) -> np.ndarray:
    """How much the reading weights move with the question, at every block.

    Args:
        reader: The LLM.
        texts: Texts to read, each under both questions.
        questions: Two questions about the texts.
        read_options: Passed on to `LLMReader.read`.

    Returns:
        For every block, the total variation distance between the reading weights under the two
        questions, averaged over the texts.
    """
    first, second = questions
    blocks = range(reader.n_blocks)
    under_first = reader.read(texts, first, blocks, **read_options)
    under_second = reader.read(texts, second, blocks, **read_options)
    distances = [
        0.5 * np.abs(a.weights - b.weights).sum(axis=1)
        for a, b in zip(under_first, under_second, strict=True)
    ]
    return np.mean(distances, axis=0)


def choose_block(sensitivity: Sequence[float]) -> int:
    """The most question-sensitive block in the LLM's second half, excluding the last block."""
    n_blocks = len(sensitivity)
    candidates = range(math.ceil(n_blocks / 2), n_blocks - 1)
    return max(candidates, key=lambda block: sensitivity[block])


def select_block(
    reader: LLMReader, *datasets: tuple[Sequence[str], tuple[str, str]], **read_options
) -> int:
    """Choose an LLM's block: the most question-sensitive one in its second half, before its last
    block. No labels are read.

    Args:
        reader: The LLM.
        datasets: One or more pairs of texts and two questions about them. The question
            sensitivity is averaged over the datasets.
        read_options: Passed on to `LLMReader.read`.
    """
    curves = [
        question_sensitivity(reader, texts, questions, **read_options)
        for texts, questions in datasets
    ]
    return choose_block(np.mean(curves, axis=0))
