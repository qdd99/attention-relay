"""Carry weights from the LLM's tokens to the embedder's tokens through characters.

The LLM and the embedder split the same text into different tokens. Each LLM token's weight is
spread evenly over its non-space characters, and each embedder token receives the weight of the
characters it covers. Tokens outside the text, such as special tokens or a fixed prefix like e5's
"query: ", cover none of its characters and receive no weight.
"""

from collections.abc import Sequence

import numpy as np

Span = tuple[int, int]


def alignment_matrix(
    text: str,
    llm_offsets: Sequence[Span],
    embedder_offsets: Sequence[Span],
    embedder_special: Sequence[bool],
    prefix_length: int = 0,
) -> np.ndarray:
    """Build the matrix that carries weights on the LLM's text tokens to the embedder's tokens.

    Entry [i, j] is the share of LLM token j's non-space characters that embedder token i covers,
    so the matrix times the LLM's weights gives each embedder token the weight of its characters.
    When several tokens of one model cover the same character, the last of them gets it.

    Args:
        text: The text both models read.
        llm_offsets: The character span of each of the LLM's text tokens in `text`.
        embedder_offsets: The character span of each embedder token in `prefix + text`.
        embedder_special: Whether each embedder token is a special token.
        prefix_length: The number of characters the embedder reads before the text.

    Returns:
        An array of shape [len(embedder_offsets), len(llm_offsets)].
    """
    llm_token_at = np.full(len(text), -1)  # the LLM token that covers each character
    for j, (start, end) in enumerate(llm_offsets):
        llm_token_at[start:end] = j

    embedder_token_at = np.full(len(text), -1)  # the embedder token that covers each character
    for i, ((start, end), special) in enumerate(
        zip(embedder_offsets, embedder_special, strict=True)
    ):
        if special or end <= prefix_length:
            continue
        embedder_token_at[max(start - prefix_length, 0) : end - prefix_length] = i

    counted = np.array([not c.isspace() for c in text], dtype=bool) & (llm_token_at >= 0)
    chars_per_llm_token = np.bincount(llm_token_at[counted], minlength=len(llm_offsets))

    shared = counted & (embedder_token_at >= 0)
    matrix = np.zeros((len(embedder_offsets), len(llm_offsets)))
    np.add.at(
        matrix,
        (embedder_token_at[shared], llm_token_at[shared]),
        1.0 / chars_per_llm_token[llm_token_at[shared]],
    )
    return matrix


def carry_weights(
    matrix: np.ndarray,
    llm_weights: np.ndarray,
    text_mask: np.ndarray,
) -> np.ndarray:
    """Carry the LLM's weights to the embedder's text tokens and renormalize them.

    Args:
        matrix: The alignment matrix from `alignment_matrix`.
        llm_weights: The LLM's weights on its text tokens.
        text_mask: The embedder tokens that count as text; the weights land only on these.

    Returns:
        Weights over the embedder's tokens that sum to one on `text_mask` and are zero elsewhere.
        If no weight lands on `text_mask`, the weights are uniform on it.

    Raises:
        ValueError: If `text_mask` selects no token.
    """
    text_mask = np.asarray(text_mask, dtype=bool)
    if not text_mask.any():
        raise ValueError("text_mask selects no embedder token")
    weights = np.where(text_mask, matrix @ llm_weights, 0.0)
    total = weights.sum()
    if total > 0:
        return weights / total
    return text_mask / text_mask.sum()
