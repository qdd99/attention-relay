"""Batches of token sequences of different lengths."""

from collections.abc import Iterator, Sequence

import torch


def batches(lengths: Sequence[int], max_tokens: int, max_size: int) -> Iterator[list[int]]:
    """Group sequences, longest first, so that no batch exceeds `max_tokens` with padding."""
    order = sorted(range(len(lengths)), key=lambda index: -lengths[index])
    batch: list[int] = []
    longest = 0
    for index in order:
        length = lengths[index]
        too_long = max(longest, length) * (len(batch) + 1) > max_tokens
        if batch and (too_long or len(batch) >= max_size):
            yield batch
            batch, longest = [], 0
        batch.append(index)
        longest = max(longest, length)
    if batch:
        yield batch


def pad(sequences: Sequence[Sequence], fill, side: str, dtype=torch.long) -> torch.Tensor:
    """Stack sequences of different lengths into one tensor, filling on the left or the right."""
    width = max(len(sequence) for sequence in sequences)
    stacked = torch.full((len(sequences), width), fill, dtype=dtype)
    for row, sequence in enumerate(sequences):
        values = torch.as_tensor(list(sequence), dtype=dtype)
        if side == "left":
            stacked[row, width - len(sequence) :] = values
        else:
            stacked[row, : len(sequence)] = values
    return stacked
