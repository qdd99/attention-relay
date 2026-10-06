"""Where a run keeps its arrays and results, and how it reads them back.

A run directory holds, as the paper's pipeline stored them:
    reading/<llm>/<dataset>/        the LLM's texts, token offsets and attention on the text
    embeddings/<experiment>/...     embeddings, one float16 array per condition
    results/<experiment>.json       the scores
"""

import json
import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Run:
    """A run directory."""

    root: str

    def path(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    # ---- embeddings ------------------------------------------------------------------------------

    def save_embeddings(self, embeddings: np.ndarray, *parts: str) -> None:
        """Save embeddings in float16, as the paper's pipeline stored them."""
        path = self.path(*parts[:-1], parts[-1] + ".npy")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, np.asarray(embeddings, np.float32).astype(np.float16))

    def load_embeddings(self, *parts: str) -> np.ndarray:
        """Stored embeddings, read back as float32 for scoring."""
        return np.load(self.path(*parts[:-1], parts[-1] + ".npy")).astype(np.float32)

    def has_embeddings(self, *parts: str) -> bool:
        return os.path.exists(self.path(*parts[:-1], parts[-1] + ".npy"))

    # ---- finished work ---------------------------------------------------------------------------

    def is_done(self, *parts: str) -> bool:
        return os.path.exists(self.path(*parts, ".done"))

    def mark_done(self, *parts: str) -> None:
        os.makedirs(self.path(*parts), exist_ok=True)
        open(self.path(*parts, ".done"), "w").close()

    # ---- results ---------------------------------------------------------------------------------

    def load_results(self, name: str) -> dict:
        path = self.path("results", f"{name}.json")
        if not os.path.exists(path):
            return {}
        with open(path) as handle:
            return json.load(handle)

    def save_results(self, name: str, results: dict) -> None:
        path = self.path("results", f"{name}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(results, handle, indent=1, default=_plain)


@dataclass(frozen=True)
class StoredReading:
    """One LLM's reading of one dataset: its texts as read, their LLM token offsets, and for each
    question (or for the items' own instructions) the head-summed attention on every text token at
    every block, in float16."""

    run: Run
    llm: str
    dataset: str

    @property
    def directory(self) -> str:
        return self.run.path("reading", self.llm, self.dataset)

    def exists(self, question: str) -> bool:
        return os.path.exists(os.path.join(self.directory, f"{question}.npy"))

    def save_texts(self, texts, offsets) -> None:
        os.makedirs(self.directory, exist_ok=True)
        lengths = np.array([len(spans) for spans in offsets], np.int32)
        padded = np.zeros((len(offsets), max(lengths.max(initial=0), 1), 2), np.int32)
        for index, spans in enumerate(offsets):
            padded[index, : len(spans)] = spans
        np.save(os.path.join(self.directory, "lengths.npy"), lengths)
        np.save(os.path.join(self.directory, "offsets.npy"), padded)
        with open(os.path.join(self.directory, "texts.json"), "w") as handle:
            json.dump(list(texts), handle)

    def save_attention(self, question: str, attention_by_text, n_blocks: int) -> None:
        """attention_by_text: for each text, an array [n_blocks, text tokens]."""
        width = max((a.shape[1] for a in attention_by_text), default=1)
        stored = np.zeros((len(attention_by_text), n_blocks, max(width, 1)), np.float16)
        for index, attention in enumerate(attention_by_text):
            stored[index, :, : attention.shape[1]] = attention.astype(np.float16)
        np.save(os.path.join(self.directory, f"{question}.npy"), stored)

    def texts(self) -> list[str]:
        with open(os.path.join(self.directory, "texts.json")) as handle:
            return json.load(handle)

    def instructions(self) -> list[str]:
        """Each item's own instruction, for datasets read item by item."""
        with open(os.path.join(self.directory, "instructions.json")) as handle:
            return json.load(handle)

    def lengths(self) -> np.ndarray:
        return np.load(os.path.join(self.directory, "lengths.npy"))

    def offsets(self) -> list[np.ndarray]:
        padded = np.load(os.path.join(self.directory, "offsets.npy"))
        return [padded[index, :length] for index, length in enumerate(self.lengths())]

    def weights(self, question: str, block: int | None = None):
        """Reading weights renormalized over each text in float64 (eq. 1): for each text, a vector
        at `block`, or with block None an array [n_blocks, text tokens]."""
        stored = np.load(os.path.join(self.directory, f"{question}.npy"), mmap_mode="r")
        weights = []
        for index, length in enumerate(self.lengths()):
            if block is None:
                rows = np.asarray(stored[index, :, :length], np.float64)
                weights.append(rows / rows.sum(axis=1, keepdims=True))
            else:
                row = np.asarray(stored[index, block, :length], np.float64)
                total = row.sum()
                weights.append(row / total if total > 0 else np.full(len(row), 1.0 / len(row)))
        return weights


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot store {type(value).__name__}")
