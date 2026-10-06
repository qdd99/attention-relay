"""The controls of Section 4 (Appendix C): weights that keep part of the LLM's reading and lose the
rest, and texts with words removed.

Weights over the LLM's text tokens:
    shuffled within the text: the same values at other positions, one permutation per text;
    another text's weights: those of another text with the same number of tokens;
    positional profile: the average weight at each relative position, over the other texts.
Word removal takes the words the LLM weights most, or as many random words, out of the text before
the embedder reads it. Place names: whether an article names one of NYT's frequent places.
"""

import math
import re
from collections.abc import Sequence

import numpy as np

SEEDS = 3
"""The shuffled and other-text controls use seeds 0, 1 and 2."""
OTHER_TEXT_SEED_OFFSET = 100
"""The other-text control draws its partners with seed 100 + s."""
POSITION_BINS = 20
REMOVED_FRACTION = 0.1
"""Word removal takes out a tenth of each text's words."""

PLACES = (
    r"america|americans?|u\.s\.|united states|iraq|iraqis?|china|chinese|germany|germans?|france|"
    r"french|japan|japanese|russia|russians?|italy|italians?|canada|canadians?|britain|british|u\.k\."
)
PLACE_PATTERN = re.compile(r"(?<!\w)(" + PLACES + r")(?!\w)", re.I)
"""The place words that name NYT's frequent locations."""


def normalized(weights: np.ndarray) -> np.ndarray:
    """Weights that sum to one; uniform when they sum to zero."""
    total = float(weights.sum())
    if total > 0:
        return weights / total
    return np.full(len(weights), 1.0 / max(len(weights), 1))


def shuffled(weights: np.ndarray, text_index: int, seed: int) -> np.ndarray:
    """The LLM's weights permuted within the text, the same permutation under every question."""
    return weights[np.random.default_rng([seed, text_index]).permutation(len(weights))]


def other_text_partners(lengths: Sequence[int], seed: int) -> np.ndarray:
    """For each text, another text with the same number of LLM tokens, or with the nearest number
    when none has the same."""
    generator = np.random.default_rng(seed)
    by_length: dict[int, list[int]] = {}
    for index, length in enumerate(lengths):
        by_length.setdefault(int(length), []).append(index)
    known_lengths = np.array(sorted(by_length))
    partners = np.zeros(len(lengths), int)
    for index, length in enumerate(lengths):
        candidates = [other for other in by_length[int(length)] if other != index]
        if not candidates:
            for nearest in known_lengths[np.argsort(np.abs(known_lengths - int(length)))]:
                candidates = [other for other in by_length[int(nearest)] if other != index]
                if candidates:
                    break
        partners[index] = generator.choice(candidates)
    return partners


def fitted(weights: np.ndarray, length: int) -> np.ndarray:
    """Another text's weights cut or padded with zeros to `length` tokens, then renormalized."""
    out = np.zeros(int(length))
    kept = min(len(weights), int(length))
    out[:kept] = weights[:kept]
    return normalized(out)


def positional_profile(weights_by_text: Sequence[np.ndarray], bins: int = POSITION_BINS):
    """Each text's weights replaced by the average weight per token at each relative position (in
    `bins` bins), averaged over the other texts: blind to the text's content."""
    density = np.zeros((len(weights_by_text), bins))
    present = np.zeros((len(weights_by_text), bins), bool)
    positions = []
    for index, weights in enumerate(weights_by_text):
        slot = np.minimum((np.arange(len(weights)) * bins) // max(len(weights), 1), bins - 1)
        counts = np.bincount(slot, minlength=bins)
        density[index] = np.bincount(slot, weights=weights, minlength=bins) / np.maximum(counts, 1)
        present[index] = counts > 0
        positions.append(slot)
    total, number = density.sum(0), present.sum(0)
    profiles = []
    for index, slot in enumerate(positions):
        others = (total - density[index]) / np.maximum(number - present[index], 1)
        profiles.append(normalized(others[slot]))
    return profiles


def character_weights(text: str, offsets, n_tokens: int, weights: np.ndarray) -> np.ndarray:
    """Each LLM token's weight spread evenly over its non-space characters."""
    per_character = np.zeros(len(text))
    for token in range(int(n_tokens)):
        start, end = offsets[token]
        characters = [c for c in range(start, end) if not text[c].isspace()]
        if characters:
            per_character[characters] += weights[token] / len(characters)
    return per_character


def remove_words(text: str, per_character: np.ndarray, fraction: float, generator=None):
    """The text without the words that hold the top `fraction` of its words by weight, or, with a
    generator, without as many random words. Each removed word takes one following space with it;
    a text keeps at least one word.

    Returns:
        The reduced text and its characters' weights, renormalized.
    """
    spans = [match.span() for match in re.finditer(r"\S+", text)]
    if len(spans) < 2:
        return text, per_character
    n_removed = min(max(1, math.ceil(fraction * len(spans))), len(spans) - 1)
    word_weights = np.array([per_character[start:end].sum() for start, end in spans])
    if generator is None:
        chosen = np.argsort(-word_weights, kind="stable")[:n_removed]
    else:
        chosen = generator.choice(len(spans), n_removed, replace=False)
    kept = np.ones(len(text), bool)
    for word in chosen:
        start, end = spans[word]
        kept[start:end] = False
        if end < len(text) and text[end].isspace():
            kept[end] = False
    reduced = "".join(character for character, keep in zip(text, kept, strict=True) if keep)
    remaining = per_character[kept]
    return reduced, (remaining / remaining.sum() if remaining.sum() > 0 else remaining)


def weights_from_characters(per_character, encoding, prefix_length: int = 0) -> np.ndarray:
    """The weights of the characters each embedder token covers, over the relay's text tokens."""
    weights = np.zeros(len(encoding.ids))
    for token, (start, end) in enumerate(encoding.offsets):
        if encoding.text_mask[token]:
            low, high = max(start - prefix_length, 0), max(end - prefix_length, 0)
            weights[token] = per_character[low:high].sum()
    if weights.sum() > 0:
        return weights / weights.sum()
    text_mask = np.array(encoding.text_mask, float)
    if text_mask.sum() > 0:
        return text_mask / text_mask.sum()
    return np.ones(len(weights)) / len(weights)


def names_a_place(text: str) -> bool:
    return bool(PLACE_PATTERN.search(text))
