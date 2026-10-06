"""What the stages share: the paper's models, the LLM's weights carried to an embedder, embedded
conditions, and the scores of an (LLM, embedder) pair."""

import numpy as np

from attention_relay import Embedder, LLMReader, alignment_matrix, carry_weights
from attention_relay.relay import mean_pool
from reproduce import controls, data, metrics, settings

RELAYED = "relayed"
ALONE = "alone"


def load_reader(key: str, device) -> LLMReader:
    """An LLM of the paper; a base checkpoint reads in its instruction-tuned sibling's template."""
    llm = settings.LLMS[key]
    sibling = settings.LLMS[llm.template_from] if llm.template_from else None
    return LLMReader(
        llm.repository,
        sibling.repository if sibling else None,
        revision=llm.revision,
        tokenizer_revision=sibling.revision if sibling else None,
        device=device,
        max_text_tokens=settings.MAX_TEXT_TOKENS,
    )


def load_embedder(key: str, device) -> Embedder:
    """An embedder of the paper, with its pooling, maximum length and prefix."""
    spec = settings.EMBEDDERS[key]
    return Embedder(
        spec.repository,
        pooling=spec.pooling,
        revision=spec.revision,
        prefix=spec.prefix,
        max_tokens=spec.max_tokens,
        device=device,
        trust_remote_code=spec.code_revision is not None,
        code_revision=spec.code_revision,
    )


class Carrier:
    """One dataset's texts in one embedder's tokens, aligned once to the LLM's tokens (eq. 2)."""

    def __init__(self, embedder: Embedder, texts, offsets):
        self.encodings = embedder.tokenize(texts)
        self.matrices = [
            alignment_matrix(text, spans, encoding.offsets, encoding.special, len(embedder.prefix))
            for text, spans, encoding in zip(texts, offsets, self.encodings, strict=True)
        ]

    def carry(self, weights_by_text) -> list[np.ndarray]:
        """Weights on the LLM's tokens, carried to the embedder's text tokens."""
        return [
            carry_weights(matrix, weights, np.array(encoding.text_mask))
            for matrix, weights, encoding in zip(
                self.matrices, weights_by_text, self.encodings, strict=True
            )
        ]

    def uniform(self) -> list[np.ndarray]:
        """Uniform weights on each text's tokens: mean pooling without the special tokens."""
        return [np.array(e.text_mask, float) / sum(e.text_mask) for e in self.encodings]


def embed_conditions(
    run, parts, embedder: Embedder, encodings, conditions: dict, **overrides
) -> None:
    """Embed every condition not yet stored. A condition is None (the embedder alone) or weights on
    the embedder's tokens, relayed with the paper's settings (settings.RELAY) except where
    `overrides` sets beta, bias or layers; mean pooling computes the token states once for all of
    them."""
    unknown = set(overrides) - set(settings.RELAY)
    if unknown:
        raise ValueError(f"unknown relay settings: {sorted(unknown)}")
    relay = {**settings.RELAY, **overrides}
    missing = {name: w for name, w in conditions.items() if not run.has_embeddings(*parts, name)}
    if not missing:
        return
    if embedder.pooling == "mean":
        states = embedder.token_states(encodings)
        for name, weights in missing.items():
            pooled = mean_pool(states, encodings, weights, relay["beta"])
            run.save_embeddings(pooled, *parts, name)
        return
    for name, weights in missing.items():
        run.save_embeddings(embedder.embed(encodings, weights, **relay), *parts, name)


def relay_conditions(
    carrier: Carrier, weights: dict, lengths, with_controls=False, positional=False
):
    """The relayed conditions of one dataset: each question's weights and, if asked, the controls.

    Args:
        carrier: The dataset's alignment to the embedder.
        weights: For each aspect, the LLM's reading weights of every text.
        lengths: Each text's number of LLM tokens.
        with_controls: Add the weights shuffled within the text and another text's weights,
            for seeds 0, 1 and 2.
        positional: Add the positional profile.
    """
    conditions = {}
    for aspect, by_text in weights.items():
        conditions[f"{RELAYED}_{aspect}"] = carrier.carry(by_text)
        if with_controls:
            for seed in range(controls.SEEDS):
                shuffled = [controls.shuffled(w, i, seed) for i, w in enumerate(by_text)]
                conditions[f"shuffled{seed}_{aspect}"] = carrier.carry(shuffled)
        if positional:
            conditions[f"positional_{aspect}"] = carrier.carry(controls.positional_profile(by_text))
    if with_controls:
        for seed in range(controls.SEEDS):
            partners = controls.other_text_partners(lengths, controls.OTHER_TEXT_SEED_OFFSET + seed)
            for aspect, by_text in weights.items():
                others = [controls.fitted(by_text[p], lengths[i]) for i, p in enumerate(partners)]
                conditions[f"other_text{seed}_{aspect}"] = carrier.carry(others)
    return conditions


def score_pair(run, parts, dataset: str, nyt=None, ie=None, workers: int = 1) -> dict:
    """The scores of a pair on NYT or IE: the embedder alone, relayed, and every control stored,
    with the 95% interval of the gain (Appendix B)."""
    aspects = list(data.QUESTIONS[dataset])

    def load(name):
        return run.load_embeddings(*parts, dataset, name)

    def exists(name):
        return run.has_embeddings(*parts, dataset, name)

    if dataset == "nyt":

        def score(first, second):
            return metrics.nyt_scores(first, second, nyt.labels)
    else:

        def score(first, second):
            return metrics.ie_scores(first, second, ie.texts, ie.triplets)[0]

    alone = load(ALONE)
    relayed = tuple(load(f"{RELAYED}_{aspect}") for aspect in aspects)
    scores = {
        ALONE: score(alone, alone),
        RELAYED: score(*relayed),
        "shuffled": [],
        "other_text": [],
    }
    for seed in range(controls.SEEDS):
        for control in ("shuffled", "other_text"):
            names = [f"{control}{seed}_{aspect}" for aspect in aspects]
            if all(exists(name) for name in names):
                scores[control].append(score(*(load(name) for name in names)))
    if exists("uniform"):
        uniform = load("uniform")
        scores["uniform"] = score(uniform, uniform)
    if all(exists(f"positional_{aspect}") for aspect in aspects):
        scores["positional"] = score(*(load(f"positional_{aspect}") for aspect in aspects))
    if dataset == "nyt":
        scores["interval"] = metrics.nyt_gain_interval(*relayed, alone, nyt.labels, workers=workers)
    else:
        relayed_passes = metrics.ie_scores(*relayed, ie.texts, ie.triplets)[1]
        alone_passes = metrics.ie_scores(alone, alone, ie.texts, ie.triplets)[1]
        scores["interval"] = metrics.ie_gain_interval(relayed_passes, alone_passes)
    return scores
