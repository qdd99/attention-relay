"""The analyses of Section 4, all with Qwen3-1.7B's weights at its block.

Relay range: the relay acts in the first half, the second half or all of the embedder's layers.
Plain bias: the alternative bias of Section 3.1, against the mass-preserving bias.
Word removal: a tenth of each text's words, the most weighted or random, removed before embedding.
Place names: NYT's location score on the articles that name a place and on the rest.
Surfacing: every clustering metric and the probe of each aspect under its own question.
"""

import numpy as np

from reproduce import controls, data, metrics, settings
from reproduce.stages.common import (
    ALONE,
    RELAYED,
    Carrier,
    embed_conditions,
    load_embedder,
    score_pair,
)
from reproduce.storage import StoredReading

LLM = "qwen3-1.7b"
RANGE_EMBEDDERS = ("qwen3-emb-0.6b", "gte-large-en-v1.5")
BIAS_EMBEDDER = "bge-large-en-v1.5"


def _block() -> int:
    return settings.LLMS[LLM].block


def _layer_ranges(n_layers: int) -> dict[str, list[int]]:
    half = n_layers // 2
    return {
        "none": [],
        "first": list(range(half)),
        "second": list(range(half, n_layers)),
        "all": list(range(n_layers)),
    }


def relay(run, device, embedders=settings.EMBEDDERS, log=print) -> None:
    for emb in [x for x in RANGE_EMBEDDERS if x in embedders]:
        embedder = None
        for name in ("none", "first", "second", "all"):
            for dataset in ("nyt", "ie"):
                parts = ("embeddings", "range", emb, name, dataset)
                if run.is_done(*parts):
                    continue
                embedder = embedder or load_embedder(emb, device)
                layers = _layer_ranges(embedder.n_layers)[name]
                _relay_dataset(run, parts, embedder, dataset, layers=layers)
                log(f"  relay range {emb} {name} {dataset}")

    embedder = None
    for dataset in ("nyt", "ie") if BIAS_EMBEDDER in embedders else ():
        parts = ("embeddings", "bias", LLM, BIAS_EMBEDDER, dataset)
        if not run.is_done(*parts):
            embedder = embedder or load_embedder(BIAS_EMBEDDER, device)
            _relay_dataset(run, parts, embedder, dataset, bias="plain")
            log(f"  plain bias {dataset}")

    for emb in [x for x in settings.CORE_EMBEDDERS if x in embedders]:
        embedder = None
        for dataset in ("nyt", "ie"):
            parts = ("embeddings", "word_removal", emb, dataset)
            if run.is_done(*parts):
                continue
            embedder = embedder or load_embedder(emb, device)
            _remove_words(run, parts, embedder, dataset)
            log(f"  word removal {emb} {dataset}")


def _relay_dataset(run, parts, embedder, dataset, **overrides):
    store = StoredReading(run, LLM, dataset)
    carrier = Carrier(embedder, store.texts(), store.offsets())
    conditions = {ALONE: None}
    for aspect in data.QUESTIONS[dataset]:
        conditions[f"{RELAYED}_{aspect}"] = carrier.carry(store.weights(aspect, _block()))
    embed_conditions(run, parts, embedder, carrier.encodings, conditions, **overrides)
    run.mark_done(*parts)


def _remove_words(run, parts, embedder, dataset):
    store = StoredReading(run, LLM, dataset)
    texts, offsets, lengths = store.texts(), store.offsets(), store.lengths()
    removed_share = {}
    for position, aspect in enumerate(data.QUESTIONS[dataset], start=1):
        weights = store.weights(aspect, _block())
        per_character = [
            controls.character_weights(texts[i], offsets[i], lengths[i], weights[i])
            for i in range(len(texts))
        ]
        for how in ("top", "random"):
            reduced = []
            for index, text in enumerate(texts):
                generator = None if how == "top" else np.random.default_rng([0, index, position])
                reduced.append(
                    controls.remove_words(
                        text, per_character[index], controls.REMOVED_FRACTION, generator
                    )
                )
            encodings = embedder.tokenize([text for text, _ in reduced])
            carried = [
                controls.weights_from_characters(chars, encoding, len(embedder.prefix))
                for (_, chars), encoding in zip(reduced, encodings, strict=True)
            ]
            conditions = {
                f"removed_{how}_{ALONE}_{aspect}": None,
                f"removed_{how}_{RELAYED}_{aspect}": carried,
            }
            embed_conditions(run, parts, embedder, encodings, conditions)
            removed_share[f"{how}_{aspect}"] = float(
                np.mean(
                    [1 - len(r) / max(len(t), 1) for (r, _), t in zip(reduced, texts, strict=True)]
                )
            )
    results = run.load_results("word_removal_characters")
    results[f"{parts[2]}|{dataset}"] = removed_share
    run.save_results("word_removal_characters", results)
    run.mark_done(*parts)


def score(run, nyt, ie, embedders=settings.EMBEDDERS, workers=1, log=print) -> None:
    """Score the analyses into results/{range,bias,word_removal,places,surfacing}.json."""
    core = [x for x in settings.CORE_EMBEDDERS if x in embedders]
    ranges = run.load_results("range")
    for emb in [x for x in RANGE_EMBEDDERS if x in embedders]:
        if emb in ranges:
            continue
        ranges[emb] = {
            name: {
                d: score_pair(run, ("embeddings", "range", emb, name), d, nyt, ie, workers)
                for d in ("nyt", "ie")
            }
            for name in ("first", "second", "all")
        }
        ranges[emb]["checks"] = _range_checks(run, emb)
        run.save_results("range", ranges)
        log(f"  scored the relay range of {emb}")

    bias = run.load_results("bias")
    if "plain" not in bias and BIAS_EMBEDDER in embedders:
        parts = ("embeddings", "bias", LLM, BIAS_EMBEDDER)
        bias["plain"] = {d: score_pair(run, parts, d, nyt, ie, workers) for d in ("nyt", "ie")}
        bias["difference"] = _bias_difference(run, nyt, ie)
        run.save_results("bias", bias)
        log("  scored the plain bias")

    removal = run.load_results("word_removal")
    for emb in core:
        if emb not in removal:
            removal[emb] = {d: _score_removal(run, emb, d, nyt, ie) for d in ("nyt", "ie")}
            run.save_results("word_removal", removal)
            log(f"  scored word removal for {emb}")

    places, surfacing = run.load_results("places"), run.load_results("surfacing")
    texts = StoredReading(run, LLM, "nyt").texts()
    named = np.array([controls.names_a_place(text) for text in texts])
    for emb in core:
        alone, under_topic, under_location = _nyt_rows(run, emb)
        if emb not in places:
            places[emb] = _place_split(alone, under_location, named, nyt.labels["location"])
        if emb not in surfacing:
            surfacing[emb] = {
                "topic": _both(alone, under_topic, nyt.labels["topic"]),
                "location": _both(alone, under_location, nyt.labels["location"]),
            }
        run.save_results("places", places)
        run.save_results("surfacing", surfacing)
        log(f"  scored place names and surfacing for {emb}")


def _nyt_rows(run, emb):
    parts = ("embeddings", "pairs", LLM, emb, "nyt")
    names = (ALONE, f"{RELAYED}_topic", f"{RELAYED}_location")
    return tuple(run.load_embeddings(*parts, name) for name in names)


def _both(alone, relayed, labels) -> dict:
    return {
        ALONE: metrics.all_clustering_metrics(alone, labels),
        RELAYED: metrics.all_clustering_metrics(relayed, labels),
    }


def _place_split(alone, under_location, named, labels) -> dict:
    labels = np.array(labels)
    out = {}
    for name, mask in (("named", named), ("not_named", ~named)):
        rows = np.where(mask)[0]
        chosen = list(labels[rows])
        alone_v = metrics.v_measure(metrics.centered(alone, rows), chosen)
        relayed_v = metrics.v_measure(metrics.centered(under_location, rows), chosen)
        out[name] = {
            "n": len(rows),
            ALONE: alone_v,
            RELAYED: relayed_v,
            "gain": relayed_v - alone_v,
        }
    return out


def _range_checks(run, emb) -> dict:
    """The relay at no layer against the embedder alone, and in every layer against the grid."""

    def unit(embeddings):
        embeddings = embeddings.astype(np.float64)
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def smallest_cosine(first, second):
        return float((unit(first) * unit(second)).sum(1).min())

    checks = {}
    for dataset in ("nyt", "ie"):
        load = run.load_embeddings
        row = {
            "alone_vs_grid": smallest_cosine(
                load("embeddings", "range", emb, "all", dataset, ALONE),
                load("embeddings", "pairs", LLM, emb, dataset, ALONE),
            )
        }
        for aspect in data.QUESTIONS[dataset]:
            name = f"{RELAYED}_{aspect}"
            row[f"none_vs_alone_{aspect}"] = smallest_cosine(
                load("embeddings", "range", emb, "none", dataset, name),
                load("embeddings", "range", emb, "none", dataset, ALONE),
            )
            row[f"all_vs_grid_{aspect}"] = smallest_cosine(
                load("embeddings", "range", emb, "all", dataset, name),
                load("embeddings", "pairs", LLM, emb, dataset, name),
            )
        checks[dataset] = row
    return checks


def _bias_difference(run, nyt, ie) -> dict:
    """The plain bias minus the mass-preserving bias: full-set scores and paired bootstrap
    intervals of the difference (200 replicates of two k-means runs for NYT; 2,000 for IE)."""

    def relayed(experiment, dataset):
        parts = ("embeddings", experiment, LLM, BIAS_EMBEDDER, dataset)
        return tuple(run.load_embeddings(*parts, f"{RELAYED}_{a}") for a in data.QUESTIONS[dataset])

    plain = {d: relayed("bias", d) for d in ("nyt", "ie")}
    mass_preserving = {d: relayed("pairs", d) for d in ("nyt", "ie")}

    def percentiles(values):
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    full_plain = metrics.nyt_scores(*plain["nyt"], nyt.labels)
    full_mass = metrics.nyt_scores(*mass_preserving["nyt"], nyt.labels)
    n = len(nyt.labels["topic"])
    nyt_diff, topic_diff, location_diff = [], [], []
    for seed in range(200):
        rows = np.unique(np.random.default_rng(seed).integers(0, n, n))
        a = metrics.nyt_scores(*plain["nyt"], nyt.labels, rows, runs=2)
        c = metrics.nyt_scores(*mass_preserving["nyt"], nyt.labels, rows, runs=2)
        nyt_diff.append(a["nyt"] - c["nyt"])
        topic_diff.append(a["under_topic"]["topic"] - c["under_topic"]["topic"])
        location_diff.append(a["under_location"]["location"] - c["under_location"]["location"])

    plain_ie = metrics.ie_scores(*plain["ie"], ie.texts, ie.triplets)
    mass_ie = metrics.ie_scores(*mass_preserving["ie"], ie.texts, ie.triplets)
    (plain_emotion, plain_intent), (mass_emotion, mass_intent) = plain_ie[1], mass_ie[1]
    generator = np.random.default_rng(0)
    total, emotion, intent = [], [], []
    for _ in range(2000):
        x = generator.integers(0, len(plain_emotion), len(plain_emotion))
        y = generator.integers(0, len(plain_intent), len(plain_intent))
        emotion.append(100 * (plain_emotion[x].mean() - mass_emotion[x].mean()))
        intent.append(100 * (plain_intent[y].mean() - mass_intent[y].mean()))
        total.append(emotion[-1] + intent[-1])
    return {
        "nyt": {
            "plain": full_plain["nyt"],
            "mass_preserving": full_mass["nyt"],
            "difference_interval": percentiles(nyt_diff),
            "topic_difference_interval": percentiles(topic_diff),
            "location_difference_interval": percentiles(location_diff),
            "share_of_replicates_above_zero": float(np.mean(np.array(nyt_diff) > 0)),
        },
        "ie": {
            "plain": plain_ie[0]["ie"],
            "mass_preserving": mass_ie[0]["ie"],
            "difference_interval": percentiles(total),
            "emotion_difference_interval": percentiles(emotion),
            "intent_difference_interval": percentiles(intent),
            "triplets_changed": {
                "emotion": int((plain_emotion != mass_emotion).sum()),
                "intent": int((plain_intent != mass_intent).sum()),
                "of": [len(plain_emotion), len(plain_intent)],
            },
        },
    }


def _score_removal(run, emb, dataset, nyt, ie) -> dict:
    parts = ("embeddings", "word_removal", emb, dataset)
    aspects = list(data.QUESTIONS[dataset])
    out = {}
    for how in ("top", "random"):
        for condition in (ALONE, RELAYED):
            first, second = (
                run.load_embeddings(*parts, f"removed_{how}_{condition}_{a}") for a in aspects
            )
            if dataset == "nyt":
                out[f"{how}_{condition}"] = metrics.nyt_scores(first, second, nyt.labels)
            else:
                out[f"{how}_{condition}"] = metrics.ie_scores(first, second, ie.texts, ie.triplets)[
                    0
                ]
    return out
