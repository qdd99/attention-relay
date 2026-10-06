"""The evaluation metrics of Appendix B.

Scores are computed on embeddings as the stages store them: float16, read back as float32.
Clustering scores run k-means with as many clusters as classes on centered, unit-length embeddings,
average ten runs (random states 0 to 9) and report 100 times the V-measure. A gain is a score of
the relayed embedder minus the same score of the embedder alone; its 95% interval comes from a
paired bootstrap over the items.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np

K_MEANS_RUNS = 10
"""Every clustering score averages this many k-means runs."""

NYT_ASPECTS = ("topic", "location")
IE_ASPECTS = ("emotion", "intent")


def unit_length(embeddings: np.ndarray) -> np.ndarray:
    return embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)


def centered(embeddings: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """The chosen rows, in float32, centered on their mean and scaled to unit length."""
    points = np.asarray(embeddings if rows is None else embeddings[rows], np.float32)
    return unit_length(points - points.mean(0))


def harmonic_mean(a: float, b: float) -> float:
    return 2 * a * b / (a + b + 1e-9)


def v_measure(points: np.ndarray, labels, runs: int = K_MEANS_RUNS) -> float:
    """100 V of k-means clusterings of the points (eq. B.1), averaged over `runs` runs."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import v_measure_score

    k = len(set(labels))
    scores = [
        v_measure_score(
            labels, KMeans(n_clusters=k, n_init=1, random_state=run).fit_predict(points)
        )
        for run in range(runs)
    ]
    return 100 * float(np.mean(scores))


# ---- NYT ----------------------------------------------------------------------------------------


def nyt_scores(under_topic, under_location, labels, rows=None, runs: int = K_MEANS_RUNS) -> dict:
    """NYT (eq. B.2): each aspect's V-measure under each question, and their harmonic mean.

    Args:
        under_topic: Embeddings made under the topic question.
        under_location: Embeddings made under the location question; for the embedder alone,
            pass the same array as `under_topic`.
        labels: The topic and location label of every article.
        rows: The articles to score; all by default.
        runs: k-means runs per V-measure.

    Returns:
        "under_topic" and "under_location" (each a V-measure per aspect), "switch" (how much each
        aspect's score moves between the questions) and "nyt", the harmonic mean of the topic score
        under the topic question and the location score under the location question.
    """
    rows = np.arange(len(labels["topic"])) if rows is None else rows
    chosen = {aspect: [labels[aspect][row] for row in rows] for aspect in NYT_ASPECTS}
    topic_points = centered(under_topic, rows)
    scores = {
        "under_topic": {
            aspect: v_measure(topic_points, chosen[aspect], runs) for aspect in NYT_ASPECTS
        }
    }
    if under_topic is under_location:
        scores["under_location"] = dict(scores["under_topic"])
    else:
        location_points = centered(under_location, rows)
        scores["under_location"] = {
            aspect: v_measure(location_points, chosen[aspect], runs) for aspect in NYT_ASPECTS
        }
    topic, location = scores["under_topic"], scores["under_location"]
    scores["switch"] = {
        "topic": topic["topic"] - location["topic"],
        "location": location["location"] - topic["location"],
    }
    scores["nyt"] = harmonic_mean(topic["topic"], location["location"])
    return scores


def nyt_gain_interval(
    relayed_under_topic,
    relayed_under_location,
    alone,
    labels,
    replicates: int = 200,
    runs: int = 2,
    workers: int = 1,
) -> list[float]:
    """The 95% interval of the NYT gain, from a paired bootstrap over articles.

    Replicate b resamples the articles with the generator seeded b and drops duplicates; both
    embedders are scored on that resample with `runs` k-means runs each. Replicates are
    independent, so running them in `workers` processes gives the same interval.
    """
    arrays = (relayed_under_topic, relayed_under_location, alone, labels, runs)
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_share, initargs=arrays) as pool:
            gains = list(pool.map(_shared_nyt_replicate, range(replicates)))
    else:
        gains = [_nyt_replicate(seed, *arrays) for seed in range(replicates)]
    return [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))]


def _nyt_replicate(seed, under_topic, under_location, alone, labels, runs) -> float:
    n = len(labels["topic"])
    rows = np.unique(np.random.default_rng(seed).integers(0, n, n))
    relayed = nyt_scores(under_topic, under_location, labels, rows, runs)["nyt"]
    return relayed - nyt_scores(alone, alone, labels, rows, runs)["nyt"]


_SHARED: tuple = ()


def _share(*arrays):
    """Give a worker process the embeddings once, with one BLAS thread."""
    global _SHARED
    _SHARED = arrays
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except ImportError:
        pass


def _shared_nyt_replicate(seed: int) -> float:
    return _nyt_replicate(seed, *_SHARED)


# ---- IntentEmotion ------------------------------------------------------------------------------


def triplet_passes(embeddings, texts, triplets) -> np.ndarray:
    """1 for each triplet whose anchor is closer, by cosine, to its positive than its negative."""
    position = {text: index for index, text in enumerate(texts)}
    points = unit_length(np.asarray(embeddings, np.float32))
    anchors = points[[position[anchor] for anchor, _, _ in triplets]]
    positives = points[[position[positive] for _, positive, _ in triplets]]
    negatives = points[[position[negative] for _, _, negative in triplets]]
    return ((anchors * positives).sum(1) > (anchors * negatives).sum(1)).astype(np.float64)


def ie_scores(under_emotion, under_intent, texts, triplets) -> tuple[dict, tuple]:
    """IntentEmotion (eq. B.3): the share of triplets that follow the question, per question.

    Args:
        under_emotion: Embeddings made under the emotion question.
        under_intent: Embeddings made under the intent question.
        texts: The messages, in the order of the embeddings' rows.
        triplets: The triplets by aspect.

    Returns:
        The scores: "under_emotion" and "under_intent" (the pass rate of each aspect's triplets),
        "ie" (the emotion pass rate under the emotion question plus the intent pass rate under the
        intent question) and "switch". Also the two pass vectors that make "ie", for the interval.
    """
    passes = {
        "under_emotion": {a: triplet_passes(under_emotion, texts, triplets[a]) for a in IE_ASPECTS},
        "under_intent": {a: triplet_passes(under_intent, texts, triplets[a]) for a in IE_ASPECTS},
    }
    scores = {
        question: {aspect: 100 * rates.mean() for aspect, rates in by_aspect.items()}
        for question, by_aspect in passes.items()
    }
    emotion, intent = scores["under_emotion"], scores["under_intent"]
    scores["ie"] = emotion["emotion"] + intent["intent"]
    scores["switch"] = {
        "emotion": emotion["emotion"] - intent["emotion"],
        "intent": intent["intent"] - emotion["intent"],
    }
    return scores, (passes["under_emotion"]["emotion"], passes["under_intent"]["intent"])


def ie_gain_interval(relayed_passes, alone_passes, replicates: int = 2000, seed: int = 0):
    """The 95% interval of the IE gain, from a paired bootstrap over each question's triplets."""
    generator = np.random.default_rng(seed)
    (relayed_emotion, relayed_intent), (alone_emotion, alone_intent) = relayed_passes, alone_passes
    gains = []
    for _ in range(replicates):
        emotion_rows = generator.integers(0, len(relayed_emotion), len(relayed_emotion))
        intent_rows = generator.integers(0, len(relayed_intent), len(relayed_intent))
        # this order of operations reproduces the paper's interval to the last digit
        gains.append(
            100
            * (
                relayed_emotion[emotion_rows].mean()
                + relayed_intent[intent_rows].mean()
                - alone_emotion[emotion_rows].mean()
                - alone_intent[intent_rows].mean()
            )
        )
    return [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))]


# ---- InBedder's other sets ----------------------------------------------------------------------


def pair_cosines(embeddings, pairs) -> np.ndarray:
    """The cosine of every sentence pair, in float32."""
    points = np.asarray(embeddings, np.float32)
    first = points[[a for a, _, _ in pairs]]
    second = points[[b for _, b, _ in pairs]]
    norms = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return (first * second).sum(1) / norms


def stsb_spearman(embeddings, pairs) -> float:
    """InstructSTSB: 100 times the Spearman correlation of the pairs' cosines with their labels."""
    from scipy.stats import spearmanr

    labels = np.array([label for _, _, label in pairs])
    return 100 * float(spearmanr(pair_cosines(embeddings, pairs), labels).correlation)


def stsb_gain_interval(relayed, alone, pairs, replicates: int = 1000, seed: int = 0) -> list[float]:
    """The 95% interval of the InstructSTSB gain, from a paired bootstrap over sentence pairs."""
    from scipy.stats import spearmanr

    labels = np.array([label for _, _, label in pairs])
    relayed_cosines, alone_cosines = pair_cosines(relayed, pairs), pair_cosines(alone, pairs)
    generator = np.random.default_rng(seed)
    gains = []
    for _ in range(replicates):
        rows = generator.integers(0, len(labels), len(labels))
        relayed_rho = spearmanr(relayed_cosines[rows], labels[rows]).correlation
        gains.append(100 * (relayed_rho - spearmanr(alone_cosines[rows], labels[rows]).correlation))
    return [float(np.nanpercentile(gains, 2.5)), float(np.nanpercentile(gains, 97.5))]


def clustering_scores(embeddings, labels, runs: int = K_MEANS_RUNS) -> dict:
    """V-measure, adjusted Rand index and normalized mutual information (100 times each)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score

    points = centered(embeddings)
    labels = np.asarray(labels)
    k = len(set(labels))
    scores = {"v": [], "ari": [], "nmi": []}
    for run in range(runs):
        clusters = KMeans(n_clusters=k, n_init=1, random_state=run).fit_predict(points)
        scores["v"].append(v_measure_score(labels, clusters))
        scores["ari"].append(adjusted_rand_score(labels, clusters))
        scores["nmi"].append(normalized_mutual_info_score(labels, clusters))
    return {name: 100 * float(np.mean(values)) for name, values in scores.items()}


def neighbour_scores(embeddings, labels, k: int = 10, chunk: int = 1024) -> dict:
    """Precision at k and mean average precision of same-label neighbours by cosine (100 times
    each), with every text as the query once."""
    points = centered(embeddings)
    labels = np.asarray(labels)
    n = len(labels)
    precision, average_precision = [], []
    for start in range(0, n, chunk):
        similarity = points[start : start + chunk] @ points.T
        for offset in range(similarity.shape[0]):
            query = start + offset
            similarity[offset, query] = -np.inf
            order = np.argsort(-similarity[offset])[: n - 1]
            relevant = (labels[order] == labels[query]).astype(np.float64)
            precision.append(relevant[:k].mean())
            n_relevant = relevant.sum()
            if n_relevant > 0:
                hits = np.cumsum(relevant) / np.arange(1, n)
                average_precision.append(float(hits @ relevant / n_relevant))
            else:
                average_precision.append(0.0)
    return {
        "precision_at_10": 100 * float(np.mean(precision)),
        "map": 100 * float(np.mean(average_precision)),
    }


def probe_accuracy(embeddings, labels, folds: int = 5, seed: int = 0) -> float:
    """The linear probe: five-fold stratified accuracy of a logistic regression (100 times), over
    the classes with at least `folds` members."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    labels = np.asarray(labels)
    counts = {label: (labels == label).sum() for label in set(labels)}
    kept = np.array([counts[label] >= folds for label in labels])
    if len(set(labels[kept])) < 2:  # only in small trial subsets
        return float("nan")
    classifier = LogisticRegression(max_iter=2000, C=1.0)
    splits = StratifiedKFold(folds, shuffle=True, random_state=seed)
    accuracy = cross_val_score(classifier, centered(embeddings)[kept], labels[kept], cv=splits)
    return 100 * float(accuracy.mean())


def all_clustering_metrics(embeddings, labels) -> dict:
    return {
        **clustering_scores(embeddings, labels),
        **neighbour_scores(embeddings, labels),
        "probe": probe_accuracy(embeddings, labels),
    }


def clustering_gain_interval(relayed, alone, labels, replicates: int = 100, seed: int = 0):
    """The 95% interval of a clustering set's V-measure gain, from a paired bootstrap over texts;
    one k-means run (random state 0) per embedder and replicate."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import v_measure_score

    labels = np.asarray(labels)
    k = len(set(labels))
    generator = np.random.default_rng(seed)
    gains = []
    for _ in range(replicates):
        rows = np.unique(generator.integers(0, len(labels), len(labels)))
        scores = []
        for embeddings in (relayed, alone):
            points = centered(embeddings, rows)
            n_clusters = min(k, len(rows))  # k on the paper's sets; fewer only in trial subsets
            clusters = KMeans(n_clusters=n_clusters, n_init=1, random_state=0).fit_predict(points)
            scores.append(v_measure_score(labels[rows], clusters))
        gains.append(100 * (scores[0] - scores[1]))
    return [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))]
