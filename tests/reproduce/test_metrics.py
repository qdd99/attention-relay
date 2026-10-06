"""Tests for the metrics of Appendix B, on synthetic embeddings."""

import numpy as np
import pytest

from reproduce import metrics


def clusters(n_per_class=20, n_classes=3, dimension=8, spread=0.05, seed=0):
    """Well-separated clusters around random centres, and their labels."""
    generator = np.random.default_rng(seed)
    centres = generator.normal(size=(n_classes, dimension)) * 5
    labels = np.repeat(np.arange(n_classes), n_per_class)
    points = centres[labels] + generator.normal(scale=spread, size=(len(labels), dimension))
    return points.astype(np.float32), labels.tolist()


def test_separated_clusters_score_100_and_noise_scores_near_0():
    points, labels = clusters()
    assert metrics.v_measure(metrics.centered(points), labels) == pytest.approx(100)
    noise = np.random.default_rng(1).normal(size=points.shape).astype(np.float32)
    assert metrics.v_measure(metrics.centered(noise), labels) < 20


def test_nyt_is_the_harmonic_mean_and_an_agnostic_embedding_does_not_switch():
    by_topic, topics = clusters(seed=0)
    _, places = clusters(seed=1)
    labels = {"topic": topics, "location": list(reversed(places))}
    alone = metrics.nyt_scores(by_topic, by_topic, labels)
    assert alone["switch"] == {"topic": 0.0, "location": 0.0}
    topic, location = alone["under_topic"]["topic"], alone["under_location"]["location"]
    assert alone["nyt"] == pytest.approx(2 * topic * location / (topic + location))


def ie_triplets(n=30, seed=0):
    """Messages whose emotion and intent vary independently, and the two sets of triplets: the
    same messages, with positive and negative swapped between the questions."""
    generator = np.random.default_rng(seed)
    emotion = generator.integers(0, 2, n)
    intent = generator.integers(0, 2, n)
    texts = [f"message {i}" for i in range(n)]
    triplets = {"emotion": [], "intent": []}
    for a in range(n):
        for p in range(n):
            for m in range(n):
                shares_emotion_only = emotion[p] == emotion[a] and intent[p] != intent[a]
                shares_intent_only = intent[m] == intent[a] and emotion[m] != emotion[a]
                if a != p and a != m and shares_emotion_only and shares_intent_only:
                    triplets["emotion"].append((texts[a], texts[p], texts[m]))
                    triplets["intent"].append((texts[a], texts[m], texts[p]))
    return texts, emotion, intent, triplets


def test_an_embedding_that_ignores_the_question_scores_100_and_one_that_follows_it_200():
    texts, emotion, intent, triplets = ie_triplets()
    generator = np.random.default_rng(3)
    agnostic = generator.normal(size=(len(texts), 16)).astype(np.float32)
    scores, _ = metrics.ie_scores(agnostic, agnostic, texts, triplets)
    assert scores["ie"] == pytest.approx(100)

    by_emotion = (
        np.stack([emotion, 1 - emotion], axis=1).astype(np.float32) + 0.01 * agnostic[:, :2]
    )
    by_intent = np.stack([intent, 1 - intent], axis=1).astype(np.float32) + 0.01 * agnostic[:, :2]
    scores, _ = metrics.ie_scores(by_emotion, by_intent, texts, triplets)
    assert scores["ie"] == pytest.approx(200)


def test_a_gain_between_identical_embeddings_has_the_interval_zero():
    points, labels = clusters()
    nyt_labels = {"topic": labels, "location": labels}
    assert metrics.nyt_gain_interval(points, points, points, nyt_labels, replicates=5) == [0, 0]
    assert metrics.clustering_gain_interval(points, points, labels, replicates=5) == [0, 0]
    passes = (np.ones(10), np.zeros(10))
    assert metrics.ie_gain_interval(passes, passes, replicates=5) == [0, 0]


def test_a_better_embedding_has_an_interval_above_zero():
    points, labels = clusters()
    noise = np.random.default_rng(1).normal(size=points.shape).astype(np.float32)
    low, _ = metrics.clustering_gain_interval(points, noise, labels, replicates=20)
    assert low > 50


def test_stsb_rewards_cosines_that_follow_the_labels():
    generator = np.random.default_rng(0)
    base = generator.normal(size=(40, 8)).astype(np.float32)
    pairs = [(2 * i, 2 * i + 1, float(i % 2)) for i in range(20)]
    embeddings = base.copy()
    for first, second, label in pairs:
        if label == 1.0:  # similar pairs: the second sentence close to the first
            embeddings[second] = embeddings[first] + 0.01 * base[second]
    assert metrics.stsb_spearman(embeddings, pairs) > 80
    assert metrics.stsb_gain_interval(embeddings, embeddings, pairs, replicates=10) == [0, 0]


def test_the_probe_separates_separated_clusters():
    points, labels = clusters(n_per_class=10)
    assert metrics.probe_accuracy(points, labels) == pytest.approx(100)
