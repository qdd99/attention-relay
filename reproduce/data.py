"""The paper's datasets and the text inputs that go with them (Appendix A).

Six datasets from InBedder's benchmark, at the revisions every run used. NYT and IntentEmotion each
come with two questions, one per aspect of their texts; FewRel, FewNerd and FewEvent come with
InBedder's instruction for the set, and InstructSTSB with an instruction for each pair.
"""

from dataclasses import dataclass

DATASETS = {
    "nyt": ("BrandonZYW/NYTClustering", "874caa290866b25a86bfa94a4e457c72fb5bdcb2"),
    "ie": ("BrandonZYW/IntentEmotion", "d0101c15083166b77371cf542c4045decc0a6e92"),
    "fewrel": ("BrandonZYW/FewRelClustering", "d951bf6cb10522c042bf8db3a837028b7e69388f"),
    "fewnerd": ("BrandonZYW/FewNerdClustering", "9dff096e4ca4dd38688236fa1d4e4c17cb547e66"),
    "fewevent": ("BrandonZYW/FewEventClustering", "aecc9c3ad7ae8d05b8621390d1919f1816eab777"),
    "instructstsb": ("BrandonZYW/InstructSTSB", "34dda81eca082ba6b3a259899fe9ae435b6ea1ae"),
}
"""Each dataset's Hugging Face repository and revision."""

QUESTIONS = {
    "nyt": {"topic": "What is the topic of news?", "location": "Where did the news happen?"},
    "ie": {"emotion": "How does the customer feel?", "intent": "What does the customer need?"},
}
"""The two questions about every text of NYT and IntentEmotion, by the aspect each asks about."""

INSTRUCTIONS = {
    "fewrel": "Here is a sentence. Please tell me the relation type between two specified entities "
    "appended after the sentence.",
    "fewnerd": "Here is a sentence. Please tell me the type of the specified entity appended after "
    "the sentence.",
    "fewevent": "Here is a sentence. Please tell me the type of the specified event according to "
    "the trigger words appended after the sentence.",
}
"""InBedder's instruction for each clustering set, verbatim."""

TASK_DESCRIPTIONS = {
    "nyt": {
        "topic": "Identify the topic of the given news article",
        "location": "Identify the location where the given news article happened",
    },
    "ie": {
        "emotion": "Identify the emotion the customer expresses in the given message",
        "intent": "Identify what the customer needs in the given message",
    },
}
"""What Qwen3-Embedding reads in its instruction mode for NYT and IntentEmotion, by aspect."""

INSTRUCTION_MODE = "Instruct: {instruction}\nQuery:{text}"
"""Qwen3-Embedding's input in its instruction mode."""

QUESTION_IN_INPUT = {"before": "{question} {text}", "after": "{text} {question}"}
"""The question placed in an instruction-agnostic embedder's input, before or after the text."""

IE_ANCHORS = 2000
"""IntentEmotion is scored on the triplets of its first 2,000 intent anchors in sorted order."""


@dataclass(frozen=True)
class NYT:
    """NYT's articles, in sorted order, with each article's answer under each question."""

    texts: list[str]
    labels: dict[str, list[str]]
    """By aspect ("topic", "location"), one label per text."""


@dataclass(frozen=True)
class IntentEmotion:
    """IntentEmotion's messages and triplets: under each question, (anchor, positive, negative)."""

    texts: list[str]
    """Every message of the triplets, in sorted order."""
    triplets: dict[str, list[tuple[str, str, str]]]
    """By aspect ("emotion", "intent"); the positive shares only the anchor's answer to that
    question, the negative only its answer to the other."""


@dataclass(frozen=True)
class ClusteringSet:
    """A clustering set: its texts, their labels and the instruction for the set."""

    texts: list[str]
    labels: list
    instruction: str


@dataclass(frozen=True)
class InstructSTSB:
    """InstructSTSB: each distinct (text, instruction) item once, and the pairs to score."""

    items: list[tuple[str, str]]
    pairs: list[tuple[int, int, float]]
    """(index of the first item, index of the second item, the pair's binary label)."""


def load_nyt(limit: int = 0) -> NYT:
    """NYT's articles; `limit` keeps the first ones, for a trial run."""
    topic = _by_text("nyt", "topic")
    location = _by_text("nyt", "location")
    texts = sorted(topic)[: limit or None]
    labels = {
        "topic": [topic[text] for text in texts],
        "location": [location[text] for text in texts],
    }
    return NYT(texts, labels)


def load_intent_emotion(anchors: int = IE_ANCHORS) -> IntentEmotion:
    """The triplets of the first `anchors` intent anchors in sorted order, and their messages."""
    every = {aspect: _triplets(aspect) for aspect in ("emotion", "intent")}
    anchors = set(sorted({anchor for anchor, _, _ in every["intent"]})[:anchors])
    triplets = {
        aspect: [triplet for triplet in rows if triplet[0] in anchors]
        for aspect, rows in every.items()
    }
    texts = set(anchors)
    for rows in triplets.values():
        for _, positive, negative in rows:
            texts.update((positive, negative))
    return IntentEmotion(sorted(texts), triplets)


def load_clustering_set(name: str, limit: int = 0) -> ClusteringSet:
    """A clustering set; `limit` keeps its first texts, for a trial run."""
    rows = _load(name)
    rows = [rows[index] for index in range(min(limit, len(rows)) if limit else len(rows))]
    return ClusteringSet(
        [row["text"] for row in rows], [row["cluster"] for row in rows], INSTRUCTIONS[name]
    )


def load_instruct_stsb(limit: int = 0) -> InstructSTSB:
    """InstructSTSB's pairs; `limit` keeps the first pairs, for a trial run."""
    index: dict[tuple[str, str], int] = {}
    items: list[tuple[str, str]] = []
    pairs = []
    rows = _load("instructstsb")
    for row in (rows[i] for i in range(min(limit, len(rows)) if limit else len(rows))):
        instruction = row["instruction"]
        for text in (row["sentence1"], row["sentence2"]):
            if (text, instruction) not in index:
                index[(text, instruction)] = len(items)
                items.append((text, instruction))
        first = index[(row["sentence1"], instruction)]
        second = index[(row["sentence2"], instruction)]
        pairs.append((first, second, float(row["score"])))
    return InstructSTSB(items, pairs)


def _load(name: str, configuration: str | None = None):
    from datasets import load_dataset

    repository, revision = DATASETS[name]
    return load_dataset(repository, configuration, revision=revision)["test"]


def _by_text(name: str, configuration: str) -> dict[str, str]:
    return {row["text"]: row["cluster"] for row in _load(name, configuration)}


def _triplets(aspect: str) -> list[tuple[str, str, str]]:
    return [(row["anchor"], row["positive"], row["negative"]) for row in _load("ie", aspect)]
