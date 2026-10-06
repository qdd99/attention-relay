"""The LLM passes and the block rule.

Every LLM reads every NYT and IntentEmotion text under each of the dataset's two questions; one
LLM of each family also reads InBedder's other sets, each item under its own instruction. The
head-summed attention on the text is stored at every block, in float16, before renormalizing. The
block rule then reads these arrays (Section 3.1; no labels).
"""

import json
import os

import numpy as np

from attention_relay import choose_block
from reproduce import data, settings
from reproduce.stages.common import load_reader
from reproduce.storage import StoredReading


def texts_and_questions(limit: int = 0):
    """NYT and IE, each with its texts and its two questions by aspect."""
    nyt = data.load_nyt(limit)
    ie = data.load_intent_emotion(limit or data.IE_ANCHORS)
    return {"nyt": (nyt.texts, data.QUESTIONS["nyt"]), "ie": (ie.texts, data.QUESTIONS["ie"])}


def item_set(name: str, limit: int = 0) -> list[tuple[str, str]]:
    """InBedder's other sets as (text, instruction) items."""
    if name == "instructstsb":
        return data.load_instruct_stsb(limit).items
    dataset = data.load_clustering_set(name, limit)
    return [(text, dataset.instruction) for text in dataset.texts]


def read(run, llm: str, device, limit: int = 0, log=print) -> None:
    """One LLM's pass over every dataset it reads in the paper."""
    reader = None
    datasets = texts_and_questions(limit)
    item_sets = settings.ITEM_SETS if llm in settings.FAMILY_LLMS else []
    for dataset, (texts, questions) in datasets.items():
        store = StoredReading(run, llm, dataset)
        for aspect, question in questions.items():
            if store.exists(aspect):
                continue
            reader = reader or load_reader(llm, device)
            readings = _read(reader, texts, question)
            store.save_texts([r.text for r in readings], [r.offsets for r in readings])
            store.save_attention(aspect, [r.weights for r in readings], reader.n_blocks)
            log(f"  {llm} read {dataset} under the {aspect} question")
    for name in item_sets:
        store = StoredReading(run, llm, name)
        if store.exists("items"):
            continue
        reader = reader or load_reader(llm, device)
        items = item_set(name, limit)
        readings = _read(reader, [text for text, _ in items], [q for _, q in items])
        store.save_texts([r.text for r in readings], [r.offsets for r in readings])
        store.save_attention("items", [r.weights for r in readings], reader.n_blocks)
        with open(os.path.join(store.directory, "instructions.json"), "w") as handle:
            json.dump([q for _, q in items], handle)
        log(f"  {llm} read {name}")


def _read(reader, texts, instructions):
    blocks = range(reader.n_blocks)
    return reader.read(texts, instructions, blocks=blocks, normalize=False)


def block_rule(run, llm: str) -> dict:
    """The question sensitivity of every block on NYT and IE, and the block the rule chooses."""
    curves = {}
    for dataset in ("nyt", "ie"):
        store = StoredReading(run, llm, dataset)
        first, second = (store.weights(aspect) for aspect in data.QUESTIONS[dataset])
        distances = [0.5 * np.abs(a - b).sum(axis=1) for a, b in zip(first, second, strict=True)]
        curves[dataset] = np.mean(distances, axis=0).tolist()
    mean = np.mean([curves["nyt"], curves["ie"]], axis=0)
    return {
        "sensitivity": curves,
        "mean": mean.tolist(),
        "chosen": choose_block(mean),
        "paper_block": settings.LLMS[llm].block,
    }
