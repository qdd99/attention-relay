"""InBedder's other sets: FewRel, FewNerd, FewEvent and InstructSTSB (Section 4, Appendix C).

One LLM of each family relays its weights into the five core embedders, each item read under its
own instruction. Conditions: the embedder alone, relayed, relayed with the weights shuffled within
the text (seed 0), and the instruction in the embedder's input (for Qwen3-Embedding, its
instruction mode).
"""

import numpy as np

from reproduce import controls, data, metrics, settings
from reproduce.stages.common import ALONE, RELAYED, Carrier, embed_conditions, load_embedder
from reproduce.storage import StoredReading

INSTRUCTION_IN_INPUT = "instruction_in_input"


def relay(run, device, llms=settings.LLMS, embedders=settings.EMBEDDERS, log=print) -> None:
    for emb in [x for x in settings.CORE_EMBEDDERS if x in embedders]:
        embedder = None
        for llm in [x for x in settings.FAMILY_LLMS if x in llms]:
            for name in settings.ITEM_SETS:
                parts = ("embeddings", "items", llm, emb, name)
                if run.is_done(*parts):
                    continue
                embedder = embedder or load_embedder(emb, device)
                store = StoredReading(run, llm, name)
                texts, instructions = store.texts(), store.instructions()
                carrier = Carrier(embedder, texts, store.offsets())
                weights = store.weights("items", settings.LLMS[llm].block)
                shuffled = [controls.shuffled(w, index, 0) for index, w in enumerate(weights)]
                conditions = {
                    ALONE: None,
                    RELAYED: carrier.carry(weights),
                    "shuffled0": carrier.carry(shuffled),
                }
                embed_conditions(run, parts, embedder, carrier.encodings, conditions)
                if not run.has_embeddings(*parts, INSTRUCTION_IN_INPUT):
                    inputs = embedder.tokenize(_with_instruction(emb, texts, instructions))
                    run.save_embeddings(embedder.embed(inputs), *parts, INSTRUCTION_IN_INPUT)
                run.mark_done(*parts)
                log(f"  {llm} -> {emb} on {name}")


def _with_instruction(embedder_key: str, texts, instructions) -> list[str]:
    """The item with its instruction in the embedder's input (Table A.3)."""
    if settings.EMBEDDERS[embedder_key].pooling == "last":
        return [
            data.INSTRUCTION_MODE.format(instruction=q, text=t)
            for t, q in zip(texts, instructions, strict=True)
        ]
    before = data.QUESTION_IN_INPUT["before"]
    return [before.format(question=q, text=t) for t, q in zip(texts, instructions, strict=True)]


def score(run, limit=0, llms=settings.LLMS, embedders=settings.EMBEDDERS, log=print) -> None:
    """Every condition's metrics and the interval of the relay's gain, into results/items.json."""
    results = run.load_results("items")
    for llm in [x for x in settings.FAMILY_LLMS if x in llms]:
        for emb in [x for x in settings.CORE_EMBEDDERS if x in embedders]:
            for name in settings.ITEM_SETS:
                key = f"{llm}|{emb}|{name}"
                if key in results:
                    continue
                parts = ("embeddings", "items", llm, emb, name)
                conditions = (ALONE, INSTRUCTION_IN_INPUT, RELAYED, "shuffled0")
                rows = {c: run.load_embeddings(*parts, c) for c in conditions}
                results[key] = _score_set(name, rows, limit)
                run.save_results("items", results)
                log(f"  scored {key}")


def _score_set(name: str, rows: dict, limit: int) -> dict:
    if name == "instructstsb":
        pairs = data.load_instruct_stsb(limit).pairs
        scores = {condition: metrics.stsb_spearman(E, pairs) for condition, E in rows.items()}
        scores["interval"] = metrics.stsb_gain_interval(rows[RELAYED], rows[ALONE], pairs)
        return scores
    labels = data.load_clustering_set(name, limit).labels
    scores = {c: metrics.all_clustering_metrics(E, labels) for c, E in rows.items()}
    scores["interval"] = metrics.clustering_gain_interval(
        rows[RELAYED], rows[ALONE], np.asarray(labels)
    )
    return scores
