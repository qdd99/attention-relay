"""Every (LLM, embedder) pair, and the relay from every block (Section 4).

Each of the six instruction-tuned LLMs is paired with all ten embedders at its block; each base
checkpoint, at its sibling's block, with the five core embedders. Every pair embeds NYT and IE
alone and relayed under each question. Controls (Figure 7): for bge-large-en-v1.5, the weights
shuffled within each text and another text's weights (seeds 0-2); for bge-large-en-v1.5 and the two
mean-pooling core embedders, the positional profile; for every mean-pooling embedder, uniform
weights on the text. The sweep relays every block of each instruction-tuned LLM into the two
mean-pooling core embedders.
"""

from reproduce import data, settings
from reproduce.stages.common import (
    ALONE,
    RELAYED,
    Carrier,
    embed_conditions,
    load_embedder,
    relay_conditions,
    score_pair,
)
from reproduce.storage import StoredReading

CONTROLS = ("bge-large-en-v1.5",)
POSITIONAL = ("bge-large-en-v1.5", "all-mpnet-base-v2", "e5-large-v2")
SWEEP = ("all-mpnet-base-v2", "e5-large-v2")


def pairs(llms=settings.LLMS, embedders=settings.EMBEDDERS):
    """(LLM, embedder) for every pair of the paper among the chosen models."""
    rows = [(llm, emb) for llm in settings.INSTRUCTION_TUNED for emb in settings.EMBEDDERS]
    rows += [(llm, emb) for llm in settings.BASE_CHECKPOINTS for emb in settings.CORE_EMBEDDERS]
    return [(llm, emb) for llm, emb in rows if llm in llms and emb in embedders]


def relay(run, device, llms=settings.LLMS, embedders=settings.EMBEDDERS, log=print) -> None:
    """Embed every pair's conditions, one embedder at a time."""
    for embedder_key in embedders:
        embedder = None
        for llm, emb in pairs(llms, embedders):
            if emb != embedder_key:
                continue
            instruction_tuned = llm in settings.INSTRUCTION_TUNED
            block = settings.LLMS[llm].block
            for dataset in ("nyt", "ie"):
                store = StoredReading(run, llm, dataset)
                parts = ("embeddings", "pairs", llm, emb, dataset)
                if run.is_done(*parts):
                    continue
                embedder = embedder or load_embedder(emb, device)
                carrier = Carrier(embedder, store.texts(), store.offsets())
                weights = {a: store.weights(a, block) for a in data.QUESTIONS[dataset]}
                conditions = {ALONE: None}
                conditions |= relay_conditions(
                    carrier,
                    weights,
                    store.lengths(),
                    with_controls=instruction_tuned and emb in CONTROLS,
                    positional=instruction_tuned and emb in POSITIONAL,
                )
                if embedder.pooling == "mean":
                    conditions["uniform"] = carrier.uniform()
                if instruction_tuned and emb in SWEEP:
                    for source in range(settings.LLMS[llm].n_blocks):
                        for aspect in data.QUESTIONS[dataset]:
                            by_text = store.weights(aspect, source)
                            conditions[f"{RELAYED}_{aspect}_block{source}"] = carrier.carry(by_text)
                embed_conditions(run, parts, embedder, carrier.encodings, conditions)
                run.mark_done(*parts)
            log(f"  {llm} -> {emb}")


def score(run, nyt, ie, llms=settings.LLMS, embedders=settings.EMBEDDERS, workers=1, log=print):
    """Score every pair, and the sweep, into results/pairs.json and results/sweep.json."""
    results = run.load_results("pairs")
    for llm, emb in pairs(llms, embedders):
        key = f"{llm}|{emb}"
        if key in results:
            continue
        parts = ("embeddings", "pairs", llm, emb)
        results[key] = {
            "block": settings.LLMS[llm].block,
            **{d: score_pair(run, parts, d, nyt, ie, workers) for d in ("nyt", "ie")},
        }
        run.save_results("pairs", results)
        log(f"  scored {key}")

    sweep = run.load_results("sweep")
    for llm in [x for x in settings.INSTRUCTION_TUNED if x in llms]:
        for emb in [x for x in SWEEP if x in embedders]:
            key = f"{llm}|{emb}"
            if key in sweep:
                continue
            sweep[key] = {}
            for dataset in ("nyt", "ie"):
                sweep[key][dataset] = {
                    str(block): _score_block(run, llm, emb, dataset, block, nyt, ie)
                    for block in range(settings.LLMS[llm].n_blocks)
                }
                sweep[key][dataset][ALONE] = results[key][dataset][ALONE]
            run.save_results("sweep", sweep)
            log(f"  scored the sweep {key}")


def _score_block(run, llm, emb, dataset, block, nyt, ie):
    from reproduce import metrics

    first, second = (
        run.load_embeddings("embeddings", "pairs", llm, emb, dataset, f"{RELAYED}_{a}_block{block}")
        for a in data.QUESTIONS[dataset]
    )
    if dataset == "nyt":
        return metrics.nyt_scores(first, second, nyt.labels)
    return metrics.ie_scores(first, second, ie.texts, ie.triplets)[0]
