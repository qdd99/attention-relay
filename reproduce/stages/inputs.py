"""Embedders given the question in their input: the baselines of Table 1 and Figure 2.

Instruction mode: Qwen3-Embedding-0.6B reads "Instruct: <instruction>\nQuery:<text>", with the
paper's questions or with task descriptions in its own style, and in its document mode without them.
Question in the input: four embedders trained without instructions read the question before or
after the text (Figure 2); the texts are cut so that the question fits in 512 tokens.
"""

import torch

from reproduce import data, metrics, settings
from reproduce.stages.common import load_embedder
from reproduce.storage import StoredReading

INSTRUCTION_MODE_EMBEDDER = "qwen3-emb-0.6b"
INSTRUCTION_MODE_MAX_TOKENS = 1024
QUESTION_IN_INPUT_EMBEDDERS = (
    "all-mpnet-base-v2",
    "e5-large-v2",
    "bge-large-en-v1.5",
    "gte-large-en-v1.5",
)
QUESTION_IN_INPUT_MAX_TOKENS = 512
READER = "qwen3-1.7b"
"""NYT's articles as this LLM reads them, truncated to 400 of its tokens."""


def _texts(run, limit: int) -> dict[str, list[str]]:
    ie = data.load_intent_emotion(limit or data.IE_ANCHORS)
    return {"nyt": StoredReading(run, READER, "nyt").texts(), "ie": ie.texts}


def embed(run, device, limit=0, embedders=settings.EMBEDDERS, log=print) -> None:
    texts = _texts(run, limit)
    parts = ("embeddings", "instruction_mode", INSTRUCTION_MODE_EMBEDDER)
    if not run.is_done(*parts) and INSTRUCTION_MODE_EMBEDDER in embedders:
        embedder = load_embedder(INSTRUCTION_MODE_EMBEDDER, device)
        embedder.max_tokens = INSTRUCTION_MODE_MAX_TOKENS
        for dataset, dataset_texts in texts.items():
            conditions = {"none": dataset_texts}
            for aspect, question in data.QUESTIONS[dataset].items():
                task = data.TASK_DESCRIPTIONS[dataset][aspect]
                for name, instruction in (("question", question), ("task_description", task)):
                    conditions[f"{name}_{aspect}"] = [
                        data.INSTRUCTION_MODE.format(instruction=instruction, text=t)
                        for t in dataset_texts
                    ]
            for name, inputs in conditions.items():
                if not run.has_embeddings(*parts, dataset, name):
                    embeddings = embedder.embed(embedder.tokenize(inputs))
                    run.save_embeddings(embeddings, *parts, dataset, name)
        run.mark_done(*parts)
        log("  instruction mode")

    for emb in [x for x in QUESTION_IN_INPUT_EMBEDDERS if x in embedders]:
        parts = ("embeddings", "question_in_input", emb)
        if run.is_done(*parts):
            continue
        embedder = _float32_embedder(emb, device)
        embedder.max_tokens = QUESTION_IN_INPUT_MAX_TOKENS
        room = _room_for_the_question(embedder)
        for dataset, dataset_texts in texts.items():
            cut = [_cut(embedder.tokenizer, text, room) for text in dataset_texts]
            conditions = {"none": cut}
            for aspect, question in data.QUESTIONS[dataset].items():
                for where, pattern in data.QUESTION_IN_INPUT.items():
                    conditions[f"{where}_{aspect}"] = [
                        pattern.format(question=question, text=t) for t in cut
                    ]
            for name, inputs in conditions.items():
                if not run.has_embeddings(*parts, dataset, name):
                    embeddings = embedder.embed(embedder.tokenize(inputs))
                    run.save_embeddings(embeddings, *parts, dataset, name)
        run.mark_done(*parts)
        log(f"  question in the input of {emb}")


def _float32_embedder(key, device):
    """Figure 2's runs used sentence-transformers' default precision, float32."""
    from attention_relay import Embedder

    spec = settings.EMBEDDERS[key]
    return Embedder(
        spec.repository,
        pooling=spec.pooling,
        revision=spec.revision,
        prefix=spec.prefix,
        max_tokens=spec.max_tokens,
        device=device,
        dtype=torch.float32,
        trust_remote_code=spec.code_revision is not None,
        code_revision=spec.code_revision,
    )


def _room_for_the_question(embedder) -> int:
    """Tokens left for the text: 512, minus 8, minus the longest question with the prefix."""
    questions = [q for by_aspect in data.QUESTIONS.values() for q in by_aspect.values()]
    tokenizer = embedder.tokenizer
    longest = max(
        len(tokenizer(embedder.prefix + q, add_special_tokens=False)["input_ids"])
        for q in questions
    )
    return QUESTION_IN_INPUT_MAX_TOKENS - 8 - longest


def _cut(tokenizer, text: str, room: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:room]
    return tokenizer.decode(ids, skip_special_tokens=True)


def score(run, nyt, ie, embedders=settings.EMBEDDERS, log=print) -> None:
    """Scores into results/instruction_mode.json and results/question_in_input.json."""
    results = run.load_results("instruction_mode")
    if not results and INSTRUCTION_MODE_EMBEDDER in embedders:
        parts = ("embeddings", "instruction_mode", INSTRUCTION_MODE_EMBEDDER)
        for condition in ("none", "question", "task_description"):
            results[condition] = {
                d: _score(run, parts, d, condition, nyt, ie) for d in ("nyt", "ie")
            }
        run.save_results("instruction_mode", results)
        log("  scored the instruction mode")

    results = run.load_results("question_in_input")
    for emb in [x for x in QUESTION_IN_INPUT_EMBEDDERS if x in embedders]:
        if emb in results:
            continue
        parts = ("embeddings", "question_in_input", emb)
        results[emb] = {
            condition: {"ie": _score(run, parts, "ie", condition, nyt, ie)}
            for condition in ("none", "before", "after")
        }
        run.save_results("question_in_input", results)
        log(f"  scored the question in the input of {emb}")


def _score(run, parts, dataset, condition, nyt, ie):
    aspects = list(data.QUESTIONS[dataset])
    if condition == "none":
        both = (run.load_embeddings(*parts, dataset, "none"),) * 2  # the same array twice
    else:
        both = tuple(run.load_embeddings(*parts, dataset, f"{condition}_{a}") for a in aspects)
    if dataset == "nyt":
        return metrics.nyt_scores(*both, nyt.labels)
    return metrics.ie_scores(*both, ie.texts, ie.triplets)[0]
