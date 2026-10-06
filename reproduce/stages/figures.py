"""Figure 1's worked example and the cost table (Appendix A).

The example: one IntentEmotion message read by Qwen3-1.7B under both questions, its reading weights
carried to bge-large-en-v1.5's tokens, and one of bge's heads' [CLS] attention before and after
relaying. The cost: wall time per text and question for the LLM's pass and for each core embedder
alone and relayed, on the first 256 NYT articles under the location question.
"""

import time

import numpy as np
import torch

from attention_relay import alignment_matrix
from reproduce import data, settings
from reproduce.stages.common import Carrier, load_embedder, load_reader
from reproduce.storage import StoredReading

LLM = "qwen3-1.7b"
EXAMPLE = {"text_index": 4543, "layer": 17, "head": 14, "embedder": "bge-large-en-v1.5"}
"""The paper's example: chosen by hand among messages of 12 to 22 LLM tokens whose two questions'
weights differ most; the head is one second-half head, for illustration."""
COST_TEXTS = 256


def example(run, log=print) -> None:
    results = run.load_results("example")
    if results:
        return
    store = StoredReading(run, LLM, "ie")
    texts, offsets = store.texts(), store.offsets()
    index = EXAMPLE["text_index"]
    if index >= len(texts):
        log("  the example needs the full IntentEmotion set; skipped")
        return
    block = settings.LLMS[LLM].block
    weights = {aspect: store.weights(aspect, block)[index] for aspect in data.QUESTIONS["ie"]}
    text, spans = texts[index], offsets[index]

    from transformers import AutoModel, AutoTokenizer

    spec = settings.EMBEDDERS[EXAMPLE["embedder"]]
    tokenizer = AutoTokenizer.from_pretrained(spec.repository, revision=spec.revision)
    model = AutoModel.from_pretrained(
        spec.repository, revision=spec.revision, attn_implementation="eager"
    ).eval()
    encoded = tokenizer(
        text, return_offsets_mapping=True, return_special_tokens_mask=True, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_attentions=True,
        )
    layer, head = EXAMPLE["layer"], EXAMPLE["head"]
    before = outputs.attentions[layer][0, head, 0].double().numpy()  # [CLS]'s attention row
    special = encoded["special_tokens_mask"][0].numpy().astype(bool)
    embedder_offsets = encoded["offset_mapping"][0].tolist()
    matrix = alignment_matrix(text, spans, embedder_offsets, special, 0)
    on_text = float(before[~special].sum())
    share = np.where(~special, before, 0.0) / on_text
    carried, after = {}, {}
    for aspect, llm_weights in weights.items():
        w = np.where(~special, matrix @ llm_weights, 0.0)
        carried[aspect] = w / w.sum()
        relayed = share * carried[aspect]
        relayed /= relayed.sum()
        after[aspect] = np.where(~special, on_text * relayed, before)

    rank, of, distance = _rank(store, block, index)
    results = {
        "text": text,
        "text_index": index,
        "questions": data.QUESTIONS["ie"],
        "llm": {
            "model": settings.LLMS[LLM].repository,
            "block": block,
            "tokens": [text[s:e] for s, e in spans.tolist()],
            "offsets": spans.tolist(),
            "weights": {aspect: w.tolist() for aspect, w in weights.items()},
        },
        "embedder": {
            "model": spec.repository,
            "revision": spec.revision,
            "tokens": tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]),
            "special": special.tolist(),
            "offsets": embedder_offsets,
            "layer": layer,
            "head": head,
            "n_layers": model.config.num_hidden_layers,
            "n_heads": model.config.num_attention_heads,
            "carried_weights": {aspect: w.tolist() for aspect, w in carried.items()},
            "attention_before": before.tolist(),
            "attention_after": {aspect: row.tolist() for aspect, row in after.items()},
            "text_share": on_text,
        },
        "selection": {"rank": rank, "of": of, "total_variation": distance, "tokens": [12, 22]},
    }
    run.save_results("example", results)
    log(f"  example: message {index}, rank {rank} of {of} by the questions' difference")


def _rank(store, block, index, shortest=12, longest=22):
    """The message's rank by total variation between the two questions' weights, among messages
    of `shortest` to `longest` LLM tokens."""
    first, second = (store.weights(aspect, block) for aspect in data.QUESTIONS["ie"])
    lengths = store.lengths()
    distances = {
        k: 0.5 * np.abs(first[k] - second[k]).sum()
        for k in np.where((lengths >= shortest) & (lengths <= longest))[0]
    }
    order = sorted(distances, key=lambda k: -distances[k])
    return order.index(index) + 1, len(order), float(distances[index])


def cost(run, device, limit=0, embedders=settings.EMBEDDERS, log=print) -> None:
    results = run.load_results("cost")
    if results:
        return
    n = min(COST_TEXTS, limit) if limit else COST_TEXTS
    texts = data.load_nyt(limit).texts[:n]
    question = data.QUESTIONS["nyt"]["location"]
    reader = load_reader(LLM, device)
    blocks = range(reader.n_blocks)  # the whole model, as the paper's pass ran
    reader.read(texts[:16], question, blocks, normalize=False)
    started = _now(device)
    readings = reader.read(texts, question, blocks, normalize=False)
    results = {"n": n, "device": str(device), "llm_ms": 1000 * (_now(device) - started) / n}
    block = settings.LLMS[LLM].block
    llm_weights = [r.weights[block] / r.weights[block].sum() for r in readings]
    read_texts, offsets = [r.text for r in readings], [r.offsets for r in readings]
    del reader
    for emb in [x for x in settings.CORE_EMBEDDERS if x in embedders]:
        embedder = load_embedder(emb, device)
        carrier = Carrier(embedder, read_texts, offsets)
        carried = carrier.carry(llm_weights)
        timing = {}
        for name, weights in (("alone", None), ("relayed", carried)):
            embedder.embed(carrier.encodings, weights, **settings.RELAY)  # warm-up
            started = _now(device)
            embedder.embed(carrier.encodings, weights, **settings.RELAY)
            timing[f"{name}_ms"] = 1000 * (_now(device) - started) / n
        results[emb] = timing
        log(f"  cost {emb}: {timing}")
    run.save_results("cost", results)


def _now(device) -> float:
    kind = torch.device(device).type
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        torch.mps.synchronize()
    return time.time()
