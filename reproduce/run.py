"""Reproduce the paper's results with one command: every stage, one after another, on one machine.

    python -m reproduce.run --out runs/paper
    python -m reproduce.run --out runs/trial --limit 16 --device mps      (a short trial)

Stages, in order: reading (the LLM passes), blocks (the label-free rule), grid (every pair, the
controls, the sweep), items (InBedder's other sets), analyses (relay range, plain bias, word
removal, place names, surfacing), inputs (the instruction-mode and question-in-the-input
baselines), figures (Figure 1's example and the cost table). Each stage keeps what is already
stored under --out, so an interrupted run resumes where it stopped. The embeddings and the LLM's
attention arrays take tens of gigabytes; the scores go to <out>/results/*.json.
"""

import argparse
import os
import platform
import sys
import time

import torch

from reproduce import data, settings
from reproduce.stages import analyses, figures, grid, inputs, items, reading
from reproduce.storage import Run

STAGES = ("reading", "blocks", "grid", "items", "analyses", "inputs", "figures")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, help="the run directory")
    parser.add_argument("--device", default=None, help="cuda, mps or cpu; the default is cuda")
    parser.add_argument("--limit", type=int, default=0, help="read only the first texts (trial)")
    parser.add_argument("--only", default="", help="comma-separated stages to run")
    parser.add_argument("--workers", type=int, default=0, help="processes for the NYT intervals")
    parser.add_argument("--llms", default="", help="comma-separated LLMs (default: all ten)")
    parser.add_argument("--embedders", default="", help="comma-separated embedders (default: all)")
    args = parser.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    workers = args.workers or (max(1, (os.cpu_count() or 2) - 1) if sys.platform == "linux" else 1)
    stages = [s for s in args.only.split(",") if s] or list(STAGES)
    unknown = sorted(set(stages) - set(STAGES))
    if unknown:
        parser.error(f"unknown stages: {unknown}; choose from {STAGES}")
    llms = _choose(args.llms, settings.LLMS, parser)
    embedders = _choose(args.embedders, settings.EMBEDDERS, parser)
    run = Run(args.out)
    run.save_results("run", _describe(args, device, workers))
    nyt = data.load_nyt(args.limit)
    ie = data.load_intent_emotion(args.limit or data.IE_ANCHORS)

    for stage in STAGES:
        if stage not in stages:
            continue
        started = time.time()
        _log(f"stage {stage}")
        if stage == "reading":
            for llm in llms:
                reading.read(run, llm, device, args.limit, _log)
        elif stage == "blocks":
            blocks = {llm: reading.block_rule(run, llm) for llm in llms}
            run.save_results("blocks", blocks)
            for llm in [x for x in settings.INSTRUCTION_TUNED if x in llms]:
                if blocks[llm]["chosen"] != blocks[llm]["paper_block"]:
                    _log(
                        f"  note: the rule chooses block {blocks[llm]['chosen']} for {llm}; "
                        f"the paper's is {blocks[llm]['paper_block']}, which the stages keep"
                    )
        elif stage == "grid":
            grid.relay(run, device, llms, embedders, _log)
            grid.score(run, nyt, ie, llms, embedders, workers, _log)
        elif stage == "items":
            items.relay(run, device, llms, embedders, _log)
            items.score(run, args.limit, llms, embedders, _log)
        elif stage == "analyses" and analyses.LLM in llms:
            analyses.relay(run, device, embedders, _log)
            analyses.score(run, nyt, ie, embedders, workers, _log)
        elif stage == "inputs" and inputs.READER in llms:
            inputs.embed(run, device, args.limit, embedders, _log)
            inputs.score(run, nyt, ie, embedders, _log)
        elif stage == "figures" and figures.LLM in llms:
            if figures.EXAMPLE["embedder"] in embedders:
                figures.example(run, _log)
            figures.cost(run, device, args.limit, embedders, _log)
        _log(f"stage {stage} done in {(time.time() - started) / 60:.1f} min")


def _choose(names: str, known, parser) -> list[str]:
    chosen = [name for name in names.split(",") if name] or list(known)
    unknown = sorted(set(chosen) - set(known))
    if unknown:
        parser.error(f"unknown models: {unknown}")
    return chosen


def _describe(args, device, workers) -> dict:
    import transformers

    return {
        "limit": args.limit,
        "device": device,
        "workers": workers,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
    }


def _log(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    main()
