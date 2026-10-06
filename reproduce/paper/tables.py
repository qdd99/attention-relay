"""The paper's tables, written as LaTeX from a run's results files and the settings."""

import os

from reproduce import data, settings
from reproduce.paper.style import SCORE, number

QWEN = "qwen3-1.7b"
MINUS = "$-$"
"""The minus sign in the tables."""
DATASET_NAMES = {
    "nyt": "NYT",
    "ie": "IntentEmotion (IE)",
    "fewrel": "FewRel",
    "fewnerd": "FewNerd",
    "fewevent": "FewEvent",
    "instructstsb": "InstructSTSB",
}


def _number(value: float, digits: int = 1) -> str:
    return number(value, digits, minus=MINUS)


def _signed(value: float) -> str:
    return number(value, signed=True, minus=MINUS)


def _interval(bounds) -> str:
    return f"[{_number(bounds[0])}, {_number(bounds[1])}]"


def _write(lines, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def _item_score(entry, dataset):
    """An item set's score: the V-measure, or InstructSTSB's Spearman."""
    return entry if dataset == "instructstsb" else entry["v"]


def main(results, out):
    """Table 1: every core embedder alone, relayed with Qwen3-1.7B's weights and the gain, on
    InBedder's six sets; below, Qwen3-Embedding-0.6B in its instruction mode."""
    sets = ("nyt", "fewrel", "fewnerd", "fewevent", "instructstsb", "ie")
    pairs, items = results["pairs"], results["items"]

    def value(embedder, dataset, condition):
        if dataset in ("nyt", "ie"):
            return pairs[f"{QWEN}|{embedder}"][dataset][condition][SCORE[dataset]]
        return _item_score(items[f"{QWEN}|{embedder}|{dataset}"][condition], dataset)

    def gain(v):
        return r"\textcolor{gainblue}{" + _signed(v) + "}"

    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & \multicolumn{4}{c}{Clustering, V-measure $\uparrow$}"
        r" & STS, Spearman $\uparrow$ & Triplets $\uparrow$ \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-6}\cmidrule(lr){7-7}",
        r"Embedder & NYT & FewRel & FewNerd & FewEvent & InstructSTSB & IE \\",
        r"\midrule",
    ]
    for index, embedder in enumerate(settings.CORE_EMBEDDERS):
        alone = [value(embedder, d, "alone") for d in sets]
        relayed = [value(embedder, d, "relayed") for d in sets]
        name = settings.EMBEDDERS[embedder].name
        name += " (document mode)" if embedder == "qwen3-emb-0.6b" else ""
        lines += [r"\midrule"] if index else []
        lines += [
            " & ".join([name] + [_number(x) for x in alone]) + r" \\",
            " & ".join([r"\quad + relay"] + [_number(y) for y in relayed]) + r" \\",
            " & ".join([r"\quad Gain"] + [gain(y - x) for x, y in zip(alone, relayed, strict=True)])
            + r" \\",
        ]
    task = results["instruction_mode"]["task_description"]
    reference = [_number(task["nyt"]["nyt"])]
    for dataset in sets[1:5]:
        entry = items[f"{QWEN}|qwen3-emb-0.6b|{dataset}"]["instruction_in_input"]
        reference.append(_number(_item_score(entry, dataset)))
    reference.append(_number(task["ie"]["ie"]))
    lines += [
        r"\midrule",
        " & ".join(["Qwen3-Embedding-0.6B (instruction mode)"] + reference) + r" \\",
    ]
    return _write(lines + [r"\bottomrule", r"\end{tabular}"], out)


def main_ci(results, out):
    """Table 9: the gains of Table 1 with their 95% paired bootstrap intervals, laid out as in
    Table 1 (embedders as rows, the sets as columns), each interval under its gain in smaller
    type."""
    pairs, items = results["pairs"], results["items"]
    names = {
        "nyt": "NYT",
        "fewrel": "FewRel",
        "fewnerd": "FewNerd",
        "fewevent": "FewEvent",
        "instructstsb": "InstructSTSB",
        "ie": "IE",
    }

    def gain_and_interval(embedder, dataset):
        if dataset in ("nyt", "ie"):
            entry = pairs[f"{QWEN}|{embedder}"][dataset]
            score = SCORE[dataset]
            value = entry["relayed"][score] - entry["alone"][score]
        else:
            entry = items[f"{QWEN}|{embedder}|{dataset}"]
            relayed = _item_score(entry["relayed"], dataset)
            value = relayed - _item_score(entry["alone"], dataset)
        return _signed(value), _interval(entry["interval"])

    lines = [
        r"\begin{tabular}{l" + "c" * len(names) + "}",
        r"\toprule",
        "Embedder & " + " & ".join(names.values()) + r" \\",
        r"\midrule",
    ]
    for index, embedder in enumerate(settings.CORE_EMBEDDERS):
        cells = [gain_and_interval(embedder, dataset) for dataset in names]
        lines += [r"\addlinespace"] if index else []
        lines += [
            settings.EMBEDDERS[embedder].name
            + " & "
            + " & ".join(gain for gain, _ in cells)
            + r" \\",
            " & " + " & ".join(r"{\scriptsize " + interval + "}" for _, interval in cells) + r" \\",
        ]
    return _write(lines + [r"\bottomrule", r"\end{tabular}"], out)


BREADTH_SETS = ("fewrel", "fewnerd", "fewevent", "instructstsb")
INSTRUCTION_SECTION = "Given the instruction in its input (Qwen3-Embedding: its instruction mode)"


def _breadth_score(results, run, embedder, dataset, condition):
    entry = results["items"][f"{run}|{embedder}|{dataset}"][condition]
    return _number(_item_score(entry, dataset))


def breadth(results, out):
    """Table 10: InBedder's other sets. One section per LLM gives each embedder alone and relayed
    with that LLM's weights; a last section gives each embedder with the instruction in its input
    (Qwen3-Embedding in its instruction mode).

    The LLM plays no part in the embedder alone or given the instruction, so the table shows the
    Qwen3-1.7B run's numbers for them; separate runs of these conditions differ only through
    k-means' sensitivity to last-bit differences in the embeddings.
    """
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Embedder & FewRel & FewNerd & FewEvent & InstructSTSB \\",
    ]
    sections = [(settings.LLMS[llm].name, llm) for llm in settings.FAMILY_LLMS]
    sections.append((INSTRUCTION_SECTION, None))
    for title, llm in sections:
        lines += [r"\midrule", f"\\multicolumn{{5}}{{c}}{{\\emph{{{title}}}}} \\\\", r"\midrule"]
        for embedder in settings.CORE_EMBEDDERS:
            if llm is None:
                cells = [
                    _breadth_score(results, QWEN, embedder, dataset, "instruction_in_input")
                    for dataset in BREADTH_SETS
                ]
            else:
                cells = [
                    _breadth_score(results, QWEN, embedder, dataset, "alone")
                    + r" $\to$ "
                    + _breadth_score(results, llm, embedder, dataset, "relayed")
                    for dataset in BREADTH_SETS
                ]
            lines.append(settings.EMBEDDERS[embedder].name + " & " + " & ".join(cells) + r" \\")
    return _write(lines + [r"\bottomrule", r"\end{tabular}"], out)


def cost(results, out):
    """Table 11: wall time per text and question on one H100: the full pass of Qwen3-1.7B, and
    each core embedder alone and relayed."""
    timing = results["cost"]
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Model & alone & relayed \\",
        r"\midrule",
        f"Qwen3-1.7B, the full pass & {_number(timing['llm_ms'], 2)} & \\\\",
        r"\midrule",
    ]
    lines += [
        f"{settings.EMBEDDERS[e].name} & {_number(timing[e]['alone_ms'], 2)}"
        f" & {_number(timing[e]['relayed_ms'], 2)} \\\\"
        for e in settings.CORE_EMBEDDERS
    ]
    return _write(lines + [r"\bottomrule", r"\end{tabular}"], out)


def _spaced(rows):
    """Data rows with a little space between them, so that a cell that wraps reads as one row."""
    lines = []
    for index, row in enumerate(rows):
        if index:
            lines.append(r"\addlinespace")
        lines.append(row)
    return lines


def _pointer_table(groups, rows, first):
    """A two-column pointer table (name, Hugging Face repository), rows grouped under headers."""
    lines = [r"\begin{tabular}{ll}", r"\toprule", first + r" & Hugging Face \\", r"\midrule"]
    for index, (group, header) in enumerate(groups):
        lines += ([r"\addlinespace"] if index else []) + [
            f"\\multicolumn{{2}}{{l}}{{\\emph{{{header}}}}} \\\\"
        ]
        lines += [
            f"{name} & \\texttt{{{repository}}} \\\\" for name, repository, g in rows if g == group
        ]
    return lines + [r"\bottomrule", r"\end{tabular}"]


def models(results, out):
    """Table 2: every model of the paper and its Hugging Face repository."""
    rows = [
        (settings.LLMS[k].name, settings.LLMS[k].repository, "llm")
        for k in settings.INSTRUCTION_TUNED
    ]
    rows += [
        (settings.LLMS[k].name, settings.LLMS[k].repository, "base")
        for k in settings.BASE_CHECKPOINTS
    ]
    embedders = settings.CORE_EMBEDDERS + settings.MORE_EMBEDDERS
    rows += [
        (settings.EMBEDDERS[k].name, settings.EMBEDDERS[k].repository, "embedder")
        for k in embedders
    ]
    groups = [
        ("llm", "Instruction-tuned LLMs"),
        ("base", "Base checkpoints"),
        ("embedder", "Embedders, used without an instruction"),
    ]
    return _write(_pointer_table(groups, rows, "Model"), out)


def datasets(results, out):
    """Table 3: every dataset and its Hugging Face repository."""
    rows = [
        (name, data.DATASETS[key][0], "all" if key in ("nyt", "ie") else "exp")
        for key, name in DATASET_NAMES.items()
    ]
    groups = [("all", "Used throughout"), ("exp", r"Used in \S\ref{sec:exp}")]
    return _write(_pointer_table(groups, rows, "Dataset"), out)


def llms(results, out):
    """Table 4: each LLM's number of blocks and the block its weights are read at."""
    lines = [r"\begin{tabular}{lll}", r"\toprule", r"LLM & Blocks & Block read \\", r"\midrule"]
    lines += [f"{llm.name} & {llm.n_blocks} & {llm.block} \\\\" for llm in settings.LLMS.values()]
    return _write(lines + [r"\bottomrule", r"\end{tabular}"], out)


def pair_settings(results, out):
    """Table 5: each embedder's pooling, where the weights enter it, and its maximum length."""
    pooling = {"last": "last token", "cls": "[CLS]", "mean": "mean"}
    enters = {"last": "attention", "cls": "attention", "mean": "pooling weights"}
    order = [
        "qwen3-emb-0.6b",
        "qwen3-emb-4b",
        "qwen3-emb-8b",
        "bge-large-en-v1.5",
        "bge-small-en-v1.5",
        "bge-base-en-v1.5",
        "gte-large-en-v1.5",
        "all-mpnet-base-v2",
        "e5-large-v2",
        "all-MiniLM-L6-v2",
    ]
    lines = [
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Embedder & Pooling & Relayed into & Max.\ tokens \\",
        r"\midrule",
    ]
    for key in order:
        embedder = settings.EMBEDDERS[key]
        maximum = embedder.max_tokens or "--"
        lines.append(
            f"{embedder.name} & {pooling[embedder.pooling]} & {enters[embedder.pooling]}"
            f" & {maximum} \\\\"
        )
    return _write(lines + [r"\bottomrule", r"\end{tabular}"], out)


def instructions(results, out):
    """Table 6: the instruction each dataset puts in the chat template, verbatim."""

    def escaped(text):
        return text.replace("&", r"\&").replace("%", r"\%")

    rows = [
        ("NYT, topic", data.QUESTIONS["nyt"]["topic"]),
        ("NYT, location", data.QUESTIONS["nyt"]["location"]),
        ("IntentEmotion, emotion", data.QUESTIONS["ie"]["emotion"]),
        ("IntentEmotion, intent", data.QUESTIONS["ie"]["intent"]),
    ]
    rows += [
        (DATASET_NAMES[key], data.INSTRUCTIONS[key]) for key in ("fewrel", "fewnerd", "fewevent")
    ]
    lines = [
        r"\begin{tabular}{lp{0.72\linewidth}}",
        r"\toprule",
        r"Dataset & Instruction \\",
        r"\midrule",
    ]
    data_rows = [f"{name} & {escaped(text)} \\\\" for name, text in rows]
    data_rows += [r"InstructSTSB & each pair's own instruction, from the dataset \\"]
    lines += _spaced(data_rows)
    return _write(lines + [r"\bottomrule", r"\end{tabular}"], out)


def embedder_inputs(results, out):
    """Table A.3: what each embedder reads around the text in each condition."""

    def typed(text):
        text = text.replace("{", r"\{").replace("}", r"\}").replace(" ", "~")
        return r"\texttt{" + text.replace("\n", r"\textbackslash{}n") + "}"

    head, tail = data.INSTRUCTION_MODE.split(
        "\n"
    )  # set as two lines, broken where the input breaks
    e5 = settings.EMBEDDERS["e5-large-v2"].prefix

    def instruction_mode(slot):
        return typed(head.replace("{instruction}", slot) + "\n") + r"\newline " + typed(tail)

    rows = [
        ("Every embedder but e5-large-v2", "alone and relayed", typed("{text}")),
        ("e5-large-v2", "alone and relayed", typed(e5 + "{text}")),
        (
            "all-mpnet-base-v2, bge-large, gte-large",
            r"the question in the input, before or after the text"
            r" (Figure~\ref{fig:question-input})",
            typed("{question} {text}") + r"\newline " + typed("{text} {question}"),
        ),
        (
            "e5-large-v2",
            "the same",
            typed(e5 + "{question} {text}") + r"\newline " + typed(e5 + "{text} {question}"),
        ),
        (
            "bge-large, gte-large, all-mpnet-base-v2",
            r"the set's instruction in the input (Table~\ref{tab:breadth-all})",
            typed("{question} {text}"),
        ),
        ("e5-large-v2", "the same", typed(e5 + "{question} {text}")),
        (
            "Qwen3-Embedding-0.6B, instruction mode",
            r"the other sets' instructions (Tables~\ref{tab:main} and~\ref{tab:breadth-all})",
            instruction_mode("{question}"),
        ),
        (
            "Qwen3-Embedding-0.6B, instruction mode",
            r"NYT and IE (Table~\ref{tab:main} and Figure~\ref{fig:question-input})",
            instruction_mode("{task}"),
        ),
    ]

    def column(width):
        return r">{\raggedright\arraybackslash}p{" + width + r"\linewidth}"

    lines = [
        r"\begin{tabular}{" + column("0.24") + column("0.31") + column("0.35") + "}",
        r"\toprule",
        r"Embedder & Condition & Input \\",
        r"\midrule",
    ]
    lines += _spaced([f"{a} & {b} & {c} \\\\" for a, b, c in rows])
    return _write(lines + [r"\bottomrule", r"\end{tabular}"], out)


def task_descriptions(results, out):
    """Table A.4: the task descriptions Qwen3-Embedding-0.6B reads for NYT and IE, verbatim."""
    rows = [
        ("NYT, topic", data.TASK_DESCRIPTIONS["nyt"]["topic"]),
        ("NYT, location", data.TASK_DESCRIPTIONS["nyt"]["location"]),
        ("IE, emotion", data.TASK_DESCRIPTIONS["ie"]["emotion"]),
        ("IE, intent", data.TASK_DESCRIPTIONS["ie"]["intent"]),
    ]
    lines = [r"\begin{tabular}{ll}", r"\toprule", r"Question & Task description \\", r"\midrule"]
    lines += [f"{a} & {b} \\\\" for a, b in rows]
    return _write(lines + [r"\bottomrule", r"\end{tabular}"], out)


TABLES = {
    "main": main,
    "models": models,
    "datasets": datasets,
    "llms": llms,
    "settings": pair_settings,
    "instructions": instructions,
    "embedder_inputs": embedder_inputs,
    "task_descriptions": task_descriptions,
    "main_ci": main_ci,
    "breadth": breadth,
    "cost": cost,
}
"""Every table of the paper, by its file name."""


def write(results, directory, names=None):
    """Write the chosen tables (all by default) into `directory`."""
    return [
        TABLES[name](results, os.path.join(directory, f"{name}.tex")) for name in names or TABLES
    ]
