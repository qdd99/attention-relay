"""The paper's figures, drawn from a run's results files."""

import os

import matplotlib.pyplot as plt
import numpy as np

from reproduce import settings
from reproduce.paper.style import (
    INK,
    LLM_COLOURS,
    LLM_NAMES,
    MUTED,
    S1,
    S2,
    SCORE,
    SHORT,
    clean,
    legend_marks,
    number,
    save_figure,
    shade,
)

QWEN = "qwen3-1.7b"


def intro(results, out):
    """Figure 1: one IntentEmotion message in Qwen3-1.7B's tokens, shaded by its reading weights
    under the two questions; the shading follows the square root of the weight within each row."""
    example = results["example"]
    tokens = example["llm"]["tokens"]
    weights = {aspect: np.array(w) for aspect, w in example["llm"]["weights"].items()}
    width_inches, height = 5.5, 0.5
    fig = plt.figure(figsize=(width_inches, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, width_inches)
    ax.set_ylim(0, height)
    renderer = fig.canvas.get_renderer()
    inverse = ax.transData.inverted()
    fontsize, label_right = 7.0, 1.75

    def text_width(text):
        drawn = ax.text(0, 0, text, fontsize=fontsize)
        box = drawn.get_window_extent(renderer=renderer)
        drawn.remove()
        (x0, _), (x1, _) = inverse.transform([[box.x0, box.y0], [box.x1, box.y1]])
        return x1 - x0

    shown = [token.strip() for token in tokens]
    x, centres = label_right + 0.1, []
    for index, text in enumerate(shown):
        w = text_width(text)
        if index > 0:
            x += 0.045 if tokens[index].startswith(" ") else 0.006
        centres.append(x + w / 2)
        x += w
    for aspect, y, colour in (("emotion", height - 0.14, S1), ("intent", height - 0.37, S2)):
        question = "“" + example["questions"][aspect] + "”"
        ax.text(
            label_right,
            y,
            question,
            ha="right",
            va="center",
            fontsize=6.6,
            color=colour,
            style="italic",
        )
        row, top = weights[aspect], weights[aspect].max()
        for index, text in enumerate(shown):
            background = shade(colour, 0.9 * (row[index] / top) ** 0.5)
            ax.text(
                centres[index],
                y,
                text,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=INK,
                bbox=dict(boxstyle="square,pad=0.14", fc=background, ec="none"),
            )
    return save_figure(fig, out)


def question_input(results, out):
    """Figure 2: the share of IntentEmotion triplets that follow the question (IE / 2), for four
    embedders trained without instructions given the question before (dot) or after (ring) the
    text, and Qwen3-Embedding-0.6B in its instruction mode with the task descriptions."""
    rows = ["all-mpnet-base-v2", "e5-large-v2", "bge-large-en-v1.5", "gte-large-en-v1.5"]
    in_input = results["question_in_input"]
    ys = -np.arange(len(rows))
    y_reference = ys[-1] - 1.35
    fig = plt.figure(figsize=(2.37, 1.8))
    ax = fig.add_axes([0.34, 0.23, 0.6, 0.57])
    ax.axvline(50, color=MUTED, lw=0.6, ls=(0, (2, 1.5)), zorder=1)
    ax.axvline(100, color=MUTED, lw=0.6, ls=(0, (2, 1.5)), zorder=1)
    ax.text(50.8, 0.55, "question\nignored", fontsize=5.4, ha="left", va="bottom", color=MUTED)
    ax.text(99.2, 0.55, "always\nfollowed", fontsize=5.4, ha="right", va="bottom", color=MUTED)
    for y, embedder in zip(ys, rows, strict=True):
        ax.axhline(y, color=shade(MUTED, 0.12), lw=0.5, zorder=0)
        before, after = (in_input[embedder][w]["ie"]["ie"] / 2 for w in ("before", "after"))
        ax.scatter([after], [y], s=26, facecolors="white", edgecolors=S1, linewidths=0.8, zorder=3)
        ax.scatter([before], [y], s=9, color=S1, zorder=4)
    instruction_mode = results["instruction_mode"]["task_description"]["ie"]["ie"] / 2
    ax.axhline(y_reference, color=shade(MUTED, 0.12), lw=0.5, zorder=0)
    ax.scatter([instruction_mode], [y_reference], s=9, color=S1, zorder=4)
    ax.set_yticks(list(ys) + [y_reference])
    ax.set_yticklabels([SHORT[e] for e in rows] + ["Qwen3-Emb-0.6B\n(instruction mode)"])
    ax.set_ylim(y_reference - 0.6, 0.6)
    ax.set_xlim(45, 102.5)
    ax.set_xticks([50, 75, 100])
    ax.set_xlabel("triplets that follow the question (%)", fontsize=6.2, labelpad=1.5)
    clean(ax)
    marks = (
        (-0.36, dict(s=9, color=S1), "question before the text"),
        (
            0.4,
            dict(s=26, facecolors="white", edgecolors=S1, linewidths=0.8),
            "question after the text",
        ),
    )
    for x, style, label in marks:
        ax.scatter([x], [1.3], transform=ax.transAxes, clip_on=False, zorder=5, **style)
        ax.text(x + 0.05, 1.3, label, transform=ax.transAxes, fontsize=5.6, va="center")
    return save_figure(fig, out)


def premises(results, out):
    """Figure 3: how much the reading weights move with the question at every block (the mean
    total variation between two questions' weights over NYT and IE), for the six instruction-tuned
    LLMs (solid) and the four base checkpoints (dashed, in their sibling's colour)."""
    blocks = results["blocks"]
    fig = plt.figure(figsize=(2.97, 1.62))
    ax = fig.add_axes([0.14, 0.26, 0.83, 0.7])
    for base in settings.BASE_CHECKPOINTS:
        sibling = settings.LLMS[base].template_from
        y = np.array(blocks[base]["mean"])
        x = np.arange(len(y)) / (len(y) - 1)
        ax.plot(x, y, color=LLM_COLOURS[sibling], lw=0.7, ls=(0, (2.5, 1.5)), zorder=2)
    for llm in LLM_NAMES:
        y = np.array(blocks[llm]["mean"])
        x = np.arange(len(y)) / (len(y) - 1)
        ax.plot(x, y, color=LLM_COLOURS[llm], lw=1.0, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.72)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.tick_params(length=2, labelsize=6)
    ax.set_xlabel("relative depth of the block", fontsize=6.3)
    ax.set_ylabel("question sensitivity", fontsize=6.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for index, llm in enumerate(LLM_NAMES):
        y = 0.66 - index * 0.058
        ax.plot([0.03, 0.08], [y, y], color=LLM_COLOURS[llm], lw=1.0)
        ax.text(0.095, y, LLM_NAMES[llm], fontsize=5.8, va="center")
    y = 0.66 - 6 * 0.058
    ax.plot([0.03, 0.08], [y, y], color=MUTED, lw=0.7, ls=(0, (2.5, 1.5)))
    ax.text(0.095, y, "base checkpoints", fontsize=5.8, va="center", color=MUTED)
    return save_figure(fig, out)


def method(results, out):
    """Figure 4: the method's flow; only the weights cross from the LLM to the embedder."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    width_inches, height = 5.5, 0.72
    fig = plt.figure(figsize=(width_inches, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, width_inches)
    ax.set_ylim(0, height)

    def box(x, w, label, sub, fill="white"):
        y, h = 0.3, 0.34
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=fill, ec=INK, lw=0.6
            )
        )
        ax.text(
            x + w / 2,
            y + h * 0.64,
            label,
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color=INK,
        )
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=5.8, color=MUTED)
        return x, x + w, y, y + h

    def arrow(start, end, above=None, below=None):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=6,
                lw=0.6,
                color=INK,
                shrinkA=0,
                shrinkB=0,
            )
        )
        xm, ym = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        if above:
            ax.text(xm, ym + 0.035, above, ha="center", va="bottom", fontsize=6.3, color=INK)
        if below:
            ax.text(xm, ym - 0.035, below, ha="center", va="top", fontsize=5.8, color=MUTED)

    llm_left, llm_right, bottom, top = box(0.45, 0.95, "LLM", "instruction-tuned", fill="#E3EEFB")
    middle = (bottom + top) / 2
    align_left, align_right, _, _ = box(
        2.00, 0.95, "alignment", "through characters", fill="#EFEFEF"
    )
    embed_left, embed_right, _, _ = box(
        3.55, 1.05, "embedder", "instruction-agnostic", fill="#E4F2E6"
    )
    arrow((llm_right, middle), (align_left, middle), "$w$", "reading weights")
    arrow((align_right, middle), (embed_left, middle), r"$\tilde w$", "into its pooling")
    arrow((embed_right, middle), (width_inches - 0.06, middle), "instruction-aware", "embedding")
    inputs = (
        ((llm_left + llm_right) / 2, "text, then question"),
        ((embed_left + embed_right) / 2, "text only"),
    )
    for x, label in inputs:
        arrow((x, 0.03), (x, bottom))
        ax.text(x + 0.05, 0.12, label, ha="left", va="center", fontsize=6, color=INK)
    return save_figure(fig, out)


def _heatmaps(
    results,
    rows,
    row_labels,
    columns,
    column_labels,
    groups,
    out,
    height,
    row_rules=(),
    axes_height=0.5,
):
    """Two annotated heatmaps, NYT (a) and IE (b), of the relay's gain over each embedder alone,
    one cell per (LLM, embedder)."""
    from matplotlib.colors import LinearSegmentedColormap

    fig = plt.figure(figsize=(5.5, height))
    colours = LinearSegmentedColormap.from_list("div", [S2, "white", S1])
    titles = (
        ("nyt", "NYT: gain over the embedder alone"),
        ("ie", "IE: gain over the embedder alone"),
    )
    for panel, (dataset, title) in enumerate(titles):
        gains = np.array([[results.gain(llm, e, dataset) for e in columns] for llm in rows])
        top = np.abs(gains).max()
        ax = fig.add_axes([0.14 + panel * 0.44, 0.03, 0.4, axes_height])
        ax.imshow(gains, cmap=colours, vmin=-top, vmax=top, aspect="auto")
        for r in range(len(rows)):
            for c in range(len(columns)):
                value = gains[r, c]
                label = number(value, digits=0, signed=True)
                colour = "white" if abs(value) > 0.62 * top else INK
                ax.text(c, r, label, ha="center", va="center", fontsize=5.2, color=colour)
        for y in row_rules:
            ax.axhline(y, color="white", lw=1.6)
        for first, _, _ in groups[1:]:
            ax.axvline(first - 0.5, color="white", lw=1.2)
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels(column_labels, rotation=55, ha="left", fontsize=5.4)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(row_labels if panel == 0 else [], fontsize=5.6)
        ax.tick_params(length=0, pad=1.5, labelcolor="black")
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.text(
            0.14 + panel * 0.44, 0.99, f"({'ab'[panel]})", fontsize=7, fontweight="bold", va="top"
        )
        fig.text(0.165 + panel * 0.44, 0.99, title, fontsize=6.3, va="top")
        for first, last, name in groups:
            x = 0.14 + panel * 0.44 + 0.4 * ((first + last) / 2) / len(columns)
            fig.text(x, 0.925, name, fontsize=5.6, ha="center", va="top", color="black")
    return save_figure(fig, out)


def gains(results, out):
    """Figure 5: the relay's gain for the six instruction-tuned LLMs and the ten embedders."""
    columns = [
        "qwen3-emb-0.6b",
        "qwen3-emb-4b",
        "qwen3-emb-8b",
        "bge-small-en-v1.5",
        "bge-base-en-v1.5",
        "bge-large-en-v1.5",
        "gte-large-en-v1.5",
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "e5-large-v2",
    ]
    labels = [
        "Qwen3-Emb-0.6B",
        "Qwen3-Emb-4B",
        "Qwen3-Emb-8B",
        "bge-small",
        "bge-base",
        "bge-large",
        "gte-large",
        "MiniLM-L6",
        "mpnet-base",
        "e5-large",
    ]
    groups = [(0, 3, "last token"), (3, 7, "[CLS]"), (7, 10, "mean")]
    rows = list(LLM_NAMES)
    row_labels = [LLM_NAMES[llm] for llm in LLM_NAMES]
    return _heatmaps(
        results, rows, row_labels, columns, labels, groups, out, height=2.05, axes_height=0.5
    )


def base(results, out):
    """Figure 6: each base checkpoint's weights against its instruction-tuned sibling's, relayed
    into the five core embedders; each family's instruction-tuned row above its base row."""
    rows, row_labels = [], []
    for base_checkpoint in settings.BASE_CHECKPOINTS:
        sibling = settings.LLMS[base_checkpoint].template_from
        rows += [sibling, base_checkpoint]
        row_labels += [LLM_NAMES[sibling], f"{LLM_NAMES[sibling]} base"]
    columns = settings.CORE_EMBEDDERS
    groups = [(0, 1, "last token"), (1, 3, "[CLS]"), (3, 5, "mean")]
    return _heatmaps(
        results,
        rows,
        row_labels,
        columns,
        [SHORT[e] for e in columns],
        groups,
        out,
        height=2.35,
        row_rules=(1.5, 3.5, 5.5),
        axes_height=0.56,
    )


def blocks(results, out):
    """Figure 9: the gain of relaying from each block of each LLM, averaged over all-mpnet-base-v2
    and e5-large-v2, against the block's relative depth; the dot marks the block the rule reads."""
    sweep, chosen = results["sweep"], results["blocks"]
    fig = plt.figure(figsize=(5.5, 1.85))
    titles = (
        ("nyt", "NYT: gain over the embedder alone"),
        ("ie", "IE: gain over the embedder alone"),
    )
    for panel, (dataset, title) in enumerate(titles):
        ax = fig.add_axes([0.075 + panel * 0.5, 0.2, 0.41, 0.64])
        score = SCORE[dataset]
        for llm in LLM_NAMES:
            by_embedder = []
            for embedder in ("all-mpnet-base-v2", "e5-large-v2"):
                entry = sweep[f"{llm}|{embedder}"][dataset]
                alone = entry["alone"][score]
                by_embedder.append(
                    {int(b): v[score] - alone for b, v in entry.items() if b != "alone"}
                )
            block_list = sorted(by_embedder[0])
            y = np.array([(by_embedder[0][b] + by_embedder[1][b]) / 2 for b in block_list])
            n = max(block_list) + 1
            x = np.array(block_list) / (n - 1)
            ax.plot(x, y, color=LLM_COLOURS[llm], lw=1.0, zorder=3)
            read = chosen[llm]["chosen"]
            ax.scatter(
                [read / (n - 1)],
                [y[block_list.index(read)]],
                s=10,
                color=LLM_COLOURS[llm],
                zorder=4,
                edgecolors="white",
                linewidths=0.4,
            )
        ax.axhline(0, color=MUTED, lw=0.5, zorder=1)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.tick_params(length=2, labelsize=6)
        ax.set_xlabel("relative depth of the block the weights come from", fontsize=6.3)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.text(
            0.01 + panel * 0.5, 0.97, f"({'ab'[panel]})", fontsize=7, fontweight="bold", va="top"
        )
        fig.text(0.045 + panel * 0.5, 0.97, title, fontsize=6.3, va="top")
    ax = fig.axes[0]
    for index, llm in enumerate(LLM_NAMES):
        y = 0.93 - index * 0.085
        ax.plot([0.03, 0.08], [y, y], color=LLM_COLOURS[llm], lw=1.0, transform=ax.transAxes)
        ax.text(0.095, y, LLM_NAMES[llm], fontsize=5.6, va="center", transform=ax.transAxes)
    return save_figure(fig, out)


def controls(results, out):
    """Figure 7: the share of the relay's gain each change to the weights or the text keeps,
    Qwen3-1.7B into bge-large-en-v1.5."""
    pair = results["pairs"][f"{QWEN}|bge-large-en-v1.5"]
    removal = results["word_removal"]["bge-large-en-v1.5"]
    alone = {d: pair[d]["alone"][SCORE[d]] for d in SCORE}
    full = {d: pair[d]["relayed"][SCORE[d]] - alone[d] for d in SCORE}
    rows = [
        (
            "shuffled within the text",
            {d: np.mean([x[SCORE[d]] for x in pair[d]["shuffled"]]) - alone[d] for d in SCORE},
        ),
        (
            "another text’s weights",
            {d: np.mean([x[SCORE[d]] for x in pair[d]["other_text"]]) - alone[d] for d in SCORE},
        ),
        ("a profile of positions", {d: pair[d]["positional"][SCORE[d]] - alone[d] for d in SCORE}),
        (
            "top tenth of words removed",
            {
                d: removal[d]["top_relayed"][SCORE[d]] - removal[d]["top_alone"][SCORE[d]]
                for d in SCORE
            },
        ),
        (
            "random tenth of words removed",
            {
                d: removal[d]["random_relayed"][SCORE[d]] - removal[d]["random_alone"][SCORE[d]]
                for d in SCORE
            },
        ),
    ]
    ys = -np.arange(len(rows))
    fig = plt.figure(figsize=(2.3, 1.4))
    ax = fig.add_axes([0.52, 0.24, 0.45, 0.58])
    ax.axvline(0, color=MUTED, lw=0.5, zorder=1)
    ax.axvline(100, color=INK, lw=0.6, ls=(0, (2, 1.5)), zorder=1)
    for y, (_, gain) in zip(ys, rows, strict=True):
        ax.axhline(y, color=shade(MUTED, 0.12), lw=0.5, zorder=0)
        ax.scatter([100 * gain["nyt"] / full["nyt"]], [y], s=13, color=S1, zorder=3)
        ax.scatter([100 * gain["ie"] / full["ie"]], [y], s=13, color=S2, marker="D", zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([label for label, _ in rows])
    ax.set_ylim(ys[-1] - 0.6, 0.6)
    ax.set_xlim(-30, 130)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("share of the relay’s gain kept (%)", fontsize=6.2, labelpad=1.5)
    clean(ax)
    legend_marks(
        ax,
        [(dict(s=13, color=S1), "NYT"), (dict(s=13, color=S2, marker="D"), "IE")],
        0.03,
        1.25,
        0.14,
    )
    return save_figure(fig, out)


def split(results, out):
    """Figure 8: the gain in NYT's location score in the articles that name a place and in the
    rest, each subset clustered on its own; Qwen3-1.7B's weights."""
    places = results["places"]
    counts = places[settings.CORE_EMBEDDERS[0]]
    ys = -np.arange(len(settings.CORE_EMBEDDERS))
    fig = plt.figure(figsize=(2.2, 1.3))
    ax = fig.add_axes([0.35, 0.25, 0.61, 0.55])
    ax.axvline(0, color=MUTED, lw=0.5, zorder=1)
    for y, embedder in zip(ys, settings.CORE_EMBEDDERS, strict=True):
        named, implied = places[embedder]["named"]["gain"], places[embedder]["not_named"]["gain"]
        ax.plot([implied, named], [y, y], color=shade(MUTED, 0.3), lw=0.7, zorder=2)
        ax.scatter([named], [y], s=13, color=S1, zorder=3)
        ax.scatter([implied], [y], s=13, color=S2, marker="D", zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([SHORT[e] for e in settings.CORE_EMBEDDERS])
    ax.set_ylim(ys[-1] - 0.6, 0.6)
    ax.set_xlim(0, 30)
    ax.set_xlabel("location V-measure gain (points)", fontsize=6.2, labelpad=1.5)
    clean(ax)
    legend_marks(
        ax,
        [
            (dict(s=13, color=S1), f"place named ({counts['named']['n']:,} articles)"),
            (dict(s=13, color=S2, marker="D"), f"only implied ({counts['not_named']['n']:,})"),
        ],
        0.03,
        1.3,
        0.14,
    )
    return save_figure(fig, out)


def probe(results, out):
    """Figure 10: a linear probe's accuracy and the V-measure of the asked aspect, each embedder
    alone (dot) and relayed (arrow head): NYT's location under the location question and
    FewNerd's entity type. Every panel spans 40 points."""
    surfacing, items = results["surfacing"], results["items"]
    ys = -np.arange(len(settings.CORE_EMBEDDERS))

    def values(dataset, embedder, metric):
        if dataset == "nyt":
            location = surfacing[embedder]["location"]
            return location["alone"][metric], location["relayed"][metric]
        entry = items[f"{QWEN}|{embedder}|fewnerd"]
        return entry["alone"][metric], entry["relayed"][metric]

    panels = [
        ("nyt", "probe", "NYT location: probe accuracy (%)", (60, 100)),
        ("nyt", "v", "NYT location: V-measure", (45, 85)),
        ("fewnerd", "probe", "FewNerd: probe accuracy (%)", (40, 80)),
        ("fewnerd", "v", "FewNerd: V-measure", (35, 75)),
    ]
    fig = plt.figure(figsize=(5.5, 1.45))
    left, width, gap = 0.15, 0.185, 0.028
    for panel, (dataset, metric, label, limits) in enumerate(panels):
        ax = fig.add_axes([left + panel * (width + gap), 0.27, width, 0.6])
        colour = S2 if metric == "probe" else S1
        for y, embedder in zip(ys, settings.CORE_EMBEDDERS, strict=True):
            alone, relayed = values(dataset, embedder, metric)
            ax.axhline(y, color=shade(MUTED, 0.12), lw=0.5, zorder=0)
            ax.scatter([alone], [y], s=7, color=colour, zorder=3)
            ax.annotate(
                "",
                xy=(relayed, y),
                xytext=(alone, y),
                arrowprops=dict(
                    arrowstyle="-|>", color=colour, lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=6
                ),
                zorder=4,
            )
        ax.set_xlim(*limits)
        ax.set_ylim(ys[-1] - 0.6, 0.6)
        ax.set_yticks(ys)
        ax.set_yticklabels([SHORT[e] for e in settings.CORE_EMBEDDERS] if panel == 0 else [])
        ax.set_xticks([limits[0], limits[0] + 20, limits[1]])
        ax.set_xlabel(label, fontsize=6.0, labelpad=1.5)
        clean(ax)
        fig.text(
            left + panel * (width + gap),
            0.97,
            f"({'abcd'[panel]})",
            fontsize=7,
            fontweight="bold",
            va="top",
        )
    return save_figure(fig, out)


def _gain(entry, dataset):
    """The relay's gain in one entry's scores on NYT or IE."""
    return entry[dataset]["relayed"][SCORE[dataset]] - entry[dataset]["alone"][SCORE[dataset]]


def layers(results, out):
    """Figure 11: the gain of relaying into only the first or only the second half of the
    embedder's layers, as a share of the gain of relaying into every layer."""
    ranges = results["range"]
    rows = []
    for embedder in ("qwen3-emb-0.6b", "gte-large-en-v1.5"):
        for dataset in ("nyt", "ie"):
            every = results.gain(QWEN, embedder, dataset)
            first = 100 * _gain(ranges[embedder]["first"], dataset) / every
            second = 100 * _gain(ranges[embedder]["second"], dataset) / every
            rows.append((embedder, dataset, first, second))
    ys = np.array([0, -1, -2.5, -3.5])
    fig = plt.figure(figsize=(2.2, 1.3))
    ax = fig.add_axes([0.45, 0.25, 0.52, 0.55])
    ax.axvline(0, color=MUTED, lw=0.5, zorder=1)
    ax.axvline(100, color=INK, lw=0.6, ls=(0, (2, 1.5)), zorder=1)
    for y, (_, _, first, second) in zip(ys, rows, strict=True):
        ax.plot([first, second], [y, y], color=shade(S1, 0.3), lw=0.7, zorder=2)
        ax.scatter([first], [y], s=13, facecolors="white", edgecolors=S1, linewidths=0.8, zorder=3)
        ax.scatter([second], [y], s=13, color=S1, zorder=3)
        ax.text(first, y + 0.33, number(first, digits=0), fontsize=5.4, ha="center", va="bottom")
        ax.text(second, y + 0.33, number(second, digits=0), fontsize=5.4, ha="center", va="bottom")
    ax.set_yticks(ys)
    ax.set_yticklabels(["NYT", "IE", "NYT", "IE"])
    ax.set_ylim(-4.0, 0.75)
    ax.set_xlim(-15, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("share of the every-layer gain (%)", fontsize=6.2, labelpad=1.5)
    clean(ax)
    for y, embedder in ((-0.5, "qwen3-emb-0.6b"), (-3.0, "gte-large-en-v1.5")):
        ax.text(
            -0.2,
            (y - (-4.0)) / 4.75,
            SHORT[embedder],
            transform=ax.transAxes,
            fontsize=6,
            ha="right",
            va="center",
        )
    legend_marks(
        ax,
        [
            (
                dict(s=13, facecolors="white", edgecolors=S1, linewidths=0.8),
                "first half of the layers",
            ),
            (dict(s=13, color=S1), "second half"),
        ],
        0.0,
        1.24,
        0.14,
    )
    return save_figure(fig, out)


def bias(results, out):
    """Figure 12: the plain bias minus the mass-preserving bias, Qwen3-1.7B into bge-large-en-v1.5,
    NYT and IE, with the 95% paired bootstrap interval of the difference."""
    mass_preserving = results["pairs"][f"{QWEN}|bge-large-en-v1.5"]
    plain, difference = results["bias"]["plain"], results["bias"]["difference"]
    ys = [0, -1]
    fig = plt.figure(figsize=(2.2, 1.3))
    ax = fig.add_axes([0.13, 0.25, 0.84, 0.5])
    ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
    for y, dataset in zip(ys, ("nyt", "ie"), strict=True):
        score = SCORE[dataset]
        value = plain[dataset]["relayed"][score] - mass_preserving[dataset]["relayed"][score]
        low, high = difference[dataset]["difference_interval"]
        ax.plot([low, high], [y, y], color=S1, lw=0.9, zorder=2)
        ax.scatter([value], [y], s=13, color=S1, zorder=3)
        label = f"{number(value, signed=True)} [{number(low)}, {number(high)}]"
        ax.text(value, y + 0.2, label, fontsize=5.6, ha="center", va="bottom")
    ax.set_yticks(ys)
    ax.set_yticklabels(["NYT", "IE"])
    ax.set_ylim(-1.45, 0.65)
    ax.set_xlim(-6, 6)
    ax.set_xlabel("plain minus mass-preserving bias (points)", fontsize=6.2, labelpad=1.5)
    clean(ax)
    return save_figure(fig, out)


FIGURES = {
    "fig_intro": intro,
    "fig_question_input": question_input,
    "fig_premises": premises,
    "fig_method": method,
    "fig_gains": gains,
    "fig_base": base,
    "fig_blocks": blocks,
    "fig_controls": controls,
    "fig_split": split,
    "fig_probe": probe,
    "fig_layers": layers,
    "fig_bias": bias,
}
"""Every figure of the paper, by its file name, in the paper's order."""


def draw(results, directory, names=None):
    """Draw the chosen figures (all by default) into `directory`."""
    return [
        FIGURES[name](results, os.path.join(directory, f"{name}.pdf")) for name in names or FIGURES
    ]
