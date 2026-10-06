"""The look shared by every figure, and the results the figures and tables read.

Conventions: palette s1 blue #2a78d6, s2 orange #eb6834, ink #1E2126, muted #6B6F77; serif type and
Computer Modern math; one axis per chart; direct labels. Every figure is saved as a PDF at 400 dpi
and a PNG at 200 dpi.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

INK, MUTED, S1, S2 = "#1E2126", "#6B6F77", "#2a78d6", "#eb6834"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7,
        "mathtext.fontset": "cm",
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "pdf.fonttype": 42,
    }
)

LLM_NAMES = {
    "qwen3-0.6b": "Qwen3-0.6B",
    "qwen3-1.7b": "Qwen3-1.7B",
    "qwen3-4b": "Qwen3-4B",
    "qwen3-8b": "Qwen3-8B",
    "llama-3.1-8b-instruct": "Llama-3.1-8B",
    "olmo-3-7b-instruct": "OLMo-3-7B",
}
"""The instruction-tuned LLMs as the figures name them, in the figures' order."""

SHORT = {
    "qwen3-emb-0.6b": "Qwen3-Emb-0.6B",
    "bge-large-en-v1.5": "bge-large",
    "gte-large-en-v1.5": "gte-large",
    "all-mpnet-base-v2": "mpnet-base",
    "e5-large-v2": "e5-large",
}
"""Short names of the five core embedders."""

SCORE = {"nyt": "nyt", "ie": "ie"}
"""The score each dataset reports: NYT's harmonic mean, IE's sum over the two questions."""


def shade(colour, weight):
    """Blend a colour with white, by a weight between 0 (white) and 1 (the colour)."""
    rgb = np.array(to_rgb(colour))
    return tuple(1 - weight * (1 - rgb))


LLM_COLOURS = {
    "qwen3-0.6b": shade(S1, 0.3),
    "qwen3-1.7b": shade(S1, 0.58),
    "qwen3-4b": S1,
    "qwen3-8b": "#16457f",
    "llama-3.1-8b-instruct": S2,
    "olmo-3-7b-instruct": INK,
}


def number(value: float, digits: int = 1, signed: bool = False, minus: str = "−") -> str:
    """A number as the paper prints it: rounded to `digits` places, with a typographic minus sign.

    The value is rounded before its sign is set, so a value that rounds to zero prints as zero,
    with no sign: never -0.0, nor +0.0 when signed. The tables pass minus="$-$", for LaTeX.
    """
    text = format(value, f"+.{digits}f" if signed else f".{digits}f")
    if float(text) == 0:
        text = text.lstrip("+-")
    return text.replace("-", minus)


def clean(ax, left=True):
    """No top or right spine (and no left one, if asked); small ticks."""
    for spine in ("top", "right") + (() if left else ("left",)):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=2, labelsize=6, pad=1.5)


def save_figure(fig, path):
    """Save a figure as a PDF at 400 dpi and a PNG at 200 dpi, then close it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=400)
    fig.savefig(path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    return path


def legend_marks(ax, items, x, y, step, fontsize=5.8):
    """Legend entries drawn with the plot's own marks, in axes coordinates.

    items: pairs of scatter keyword arguments and a label, listed top to bottom.
    """
    for index, (marks, label) in enumerate(items):
        row = y - index * step
        ax.scatter([x], [row], transform=ax.transAxes, clip_on=False, zorder=5, **marks)
        ax.text(x + 0.05, row, label, transform=ax.transAxes, fontsize=fontsize, va="center")


class Results:
    """The results files of a run: results/<name>.json, read once each."""

    def __init__(self, directory: str):
        self.directory = directory
        self._loaded: dict[str, dict] = {}

    def __getitem__(self, name: str) -> dict:
        if name not in self._loaded:
            with open(os.path.join(self.directory, f"{name}.json")) as handle:
                self._loaded[name] = json.load(handle)
        return self._loaded[name]

    def gain(self, llm: str, embedder: str, dataset: str) -> float:
        """The relay's gain over the embedder alone, for one pair on NYT or IE."""
        pair = self["pairs"][f"{llm}|{embedder}"][dataset]
        return pair["relayed"][SCORE[dataset]] - pair["alone"][SCORE[dataset]]
