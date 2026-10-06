"""Tests for the paper's figures and tables, drawn from the paper's own results files."""

import os
import re

import pytest

pytest.importorskip("matplotlib")

from reproduce.paper import figures, tables  # noqa: E402
from reproduce.paper.style import Results, number  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results", "paper")


@pytest.fixture(scope="module")
def drawn(tmp_path_factory):
    out = tmp_path_factory.mktemp("paper")
    results = Results(RESULTS)
    return (
        out,
        figures.draw(results, str(out / "figures")),
        tables.write(results, str(out / "tables")),
    )


def test_every_figure_is_drawn_as_a_pdf_and_a_png(drawn):
    _, drawn_figures, _ = drawn
    assert len(drawn_figures) == len(figures.FIGURES) == 12
    for path in drawn_figures:
        assert os.path.getsize(path) > 0 and os.path.getsize(path.replace(".pdf", ".png")) > 0


def test_every_table_is_written(drawn):
    _, _, written = drawn
    assert len(written) == len(tables.TABLES) == 11
    with open([p for p in written if p.endswith("main.tex")][0]) as handle:
        main = handle.read()
    assert main.startswith(r"\begin{tabular}{lcccccc}")
    assert "Qwen3-Embedding-0.6B (instruction mode)" in main


def test_numbers_are_rounded_before_the_sign_is_set():
    assert number(-0.04) == "0.0"
    assert number(-0.04, signed=True) == number(0.04, signed=True) == "0.0"
    assert number(-0.4, digits=0, signed=True) == "0"
    assert number(-1.26) == "−1.3"
    assert number(1.26, signed=True) == "+1.3"
    assert number(-2.0, minus="$-$") == "$-$2.0"
    assert number(12.3456, digits=2) == "12.35"


def test_no_table_prints_a_negative_zero(drawn):
    _, _, written = drawn
    for path in written:
        with open(path) as handle:
            assert not re.search(r"\$-\$0\.0+(?![0-9])", handle.read()), path
