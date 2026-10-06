"""Draw the paper's figures and write its tables from a run's results files.

python -m reproduce.paper                                  (the paper's results, into paper/)
python -m reproduce.paper --results runs/paper/results --out paper fig_gains main
"""

import argparse
import os

from reproduce.paper import figures, tables
from reproduce.paper.style import Results


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", default="results/paper", help="a directory of results files")
    parser.add_argument(
        "--out", default="paper", help="figures go to <out>/figures, tables to <out>/tables"
    )
    parser.add_argument("names", nargs="*", help="figures or tables to make (default: all)")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.names) - set(figures.FIGURES) - set(tables.TABLES))
    if unknown:
        parser.error(f"unknown figures or tables: {unknown}")
    results = Results(args.results)
    chosen_figures = [n for n in args.names if n in figures.FIGURES] if args.names else None
    chosen_tables = [n for n in args.names if n in tables.TABLES] if args.names else None
    if chosen_figures is None or chosen_figures:
        for path in figures.draw(results, os.path.join(args.out, "figures"), chosen_figures):
            print(path)
    if chosen_tables is None or chosen_tables:
        for path in tables.write(results, os.path.join(args.out, "tables"), chosen_tables):
            print(path)


if __name__ == "__main__":
    main()
