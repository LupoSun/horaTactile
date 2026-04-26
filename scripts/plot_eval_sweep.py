#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hora.utils.eval_plots import (
    DEFAULT_METRICS,
    resolve_results_csv,
    write_eval_summary_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and plot results from an evaluation sweep.")
    parser.add_argument(
        "results_path",
        type=Path,
        help="Path to results.csv or to an eval sweep output directory containing results.csv.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to summarize/plot. Defaults to the standard eval metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write the summary tables and plots into. Defaults to <eval_dir>/plots/",
    )
    args = parser.parse_args()

    results_csv = resolve_results_csv(args.results_path)
    eval_dir = results_csv.parent
    output_dir = args.output_dir or (eval_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows, plot_paths = write_eval_summary_outputs(results_csv, output_dir=output_dir, metrics=args.metrics)
    if not summary_rows:
        raise SystemExit("No successful eval rows were found to summarize.")

    summary_csv = output_dir / "summary.csv"
    summary_md = output_dir / "summary.md"

    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote summary Markdown: {summary_md}")
    for plot_path in plot_paths:
        print(f"Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()
