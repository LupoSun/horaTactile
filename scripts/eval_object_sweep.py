#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hora.utils.eval_sweep import default_output_dir, finalize_sweep_outputs, run_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a manifest-driven evaluation sweep over models and object variants.")
    parser.add_argument("manifest", type=Path, help="Path to a JSON manifest describing models, objects, and seeds.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write logs and results into. Defaults to outputs/eval_sweeps/<manifest>_<timestamp>/",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        help="Python executable to use for train.py subprocesses. Defaults to the interpreter running this script.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print and record commands without executing them.")
    parser.add_argument("--wandb-name", default="", help="Optional W&B run name for the sweep summary.")
    parser.add_argument("--wandb-group", default="eval", help="W&B group for the sweep summary run.")
    args = parser.parse_args()
    output_dir = args.output_dir or default_output_dir(args.manifest)

    results = run_sweep(
        manifest_path=args.manifest,
        output_dir=output_dir,
        python_executable=args.python_executable,
        dry_run=args.dry_run,
        wandb_name=args.wandb_name,
        wandb_group=args.wandb_group,
    )
    if args.dry_run:
        print(f"Prepared {len(results)} cases.")
    else:
        ok_count = sum(result["status"] == "ok" for result in results)
        print(f"Completed {ok_count} / {len(results)} cases successfully.")
        summary_rows, plot_paths = finalize_sweep_outputs(
            output_dir,
            wandb_name=args.wandb_name,
            wandb_group=args.wandb_group,
        )
        print(f"Wrote {len(summary_rows)} summary rows and {len(plot_paths)} plots.")


if __name__ == "__main__":
    main()
