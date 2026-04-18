#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hora.utils.eval_sweep import run_sweep


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
    args = parser.parse_args()

    results = run_sweep(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        python_executable=args.python_executable,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"Prepared {len(results)} cases.")
    else:
        ok_count = sum(result["status"] == "ok" for result in results)
        print(f"Completed {ok_count} / {len(results)} cases successfully.")


if __name__ == "__main__":
    main()
