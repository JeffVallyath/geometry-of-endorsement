from __future__ import annotations

import argparse
from pathlib import Path

from geometry_of_truth.truth.reproduce import reproduce_analysis, reproduce_full


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    analysis = subparsers.add_parser("analysis")
    analysis.add_argument("--cache-dir", type=Path, required=True)
    analysis.add_argument("--output-dir", type=Path, required=True)
    full = subparsers.add_parser("full")
    full.add_argument("--output-dir", type=Path, required=True)
    full.add_argument("--persistent-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "analysis":
        result = reproduce_analysis(args.cache_dir, args.output_dir)
        print(result["comparison"].to_string(index=False))
    else:
        result = reproduce_full(args.output_dir, args.persistent_dir)
        print(result["checks"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
