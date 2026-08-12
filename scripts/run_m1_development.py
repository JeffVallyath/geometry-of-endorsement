#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from geometry_of_truth.m1.runner import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen model-specific M1 moral-relation vertical slice."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--truth-results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--persistent-dir", type=Path)
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument(
        "--m0-dir",
        required=True,
        type=Path,
        help="ValuePrism reconstruction directory containing the frozen M0 manifests.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Require a smoke-mode config and run the same provenance gates.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the plan without GPU, network, or gated-data access.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(
        args.config,
        args.output_dir,
        truth_results=args.truth_results,
        persistent_dir=args.persistent_dir,
        manifest_dir=args.manifest_dir,
        m0_dir=args.m0_dir,
        dry_run=args.dry_run,
        require_smoke_mode=args.smoke_test,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
