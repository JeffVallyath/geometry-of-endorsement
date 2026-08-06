from __future__ import annotations

import argparse
from pathlib import Path

from geometry_of_truth.leakage.reproduce import reproduce


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = reproduce(args.output_dir)
    print(result["comparison"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
