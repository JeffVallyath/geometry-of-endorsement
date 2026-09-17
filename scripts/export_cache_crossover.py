"""Portable adaptation of the supplied CPU-only reporting exporter.

Reuses the current repo's independent replay and already hash-verified scientific
inputs. No separate historical analysis checkout, model loader or archive is
needed. The original panel and one-case extension remain separate tables.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from repro.cache_crossover_extension import export


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='Fresh output location')
    args = parser.parse_args()
    print(json.dumps(export(args.output), indent=2))
