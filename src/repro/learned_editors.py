"""Check selected editor parameters and synthetic CPU serving boundaries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .common import ROOT, load_json, new_output
from .release_inputs import verify

BASE = ROOT / 'reproducibility/learned_editors'


def check(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a fresh output directory')
    verify(BASE, 'learned_editors')
    destination = new_output(output)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    receipt = destination / 'CPU_CHECK.json'
    completed = subprocess.run([sys.executable, '-B', str(BASE / 'check_cpu.py'), '--output', str(receipt)],
                               env=environment, capture_output=True, text=True, timeout=600)
    if not receipt.is_file():
        raise RuntimeError('Editor CPU checks did not produce a receipt')
    result = load_json(receipt)
    if completed.returncode or result['status'] != 'PASS' or result['cuda_initialized'] or result['pretrained_models_loaded']:
        raise RuntimeError('Editor CPU checks failed; see the output receipt')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    result = check(parser.parse_args().output)
    print(json.dumps({k: result[k] for k in ('status', 'tests_run', 'passed', 'failures', 'errors', 'cuda_initialized')}, indent=2))


if __name__ == '__main__':
    main()
