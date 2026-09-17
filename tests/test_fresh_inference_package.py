"""The portable entry point must default to no pretrained model execution."""
import importlib.util
import json
import subprocess
import sys

import pytest

from repro.common import ROOT


def test_fresh_entry_point_is_cpu_only_by_default(tmp_path):
    if importlib.util.find_spec('torch') is None:
        pytest.skip('Optional original scientific CPU smoke requires torch; saved-score suite does not')
    result = subprocess.run([sys.executable, str(ROOT / 'reproducibility/fresh_inference/run.py')],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    value = json.loads(result.stdout)
    assert value['status'] == 'CPU_SCIENTIFIC_PACKAGE_SMOKE_PASS'
    assert value['neural_forward_passes'] == 0
    assert value['checkpoints'] == 10
    assert value['origins_per_model'] == 8
    assert value['neural_readiness'].startswith('NOT_VERIFIED')


def test_inference_requires_explicit_inputs(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'reproducibility/fresh_inference/run.py'), '--execute'],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'requires a local --cache and fresh --output' in result.stderr
