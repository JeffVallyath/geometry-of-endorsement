"""Single CPU acceptance command. Never initializes a pretrained model or CUDA."""
import argparse
import contextlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='CPU_CHECK.json');args=p.parse_args()
    output=Path(args.output)
    if output.exists():
        raise FileExistsError('Choose a new CPU receipt filename; previous receipts are retained')
    import torch
    torch.set_num_threads(4)
    started=time.perf_counter(); stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test*.py')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    details=stream.getvalue()
    # Portability: test receipts contain test IDs and compact messages, not tracebacks/paths.
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
        failures=[t.id() for t,_ in result.failures],errors=[t.id() for t,_ in result.errors],
        skipped=[dict(test=t.id(),reason=reason) for t,reason in result.skipped],
        passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        test_ids=[line.split(' ... ')[0] for line in details.splitlines() if ' ... ' in line],
        seconds=time.perf_counter()-started,pretrained_models_loaded=False,cuda_initialized=torch.cuda.is_initialized(),
        model_checks='PENDING: pinned backbones, real tokenizers, stock external adapter and performance',
        python='.'.join(map(str,sys.version_info[:3])),
        versions={k:importlib.metadata.version(k) for k in ('numpy','torch','transformers','tokenizers','safetensors')})
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
    print(details)
    print(json.dumps({k:report[k] for k in ('status','tests_run','passed','failures','errors','cuda_initialized')}))
    return 0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1

if __name__=='__main__':
    raise SystemExit(main())
