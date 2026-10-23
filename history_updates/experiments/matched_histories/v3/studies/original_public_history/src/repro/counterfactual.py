"""CPU-only replay of retained representation claims from saved scientific records.

Loads selected pure function definitions by AST, not their model-loading modules.
The output records source hashes, exact function line spans and every comparison.
"""
from __future__ import annotations

import ast
import argparse
import collections
import dataclasses
import gzip
import hashlib
import json
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import ROOT as REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT / "reproducibility/representation"
SOURCES = []
CHECKS = []


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    path = Path(path)
    if not path.exists() and path.with_name(path.name + ".gz").exists():
        path = path.with_name(path.name + ".gz")
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(data)


def definitions(path, names, env=None):
    path = Path(path)
    data = path.read_bytes()
    tree = ast.parse(data.decode("utf-8-sig"))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f"pure function selection incomplete: {path}")
    ns = {"np": np, "pd": pd, "Path": Path, "Any": Any, "dataclass": dataclasses.dataclass,
          "asdict": dataclasses.asdict, "defaultdict": collections.defaultdict, "collections": collections,
          "__name__": "__main__", **(env or {})}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    SOURCES.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest(data), "functions": {n.name: [n.lineno, n.end_lineno] for n in nodes}})
    return ns


def compare(name, got, expected, atol=2e-10):
    a, b = np.asarray(got, dtype=float), np.asarray(expected, dtype=float)
    error = float(np.max(np.abs(a-b))) if a.size else 0.0
    passed = a.shape == b.shape and bool(np.allclose(a, b, rtol=0, atol=atol, equal_nan=True))
    CHECKS.append({"claim": name, "max_absolute_error": error, "tolerance": atol, "passed": passed})
    return passed


def triple(x, center="mean"):
    return [x[center], x["ci_low"], x["ci_high"]]


def counterfactual():
    root=ROOT/"counterfactual_fidelity"
    source=root/"source"
    env={"CONSEQUENCE_QUERIES":("Q2","Q3","Q4","Q5","Q6"),"FOCAL_QUERIES":("Q2","Q3","Q4"),"GROUP_WEIGHTS":{"Q2":1/3,"Q3":1/3,"Q4":1/3,"Q5":.5,"Q6":.5},"BOOTSTRAP_REPLICATES":10000,"BOOTSTRAP_SEED":202609041720}
    math=definitions(source/"crif_v1_common.py",["group_weights","squared_distance","csr","bootstrap_indices","bootstrap_mean"],env)
    C=types.SimpleNamespace(**env,QUERIES=("Q1","Q2","Q3","Q4","Q5","Q6"),TAUS=(.25,.5,.75,1.),LOGIT_CONTROL="MATCHED_FINAL_LOGIT_CONTROL",**{k:math[k] for k in ("csr","bootstrap_indices","bootstrap_mean")})
    fit=definitions(source/"stcrf_v2_fit.py",["natural_signature"],{"C":C})
    funcs=definitions(source/"stcrf_v2_analysis.py",["world_signatures","applied_mask","csr_for"],{"C":C,"F":types.SimpleNamespace(natural_signature=fit["natural_signature"])})
    summaries=[json.loads(line) for line in gzip.open(root/"scores/final_gemma.jsonl.gz","rt")]
    summaries.sort(key=lambda s:s["world_id"])
    assert len(summaries)==96 and len({s['world_id'] for s in summaries})==96
    expected=load(root/"gemma_analysis.json")
    freeze=load(root/"analysis_freeze.json")
    scales=np.asarray(freeze["actors"]["gemma"]["scales_Q2_to_Q6"])
    output={}
    for fmt,real,label in (("F2","A","primary"),("F2","B","cross_template"),("F1","A","mapped_format")):
        sigs=[funcs["world_signatures"](s,fmt,real)|{"world_id":s["world_id"]} for s in summaries]
        cache={}
        for name,record in expected["csr"].items():
            wanted=record.get("applied" if label=="primary" else label)
            if wanted is None or name in {"NATURAL","NO_INTERVENTION"}:continue
            arm,tau=name.split("@") if "@" in name else (name,"")
            mask=funcs["applied_mask"](summaries,fmt,arm,tau)
            # Whole-state patch lacks an arm calibration row; its complete query records are the witness.
            if arm=="FULL_STATE_SOURCE_PATCH":mask=np.ones(96,dtype=bool)
            got=funcs["csr_for"](sigs,(arm,tau),scales,mask,cache)
            if got is None:
                CHECKS.append({"claim":f"counterfactual/{label}/{name}","passed":False,"reason":"no complete applied score rows"});continue
            for metric in ("csr","csr_focal","csr_invariance"):
                compare(f"counterfactual/{label}/{name}/{metric}",triple(got[metric]),triple(wanted[metric]))
            output[f"{label}/{name}"]={k:got[k] for k in ("csr","csr_focal","csr_invariance","n","mean_signature")}
    hits={}
    for arm in ("ORIGINAL_FROZEN_DIM","REGIME_MATCHED_DIM"):
        rows=[s['arms']['F2'][arm]['applied']['0.75'] for s in summaries]
        hits[arm]={"within_tolerance":float(np.mean([r['within_tolerance'] for r in rows])),"hard_answer_counterfactual":float(np.mean([r['hard_answer_counterfactual'] for r in rows]))}
        for k,v in hits[arm].items():compare(f"counterfactual/hits/{arm}/{k}",v,expected['dose'][arm]['0.75']['criteria'][k])
    output['hits']=hits
    output['qualification']='Descriptive, qualification-limited Gemma comparison. The original direction reaches the Q1 calibration tolerance in 96/96 contexts but fails the full dose rule (80/96 dose-valid contexts, below 90%). Its transport disposition remains STRONG_TARGET_UNREACHABLE. The refitted direction is DIRECTION_NOT_QUALIFIED; Llama is ASSAY_NOT_QUALIFIED.'
    return output


from .common import new_output
from .release_inputs import verify


def export(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a fresh output directory")
    verify(ROOT / "counterfactual_fidelity", "counterfactual")
    CHECKS.clear()
    SOURCES.clear()
    results = counterfactual()
    if len(CHECKS) != 145 or not all(row["passed"] for row in CHECKS):
        raise ValueError("Counterfactual saved-score comparison failed")
    output = new_output(output)
    report = dict(status="PASS", comparisons=CHECKS, source_functions=SOURCES,
                  formal_classifications=load(ROOT / "counterfactual_fidelity/results.json")["actor_classifications"],
                  inference=False, tolerance=2e-10)
    for name, value in (("RESULTS.json", results), ("VERIFICATION.json", report)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Reconstruct direct-answer consequence statistics from saved scores.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = export(args.output)
    print(json.dumps(dict(status=report["status"], comparisons=len(report["comparisons"]),
                          maximum_error=max(row["max_absolute_error"] for row in report["comparisons"]),
                          tolerance=report["tolerance"], inference=False), indent=2))


if __name__ == "__main__":
    main()
