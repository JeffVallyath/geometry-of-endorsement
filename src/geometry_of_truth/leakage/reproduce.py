from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geometry_of_truth.common.artifacts import repository_root
from geometry_of_truth.leakage.reproduction.build_confirmatory_split import manifest_hash

from .contracts import DATASET_REVISION, load_bundle


MODULE = "geometry_of_truth.leakage.reproduction"

def _run(module: str, arguments: list[str], cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    started = time.perf_counter()
    command = [sys.executable, "-m", f"{MODULE}.{module}", *arguments]
    result = subprocess.run(command, cwd=cwd, env=env, text=True, encoding="utf-8", errors="replace", capture_output=True)
    record = {
        "module": module,
        "command": " ".join(command),
        "seconds": time.perf_counter() - started,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    if result.returncode:
        raise RuntimeError(f"ValuePrism reconstruction failed in {module}\n{result.stderr}")
    return record


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_hash(path: Path) -> str:
    return manifest_hash(pd.read_csv(path, dtype={"row_id": str})["row_id"])


CONFIRMATORY_ROW_COLUMNS = (
    "row_id_s1_A",
    "row_id_s1_B",
    "row_id_s2_A",
    "row_id_s2_B",
)


def _confirmatory_row_hash(path: Path) -> str:
    frame = pd.read_csv(path, dtype=str)
    missing = [column for column in CONFIRMATORY_ROW_COLUMNS if column not in frame]
    if missing:
        raise RuntimeError(f"Confirmatory manifest is missing row columns  {missing}")
    row_ids = set(frame.loc[:, CONFIRMATORY_ROW_COLUMNS].to_numpy().ravel())
    return manifest_hash(row_ids)


def compare_rebuild(output_root: str | Path) -> pd.DataFrame:
    output = Path(output_root).resolve()
    frozen = load_bundle()["results"]
    audit = _load(output / "audit.json")
    multiseed = _load(output / "multiseed.json")
    power = _load(output / "power.json")
    manifest = _load(output / "MANIFEST.json")
    ultra = _load(output / "reports" / "valueprism_sensitivity.json")
    runtime_config = _load(output / "configs" / "valueprism_sensitivity.json")
    consideration = np.array([multiseed["leak_delta_considerations"][str(index)] for index in range(5)])
    situation = np.array([multiseed["leak_delta_situations"][str(index)] for index in range(5)])
    checks: list[tuple[str, Any, Any, float | None]] = [
        ("binary rows", audit["n_rows_binary"], frozen["audit"]["binary_rows"], 0),
        ("both valence situations", audit["n_both_valence_situations"], frozen["audit"]["both_valence_situations"], 0),
        ("reversing considerations", audit["n_both_valence_considerations_exact"], frozen["audit"]["both_valence_exact_considerations"], 0),
        ("candidate checkerboards", power["n_available_boards"], frozen["audit"]["candidate_checkerboards"], 0),
        ("consideration pairs", power["n_pairs"], frozen["audit"]["consideration_pairs"], 0),
        ("consideration restoration mean pp", float(100 * consideration.mean()), frozen["stress_test"]["consideration_mean_pp"], 1e-12),
        ("situation restoration mean pp", float(100 * situation.mean()), frozen["stress_test"]["situation_mean_pp"], 1e-12),
        ("strict train rows", len(pd.read_csv(output / "manifest_strict_train.csv")), frozen["strict_split"]["train_rows"], 0),
        ("strict test rows", len(pd.read_csv(output / "manifest_strict_test.csv")), frozen["strict_split"]["test_rows"], 0),
        ("ranked candidates", manifest["candidates_selected"], frozen["candidate_audit"]["ranked_candidates"], 0),
        ("target confirmed", manifest["target_confirmed_boards"], frozen["candidate_audit"]["target_confirmed"], 0),
        ("U1 rows", ultra["coverage"]["U1"]["rows"], frozen["sensitivity"]["coverage"]["U1"]["rows"], 0),
        ("U1 comparisons", ultra["coverage"]["U1"]["within_situation_comparisons"], frozen["sensitivity"]["coverage"]["U1"]["within_situation_comparisons"], 0),
    ]
    for index, expected in enumerate(frozen["stress_test"]["strict_scores"]):
        checks.append((f"seed {index} strict score", multiseed["leak_T0"][str(index)], expected, 1e-10))
    for index, expected in enumerate(frozen["stress_test"]["consideration_deltas"]):
        checks.append((f"seed {index} consideration delta", multiseed["leak_delta_considerations"][str(index)], expected, 1e-10))
    for index, expected in enumerate(frozen["stress_test"]["situation_deltas"]):
        checks.append((f"seed {index} situation delta", multiseed["leak_delta_situations"][str(index)], expected, 1e-10))
    for key, expected in frozen["strict_split"]["overlaps"].items():
        checks.append((f"strict overlap {key}", ultra["strict_split_reproduction"]["overlaps"][key], expected, 0))
    for filename, expected in runtime_config["frozen_inputs"].items():
        checks.append((f"file hash {filename}", _sha256(output / filename), expected, None))
    checks.extend([
        ("strict train row hash", _row_hash(output / "manifest_strict_train.csv"), frozen["strict_split"]["train_row_hash"], None),
        ("strict test row hash", _row_hash(output / "manifest_strict_test.csv"), frozen["strict_split"]["test_row_hash"], None),
        ("common training row hash", _row_hash(output / "manifest_train_common.csv"), manifest["common_training_manifest"]["hash_rows"], None),
        ("confirmatory row-membership hash", _confirmatory_row_hash(output / "manifest_confirmatory.csv"), manifest["hash_confirmatory_rows"], None),
        ("U1 exclusion row hash", ultra["artifacts"]["manifests/strict_U1_high_precision_clean.csv"]["row_hash"], frozen["sensitivity"]["coverage"]["U1"]["row_hash"], None),
    ])
    rows = []
    for name, observed, expected, tolerance in checks:
        passed = observed == expected if tolerance is None else abs(float(observed) - float(expected)) <= tolerance
        rows.append({"quantity": name, "reproduced": observed, "frozen": expected, "tolerance": "exact" if tolerance is None else tolerance, "pass": passed})
    table = pd.DataFrame(rows)
    if not table["pass"].all():
        raise RuntimeError("ValuePrism headline reconstruction differs from the public aggregate")
    return table


def reproduce(output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = repository_root()
    runtime_config = output / "configs" / "valueprism_sensitivity.json"
    runtime_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "configs" / "valueprism_sensitivity.json", runtime_config)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONHASHSEED"] = "0"
    env["VALUEPRISM_DATASET_REVISION"] = DATASET_REVISION
    env["GEOMETRY_OF_TRUTH_WORKDIR"] = str(output)
    source_root = str(root / "src")
    env["PYTHONPATH"] = source_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    commands = [
        ("audit_valueprism", ["--out", "audit.json"]),
        ("split_stress_test", ["--max-train", "60000", "--max-test", "25000", "--out", "stress.json"]),
        ("multi_seed", ["--seeds", "5", "--max-train", "40000", "--out", "multiseed.json"]),
        ("relation_purity", ["--max-train", "60000", "--max-test", "25000", "--out", "purity.json"]),
        ("power_simulation", ["--trials", "1500", "--out", "power.json"]),
        ("build_confirmatory_split", ["--candidates", "1090", "--cluster-cap", "14", "--target", "800"]),
        ("sensitivity_analysis", ["--config", str(runtime_config)]),
    ]
    records = [_run(module, arguments, output, env) for module, arguments in commands]
    return {"commands": pd.DataFrame(records), "comparison": compare_rebuild(output), "output_root": output}
