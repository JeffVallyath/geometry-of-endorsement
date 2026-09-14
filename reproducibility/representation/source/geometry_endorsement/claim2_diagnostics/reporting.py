from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: str | Path, data: bytes) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_bytes(data)
    if target.exists():
        if sha256_file(target) != digest:
            raise RuntimeError(f"Refusing to overwrite immutable artifact: {target}")
        return digest
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
    except Exception:
        try:
            target.unlink()
        except OSError:
            pass
        raise
    return digest


def write_json_once(path: str | Path, value: Any) -> str:
    return write_once(path, canonical_json_bytes(value))


def write_text_once(path: str | Path, text: str) -> str:
    normalized = text.replace("\r\n", "\n")
    if not normalized.endswith("\n"):
        normalized += "\n"
    return write_once(path, normalized.encode("utf-8"))


def artifact_manifest(root: str | Path, *, exclude: Iterable[str] = ("artifact_manifest.json",)) -> dict[str, Any]:
    base = Path(root)
    excluded = set(exclude)
    files = {
        path.relative_to(base).as_posix(): sha256_file(path)
        for path in sorted(base.rglob("*"))
        if path.is_file() and path.name not in excluded
    }
    return {"schema_version": 1, "status": "ARTIFACT_MANIFEST", "files": files}


DIAGNOSTIC_TEST_FILES = (
    "tests/test_claim2_diagnostic_inputs.py",
    "tests/test_claim2_sequence_scoring.py",
    "tests/test_claim2_prompt_mappings.py",
    "tests/test_claim2_jury.py",
    "tests/test_claim2_grouped_splits.py",
    "tests/test_claim2_outcome_firewall.py",
    "tests/test_claim2_geometry_joins.py",
    "tests/test_claim2_batch_size_one.py",
)

ADJACENT_TEST_FILES = (
    "tests/test_claim1_first_pass.py",
    "tests/test_claim1_human_consensus.py",
    "tests/test_claim2_behavioral_viability.py",
    "tests/test_claim2_claude_generation.py",
    "tests/test_claim2_pilot.py",
    "tests/test_claim2_repeatability_control.py",
    "tests/test_claim2_review_closure.py",
    "tests/test_human_review_derived_sets.py",
    "tests/test_human_review_public.py",
    "tests/test_human_review_v2.py",
)

CODE_FREEZE_FILES = (
    "configs/claim2_failure_analysis.yaml",
    "notebooks/claim2_failure_analysis.ipynb",
    *tuple(
        f"src/geometry_endorsement/claim2_diagnostics/{name}"
        for name in (
            "__init__.py",
            "blind_features.py",
            "geometry_diagnostics.py",
            "input_audit.py",
            "jury.py",
            "mechanical.py",
            "outcome_diagnostics.py",
            "power_simulation.py",
            "predictor_models.py",
            "prompt_contracts.py",
            "prospective_gemma.py",
            "reporting.py",
            "sequence_scoring.py",
        )
    ),
    *DIAGNOSTIC_TEST_FILES,
    *ADJACENT_TEST_FILES,
)


def freeze_code_state(config_path: str | Path) -> dict[str, Any]:
    """Hash every task-owned source surface before GPU execution."""
    from .input_audit import load_contract

    contract = load_contract(config_path)
    repo = contract.path.parent.parent.resolve()
    output = (repo / contract.raw["paths"]["output_dir"]).resolve()
    missing = [name for name in CODE_FREEZE_FILES if not (repo / name).is_file()]
    if missing:
        raise RuntimeError(f"Code-freeze files are missing: {missing}")
    files = {name: sha256_file(repo / name) for name in CODE_FREEZE_FILES}
    existing_path = output / "CODE_FREEZE.json"
    if existing_path.is_file():
        receipt = json.loads(existing_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") != "PRE_GPU_CODE_FROZEN"
            or receipt.get("files") != files
            or receipt.get("source_config_sha256") != contract.sha256
            or receipt.get("diagnostic_contract_sha256")
            != sha256_file(output / "DIAGNOSTIC_CONTRACT.json")
            or receipt.get("input_manifest_sha256")
            != sha256_file(output / "INPUT_MANIFEST.json")
        ):
            raise RuntimeError("Existing pre-GPU code freeze does not match this checkout.")
        return {**receipt, "receipt_sha256": sha256_file(existing_path)}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    receipt = {
        "schema_version": 1,
        "status": "PRE_GPU_CODE_FROZEN",
        "repository_commit": commit,
        "contract_id": contract.raw["contract_id"],
        "source_config_sha256": contract.sha256,
        "diagnostic_contract_sha256": sha256_file(
            output / "DIAGNOSTIC_CONTRACT.json"
        ),
        "input_manifest_sha256": sha256_file(output / "INPUT_MANIFEST.json"),
        "files": files,
        "combined_source_sha256": sha256_bytes(canonical_json_bytes(files)),
        "automatic_commit_or_push": False,
    }
    digest = write_json_once(output / "CODE_FREEZE.json", receipt)
    return {**receipt, "receipt_sha256": digest}


def verify_frozen_test_receipt(config_path: str | Path) -> dict[str, Any]:
    """Verify the CPU test receipt without importing any behavioral outcome."""
    from .input_audit import load_contract

    contract = load_contract(config_path)
    repo = contract.path.parent.parent.resolve()
    output = (repo / contract.raw["paths"]["output_dir"]).resolve()
    receipt_path = output / "TEST_VERIFICATION.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "NEW_AND_ADJACENT_TESTS_PASSED"
        or int(receipt.get("failed", -1)) != 0
        or receipt.get("diagnostic_contract_sha256")
        != sha256_file(output / "DIAGNOSTIC_CONTRACT.json")
    ):
        raise RuntimeError("Frozen CPU/static test receipt is invalid.")
    mismatches = [
        name for name, digest in receipt["test_file_sha256"].items()
        if not (repo / name).is_file() or sha256_file(repo / name) != digest
    ]
    if mismatches:
        raise RuntimeError(f"Test files changed after verification: {mismatches}")
    return {**receipt, "receipt_sha256": sha256_file(receipt_path)}


def run_frozen_test_verification(config_path: str | Path) -> dict[str, Any]:
    """Run and retain the exact new-plus-adjacent CPU/static verification set."""
    from .input_audit import load_contract

    contract = load_contract(config_path)
    repo = contract.path.parent.parent.resolve()
    output = (repo / contract.raw["paths"]["output_dir"]).resolve()
    test_files = DIAGNOSTIC_TEST_FILES + ADJACENT_TEST_FILES
    missing = [name for name in test_files if not (repo / name).is_file()]
    if missing:
        raise RuntimeError(f"Required verification tests are missing: {missing}")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "pythonpath=src",
        *test_files,
        "-p",
        "no:cacheprovider",
        "-q",
    ]
    process = subprocess.run(
        command,
        cwd=repo,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    combined = process.stdout + "\n" + process.stderr
    matches = re.findall(r"(?<!\d)(\d+) passed", combined)
    passed = int(matches[-1]) if matches else 0
    if process.returncode != 0:
        raise RuntimeError(
            "Frozen diagnostic verification failed. Tail:\n"
            + "\n".join(combined.splitlines()[-80:])
        )
    receipt = {
        "schema_version": 1,
        "status": "NEW_AND_ADJACENT_TESTS_PASSED",
        "return_code": process.returncode,
        "passed": passed,
        "failed": 0,
        "test_files": list(test_files),
        "test_file_sha256": {
            name: sha256_file(repo / name) for name in test_files
        },
        "python_version": sys.version.split()[0],
        "diagnostic_contract_sha256": sha256_file(
            output / "DIAGNOSTIC_CONTRACT.json"
        ),
    }
    digest = write_json_once(output / "TEST_VERIFICATION.json", receipt)
    return {**receipt, "receipt_sha256": digest}


def run_mode(mode: str, config_path: str | Path) -> dict[str, Any]:
    if mode == "AUDIT_AND_FREEZE":
        from .input_audit import run_audit_and_freeze

        return run_audit_and_freeze(config_path)
    if mode == "GENERATE_BLIND_FEATURES":
        from .blind_features import generate_blind_features

        return generate_blind_features(config_path)
    if mode == "ANALYZE_EXISTING_LLAMA":
        from .outcome_diagnostics import analyze_existing_llama

        return analyze_existing_llama(config_path)
    if mode == "RUN_PROSPECTIVE_GEMMA":
        from .outcome_diagnostics import run_prospective_gemma

        return run_prospective_gemma(config_path)
    raise ValueError(f"Unknown Claim 2 diagnostic mode: {mode}")


def final_audit(config_path: str | Path) -> dict[str, Any]:
    from .outcome_diagnostics import final_diagnostic_audit

    return final_diagnostic_audit(config_path)
