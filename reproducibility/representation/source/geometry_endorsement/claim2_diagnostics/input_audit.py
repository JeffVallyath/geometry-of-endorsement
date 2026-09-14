from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from claim2_pilot.analysis import _model_metrics

from .reporting import sha256_bytes, sha256_file, write_json_once, write_once, write_text_once

MODES = (
    "AUDIT_AND_FREEZE",
    "GENERATE_BLIND_FEATURES",
    "ANALYZE_EXISTING_LLAMA",
    "RUN_PROSPECTIVE_GEMMA",
)
FROZEN_SENTENCE = (
    "Claim 1 geometry did not improve the preregistered held-out prediction "
    "metric beyond original text and native confidence in the representative "
    "Llama population."
)
POPULATIONS = (
    "representative_primary",
    "representative_mapping_consistent_secondary",
    "enriched_geometry_confidence_diagnostic",
)


@dataclass(frozen=True)
class Contract:
    path: Path
    raw: dict[str, Any]
    sha256: str


def load_contract(path: str | Path) -> Contract:
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if raw.get("contract_id") not in {
        "claim2_failure_analysis_v001",
        "claim2_failure_analysis_v002",
        "claim2_failure_analysis_v003",
        "claim2_failure_analysis_v004",
    }:
        raise RuntimeError("Unexpected Claim 2 diagnostic contract ID.")
    if raw.get("frozen_llama_result") != FROZEN_SENTENCE:
        raise RuntimeError("Frozen Llama result language changed.")
    if int(raw["scoring"]["behavioral_prompt_batch_size"]) != 1:
        raise RuntimeError("Candidate-sequence scoring must use batch size 1.")
    return Contract(source, raw, sha256_file(source))


def _repo_root(contract: Contract) -> Path:
    return contract.path.parent.parent.resolve()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_repo_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def resolve_authoritative_input(
    repo: Path, role: str, binding: dict[str, Any]
) -> tuple[Path, int]:
    """Resolve one explicitly hash-bound CSV without guessing or fuzzy matching."""
    expected_sha256 = str(binding["sha256"])
    expected_rows = int(binding["rows"])
    failures: list[str] = []
    for value in binding["candidates"]:
        candidate = _resolve_repo_path(repo, str(value))
        if not candidate.is_file():
            failures.append(f"{candidate}:missing")
            continue
        observed_sha256 = sha256_file(candidate)
        if observed_sha256 != expected_sha256:
            failures.append(f"{candidate}:sha256={observed_sha256}")
            continue
        rows = len(pd.read_csv(candidate, dtype=str))
        if rows != expected_rows:
            failures.append(f"{candidate}:rows={rows}")
            continue
        return candidate, rows
    raise RuntimeError(
        f"Authoritative input {role!r} did not resolve to its exact bound bytes: "
        + "; ".join(failures)
    )


def stage_authoritative_inputs(
    repo: Path, output: Path, bindings: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Retain exact portable copies of the three owner-side bindings."""
    staged: dict[str, dict[str, Any]] = {}
    staged_names: set[str] = set()
    for role, binding in bindings.items():
        source, rows = resolve_authoritative_input(repo, role, binding)
        staged_name = str(binding["staged_name"])
        if Path(staged_name).name != staged_name or staged_name in staged_names:
            raise RuntimeError("Authoritative staged names must be unique basenames.")
        staged_names.add(staged_name)
        target = output / "frozen_inputs" / staged_name
        digest = write_once(target, source.read_bytes())
        if digest != str(binding["sha256"]):
            raise RuntimeError(f"Staged authoritative input hash mismatch: {role}")
        staged[role] = {
            "source_path": str(source),
            "source_sha256": digest,
            "rows": rows,
            "staged_path": str(target),
            "staged_sha256": sha256_file(target),
        }
    return staged


def locate_frozen_geometry(contract: Contract) -> Path:
    candidates: list[Path] = []
    override = os.environ.get("CLAIM2_FROZEN_GEOMETRY_ROOT")
    if override:
        candidates.append(Path(override))
    candidates.extend(Path(value) for value in contract.raw["paths"]["frozen_geometry_candidates"])
    for candidate in candidates:
        resolved = candidate.resolve()
        required = (
            resolved / "geometry_analysis_report.json",
            resolved / "input_receipt.json",
            resolved / "code_freeze_receipt.json",
        )
        if all(path.is_file() for path in required):
            return resolved
    raise FileNotFoundError("Authoritative frozen geometry output was not found via configured manifests.")


def verify_manifest_files(root: Path, manifest_path: Path) -> dict[str, str]:
    manifest = _json(manifest_path)
    expected = manifest.get("files") or manifest.get("output_hashes_excluding_this_receipt")
    if not isinstance(expected, dict):
        raise RuntimeError(f"No file hash map in {manifest_path}")
    verified: dict[str, str] = {}
    for relative, digest in expected.items():
        candidates = []
        for base in (root, manifest_path.parent, manifest_path.parent.parent):
            candidate = (base / relative).resolve()
            if candidate not in candidates and candidate.is_file():
                candidates.append(candidate)
        if not candidates:
            raise RuntimeError(
                f"Manifest-listed input is missing from declared/manifest roots: {relative}"
            )
        observed_values = {sha256_file(candidate) for candidate in candidates}
        if observed_values != {digest}:
            raise RuntimeError(
                f"Hash mismatch for manifest entry {relative}: {sorted(observed_values)}"
            )
        observed = digest
        verified[relative] = observed
    return verified


def reconcile_48_49(frozen_root: Path, gate2_dir: Path) -> dict[str, Any]:
    items = pd.read_csv(frozen_root / "population_items_private.csv", dtype=str)
    primary = items[items["population"].eq("representative_primary")].copy()
    coverage = pd.read_csv(gate2_dir / "gate2_per_original_outcomes_private.csv", dtype=str)
    merged = primary.merge(coverage, on="base_item_id", how="left", suffixes=("_geometry", "_gate2"), validate="one_to_one")
    count_column = "accepted_primary_rewrites_gate2" if "accepted_primary_rewrites_gate2" in merged else "accepted_primary_rewrites"
    counts = pd.to_numeric(merged[count_column], errors="raise")
    extra = merged[counts.eq(1)]
    if len(primary) != 49 or int((counts >= 2).sum()) != 48 or len(extra) != 1:
        raise RuntimeError("48-versus-49 reconciliation did not match the frozen populations.")
    row = extra.iloc[0]
    return {
        "status": "RESOLVED",
        "base_item_id": str(row["base_item_id"]),
        "board_id": str(row.get("board_id_geometry", row.get("board_id", ""))),
        "stratum": str(row.get("stratum_geometry", row.get("stratum", ""))),
        "accepted_primary_rewrites": 1,
        "gate2_coverage_rule": "at_least_two_accepted_primary_rewrites",
        "frozen_geometry_analysis_rule": "at_least_one_accepted_primary_rewrite",
        "reporting_only": False,
        "affects_analysis": True,
        "explanation": "The Gate 2 headline eligibility count requires >=2 accepted rewrites; the frozen geometry primary population includes every item with n_i>0.",
    }


def reproduce_frozen_metrics(frozen_root: Path) -> dict[str, Any]:
    report = _json(frozen_root / "geometry_analysis_report.json")
    if report.get("status") != "FROZEN_CLAIM2_GEOMETRY_ANALYSIS_COMPLETE":
        raise RuntimeError("Frozen geometry report is incomplete.")
    reproduced: dict[str, Any] = {}
    for population in POPULATIONS:
        frozen = report["populations"][population]
        predictions = pd.read_csv(frozen_root / "populations" / population / "item_level_predictions_private.csv")
        if predictions.duplicated(["base_item_id", "model"]).any():
            raise RuntimeError(f"Duplicate item/model predictions in {population}")
        observed_models: dict[str, Any] = {}
        for model in ("M2", "M3"):
            metrics = _model_metrics(predictions[predictions["model"].eq(model)])
            expected = frozen["models"][model]
            for key, value in expected.items():
                if key == "ece_caveat":
                    if metrics[key] != value:
                        raise RuntimeError(f"Frozen metric mismatch: {population}/{model}/{key}")
                elif value is None:
                    if metrics[key] is not None:
                        raise RuntimeError(f"Frozen metric mismatch: {population}/{model}/{key}")
                elif not np.isclose(float(metrics[key]), float(value), rtol=0.0, atol=1e-12):
                    raise RuntimeError(f"Frozen metric mismatch: {population}/{model}/{key}")
            observed_models[model] = metrics
        bootstrap = np.load(frozen_root / "populations" / population / "m2_m3_base_item_bootstrap.npy", allow_pickle=False)
        interval = [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))]
        expected_interval = frozen["delta_log_loss_m2_minus_m3"]["interval_95"]
        if not np.allclose(interval, expected_interval, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"Frozen bootstrap interval mismatch: {population}")
        delta = observed_models["M2"]["binomial_log_loss_per_rephrasing"] - observed_models["M3"]["binomial_log_loss_per_rephrasing"]
        if not np.isclose(delta, frozen["delta_log_loss_m2_minus_m3"]["point_estimate"], rtol=0.0, atol=1e-12):
            raise RuntimeError(f"Frozen delta mismatch: {population}")
        reproduced[population] = {
            "items": int(frozen["items"]),
            "flipping_items": int(frozen["flipping_items"]),
            "accepted_rewrites": int(frozen["accepted_rewrites"]),
            "flips": int(frozen["flips"]),
            "models": observed_models,
            "delta_log_loss_m2_minus_m3": {"point_estimate": delta, "interval_95": interval},
        }
    return reproduced


def _frozen_reproduction_markdown(reproduction: dict[str, Any], discrepancy: dict[str, Any]) -> str:
    lines = [
        "# Frozen Llama reproduction",
        "",
        FROZEN_SENTENCE,
        "",
        "This is an exact reproduction of the existing frozen result, not a new analysis.",
        "",
        "## 48 versus 49 reconciliation",
        "",
        f"The difference is caused by `{discrepancy['base_item_id']}`. Gate 2's headline coverage requires at least two accepted rewrites, while the frozen geometry analysis includes items with at least one. The discrepancy affects the analysis population by one item; it is not a typographical reporting error.",
        "",
        "## Exact metrics",
        "",
        "| Population | Items | Flipping originals | Rewrites | Flips | M2 loss | M3 loss | M2-M3 | 95% item bootstrap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for population in POPULATIONS:
        value = reproduction[population]
        lines.append(
            f"| {population} | {value['items']} | {value['flipping_items']} | {value['accepted_rewrites']} | {value['flips']} | "
            f"{value['models']['M2']['binomial_log_loss_per_rephrasing']:.15g} | {value['models']['M3']['binomial_log_loss_per_rephrasing']:.15g} | "
            f"{value['delta_log_loss_m2_minus_m3']['point_estimate']:.15g} | {value['delta_log_loss_m2_minus_m3']['interval_95']} |"
        )
    lines += ["", "The primary representative population contains 13 flipping originals. The mapping-consistent secondary contains only 4."]
    return "\n".join(lines)


def run_audit_and_freeze(config_path: str | Path) -> dict[str, Any]:
    contract = load_contract(config_path)
    repo = _repo_root(contract)
    paths = contract.raw["paths"]
    output = _resolve_repo_path(repo, paths["output_dir"])
    frozen = locate_frozen_geometry(contract)
    human_manifest_path = _resolve_repo_path(repo, paths["human_manifest"])
    gate2_record_path = _resolve_repo_path(repo, paths["gate2_run_record"])
    gate2_dir = _resolve_repo_path(repo, paths["gate2_dir"])

    human_verified = verify_manifest_files(human_manifest_path.parent, human_manifest_path)
    gate2_verified = verify_manifest_files(gate2_record_path.parent.parent, gate2_record_path)
    if len(gate2_verified) != 24:
        raise RuntimeError(
            f"The repaired Gate 2 run record must verify all 24 files; observed {len(gate2_verified)}."
        )
    authoritative = stage_authoritative_inputs(
        repo, output, paths["authoritative_inputs"]
    )
    discrepancy = reconcile_48_49(frozen, gate2_dir)
    reproduction = reproduce_frozen_metrics(frozen)

    frozen_source_sha = write_once(
        output / "FROZEN_CONFIG_SOURCE.yaml", contract.path.read_bytes()
    )

    inputs = {
        "schema_version": 1,
        "status": "AUDITED_AND_FROZEN",
        "repository_root": str(repo),
        "diagnostic_config": {"path": str(contract.path), "sha256": contract.sha256},
        "human_review_manifest": {"path": str(human_manifest_path), "sha256": sha256_file(human_manifest_path), "verified_files": len(human_verified)},
        "gate2_run_record": {"path": str(gate2_record_path), "sha256": sha256_file(gate2_record_path), "verified_files": len(gate2_verified)},
        "authoritative_inputs": authoritative,
        "frozen_geometry": {
            "path": str(frozen),
            "report_sha256": sha256_file(frozen / "geometry_analysis_report.json"),
            "input_receipt_sha256": sha256_file(frozen / "input_receipt.json"),
            "code_freeze_receipt_sha256": sha256_file(frozen / "code_freeze_receipt.json"),
        },
        "llama_vectors": {"path": str(_resolve_repo_path(repo, paths["llama_compact_vectors"])), "sha256": sha256_file(_resolve_repo_path(repo, paths["llama_compact_vectors"]))},
        "gemma_status": {"provenance_sha256": sha256_file(_resolve_repo_path(repo, paths["gemma_provenance"])), "row_level_vectors_available": False},
        "reconciliation_48_49": discrepancy,
        "frozen_source_config": {
            "path": str(output / "FROZEN_CONFIG_SOURCE.yaml"),
            "sha256": frozen_source_sha,
        },
    }
    input_sha = write_json_once(output / "INPUT_MANIFEST.json", inputs)

    diagnostic = dict(contract.raw)
    diagnostic["schema_version"] = 1
    diagnostic["status"] = "DIAGNOSTIC_CONTRACT_FROZEN"
    diagnostic["source_config_sha256"] = contract.sha256
    diagnostic["input_manifest_sha256"] = input_sha
    diagnostic_sha = write_json_once(output / "DIAGNOSTIC_CONTRACT.json", diagnostic)
    amendment_sha = None
    if "supersedes" in contract.raw:
        predecessor_version = str(contract.raw["supersedes"]["contract_id"]).rsplit("_v", 1)[-1]
        predecessor_output = repo / "outputs" / "claim2-failure-analysis" / f"v{predecessor_version}"
        predecessor_source = predecessor_output / "FROZEN_CONFIG_SOURCE.yaml"
        amendment = {
            "schema_version": 1,
            "status": "PRE_EXECUTION_CONTRACT_AMENDMENT",
            "predecessor": contract.raw["supersedes"],
            "predecessor_config_copy": str(predecessor_source),
            "predecessor_config_copy_sha256": sha256_file(predecessor_source),
            "predecessor_input_manifest_sha256": sha256_file(predecessor_output / "INPUT_MANIFEST.json"),
            "predecessor_diagnostic_contract_sha256": sha256_file(predecessor_output / "DIAGNOSTIC_CONTRACT.json"),
            "successor_source_config_sha256": contract.sha256,
            "successor_input_manifest_sha256": input_sha,
            "successor_diagnostic_contract_sha256": diagnostic_sha,
            "blind_features_existed_before_amendment": False,
            "claim2_outcomes_reanalyzed_before_amendment": False,
            "changed_fields_only": list(contract.raw["supersedes"].get("changed_fields_only", [])),
            "reason": str(contract.raw["supersedes"]["reason"]),
            "human_authorization_received": bool(
                contract.raw["supersedes"].get("human_authorization_received", False)
            ),
            "human_authorization_text": str(
                contract.raw["supersedes"].get("human_authorization_text", "")
            ),
        }
        amendment_sha = write_json_once(output / "CONTRACT_AMENDMENT.json", amendment)
    reproduction_sha = write_text_once(output / "FROZEN_LLAMA_REPRODUCTION.md", _frozen_reproduction_markdown(reproduction, discrepancy))
    return {
        "status": "AUDIT_AND_FREEZE_COMPLETE",
        "output_dir": str(output),
        "input_manifest_sha256": input_sha,
        "diagnostic_contract_sha256": diagnostic_sha,
        "frozen_reproduction_sha256": reproduction_sha,
        "contract_amendment_sha256": amendment_sha,
        "reconciliation_48_49": discrepancy,
        "reproduction": reproduction,
    }


def _find_archive_with_member(
    candidates: list[Path], member: str, expected_member_sha256: str
) -> tuple[Path, bytes]:
    for archive in candidates:
        if not archive.is_file():
            continue
        with zipfile.ZipFile(archive) as zipped:
            if member not in zipped.namelist():
                continue
            payload = zipped.read(member)
        if sha256_bytes(payload) == expected_member_sha256:
            return archive.resolve(), payload
    raise FileNotFoundError(
        f"No configured archive contains {member!r} with SHA-256 {expected_member_sha256}."
    )


def stage_outcome_blind_probe_inputs(config_path: str | Path) -> dict[str, Any]:
    """Recover exact frozen probe bytes without resolving any Claim 2 outcome path."""
    contract = load_contract(config_path)
    repo = _repo_root(contract)
    output = _resolve_repo_path(repo, contract.raw["paths"]["output_dir"])
    input_manifest = output / "INPUT_MANIFEST.json"
    diagnostic_contract = output / "DIAGNOSTIC_CONTRACT.json"
    if not input_manifest.is_file() or not diagnostic_contract.is_file():
        raise RuntimeError("AUDIT_AND_FREEZE must complete before probe staging.")
    existing_receipt = output / "BLIND_INPUT_RECEIPT.json"
    existing_llama = output / "frozen_inputs" / "llama_m1_probe_parameters.npz"
    existing_gemma = output / "frozen_inputs" / "gemma_m1_probe_parameters.npz"
    if existing_receipt.is_file() and existing_llama.is_file() and existing_gemma.is_file():
        receipt = _json(existing_receipt)
        if (
            sha256_file(existing_llama) != receipt["llama"]["member_sha256"]
            or sha256_file(existing_gemma) != receipt["gemma"]["member_sha256"]
            or sha256_file(input_manifest) != receipt["input_manifest_sha256"]
            or sha256_file(diagnostic_contract) != receipt["diagnostic_contract_sha256"]
        ):
            raise RuntimeError("Existing staged blind probe inputs fail hash verification.")
        return {**receipt, "receipt_sha256": sha256_file(existing_receipt)}

    llama_provenance_path = repo / "outputs" / "claim1-first-pass-semantic-stratification" / "llama_score_provenance.json"
    gemma_provenance_path = repo / "outputs" / "claim1-first-pass-semantic-stratification" / "gemma_score_provenance.json"
    llama_provenance = _json(llama_provenance_path)
    gemma_provenance = _json(gemma_provenance_path)
    member = "m1_probe_parameters.npz"
    llama_expected = str(llama_provenance["probe_parameters_sha256"])
    gemma_expected = str(gemma_provenance["probe_parameters_sha256"])
    llama_candidates = [
        Path(os.environ.get("LLAMA_M1_AUDIT_ARCHIVE", "")),
        Path("external-artifacts"),
    ]
    gemma_candidates = [
        Path(os.environ.get("GEMMA_M1_AUDIT_ARCHIVE", "")),
        Path("external-artifacts"),
    ]
    llama_archive, llama_bytes = _find_archive_with_member(llama_candidates, member, llama_expected)
    gemma_archive, gemma_bytes = _find_archive_with_member(gemma_candidates, member, gemma_expected)
    frozen_input_dir = output / "frozen_inputs"
    llama_target = frozen_input_dir / "llama_m1_probe_parameters.npz"
    gemma_target = frozen_input_dir / "gemma_m1_probe_parameters.npz"
    write_once(llama_target, llama_bytes)
    write_once(gemma_target, gemma_bytes)
    receipt = {
        "schema_version": 1,
        "status": "OUTCOME_BLIND_PROBE_INPUTS_STAGED",
        "diagnostic_contract_sha256": sha256_file(diagnostic_contract),
        "input_manifest_sha256": sha256_file(input_manifest),
        "claim2_outcomes_loaded": False,
        "llama": {
            "source_archive": str(llama_archive),
            "source_archive_sha256": sha256_file(llama_archive),
            "source_provenance_sha256": sha256_file(llama_provenance_path),
            "member": member,
            "member_sha256": llama_expected,
            "staged_path": str(llama_target),
        },
        "gemma": {
            "source_archive": str(gemma_archive),
            "source_archive_sha256": sha256_file(gemma_archive),
            "source_provenance_sha256": sha256_file(gemma_provenance_path),
            "member": member,
            "member_sha256": gemma_expected,
            "staged_path": str(gemma_target),
            "row_level_claim1_vectors_available": False,
        },
    }
    digest = write_json_once(output / "BLIND_INPUT_RECEIPT.json", receipt)
    return {**receipt, "receipt_sha256": digest}
