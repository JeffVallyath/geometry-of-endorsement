from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml


SOURCE_CELL_FIELDS = {
    "s1_A": ("situation_1", "consideration_A", "s1_A_relation"),
    "s1_B": ("situation_1", "consideration_B", "s1_B_relation"),
    "s2_A": ("situation_2", "consideration_A", "s2_A_relation"),
    "s2_B": ("situation_2", "consideration_B", "s2_B_relation"),
}
PRIOR_OUTCOME_NAMES = (
    "verdict_pair_results.parquet",
    "intervention_item_results.parquet",
    "VERDICT_ARM_REPORT.md",
    "CAUSAL_INTERVENTION_REPORT.md",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def array_sha256(value: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(value).tobytes())


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or int(value.get("schema_version", -1)) != 1:
        raise RuntimeError("Review-payoff config schema is missing or unsupported.")
    if value.get("strategic_constraint") != "NO_NEW_LARGE_SCALE_HUMAN_REVIEW":
        raise RuntimeError("The no-new-human-review boundary is not frozen.")
    if int(value["runtime"]["batch_size"]) != 1:
        raise RuntimeError("Every fresh scoring phase must freeze batch_size=1.")
    if value["models"]["order"] != ["llama", "gemma"]:
        raise RuntimeError("Model order must remain Llama then Gemma.")
    return value


def _resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo / path).resolve()


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise RuntimeError(f"Required {label} is absent: {path}")
    return path


def _require_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: observed={observed}, expected={expected}"
        )
    return observed


def _read_csv(path: Path, *, rows: int, unique: str, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    if len(frame) != rows:
        raise RuntimeError(f"{label} row count changed: {len(frame)} != {rows}")
    if unique not in frame or frame[unique].astype(str).duplicated().any():
        raise RuntimeError(f"{label} does not have unique {unique} values.")
    return frame


def _probe_contract(path: Path, member: str, model: Mapping[str, Any]) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        try:
            payload = archive.read(member)
        except KeyError as exc:
            raise RuntimeError(f"Probe bundle lacks {member}: {path}") from exc
    arrays = np.load(io.BytesIO(payload), allow_pickle=False)
    required = {
        "difference_in_means_direction",
        "difference_in_means_midpoint",
        "logistic_coef",
        "logistic_intercept",
        "selected_layer",
        "model_revision",
        "tokenizer_revision",
    }
    if not required <= set(arrays.files):
        raise RuntimeError(f"Probe bundle lacks keys: {sorted(required - set(arrays.files))}")
    direction = np.asarray(arrays["difference_in_means_direction"], dtype=np.float32)
    logistic = np.asarray(arrays["logistic_coef"], dtype=np.float32)
    layer = int(np.asarray(arrays["selected_layer"]).item())
    revision = str(np.asarray(arrays["model_revision"]).item())
    tokenizer_revision = str(np.asarray(arrays["tokenizer_revision"]).item())
    if direction.shape != (int(model["hidden_size"]),):
        raise RuntimeError("Frozen DIM vector width changed.")
    if logistic.shape != (1, int(model["hidden_size"])):
        raise RuntimeError("Frozen logistic vector width changed.")
    if layer != int(model["selected_layer"]):
        raise RuntimeError("Frozen selected layer changed.")
    if revision != model["revision"] or tokenizer_revision != model["tokenizer_revision"]:
        raise RuntimeError("Frozen model/tokenizer revision changed.")
    return {
        "bundle_path": str(path),
        "bundle_sha256": sha256_file(path),
        "probe_member": member,
        "probe_member_sha256": sha256_bytes(payload),
        "selected_layer": layer,
        "model_revision": revision,
        "tokenizer_revision": tokenizer_revision,
        "hidden_size": int(direction.size),
        "dim_vector_sha256": array_sha256(direction),
        "logistic_vector_sha256": array_sha256(logistic),
        "dim_midpoint": float(np.asarray(arrays["difference_in_means_midpoint"]).item()),
        "logistic_intercept": float(np.asarray(arrays["logistic_intercept"]).reshape(-1)[0]),
        "dim_scaler_mu": (
            float(np.asarray(arrays["dim_scaler_mu"]).item())
            if "dim_scaler_mu" in arrays.files
            else None
        ),
        "dim_scaler_sigma": (
            float(np.asarray(arrays["dim_scaler_sigma"]).item())
            if "dim_scaler_sigma" in arrays.files
            else None
        ),
    }


def _verify_review_sources(
    cells: pd.DataFrame,
    scores: pd.DataFrame,
    reviewer_1: pd.DataFrame,
    reviewer_2: pd.DataFrame,
) -> None:
    source_columns = [
        "review_id",
        "situation_1",
        "situation_2",
        "consideration_A",
        "consideration_B",
        "s1_A_relation",
        "s1_B_relation",
        "s2_A_relation",
        "s2_B_relation",
    ]
    left = reviewer_1[source_columns].sort_values("review_id").reset_index(drop=True)
    right = reviewer_2[source_columns].sort_values("review_id").reset_index(drop=True)
    if not left.equals(right):
        raise RuntimeError("Claim 1 reviewer source fields are not exactly identical.")
    lookup = reviewer_1.set_index("review_id")
    if set(cells["review_id"]) != set(lookup.index.astype(str)):
        raise RuntimeError("Claim 1 consensus cells do not cover the reviewer IDs exactly.")
    for row in cells.itertuples(index=False):
        position = str(row.cell_position)
        if position not in SOURCE_CELL_FIELDS:
            raise RuntimeError(f"Unknown Claim 1 cell position: {position}")
        _, _, relation_field = SOURCE_CELL_FIELDS[position]
        if str(row.stored_relation) != str(lookup.loc[str(row.review_id), relation_field]):
            raise RuntimeError(f"Stored relation changed for {row.row_id}.")
    joined = cells[["row_id", "stored_relation"]].merge(
        scores[["row_id", "relation"]], on="row_id", how="outer", validate="one_to_one"
    )
    if len(joined) != 500 or not joined["stored_relation"].eq(joined["relation"]).all():
        raise RuntimeError("Frozen Llama score relation/source mapping changed.")


def _load_valueprism_types(
    target_ids: set[str],
    cells: pd.DataFrame,
    reviewer: pd.DataFrame,
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
) -> pd.DataFrame:
    try:
        from datasets import load_dataset
        from m1_vertical_slice.manifests import row_id as make_row_id
    except ImportError as exc:
        raise RuntimeError("Cached ValuePrism audit requires the datasets package.") from exc

    loaded = load_dataset(dataset_name, dataset_config)
    if dataset_split != "all":
        source_parts = [loaded[dataset_split]]
    else:
        source_parts = list(loaded.values())
    resolved: dict[str, dict[str, str]] = {}
    duplicates: set[str] = set()
    for source in source_parts:
        for record in source:
            relation = str(record["valence"]).strip().lower()
            if relation not in {"supports", "opposes"}:
                continue
            key = make_row_id(
                str(record["situation"]),
                str(record["text"]),
                relation,
                str(record["vrd"]),
            )
            if key not in target_ids:
                continue
            candidate = {
                "row_id": key,
                "situation": str(record["situation"]),
                "consideration": str(record["text"]),
                "stored_relation": relation.title(),
                "consideration_type": str(record["vrd"]),
            }
            if key in resolved and resolved[key] != candidate:
                duplicates.add(key)
            resolved.setdefault(key, candidate)
    if duplicates or set(resolved) != target_ids:
        raise RuntimeError(
            "ValuePrism row-ID join is not one-to-one and complete: "
            f"missing={len(target_ids-set(resolved))}, duplicates={len(duplicates)}"
        )
    allowed = {"Value", "Right", "Duty"}
    if {row["consideration_type"] for row in resolved.values()} - allowed:
        raise RuntimeError("ValuePrism contains an unexpected consideration type.")

    review = reviewer.set_index("review_id")
    for cell in cells.itertuples(index=False):
        situation_field, consideration_field, relation_field = SOURCE_CELL_FIELDS[
            str(cell.cell_position)
        ]
        expected = resolved[str(cell.row_id)]
        source_row = review.loc[str(cell.review_id)]
        observed = (
            str(source_row[situation_field]),
            str(source_row[consideration_field]),
            str(source_row[relation_field]),
        )
        authoritative = (
            expected["situation"],
            expected["consideration"],
            expected["stored_relation"],
        )
        if observed != authoritative:
            raise RuntimeError(f"ValuePrism exact source fields changed for {cell.row_id}.")
    return pd.DataFrame(resolved.values()).sort_values("row_id").reset_index(drop=True)


def _prior_exposure(repo: Path, output: Path) -> list[str]:
    matches: list[str] = []
    for name in PRIOR_OUTCOME_NAMES:
        for path in repo.glob(f"outputs/**/{name}"):
            if output not in path.parents:
                matches.append(str(path.resolve()))
    return sorted(set(matches))


def _write_once(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Frozen artifact differs and will not be overwritten: {path}")
    else:
        path.write_bytes(payload)
    return sha256_bytes(payload)


def _input_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def run_input_audit(
    repo_root: str | Path,
    config_path: str | Path,
    *,
    include_valueprism: bool = True,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    paths = {key: _resolve(repo, value) for key, value in config["paths"].items()}
    output = paths["output_root"]

    required = {
        "human_manifest": paths["human_derived_root"] / "artifact_manifest.json",
        "claim1_cells": paths["human_derived_root"] / "claim1_consensus_cells.csv",
        "claim1_boards": paths["human_derived_root"] / "claim1_consensus_boards.csv",
        "claim1_pairs": paths["human_derived_root"] / "claim1_consensus_reason_pairs.csv",
        "claim2_judgments": paths["human_derived_root"] / "claim2_independent_judgments.csv",
        "claim2_accepted": paths["human_derived_root"] / "claim2_strict_consensus_ordinary.csv",
        "claim1_claim2_join": paths["claim2_behavioral_root"] / "frozen_inputs" / "claim1_claim2_join.csv",
        "base_manifest": paths["claim2_behavioral_root"] / "frozen_inputs" / "base_item_manifest_private.csv",
        "rewrite_manifest": paths["claim2_behavioral_root"] / "frozen_inputs" / "rewrite_strength_manifest.csv",
        "llama_behavior": paths["claim2_behavioral_root"] / "checkpoints" / "target_llama_blind.parquet",
        "gemma_behavior": paths["claim2_behavioral_root"] / "checkpoints" / "gemma_candidate_behavior.parquet",
        "llama_compact": paths["llama_compact"],
        "llama_compact_metadata": paths["llama_compact_metadata"],
        "llama_scores": paths["llama_cell_scores"],
        "reviewer_1": paths["claim1_reviewer_1"],
        "reviewer_2": paths["claim1_reviewer_2"],
        "llama_probe": paths["llama_probe_bundle"],
        "llama_source": paths["llama_source_bundle"],
        "gemma_probe": paths["gemma_probe_bundle"],
    }
    for label, path in required.items():
        _require_file(path, label)

    expected = config["expected_inputs"]
    hash_bindings = {
        "human_derived_manifest_sha256": required["human_manifest"],
        "claim1_consensus_cells_sha256": required["claim1_cells"],
        "claim1_consensus_boards_sha256": required["claim1_boards"],
        "claim1_consensus_reason_pairs_sha256": required["claim1_pairs"],
        "claim2_independent_judgments_sha256": required["claim2_judgments"],
        "claim2_strict_consensus_ordinary_sha256": required["claim2_accepted"],
        "llama_cell_scores_sha256": required["llama_scores"],
        "llama_probe_bundle_sha256": required["llama_probe"],
        "llama_source_bundle_sha256": required["llama_source"],
        "gemma_probe_bundle_sha256": required["gemma_probe"],
    }
    for key, path in hash_bindings.items():
        _require_hash(path, str(expected[key]), key)

    cells = _read_csv(required["claim1_cells"], rows=500, unique="row_id", label="Claim 1 cells")
    boards = _read_csv(required["claim1_boards"], rows=125, unique="board_id", label="Claim 1 boards")
    pairs = pd.read_csv(required["claim1_pairs"], dtype=str).fillna("")
    if len(pairs) != 250 or pairs.duplicated(["board_id", "consideration_slot"]).any():
        raise RuntimeError("Claim 1 repeated-consideration topology changed.")
    judgments = _read_csv(
        required["claim2_judgments"], rows=481, unique="candidate_id", label="Claim 2 judgments"
    )
    accepted = _read_csv(
        required["claim2_accepted"], rows=398, unique="candidate_id", label="Claim 2 accepted rewrites"
    )
    scores = _read_csv(required["llama_scores"], rows=500, unique="row_id", label="Llama cell scores")
    reviewer_1 = _read_csv(required["reviewer_1"], rows=125, unique="review_id", label="Reviewer 1")
    reviewer_2 = _read_csv(required["reviewer_2"], rows=125, unique="review_id", label="Reviewer 2")
    _verify_review_sources(cells, scores, reviewer_1, reviewer_2)

    if set(boards["board_id"]) != set(cells["board_id"]):
        raise RuntimeError("Claim 1 board/cell membership changed.")
    if judgments["base_item_id"].nunique() != 112:
        raise RuntimeError("Claim 2 original population no longer contains 112 items.")
    ordinary = judgments["control_type"].eq("none")
    if int((~ordinary).sum()) != 33:
        raise RuntimeError("Hidden-control population no longer contains 33 rows.")
    strata = (
        judgments.loc[ordinary, ["base_item_id", "stratum"]]
        .drop_duplicates()
        .groupby("stratum")["base_item_id"]
        .nunique()
        .to_dict()
    )
    if strata != {"disagreement": 32, "representative": 80}:
        raise RuntimeError(f"Frozen 80/32 membership changed: {strata}")
    if not set(accepted["candidate_id"]).issubset(set(judgments["candidate_id"])):
        raise RuntimeError("Accepted rewrites are not a strict subset of reviewed candidates.")

    human_manifest = json.loads(required["human_manifest"].read_text(encoding="utf-8"))
    for name, digest in human_manifest.get("files", {}).items():
        file_path = paths["human_derived_root"] / name
        if not file_path.is_file() or sha256_file(file_path) != digest:
            raise RuntimeError(f"Human-derived manifest verification failed: {name}")

    llama_probe = _probe_contract(required["llama_probe"], "m1_probe_parameters.npz", config["models"]["llama"])
    gemma_probe = _probe_contract(required["gemma_probe"], "m1_probe_parameters.npz", config["models"]["gemma"])

    type_map: pd.DataFrame | None = None
    if include_valueprism:
        compact = np.load(required["llama_compact"], allow_pickle=False)
        if "row_id" not in compact.files or "activation" not in compact.files:
            raise RuntimeError("Llama compact activation artifact lacks row IDs or activations.")
        compact_ids = set(np.asarray(compact["row_id"]).astype(str))
        if len(compact_ids) != 2300 or np.asarray(compact["activation"]).shape != (2300, 4096):
            raise RuntimeError("Llama compact activation topology changed.")
        type_map = _load_valueprism_types(
            compact_ids,
            cells,
            reviewer_1,
            str(config["paths"]["valueprism_dataset"]),
            str(config["paths"]["valueprism_config"]),
            str(config["paths"]["valueprism_split"]),
        )
        type_payload = type_map.to_csv(index=False, lineterminator="\n").encode("utf-8")
        _write_once(output / "frozen_valueprism_type_map.csv", type_payload)

    prior = _prior_exposure(repo, output)
    if prior:
        raise RuntimeError(
            "Prior complete-verdict or real intervention outputs exist; prospectivity stops: "
            + "; ".join(prior)
        )

    availability = {
        "llama_row_level_claim1_scores": True,
        "llama_row_level_activation_cache": True,
        "llama_logistic_scores": True,
        "llama_native_answer_margins": True,
        "llama_text_only_row_scores": False,
        "gemma_row_level_claim1_scores": False,
        "gemma_row_level_activation_cache": False,
        "gemma_direction_and_probe": True,
        "reason_level_llama_outputs": True,
        "reason_level_gemma_outputs": True,
        "valueprism_type_metadata": type_map is not None,
    }
    manifest = {
        "schema_version": 1,
        "status": "INPUTS_AUDITED_EXACTLY",
        "study_id": config["study_id"],
        "repository_head": _git_head(repo),
        "config": _input_record(config_path),
        "inputs": {name: _input_record(path) for name, path in sorted(required.items())},
        "probe_contracts": {"llama": llama_probe, "gemma": gemma_probe},
        "counts": {
            "claim1_cells": 500,
            "claim1_boards": 125,
            "claim1_repeated_considerations": 250,
            "claim2_review_rows": 481,
            "claim2_originals": 112,
            "claim2_representative_originals": 80,
            "claim2_enriched_originals": 32,
            "claim2_hidden_controls": 33,
            "claim2_strict_accepted_ordinary": 398,
            "valueprism_types": (
                type_map["consideration_type"].value_counts().sort_index().to_dict()
                if type_map is not None
                else None
            ),
        },
        "availability": availability,
        "unavailable_reason": {
            "gemma_row_level_claim1_scores": "The frozen Gemma development audit contains probe parameters and aggregate results but no row-level score/cache payload.",
            "llama_text_only_row_scores": "No exact row-level frozen text-only comparator is retained; no imputation or refit is permitted.",
        },
        "joins": {
            "stable_ids_only": True,
            "fuzzy_matching_used": False,
            "reviewer_source_fields_exact": True,
            "valueprism_source_fields_exact": type_map is not None,
            "population_membership_exact": True,
            "population_label_alias": {
                "representative": "representative_primary",
                "disagreement": "enriched_diagnostic",
            },
        },
        "prior_exposure": {
            "complete_verdict_or_real_intervention_outputs": prior,
            "prospective_arms_clear_to_start": not prior,
            "architecture_only_intervention_harness_is_not_an_outcome": True,
        },
    }
    manifest_sha = _write_once(output / "INPUT_MANIFEST.json", canonical_json_bytes(manifest))
    contract = {
        "schema_version": 1,
        "status": "STUDY_CONTRACT_FROZEN_BEFORE_FRESH_INFERENCE",
        "study_id": config["study_id"],
        "input_manifest_sha256": manifest_sha,
        "config_sha256": sha256_file(config_path),
        "config": config,
        "prior_outcome_exposure": False,
        "contract_changes_after_fresh_inference_forbidden": True,
    }
    contract_sha = _write_once(output / "STUDY_CONTRACT.json", canonical_json_bytes(contract))
    return {
        "status": manifest["status"],
        "input_manifest_sha256": manifest_sha,
        "study_contract_sha256": contract_sha,
        "availability": availability,
        "counts": manifest["counts"],
    }


def _git_head(repo: Path) -> str:
    head = repo / ".git" / "HEAD"
    if not head.is_file():
        return "UNAVAILABLE"
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        ref = repo / ".git" / value[5:]
        if ref.is_file():
            return ref.read_text(encoding="utf-8").strip()
    return value
