#!/usr/bin/env python3
"""Freeze the prospectively frozen internal replication of the exploratory Arm C finding.

Addendum to RELATION_READOUT_REPORT_DISCRIMINATION_V1 (study owner, 2026-09-03). Writes, create-only,
reports/relation_readout_report_discrimination_v1/replication_pilot_v1/freeze/:

  REPLICATION_CONTRACT.{json,md}     hypothesis, populations, unit, baseline, primary statistic, rule
  POPULATION.jsonl                   the 1,800 pilot rows (pilot_train 1,500 + pilot_select 300) with the
                                     M1 prompt-contract fields; no outcome field of this analysis
  FOLD_ASSIGNMENTS.json              situation-grouped (pilot_train) and board-grouped (pilot_select) folds
  INPUT_MANIFEST.json                hashes of every input
  artifact_manifest.json

The label-versus-report ladder has never been run on these rows. The frozen DIM was fitted on
pilot_train and the probe layer was selected on pilot_select, so no DIM-based statistic is computed
here and the result is labelled a prospectively frozen internal replication, not untouched confirmation.
"""

from __future__ import annotations

import collections
import datetime as dt
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rrrd_v1_common as C  # noqa: E402

GEOM = Path("external-artifacts")
COMPACT = GEOM / "outputs/claim1-first-pass-semantic-stratification/source/llama-compact/claim2_m1_compact.npz"
TYPE_MAP = GEOM / "outputs/existing-human-review-payoff/v001/frozen_valueprism_type_map.csv"
GEMMA_CACHE = GEOM / "outputs/existing-human-review-payoff/v001/checkpoints/gemma_claim1_activations.npz"
ROOT_OUT = C.STUDY_ROOT / "replication_pilot_v1" / "freeze"
FOLDS = 5
INNER = 4
FOLD_SALT = "RRRD_V1_REPLICATION_FOLD|"


def entry(path: Path, role: str) -> dict[str, Any]:
    return {"role": role, "path": str(path), "bytes": path.stat().st_size, "sha256": C.sha256_file(path)}


def main() -> int:
    if ROOT_OUT.exists() and any(ROOT_OUT.iterdir()):
        raise SystemExit(f"freeze root not empty: {ROOT_OUT}")
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    a = np.load(COMPACT, allow_pickle=False)
    types = pd.read_csv(TYPE_MAP, dtype=str).fillna("").set_index("row_id")
    g = np.load(GEMMA_CACHE, allow_pickle=False)
    if list(g["row_id"].astype(str)) != list(a["row_id"].astype(str)):
        raise RuntimeError("Gemma cache row order differs from the Llama compact cache")
    rows = []
    for i in range(len(a["row_id"])):
        split = str(a["split"][i])
        if split not in ("pilot_train", "pilot_select"):
            continue
        rid, iid = str(a["row_id"][i]), str(a["item_id"][i])
        t = types.loc[rid]
        mapping = C.m1_mapping_for_item(iid)
        if mapping["name"] != str(a["mapping_name"][i]):
            raise RuntimeError(f"mapping reconstruction mismatch for {rid}")
        unit = str(a["board_id"][i]) if split == "pilot_select" else C.stable_hash("RRRD_V1_SITUATION|" + t["situation"])[:16]
        rows.append({"row_id": rid, "item_id": iid, "split": split, "population": "primary_replication" if split == "pilot_train" else "secondary_board_population", "unit_id": unit, "unit_kind": "board" if split == "pilot_select" else "situation", "board_id": str(a["board_id"][i]), "cell_position": "", "situation": t["situation"], "consideration": t["consideration"], "consideration_type": t["consideration_type"], "stored_relation": t["stored_relation"], "label": 1 if t["stored_relation"] == "Supports" else 0, "relation_sign": 1.0 if t["stored_relation"] == "Supports" else -1.0, "consensus_clear": None, "clarity_disputed": None, "mapping_name": mapping["name"], "supports_symbol": mapping["supports"], "opposes_symbol": mapping["opposes"], "reference": {"llama": {"batched_fp16_cache_native_margin": float(a["native_margin"][i]), "batched_rendered_prompt_sha256": str(a["rendered_prompt_sha256"][i]), "note": "descriptive reference only (M1 batched extraction); the frozen bs1 gate does not apply here"}, "gemma": {"note": "no retained margin; fresh extraction is the only source"}}})
    if len(rows) != 1800:
        raise RuntimeError(f"expected 1800 pilot rows, got {len(rows)}")
    # ---- folds: deterministic serpentine deal over units sorted by (label balance, mapping balance, hash)
    assignment: dict[str, dict[str, int]] = {}
    composition: dict[str, list[dict[str, Any]]] = {}
    for pop in ("primary_replication", "secondary_board_population"):
        units: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for r in rows:
            if r["population"] == pop:
                units[r["unit_id"]].append(r)
        keyed = sorted(((sum(x["label"] for x in v), sum(x["mapping_name"].endswith("reversed") for x in v), C.stable_hash(FOLD_SALT + u)), u) for u, v in units.items())
        amap = {}
        for i, (_, u) in enumerate(keyed):
            block, pos = divmod(i, FOLDS)
            amap[u] = pos if block % 2 == 0 else FOLDS - 1 - pos
        assignment[pop] = amap
        composition[pop] = [{"fold": f, "units": sum(1 for u in amap if amap[u] == f), "rows": sum(len(units[u]) for u in amap if amap[u] == f), "supports": sum(x["label"] for u in amap if amap[u] == f for x in units[u]), "reversed_mapping": sum(x["mapping_name"].endswith("reversed") for u in amap if amap[u] == f for x in units[u])} for f in range(FOLDS)]
    C.write_create_only(ROOT_OUT / "POPULATION.jsonl", C.jsonl_bytes(rows))
    C.write_create_only(ROOT_OUT / "FOLD_ASSIGNMENTS.json", C.canonical_json({"schema_version": f"{C.SCHEMA_PREFIX}_REPLICATION_FOLDS", "folds": FOLDS, "nested_inner_folds": INNER, "grouping": {"primary_replication": "situation_id (stable hash of the situation text); no situation crosses folds", "secondary_board_population": "board_id; no board crosses folds"}, "assignment": assignment, "composition": composition, "salt": FOLD_SALT}))
    contract = {
        "schema_version": f"{C.SCHEMA_PREFIX}_REPLICATION_CONTRACT", "study_id": C.STUDY_ID, "addendum": "replication_pilot_v1", "status": "FROZEN_BEFORE_EXTRACTION_AND_BEFORE_ANY_LOOK", "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evidence_status": "PROSPECTIVELY_FROZEN_INTERNAL_REPLICATION of an exploratory finding (not untouched confirmation: these rows fed M1 representation development, the frozen DIM was fitted on pilot_train and the probe layer was selected on pilot_select; the label-versus-report ladder has never been run on them)",
        "discovery": {"population": "the 500-cell human-reviewed set", "analysis": "post hoc no-position variant of Arm C", "llama_delta_logloss": 0.098, "llama_ci": [0.050, 0.142], "gemma_delta_logloss": 0.036, "gemma_ci": [-0.002, 0.068]},
        "hypothesis": "The selected-layer activation state improves out-of-fold prediction of the reviewed Supports/Opposes label beyond text, answer mapping, and the model's own native report margin.",
        "populations": {"primary_replication": {"split": "pilot_train", "rows": 1500, "unit": "situation", "units": len(assignment["primary_replication"])}, "secondary_board_population": {"split": "pilot_select", "rows": 300, "unit": "board", "units": len(assignment["secondary_board_population"]), "caveat": "M1 selected the probe layer on this split; secondary only"}, "disjointness": "no situation or consideration text is shared with the 500-cell discovery population (verified)"},
        "actors": {a: {"model_id": C.ACTORS[a]["model_id"], "revision": C.ACTORS[a]["model_revision"], "layer": C.ACTORS[a]["primary_layer"]} for a in C.ACTORS},
        "extraction": {"path": "rrrd_v1_gpu_run.py rerun-naturalistic (frozen baseline scoring path: two candidates in one right-padded batch; activations from a prompt-only batch-size-one forward)", "prompt_contract": "exact M1 primary/joint contract with the item-hash answer mapping", "layers": "primary layer used for the analysis; band retained", "dtype": "bfloat16 exact"},
        "baseline_L0": ["MiniLM(situation) (+) MiniLM(consideration) -> PCA(32) fitted inside training folds", "log character lengths, consideration_type one-hot", "answer-mapping reversed flag", "native report margin (supports minus opposes log-probability)", "NOTHING derived from cell position, row identity, board ordering, or any variable that deterministically encodes the label"],
        "augmented_L1": "L0 + selected-layer activation (standardized inside training folds)",
        "classifier": "L2 logistic, class_weight balanced, lbfgs, max_iter 5000; C from {0.001, 0.01, 0.1, 1, 10} by inner 4-fold grouped CV on training units; standardization and PCA fitted on training folds only",
        "primary_statistic": "Delta = logloss(L0) - logloss(L1) on out-of-fold label predictions, averaged within unit, 10,000-replicate unit bootstrap (seed " + str(C.BOOTSTRAP_SEED) + ")",
        "decision_rule": {"POSITIVE": "Delta > 0 and the 95% unit-bootstrap interval lies entirely above zero -> evidence for reviewed-label information beyond report", "UNDETERMINED": "interval spans zero", "NEGATIVE": "interval lies entirely below zero -> no useful incremental label information under this representation and baseline"},
        "secondaries_descriptive_only": ["AUROC and balanced accuracy of L0 and L1", "R ladder (report beyond label: R0 = text + mapping + label, R1 = R0 + activation)", "activation-only label and report probes and their coefficient cosine", "on model-label disagreement rows: which side each probe takes", "token-direction-removed variant of the primary as a robustness row (not a classifier)", "the secondary board population reported with its own interval, never pooled"],
        "not_computed": ["any DIM-based statistic (the DIM was fitted on pilot_train)", "consensus-clear subsets (no human clarity review on these rows)"],
        "licensed_wording_if_positive_in_llama": "Llama's broader activation state contains information predictive of the externally reviewed relation that is not exhausted by its reported answer; the strongest initial estimate was discovered post hoc on the 500-cell set and prospectively reproduced on a non-overlapping pre-existing population.",
        "prohibited_wording": ["the model secretly knows the correct relation", "the model believes the reviewed label", "hidden knowledge", "any change to Unit 2's READOUT_IDENTICAL_TO_REPORT"],
        "no_retry": "no change to baseline, folds, classifier grid, or rule after extraction; one extraction per actor; a failed extraction restarts from a clean output root",
    }
    C.write_create_only(ROOT_OUT / "REPLICATION_CONTRACT.json", C.canonical_json(contract))
    md = ["# Replication contract — " + C.STUDY_ID + " / replication_pilot_v1", "", f"Status: **{contract['status']}** ({contract['recorded_at']})", "", "## Hypothesis", "", contract["hypothesis"], "", "## Evidence status", "", contract["evidence_status"], "", "## Populations", ""] + [f"- {k}: {v}" for k, v in contract["populations"].items()] + ["", "## Baseline L0", ""] + [f"- {x}" for x in contract["baseline_L0"]] + ["", f"L1: {contract['augmented_L1']}", "", f"Classifier: {contract['classifier']}", "", "## Primary statistic", "", contract["primary_statistic"], "", "## Decision rule", ""] + [f"- **{k}**: {v}" for k, v in contract["decision_rule"].items()] + ["", "## Descriptive secondaries", ""] + [f"- {x}" for x in contract["secondaries_descriptive_only"]] + ["", "## Not computed", ""] + [f"- {x}" for x in contract["not_computed"]] + ["", "## Wording", "", f"- Licensed if positive in Llama: {contract['licensed_wording_if_positive_in_llama']}", "- Prohibited: " + "; ".join(contract["prohibited_wording"]), "", f"No-retry: {contract['no_retry']}", ""]
    C.write_create_only(ROOT_OUT / "REPLICATION_CONTRACT.md", "\n".join(md).encode("utf-8"))
    C.write_create_only(ROOT_OUT / "INPUT_MANIFEST.json", C.canonical_json({"schema_version": f"{C.SCHEMA_PREFIX}_REPLICATION_INPUT_MANIFEST", "inputs": [entry(COMPACT, "llama_compact_cache_row_metadata"), entry(TYPE_MAP, "valueprism_cell_text_type_map"), entry(GEMMA_CACHE, "gemma_cache_row_order_check"), entry(C.FREEZE_ROOT / "artifact_manifest.json", "parent_study_freeze_manifest"), entry(Path(__file__).resolve(), "replication_freeze_builder"), entry(C.ROOT / "scripts" / "rrrd_v1_gpu_run.py", "extraction_runner"), entry(C.ROOT / "scripts" / "rrrd_v1_q_report_v2.py", "nested_probe_code")], "population_sha256": C.sha256_file(ROOT_OUT / "POPULATION.jsonl"), "outcome_values_parsed_or_computed": False}))
    C.write_create_only(ROOT_OUT / "artifact_manifest.json", C.canonical_json({"schema_version": f"{C.SCHEMA_PREFIX}_REPLICATION_FREEZE_MANIFEST", "files": C.manifest_for(ROOT_OUT)}))
    print("REPLICATION_FREEZE_WRITTEN", ROOT_OUT, {p: len(v) for p, v in assignment.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
