#!/usr/bin/env python3
"""Build the immutable pre-outcome freeze package of RELATION_READOUT_REPORT_DISCRIMINATION_V1.

Writes (create-only) under reports/relation_readout_report_discrimination_v1/freeze/:

  STUDY_CONTRACT.json / STUDY_CONTRACT.md
  INPUT_MANIFEST.json                       every input with full SHA-256 and provenance
  ANALYSIS_PLAN.json                        metrics, thresholds, uncertainty, result->interpretation map
  TRACK_A_REELICITATION_PROMPT_FREEZE.json  the two frozen formats, the C9 scaffold adaptation, corpus hashes
  TRACK_A_DEMONSTRATION_FREEZE.jsonl        the eight demonstrations with rendered bytes for every mapping
  TRACK_A_REELICITATION_PROMPTS.jsonl       every rendered evaluation/development prompt (F1 x 4 mappings, F2)
  NATURALISTIC_CELL_INPUTS.jsonl            the 500 open ValuePrism cells with the M1 prompt contract fields
  NATURALISTIC_FOLD_ASSIGNMENTS.json        five board-grouped outer folds
  TERMINAL_CLASSIFICATIONS.json             every classification enumeration and its decision rule
  CLAIM_BOUNDARY.md
  artifact_manifest.json

No model is loaded and no outcome statistic is computed here. Reference outcome
values that already exist (Unit 2 cell table, frozen causal-baseline margins,
Unit 6/7 results) are carried as *reproduction targets* only.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import io
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rrrd_v1_common as C  # noqa: E402

GEOM = Path("external-artifacts")
HANDOFF = Path("external-artifacts")
LOOP158 = Path("external-artifacts")
PROJECTS = Path("external-artifacts")
NFS = "external-artifacts"
PROBE_BUNDLES = {
    "llama": (Path("external-artifacts"), "d1e57f79cd93e7ae7a4db44076dcbe6785c685c31e908e717012c0f509f9ac53"),
    "gemma": (Path("external-artifacts"), "bb155a092e3c33b04d6031975c2bbf43f2f57513188451022193d2ab8a9db3d8"),
}
MINILM = {"encoder_id": "sentence-transformers/all-MiniLM-L6-v2", "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"}


def git_head(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def entry(path: Path, role: str, **extra: Any) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{role}: {path}")
    return {"role": role, "path": str(path), "bytes": path.stat().st_size, "sha256": C.sha256_file(path), **extra}


# ---------------------------------------------------------------- demonstrations
DEMO_SLOTS = [
    # slot id, family, member selector, required relation/entailment, notes
    ("direct_single_holder_entailed", "DIRECT_HOLDER_TARGET_QUERY", None, "ENTAILED", "direct single-holder case"),
    ("conflicting_holder_entailed", "CONFLICTING_STANCES_IN_ONE_CONTEXT", None, "ENTAILED", "conflicting-holder context, supporting side"),
    ("conflicting_holder_not_entailed", "CONFLICTING_STANCES_IN_ONE_CONTEXT", "PAIR_OF_PREVIOUS", "NOT_ENTAILED", "same conflicting-holder context, opposing side"),
    ("target_swap_entailed", "TARGET_SWAP", None, "ENTAILED", "target-swap case"),
    ("order_reversed_not_entailed", "SENTENCE_ORDER_SWAP", 1, "NOT_ENTAILED", "order-reversed case (holder mentioned second)"),
    ("reporter_attribution_entailed", "REPORTER_VERSUS_REPORTED_HOLDER", 0, "ENTAILED", "reporter/quoted attribution: reported holder owns the stance"),
    ("negated_not_entailed", "NEGATED_STANCE", 0, "NOT_ENTAILED", "negated stance (does not support)"),
    ("withheld_not_entailed", "NEUTRAL_OR_WITHHELD_STANCE", 0, "NOT_ENTAILED", "withheld stance"),
]


def select_demonstrations(rows: list[dict[str, Any]], worlds: dict[str, dict[str, Any]], eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fit_rows = [r for r in rows if r["world_split"] == "residual_probe_train" and r["lexical_template_id"] == "EXPLICIT_DEVELOPMENT_CORE" and not r["domain_holdout"]]
    eval_names = {n for r in eval_rows for n in (r["holder_name"],)} | {worlds[r["world_id"]][k] for r in eval_rows for k in ("canonical_holder_name", "alternate_holder_name", "reporter_name", "quoter_name")}
    eval_targets = {r["target_text"] for r in eval_rows}
    eval_contexts = {C.context_block(r) for r in eval_rows}
    by_id = {r["example_id"]: r for r in rows}
    chosen: list[dict[str, Any]] = []
    used_worlds: set[str] = set()
    for slot, family, member, entail, note in DEMO_SLOTS:
        if member == "PAIR_OF_PREVIOUS":
            prev = chosen[-1]
            cid, m = prev["contrast_id"], 1 - int(prev["contrast_member"])
            cand = [r for r in fit_rows if r["contrast_id"] == cid and int(r["contrast_member"]) == m]
        else:
            cand = [r for r in fit_rows if r["variant_family"] == family and r["entailment_label"] == entail and (member is None or int(r["contrast_member"]) == member) and r["world_id"] not in used_worlds]
        cand = [r for r in cand if r["entailment_label"] == entail]
        if not cand:
            raise RuntimeError(f"no candidate for demonstration slot {slot}")
        cand.sort(key=lambda r: C.stable_hash(C.DEMO_HASH_SALT + r["example_id"]))
        pick = None
        for r in cand:
            ctx = C.context_block(r)
            names = {r["holder_name"], worlds[r["world_id"]]["canonical_holder_name"], worlds[r["world_id"]]["alternate_holder_name"], worlds[r["world_id"]]["reporter_name"], worlds[r["world_id"]]["quoter_name"]}
            if names & eval_names or r["target_text"] in eval_targets or ctx in eval_contexts:
                continue
            pick = r
            break
        if pick is None:
            raise RuntimeError(f"every candidate for {slot} overlaps evaluation worlds")
        used_worlds.add(pick["world_id"])
        chosen.append({"slot": slot, "note": note, "example_id": pick["example_id"], "world_id": pick["world_id"], "contrast_id": pick["contrast_id"], "contrast_member": int(pick["contrast_member"]), "variant_family": pick["variant_family"], "relation_label": pick["relation_label"], "entailment_label": pick["entailment_label"], "holder_name": pick["holder_name"], "queried_operator": pick["queried_operator"], "domain": pick["domain"], "context": C.context_block(pick), "claim": C.declarative_claim(pick), "selection_hash": C.stable_hash(C.DEMO_HASH_SALT + pick["example_id"])})
    # fixed presentation order for every prompt: ascending selection hash (position-neutral, deterministic)
    order = sorted(range(len(chosen)), key=lambda i: chosen[i]["selection_hash"])
    chosen = [chosen[i] for i in order]
    for pos, d in enumerate(chosen):
        d["presentation_position"] = pos
    labels = collections.Counter(d["entailment_label"] for d in chosen)
    if labels["ENTAILED"] != 4 or labels["NOT_ENTAILED"] != 4:
        raise RuntimeError(f"demonstration balance broken: {labels}")
    if len({d["example_id"] for d in chosen}) != 8:
        raise RuntimeError("demonstrations are not eight distinct rows")
    for d in chosen:
        d["rendered"] = {"F2": C.render_f2_example(d["context"], d["claim"], d["entailment_label"])}
        for mapping in C.MAPPINGS:
            d["rendered"][f"F1:{mapping}"] = C.render_f1_example(d["context"], d["claim"], mapping, C.f1_symbol_for(mapping, d["entailment_label"]))
        d["rendered_sha256"] = {k: C.sha256_text(v) for k, v in d["rendered"].items()}
    return chosen


# ---------------------------------------------------------------- prompt corpus
def render_corpus(rows: list[dict[str, Any]], demos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if r["world_split"] not in ("development", "heldout_evaluation"):
            continue
        if C.f1_symbol_for(r["answer_mapping_id"], r["entailment_label"]) != r["expected_answer_token"]:
            raise RuntimeError(f"mapping convention mismatch on {r['example_id']}")
        ctx, claim = C.context_block(r), C.declarative_claim(r)
        base = {k: r[k] for k in ("example_id", "world_id", "world_split", "readout_eligibility", "contrast_id", "contrast_member", "variant_family", "discourse_construction", "lexical_family", "lexical_template_id", "lexical_holdout", "discourse_holdout", "domain", "domain_holdout", "relation_label", "entailment_label", "queried_operator", "holder_role", "holder_name", "target_slot", "target_text", "target_polarity", "holder_mentioned_first", "answer_mapping_id", "expected_answer_token", "paired_reference_contrast_id", "paired_reference_member")}
        base["context"], base["claim"] = ctx, claim
        base["baseline_prompt_sha256"] = C.sha256_text(r["prompt_text"])
        for mapping in C.MAPPINGS:
            text = C.render_f1_prompt(demos, ctx, claim, mapping)
            sym_e, sym_n = C.MAPPING_SYMBOLS[mapping]
            out.append({**base, "format": "F1", "prompt_id": f"{r['example_id']}__F1__{mapping}", "mapping_id": mapping, "own_mapping": mapping == r["answer_mapping_id"], "candidates": [f" {sym_e}", f" {sym_n}"], "candidate_semantics": list(C.F1_SEMANTICS), "expected_symbol": C.f1_symbol_for(mapping, r["entailment_label"]), "retain_activations": mapping == r["answer_mapping_id"], "prompt_text": text, "prompt_sha256": C.sha256_text(text)})
        text = C.render_f2_prompt(demos, ctx, claim)
        out.append({**base, "format": "F2", "prompt_id": f"{r['example_id']}__F2", "mapping_id": None, "own_mapping": True, "candidates": [f" {C.F2_LABELS[0]}", f" {C.F2_LABELS[1]}"], "candidate_semantics": list(C.F2_LABELS), "expected_symbol": r["entailment_label"], "retain_activations": True, "prompt_text": text, "prompt_sha256": C.sha256_text(text)})
    return out


# ---------------------------------------------------------------- naturalistic cells
def build_cells() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v001 = GEOM / "outputs" / "existing-human-review-payoff" / "v001"
    strata = {a: pd.read_csv(v001 / f, dtype=str).fillna("") for a, f in (("llama", "claim1_human_strata_cells.csv"), ("gemma", "claim1_human_strata_gemma_cells.csv"))}
    types = pd.read_csv(v001 / "frozen_valueprism_type_map.csv", dtype=str).fillna("")
    compact = np.load(GEOM / "outputs/claim1-first-pass-semantic-stratification/source/llama-compact/claim2_m1_compact.npz", allow_pickle=False)
    comp = pd.DataFrame({"item_id": compact["item_id"].astype(str), "row_id": compact["row_id"].astype(str), "split": compact["split"].astype(str), "board_id": compact["board_id"].astype(str), "llama_batched_native_margin": compact["native_margin"].astype(float), "llama_batched_prompt_token_index": compact["prompt_token_index"].astype(int), "llama_batched_mapping_name": compact["mapping_name"].astype(str), "llama_batched_rendered_prompt_sha256": compact["rendered_prompt_sha256"].astype(str)})
    comp = comp[comp["split"] == "pilot_eval"].copy()
    if len(comp) != 500:
        raise RuntimeError("pilot_eval population is not 500 rows")
    causal = {}
    for a in ("llama", "gemma"):
        p = pd.read_parquet(v001 / "checkpoints" / f"causal_claim1_{a}.parquet")
        n = p[p["family"] == "no_intervention"][["row_id", "item_id", "board_id", "mapping", "semantic_margin", "supports_logp", "opposes_logp", "prompt_token_index", "prompt_token_ids_sha256", "layer"]].copy()
        if len(n) != 500 or n["row_id"].duplicated().any():
            raise RuntimeError(f"{a} causal baseline is not 500 unique rows")
        causal[a] = n.set_index("row_id")
    cells = []
    llama_s = strata["llama"].set_index("row_id")
    gemma_s = strata["gemma"].set_index("row_id")
    tmap = types.set_index("row_id")
    for rec in comp.sort_values(["board_id", "row_id"]).itertuples(index=False):
        rid = rec.row_id
        ls, gs, t = llama_s.loc[rid], gemma_s.loc[rid], tmap.loc[rid]
        mapping = C.m1_mapping_for_item(rec.item_id)
        if mapping["name"] != causal["llama"].loc[rid, "mapping"] or mapping["name"] != causal["gemma"].loc[rid, "mapping"]:
            raise RuntimeError(f"mapping reconstruction mismatch for {rid}")
        if rec.llama_batched_mapping_name != mapping["name"].split(":")[0] + ":" + mapping["name"].split(":")[1]:
            pass  # compact stores 'primary:standard' style too; checked below
        if ls["board_id"] != rec.board_id or gs["board_id"] != rec.board_id:
            raise RuntimeError(f"board mismatch for {rid}")
        if t["stored_relation"] != ls["stored_relation"]:
            raise RuntimeError(f"stored relation mismatch for {rid}")
        cell = {
            "row_id": rid, "item_id": rec.item_id, "board_id": rec.board_id, "review_id": ls["review_id"], "cell_position": ls["cell_position"], "consideration_slot": ls["consideration_slot"],
            "situation": t["situation"], "consideration": t["consideration"], "consideration_type": t["consideration_type"],
            "stored_relation": t["stored_relation"], "label": int(ls["label"]), "relation_sign": float(ls["relation_sign"]),
            "consensus_clear": ls["consensus_clear"] == "True", "clarity_disputed": ls["clarity_disputed"] == "True", "ordered_clarity_stratum": ls["ordered_clarity_stratum"],
            "mapping_name": mapping["name"], "supports_symbol": mapping["supports"], "opposes_symbol": mapping["opposes"],
            "reference": {
                "llama": {"unit2_native_answer_margin": float(ls["native_answer_margin"]), "unit2_difference_in_means": float(ls["difference_in_means"]), "unit2_logistic": float(ls["logistic"]), "unit2_margin_origin": "M1 batched (batch 8, right-padded) extraction, float16 activation cache", "bs1_semantic_margin": float(causal["llama"].loc[rid, "semantic_margin"]), "bs1_supports_logp": float(causal["llama"].loc[rid, "supports_logp"]), "bs1_opposes_logp": float(causal["llama"].loc[rid, "opposes_logp"]), "bs1_prompt_token_index": int(causal["llama"].loc[rid, "prompt_token_index"]), "bs1_prompt_token_ids_sha256": causal["llama"].loc[rid, "prompt_token_ids_sha256"], "batched_prompt_token_index": int(rec.llama_batched_prompt_token_index), "batched_rendered_prompt_sha256": rec.llama_batched_rendered_prompt_sha256},
                "gemma": {"unit2_native_answer_margin": float(gs["native_answer_margin"]), "unit2_difference_in_means": float(gs["difference_in_means"]), "unit2_logistic": float(gs["logistic"]), "unit2_margin_origin": "batch-size-one causal baseline (same protocol as this study)", "bs1_semantic_margin": float(causal["gemma"].loc[rid, "semantic_margin"]), "bs1_supports_logp": float(causal["gemma"].loc[rid, "supports_logp"]), "bs1_opposes_logp": float(causal["gemma"].loc[rid, "opposes_logp"]), "bs1_prompt_token_index": int(causal["gemma"].loc[rid, "prompt_token_index"]), "bs1_prompt_token_ids_sha256": causal["gemma"].loc[rid, "prompt_token_ids_sha256"]},
            },
        }
        cells.append(cell)
    if len({c["row_id"] for c in cells}) != 500 or len({c["board_id"] for c in cells}) != 125:
        raise RuntimeError("cell population topology is not 500 rows / 125 boards")
    summary = {"cells": 500, "boards": 125, "labels": dict(collections.Counter(c["stored_relation"] for c in cells)), "consensus_clear": sum(c["consensus_clear"] for c in cells), "mappings": dict(collections.Counter(c["mapping_name"] for c in cells)), "types": dict(collections.Counter(c["consideration_type"] for c in cells))}
    return cells, summary


def assign_folds(cells: list[dict[str, Any]], folds: int = 5) -> dict[str, Any]:
    """Board-grouped, deterministic, stratified as far as possible.

    Boards are sorted by a composite stratification key (label balance, report
    agreement in each actor, consensus clarity, mapping balance, then board id)
    and dealt in serpentine order so every fold receives a matched profile.
    """
    boards: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for c in cells:
        boards[c["board_id"]].append(c)
    keyed = []
    for b, rows in boards.items():
        n_sup = sum(r["label"] for r in rows)
        agree_l = sum((r["reference"]["llama"]["bs1_semantic_margin"] > 0) == bool(r["label"]) for r in rows)
        agree_g = sum((r["reference"]["gemma"]["bs1_semantic_margin"] > 0) == bool(r["label"]) for r in rows)
        clear = sum(r["consensus_clear"] for r in rows)
        rev = sum(r["mapping_name"].endswith("reversed") for r in rows)
        keyed.append(((agree_l + agree_g, clear, n_sup, rev, C.stable_hash("RRRD_V1_FOLD|" + b)), b))
    keyed.sort()
    assignment: dict[str, int] = {}
    for i, (_, b) in enumerate(keyed):
        block, pos = divmod(i, folds)
        fold = pos if block % 2 == 0 else folds - 1 - pos
        assignment[b] = fold
    composition = []
    for f in range(folds):
        rows = [c for c in cells if assignment[c["board_id"]] == f]
        composition.append({"fold": f, "boards": sum(1 for b, ff in assignment.items() if ff == f), "cells": len(rows), "supports": sum(r["label"] for r in rows), "consensus_clear": sum(r["consensus_clear"] for r in rows), "llama_report_agrees": sum((r["reference"]["llama"]["bs1_semantic_margin"] > 0) == bool(r["label"]) for r in rows), "gemma_report_agrees": sum((r["reference"]["gemma"]["bs1_semantic_margin"] > 0) == bool(r["label"]) for r in rows), "reversed_mapping": sum(r["mapping_name"].endswith("reversed") for r in rows), "cell_positions": dict(collections.Counter(r["cell_position"] for r in rows))})
    return {"schema_version": f"{C.SCHEMA_PREFIX}_NATURALISTIC_FOLDS", "grouping": "board_id (no board crosses folds)", "folds": folds, "stratification": "boards sorted by (report agreement in both actors, consensus-clear count, support count, reversed-mapping count, stable board hash) and dealt serpentine", "nested_inner_folds": 4, "seed": C.FOLD_SEED, "assignment": assignment, "composition": composition, "outcome_values_used_for_assignment": "frozen causal-baseline report sign only (existing artifact, not a new outcome)"}


# ---------------------------------------------------------------- contracts
def analysis_plan() -> dict[str, Any]:
    core = ["same_context_dual_query_reversal", "holder_swap_correct", "target_swap_correct"]
    return {
        "schema_version": f"{C.SCHEMA_PREFIX}_ANALYSIS_PLAN",
        "independent_units": {"track_a": "world_id", "naturalistic": "board_id"},
        "uncertainty": {"replicates": C.BOOTSTRAP_REPLICATES, "seed": C.BOOTSTRAP_SEED, "method": "multinomial resampling of independent units; paired differences resample the same units for both arms", "interval": [0.025, 0.975], "p_value": "fraction of bootstrap draws on the null side of the gate, (k+1)/(B+1)"},
        "multiplicity": {"holm": "within actor and within predeclared family", "families": {"arm_a_specificity_primary": "3 comparators (residual-relation minus sentiment, factual, answer_token) x the mean over the three core endpoints", "arm_a_specificity_secondary": "per endpoint comparisons, reported", "arm_b_bound": "dual-query, holder-swap, target-swap per format", "arm_c": "one primary statistic per direction (L and R) per actor; token-removed and consensus-clear variants are robustness"}},
        "arm_a": {
            "label": "PROSPECTIVELY_FROZEN_POST_HOC_DIAGNOSTICS",
            "inputs": "retained Unit 6/7 completed jobs, full selected-layer activations, frozen competitor direction bundles, world metadata, frozen fit/development/evaluation splits; no forward pass",
            "4.1_transfer_ladder": {
                "directions": ["raw_relation_dim", "jointly_residualized_relation"],
                "levels": {
                    "A_STRICT_SOURCE_CALIBRATION": "score = activation . direction - source_midpoint (M1 probe bundle difference_in_means_midpoint, orientation +=Supports); SUPPORT iff score > 0. The jointly residualized direction has no authoritative source midpoint -> SOURCE_CALIBRATION_MISSING (never replaced by zero).",
                    "B_TARGET_SCALAR_CALIBRATION": "the existing Track A one-dimensional logistic (fit rows, standardized, C on development worlds) on the projection; no vector refitting; reproduces the frozen dim_scalar readout for the raw direction",
                    "C_THRESHOLD_FREE_PAIRED_TRANSFER": "within-world paired score differences: dual-query (CONFLICTING pair: proj(SUPPORT member) > proj(OPPOSE member)), holder swap, target swap (same rule), role-permutation transfer (the same paired rule restricted to evaluation worlds whose alternate holder supports), order-crossed transfer (order-reversed rows versus every opposite-label signed row of the same world, fraction ordered correctly), sign statistic = within-world AUROC of SUPPORT over OPPOSE signed rows macro-averaged over worlds; mapping agreement NOT_DEFINED at this level",
                },
                "metrics": ["sign_balanced_accuracy (A,B) / world_auroc (C)", "same_context_dual_query_reversal", "holder_swap_correct", "target_swap_correct", "role_permutation_transfer", "order_crossed_transfer", "mapping_reversal_semantic_agreement (A,B only)"],
                "gates_for_reference": {"dual_query": C.PRACTICAL_GATE, "sign": C.PRACTICAL_GATE, "role_order": 0.60, "mapping": C.AGREEMENT_GATE},
                "reporting_rule": "never call all three levels zero-shot; the vector is frozen in every level and the scalar calibration status is named",
            },
            "4.2_paired_specificity": {
                "directions": ["raw_relation_dim", "jointly_residualized_relation", "sentiment", "factual", "answer_token", "matched_norm_random x128 per reference"],
                "endpoints": ["operator_sign (level-B calibrated, per-world BA)", "operator_sign_world_auroc (threshold-free)", "same_context_dual_query_reversal", "holder_swap_correct", "target_swap_correct", "role_permutation_transfer", "order_crossed_transfer", "abs_context_conditioned_effect (unit-norm direction; |(hs0-hs1)-(qo0-qo1)| per world)"],
                "primary_comparators": ["jointly_residualized_relation - sentiment", "jointly_residualized_relation - factual", "jointly_residualized_relation - answer_token"],
                "primary_statistic": "per comparator, the world-paired difference averaged over the three core endpoints " + str(core) + "; 10,000 world bootstraps; Holm over the three comparators within actor",
                "random_controls": "the 128 matched-norm random directions are evaluated on the calibrated sign statistic AND on every threshold-free paired endpoint; band = [q2.5, q97.5] over directions; a band is DISCRIMINATING for a statistic iff its q97.5 is below the practical gate 0.70 (so random directions cannot reach the pass region); the calibrated sign band is expected to be non-discriminating (Llama 0.83-0.87, Gemma 0.94-0.98 observed)",
                "classification": {
                    "RELATION_SPECIFICITY_FAVORED": "all three primary comparator differences have Holm-adjusted p < 0.05 with positive point estimates",
                    "RELATION_SPECIFICITY_NOT_FAVORED": "no comparator is significantly positive and at least one is significantly negative (Holm)",
                    "MIXED_SPECIFICITY": "at least one comparator significantly positive and at least one not (or significantly negative)",
                    "RANDOM_NULL_NONDISCRIMINATING": "the random band is non-discriminating on every threshold-free core endpoint (q97.5 >= 0.70) so no specificity statement can be tested",
                    "UNDERDETERMINED": "no comparator is significant in either direction",
                    "precedence": "RANDOM_NULL_NONDISCRIMINATING is evaluated first; then FAVORED; then NOT_FAVORED; then MIXED; else UNDERDETERMINED",
                },
            },
            "4.3_presence_vs_polarity": {
                "targets": {"SIGNED_POLARITY": "SUPPORT vs OPPOSE on rows with relation_label in {SUPPORT, OPPOSE}", "RELATION_PRESENCE": "PRESENT for {SUPPORT, OPPOSE}; ABSENT_OR_WITHHELD for {WITHHELD, DOES_NOT_SUPPORT, DOES_NOT_OPPOSE}", "THREE_STATE_RELATION": "{SUPPORT, OPPOSE, ABSENT_OR_WITHHELD}"},
                "models": {"1_dim_polarity_alone": "logistic on [proj, |proj|] of the raw DIM (fit rows)", "2_presence_full_activation": "L2 logistic on the full activation, C from {0.001,0.01,0.1,1,10} on development worlds", "3_presence_after_dim_and_nuisance_removal": "same on the activation residualized (fit-world coefficients) against the DIM scalar and the frozen nuisance covariates", "4_two_factor": "frozen polarity (level-B calibrated DIM sign) composed with model 2 presence: ABSENT_OR_WITHHELD if presence says absent, else the DIM sign", "5_direct_three_state": "L2 logistic three-state classifier on the full activation"},
                "evaluation": "fit on fit worlds, C on development worlds, one evaluation on held-out worlds; metrics macro-world balanced accuracy; families: negated (NEGATED_STANCE), withheld (NEUTRAL_OR_WITHHELD_STANCE, REPORTER/QUOTER member 1, QUERY_ONLY), attribution (REPORTER/QUOTER), lexical (EXPLICIT_LEXICAL_REALIZATION, LEXICAL_FREE_OR_IMPLICIT_REALIZATION incl. held-out templates), discourse (held-out QUOTER), mapping (ANSWER_MAPPING_REVERSAL)",
                "held_out_family": "leave-one-family-out refits of model 2 (fit worlds minus the family; evaluation worlds of that family) for negated, withheld, attribution, lexical",
                "classification": {
                    "PRESENCE_PLUS_POLARITY_SUPPORTED": "model-2 presence BA >= 0.70 with CI low > 0.60 on evaluation worlds AND leave-one-family-out presence BA >= 0.60 for both negated and withheld families AND paired (model 2 - model 1) presence BA CI low > 0",
                    "BIPOLAR_POLARITY_ONLY": "model-2 presence BA CI includes 0.50 or point < 0.60",
                    "PRESENCE_TEMPLATE_SPECIFIC": "model-2 in-distribution presence BA >= 0.70 but leave-one-family-out presence BA < 0.60 for negated or withheld",
                    "MIXED_PRESENCE_RESULT": "in-distribution passes, held-out passes on one of the two families only, or model 1 matches model 2",
                    "UNDERDETERMINED": "otherwise",
                },
                "boundary": "may show the distributed state is richer than DIM; may not be described as a causal relation operator",
            },
            "4.4_mapping_reversal": {
                "primary_correction": "intercept only: per mapping m, offset_m = mean over signed fit rows with mapping m of (proj - class mean of proj on fit rows); corrected proj = proj - offset_m; applied unchanged to evaluation worlds",
                "secondary_correction": "affine: per mapping, proj = a_m + b_m * sign fit on signed fit rows; corrected = (proj - a_m)/b_m * b_pooled + a_pooled; never selected on evaluation",
                "reported_before_after": ["paired raw-score correlation across ANSWER_MAPPING_REVERSAL pairs", "mean mapping-specific shift", "semantic sign agreement (level-A strict midpoint and level-B calibrated) across mapping-reversal pairs", "dual-query, holder-swap, target-swap (threshold-free)", "calibration transfer: level-B classifier fit on corrected fit rows evaluated on corrected evaluation rows"],
                "held_out_criterion": "mapping_reversal_semantic_agreement >= 0.90 on evaluation-world pairs (level B) after the prospectively fixed correction",
                "classification": {"MAPPING_OFFSET_EXPLAINS_INSTABILITY": "intercept correction meets the criterion", "MAPPING_AFFINE_SHIFT_EXPLAINS_INSTABILITY": "intercept fails, affine meets it", "DEEPER_MAPPING_DEPENDENCE": "agreement < 0.75 after both corrections", "MIXED_MAPPING_RESULT": "0.75 <= agreement < 0.90 after both", "UNDERDETERMINED": "fewer than 20 evaluable evaluation pairs"},
                "comparator": "full-activation logistic mapping agreement (existing 1.000 both actors) recomputed alongside",
            },
            "4.5_additive_holder_sign": {
                "holder_decoder": "L2 logistic on the full activation trained on QUERY_ONLY fit rows (label: queried slot CANONICAL vs ALTERNATE), C on QUERY_ONLY development rows",
                "sign": "frozen raw DIM at level B (calibrated one-dimensional logistic probabilities); level A strict as secondary",
                "combination": "P(slot, sign) = P_holder(slot) * P_sign(sign); argmax; no interaction fitted on evaluation data",
                "comparators": "existing full-activation and residual-activation joint classifiers (recomputed with the frozen code path)",
                "metric": "macro-world four-class joint binding balanced accuracy on evaluation two-actor rows; world-paired bootstrap differences",
                "classification": {"JOINT_INTERACTION_ADDS_INFORMATION": "paired (joint - additive) CI low > 0.05 for both joint classifiers", "ADDITIVE_HOLDER_PLUS_SIGN_SUFFICIENT": "paired (joint - additive) CI high < 0.05 for both joint classifiers and additive CI low > 0.25", "MIXED_ADDITIVE_RESULT": "the two joint classifiers give different verdicts", "UNDERDETERMINED": "otherwise"},
                "boundary": "the frozen G1 decoding verdict is preserved regardless",
            },
            "4.6_g3_amendment": "factual reporting amendment only; no recomputation",
        },
        "arm_b": {
            "label": "PROSPECTIVE_FOLLOW_UP_EVIDENCE",
            "formats": ["F1 C9_MAPPED_ENTAILMENT_8SHOT", "F2 DIRECT_SEMANTIC_LABEL_8SHOT"],
            "co_primary": True,
            "rows": "every development and held-out evaluation row (all families); F1 rendered under all four mappings (activations retained under the row's own frozen mapping), F2 once",
            "behavioral_prediction": {"F1_primary": "row's own frozen mapping: ENTAILED iff sequence log-prob(entailed symbol) > log-prob(not-entailed symbol); ties -> NOT_ENTAILED", "F1_secondary": "mean mapping-normalized margin across the four mappings", "F2": "ENTAILED iff sequence log-prob(' ENTAILED') > log-prob(' NOT_ENTAILED'); generated label parsed by ^\\s*(NOT_ENTAILED|ENTAILED)\\b; unparseable counts as disagreement"},
            "sign_prediction": "as the frozen native_answer readout: predicted relation = queried operator if entailed else its opposite (signed rows only)",
            "endpoints": ["direct_single_holder_entailment_ba", "operator_sign_ba", "same_context_dual_query_reversal", "holder_swap_correct", "target_swap_correct", "role_permutation_transfer", "order_crossed_transfer", "reporter_quoter_attribution", "negated_withheld_entailment_ba", "F1_mapping_semantic_agreement (all four mappings agree, per row, macro world)", "F2_candidate_vs_generated_agreement", "native_vs_frozen_dim_sign_agreement (level A and level C)", "paired_improvement_vs_zero_shot_baseline", "frozen_dim_ladder_under_new_format (levels A, C; level B calibrated on development worlds only)"],
            "definitions": {"DIRECT_TASK_COMPETENT": "direct single-holder BA >= 0.80 AND its world-bootstrap CI low > 0.70 AND (F1: mapping agreement >= 0.90) AND (F2: candidate/generated agreement >= 0.90)", "BOUND_BEHAVIOR_PASS": "dual-query, holder-swap, target-swap each >= 0.70 (points); role and order transfer reported separately", "IMPROVES_INTO_PRACTICAL_RANGE": "each of the three bound endpoints has point >= 0.60 and paired improvement over the zero-shot baseline with CI low > 0", "FROZEN_READOUT_PASS": "raw DIM under the format's activations: dual-query >= 0.70 and sign >= 0.70 at level A or level C (points, with CIs reported)", "ALIGNS_WITH_DIM": "native-vs-DIM sign agreement (level C reference: sign of the strict-source score) >= 0.80 on signed evaluation rows"},
            "classification_order": ["TASK_NOT_ELICITED if no format is DIRECT_TASK_COMPETENT", "PROMPT_SENSITIVE_INTERNAL_READOUT if a competent format fails FROZEN_READOUT_PASS while the frozen zero-shot activations pass it at the same level", "ROBUST_ELICITATION_REPAIR if both formats competent AND (BOUND_BEHAVIOR_PASS or IMPROVES_INTO_PRACTICAL_RANGE) for both AND ALIGNS_WITH_DIM for both", "VALID_TASK_INTERNAL_READOUT_BEHAVIOR_DISSOCIATION if some competent format has CI high < 0.70 on at least two of the three bound endpoints AND the frozen readout passes those endpoints with CI low > 0.70 AND the world-paired readout-minus-behavior difference CI excludes zero", "FORMAT_SPECIFIC_ELICITATION_REPAIR if exactly one format is competent AND it passes/improves the bound endpoints AND aligns with DIM", "MIXED_OR_UNDERDETERMINED otherwise"],
            "joint_two_actor": "the pair of classifications; SAME_CLASS or ACTOR_DIVERGENT",
            "no_retry": "no adaptive prompt tuning, no third format, no threshold relaxation, no post-outcome format selection; development worlds never alter a prompt",
        },
        "arm_c": {
            "label": "PROSPECTIVE_FOLLOW_UP_EVIDENCE (extraction) + frozen probe analysis",
            "population": "the open human-reviewed 125-board / 500-cell ValuePrism population of Unit 2; the sealed 60-board set is never touched",
            "extraction_verification": {"prompt_identity": "rendered prompt token ids must hash (8-byte LE signed, sha256) to the frozen batch-size-one causal-baseline prompt_token_ids_sha256 for all 500 rows in each actor", "margin_reproduction": "new margin vs frozen batch-size-one baseline: absolute difference <= 0.002 on >= 99% of rows and hard-verdict (sign) agreement 100% on rows with |baseline margin| > 0.002; the Llama Unit 2 table margin (batched origin) is compared descriptively (sign agreement, max abs diff) and cannot gate", "dim_reproduction": "new strict-source DIM score vs the Unit 2 cell table difference_in_means: sign agreement >= 0.99 and Pearson r >= 0.99 (the Unit 2 Llama values came from float16 activations)", "failure": "NATURALISTIC_EXTRACTION_INVALID; no probe is fitted"},
            "probes": {"label": "reviewed Supports/Opposes", "report": "sign of the new mapping-normalized native margin", "classifier": "L2 logistic, standardized on training folds, C from {0.001,0.01,0.1,1,10} by inner 4-fold grouped CV on training boards, max_iter 5000, class_weight balanced"},
            "conditional_information": {"L0": "G4 comparator features (MiniLM(situation) (+) MiniLM(consideration) -> PCA(32) fit in training folds, log lengths, consideration_type one-hot, cell_position one-hot) + mapping covariate (reversed) + native margin + native sign", "L1": "L0 + selected-layer activation (standardized in training folds)", "R0": "text features + mapping covariate + reviewed label", "R1": "R0 + selected-layer activation", "statistic": "held-out log-loss improvement (baseline minus augmented) averaged within board, 10,000 board bootstraps; AUROC and balanced accuracy secondary", "token_removed": "repeat after projecting out the frozen answer_token direction (fixed vector) and regressing out its projection with training-fold coefficients", "consensus_clear": "prespecified robustness subset"},
            "classification": {"BIDIRECTIONAL_LABEL_REPORT_SEPARABILITY": "both L0->L1 and R0->R1 improvements have board-bootstrap CI low > 0", "LABEL_INFORMATION_BEYOND_REPORT": "only L0->L1", "REPORT_INFORMATION_BEYOND_LABEL": "only R0->R1", "NO_DETECTED_INCREMENT": "neither CI low > 0 and neither CI high < 0", "ACTIVATION_INCREMENT_ADVERSE": "an augmented model has CI high < 0", "MIXED_OR_UNDERDETERMINED": "the token-removed variant changes the classification"},
            "layer_band": "label/report probes repeated per band layer for the trajectory CSV; no layerwise significance claim is made, so no layer is selected",
        },
        "result_to_interpretation": {"case_1": "re-elicited behavior competent and aligned with DIM -> old Track A native failure was elicitation failure; no representation-report dissociation from Track A", "case_2": "competence + consistent generation + poor multi-entity behavior + strong DIM -> internal-readout/behavioral-output dissociation; not hidden knowledge", "case_3": "behavior improves but frozen DIM effect disappears -> prompt-format dependence; weaken the general query-bound claim", "case_4": "no direct competence -> native Track A behavior uninterpretable; keep the activation result", "case_5": "naturalistic label increment beyond report -> DIM scalar report-aligned but broader state carries separable label information", "case_6": "no naturalistic label increment -> report-dominant account at the tested layer with power limits", "case_7": "presence plus polarity succeeds -> DIM is one bipolar coordinate in a richer relation state", "case_8": "additive matches joint -> binding in the decoding sense through jointly accessible holder and sign coordinates", "case_9": "joint exceeds additive -> extra holder-relation interaction information linearly available"},
    }


def terminal_classifications(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": f"{C.SCHEMA_PREFIX}_TERMINAL_CLASSIFICATIONS",
        "arm_a": {"4.2": plan["arm_a"]["4.2_paired_specificity"]["classification"], "4.3": plan["arm_a"]["4.3_presence_vs_polarity"]["classification"], "4.4": plan["arm_a"]["4.4_mapping_reversal"]["classification"], "4.5": plan["arm_a"]["4.5_additive_holder_sign"]["classification"], "4.1": {"levels": ["STRICT_SOURCE_CALIBRATION", "TARGET_SCALAR_CALIBRATION", "THRESHOLD_FREE_PAIRED_TRANSFER"], "missing": "SOURCE_CALIBRATION_MISSING"}},
        "arm_b": {"per_actor": ["ROBUST_ELICITATION_REPAIR", "FORMAT_SPECIFIC_ELICITATION_REPAIR", "VALID_TASK_INTERNAL_READOUT_BEHAVIOR_DISSOCIATION", "TASK_NOT_ELICITED", "PROMPT_SENSITIVE_INTERNAL_READOUT", "MIXED_OR_UNDERDETERMINED"], "rule_order": plan["arm_b"]["classification_order"], "definitions": plan["arm_b"]["definitions"]},
        "arm_c": {"per_actor": ["BIDIRECTIONAL_LABEL_REPORT_SEPARABILITY", "LABEL_INFORMATION_BEYOND_REPORT", "REPORT_INFORMATION_BEYOND_LABEL", "NO_DETECTED_INCREMENT", "ACTIVATION_INCREMENT_ADVERSE", "MIXED_OR_UNDERDETERMINED", "NATURALISTIC_EXTRACTION_INVALID"], "rules": plan["arm_c"]["classification"], "extraction_gate": plan["arm_c"]["extraction_verification"]},
        "terminal_status": ["RELATION_READOUT_REPORT_DISCRIMINATION_V1_COMPLETE", "RELATION_READOUT_REPORT_DISCRIMINATION_V1_PARTIAL_SCIENTIFIC_RESULT", "RELATION_READOUT_REPORT_DISCRIMINATION_V1_INVALID", "AWAITING_HUMAN_LAMBDA_INSTANCE"],
    }


CLAIM_BOUNDARY_MD = """# Claim boundary — RELATION_READOUT_REPORT_DISCRIMINATION_V1

Frozen before any new outcome computation.

## May establish, depending on results
- frozen-direction transfer with or without scalar calibration (the calibration level is always named);
- query-conditioned relation decoding on the controlled corpus;
- favorable actor-specific semantic specificity (per actor; actors need not agree);
- a bipolar polarity coordinate plus a separate relation-presence representation;
- a mapping-specific scalar offset (mapping invariance only if the prospectively fixed correction passes the frozen held-out criterion);
- additive versus interaction-bearing decoding of holder and sign;
- output elicitation failure of the original zero-shot Track A protocol;
- an internal-readout / behavioral-output dissociation under a separately competent output protocol, licensing only: "The linear internal readout tracks the constructed queried relation on cases where a separately competent output protocol fails to report it.";
- reviewed-label information beyond the model report in the broader activation state of the open naturalistic population.

## May not establish from these data alone
- objective moral truth; "the model secretly knows" or "believes";
- a compact symbolic relation operator or a query-independent scene graph;
- a unique causal layer or unique direction; causal use of the decoded holder-binding information;
- mapping invariance unless the frozen correction passes;
- universal model-family behavior;
- Q-FAMILY, causal compression, or causal fingerprints;
- absence of transverse fragility information (G5 stays ENDPOINT_AT_FLOOR);
- untouched confirmation on the sealed 60-board set;
- any change to the frozen terminal verdicts of relation-program v2, V3, Claim 2, or C9.

## Evidence status labels
- Arm A: PROSPECTIVELY_FROZEN_POST_HOC_DIAGNOSTICS (outcomes of the underlying forwards were already known).
- Arm B and Arm C extraction: prospective follow-up evidence, not an untouched confirmation of G1/G2.
- The prior zero-shot Track A native results are the frozen baseline; they are not rerun.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze-root", type=Path, default=C.FREEZE_ROOT)
    args = ap.parse_args(argv)
    root: Path = args.freeze_root
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"freeze root not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    # ---- Track A corpus and demonstrations
    rows = C.read_jsonl(C.TRACK_A_FREEZE / "MODEL_INPUTS_LLAMA.jsonl")
    rows_g = C.read_jsonl(C.TRACK_A_FREEZE / "MODEL_INPUTS_GEMMA.jsonl")
    if [r["example_id"] for r in rows] != [r["example_id"] for r in rows_g] or any(a["prompt_text"] != b["prompt_text"] for a, b in zip(rows, rows_g)):
        raise RuntimeError("Llama and Gemma model inputs are not the same prompt corpus")
    worlds = {w["world_id"]: w for w in C.read_jsonl(C.TRACK_A_FREEZE / "RELATION_BINDING_FACTORIAL_V2_WORLDS.jsonl")}
    eval_rows = [r for r in rows if r["world_split"] == "heldout_evaluation"]
    demos = select_demonstrations(rows, worlds, eval_rows)
    corpus = render_corpus(rows, demos)
    demo_names = {d["holder_name"] for d in demos} | {worlds[d["world_id"]][k] for d in demos for k in ("canonical_holder_name", "alternate_holder_name", "reporter_name", "quoter_name")}
    for p in corpus:
        if p["world_split"] == "heldout_evaluation":
            if any(n in p["context"] or n in p["claim"] for n in demo_names):
                raise RuntimeError(f"demonstration name leaks into evaluation prompt {p['prompt_id']}")
            if any(d["context"] == p["context"] or d["claim"] == p["claim"] for d in demos):
                raise RuntimeError(f"demonstration content overlaps evaluation prompt {p['prompt_id']}")
    C.write_create_only(root / "TRACK_A_DEMONSTRATION_FREEZE.jsonl", C.jsonl_bytes(demos))
    C.write_create_only(root / "TRACK_A_REELICITATION_PROMPTS.jsonl", C.jsonl_bytes(corpus))
    counts = collections.Counter((p["format"], p["world_split"]) for p in corpus)
    c9_script = C.ROOT / "scripts" / "c9_base_vs_instruct_v1.py"
    prompt_freeze = {
        "schema_version": f"{C.SCHEMA_PREFIX}_TRACK_A_REELICITATION_PROMPT_FREEZE",
        "status": "FROZEN_BEFORE_ANY_NEW_FORWARD_PASS",
        "authoritative_scaffold": {"source": "scripts/c9_base_vs_instruct_v1.py build_competence_prompts/extract (Unit 9 C9 8-shot competence prompt)", "script_sha256": C.sha256_file(c9_script), "properties_reused": ["raw text, no chat template, add_special_tokens=True (BOS)", "eight demonstrations, four per class", "demonstration rendered as the example prompt followed by a single space and the answer symbol", "demonstrations and the test example joined by a blank line (\\n\\n)", "candidate answers scored as the space-prefixed symbol immediately after 'Answer:'", "candidate comparison by log-probability of the candidate sequence (the frozen candidate-sequence scorer generalizes C9's single next-token comparison exactly when the candidate is one token)"], "minimal_adaptation": ["a fixed Task line precedes the demonstrations (the goal's required semantic skeleton)", "each example carries a Track A Context block and a declarative holder-operator-target Claim instead of the corpus Question", "F1 candidate semantics are ENTAILED / NOT_ENTAILED listed under the row's answer mapping in physical symbol order", "demonstration order is fixed (ascending selection hash) instead of C9's per-item shuffle", "C9's competence items were the development DIRECT questions; here every development and evaluation row of every family is scored"], "not_reused": ["C9 chose demonstrations by per-item RNG from fit rows; here the eight demonstrations are frozen by lowest stable hash under the goal's constraints and are identical for every prompt, actor and format"]},
        "formats": {"F1": {"id": "C9_MAPPED_ENTAILMENT_8SHOT", "task_line": C.F1_TASK_LINE, "example_template": "Context:\\n{context}\\n\\nClaim:\\n{claim}\\n\\n{sym_first}) {semantics_first}\\n{sym_second}) {semantics_second}\\nAnswer:", "mappings": {m: {"entailed_symbol": C.MAPPING_SYMBOLS[m][0], "not_entailed_symbol": C.MAPPING_SYMBOLS[m][1]} for m in C.MAPPINGS}, "candidates": "[' '+entailed_symbol, ' '+not_entailed_symbol]", "activations_retained": "row's own frozen mapping only"}, "F2": {"id": "DIRECT_SEMANTIC_LABEL_8SHOT", "task_line": C.F2_TASK_LINE, "example_template": "Context:\\n{context}\\n\\nClaim:\\n{claim}\\n\\nAnswer:", "candidates": [" ENTAILED", " NOT_ENTAILED"], "generation": {"decoding": "greedy, temperature 0", "max_new_tokens": C.F2_MAX_NEW_TOKENS, "parse": "^\\s*(NOT_ENTAILED|ENTAILED)\\b on the decoded continuation"}, "activations_retained": True}},
        "shot_separator": C.SHOT_SEPARATOR,
        "claim_reconstruction": "Question 'Is it established that {holder} {supports|opposes} the proposition that {target}?' -> '{holder} {supports|opposes} the proposition that {target}.'; verified against holder_name, target_text and queried_operator for every row",
        "demonstrations": [{k: d[k] for k in ("slot", "example_id", "world_id", "variant_family", "entailment_label", "relation_label", "presentation_position", "selection_hash")} for d in demos],
        "demonstration_file_sha256": C.sha256_file(root / "TRACK_A_DEMONSTRATION_FREEZE.jsonl"),
        "prompt_corpus_file_sha256": C.sha256_file(root / "TRACK_A_REELICITATION_PROMPTS.jsonl"),
        "prompt_counts": {f"{f}/{s}": n for (f, s), n in sorted(counts.items())},
        "rows_per_split": {"development": sum(1 for r in rows if r["world_split"] == "development"), "heldout_evaluation": len(eval_rows)},
        "activation_retention": {"layers": {a: C.ACTORS[a]["band"] for a in C.ACTORS}, "primary": {a: C.ACTORS[a]["primary_layer"] for a in C.ACTORS}, "site": C.ACTIVATION_POSITION, "dtype": "bfloat16 model, float32 stored"},
        "baseline": "the frozen zero-shot native answers of Unit 6/7 (chat-template rendered corpus prompts) are the fixed baseline and are not rerun",
        "rule": "no adaptive prompt selection; no third format; no actor-specific repair after outcomes; development worlds may not alter either prompt",
    }
    C.write_create_only(root / "TRACK_A_REELICITATION_PROMPT_FREEZE.json", C.canonical_json(prompt_freeze))

    # ---- naturalistic cells and folds
    cells, cell_summary = build_cells()
    C.write_create_only(root / "NATURALISTIC_CELL_INPUTS.jsonl", C.jsonl_bytes(cells))
    C.write_create_only(root / "NATURALISTIC_FOLD_ASSIGNMENTS.json", C.canonical_json(assign_folds(cells)))

    # ---- input manifest
    d6, d7 = HANDOFF / "unit6", HANDOFF / "unit7"
    v001 = GEOM / "outputs" / "existing-human-review-payoff" / "v001"
    inputs = [
        entry(C.TRACK_A_FREEZE / "MODEL_INPUTS_LLAMA.jsonl", "track_a_model_inputs_llama"),
        entry(C.TRACK_A_FREEZE / "MODEL_INPUTS_GEMMA.jsonl", "track_a_model_inputs_gemma"),
        entry(C.TRACK_A_FREEZE / "RELATION_BINDING_FACTORIAL_V2_WORLDS.jsonl", "track_a_worlds"),
        entry(C.TRACK_A_FREEZE / "RELATION_BINDING_FACTORIAL_V2_SPLITS.json", "track_a_splits"),
        entry(C.TRACK_A_FREEZE / "TRACK_A_ANALYSIS_FREEZE.json", "track_a_analysis_freeze"),
        entry(C.TRACK_A_FREEZE / "TRACK_A_ANALYSIS_PLAN.json", "track_a_analysis_plan"),
        entry(C.TRACK_A_FREEZE / "TRACK_A_COMPETITOR_DIRECTIONS.json", "competitor_direction_manifest"),
        entry(C.TRACK_A_FREEZE / "runtime" / "llama" / "competitor_directions_v2.npz", "competitor_directions_llama"),
        entry(C.TRACK_A_FREEZE / "runtime" / "gemma" / "competitor_directions_v2.npz", "competitor_directions_gemma"),
        entry(C.ROOT / "scripts" / "relation_binding_analysis_v2.py", "frozen_track_a_analysis_code"),
        entry(C.ROOT / "scripts" / "run_relation_binding_factorial_v2_model.py", "frozen_track_a_runner"),
        entry(c9_script, "c9_scaffold_code"),
        entry(C.ROOT / "reports" / "relation_operator_transverse_fragility_v1" / "c9_base_vs_instruct_v1" / "freeze" / "C9_FREEZE.json", "c9_freeze"),
        entry(C.ROOT / "reports" / "relation_operator_transverse_fragility_v1" / "c9_base_vs_instruct_v1" / "freeze" / "C9_COMPETENCE_PROMPTS.jsonl", "c9_competence_prompts"),
        entry(C.ROOT / "scripts" / "rrrd_v1_common.py", "study_common_code"),
        entry(Path(__file__).resolve(), "study_freeze_builder"),
        entry(d6 / "llama" / "track_a_completed_jobs.jsonl", "unit6_llama_completed_jobs", nfs_path=f"{NFS}/outputs/unit6/llama/track_a_completed_jobs.jsonl"),
        entry(d6 / "llama" / "technical_receipt.json", "unit6_llama_technical_receipt", nfs_path=f"{NFS}/outputs/unit6/llama/technical_receipt.json"),
        entry(d6 / "llama" / "artifact_manifest.json", "unit6_llama_artifact_manifest", nfs_path=f"{NFS}/outputs/unit6/llama/artifact_manifest.json"),
        entry(d6 / "llama_analysis" / "RELATION_BINDING_V2_RESULTS_LLAMA.json", "unit6_llama_results"),
        entry(d7 / "gemma" / "track_a_completed_jobs.jsonl", "unit7_gemma_completed_jobs", nfs_path=f"{NFS}/outputs/unit7/gemma/track_a_completed_jobs.jsonl"),
        entry(d7 / "gemma" / "technical_receipt.json", "unit7_gemma_technical_receipt", nfs_path=f"{NFS}/outputs/unit7/gemma/technical_receipt.json"),
        entry(d7 / "gemma" / "artifact_manifest.json", "unit7_gemma_artifact_manifest", nfs_path=f"{NFS}/outputs/unit7/gemma/artifact_manifest.json"),
        entry(d7 / "gemma_analysis" / "RELATION_BINDING_V2_RESULTS_GEMMA.json", "unit7_gemma_results"),
        entry(v001 / "claim1_human_strata_cells.csv", "unit2_strata_llama"),
        entry(v001 / "claim1_human_strata_gemma_cells.csv", "unit2_strata_gemma"),
        entry(v001 / "frozen_valueprism_type_map.csv", "valueprism_cell_text_type_map"),
        entry(v001 / "checkpoints" / "causal_claim1_llama.parquet", "frozen_bs1_baseline_llama"),
        entry(v001 / "checkpoints" / "causal_claim1_gemma.parquet", "frozen_bs1_baseline_gemma"),
        entry(v001 / "checkpoints" / "gemma_claim1_activations.npz", "gemma_claim1_activation_cache_fp16"),
        entry(GEOM / "outputs/claim1-first-pass-semantic-stratification/source/llama-compact/claim2_m1_compact.npz", "llama_compact_cache_fp16"),
        entry(GEOM / "outputs/claim1-first-pass-semantic-stratification/source/llama-compact/claim2_m1_compact.metadata.json", "llama_compact_metadata"),
        entry(GEOM / "configs" / "m1_vertical_slice.yaml", "m1_prompt_config"),
        entry(GEOM / "outputs" / "human-review-derived-sets-v3" / "claim1_consensus_cells.csv", "claim1_consensus_cells"),
        entry(HANDOFF / "unit2" / "result_v1" / "RQ1B_RESULTS.json", "unit2_results"),
        entry(PROJECTS / "geometry-relation-representation-v2-clarity" / "reports" / "clarity_text_only_comparator_v1" / "freeze" / "CLARITY_COMPARATOR_ANALYSIS_FREEZE.json", "unit3_g4_comparator_freeze"),
        entry(HANDOFF / "unit3" / "result_v1" / "CLARITY_COMPARATOR_RESULTS.json", "unit3_results"),
        entry(HANDOFF / "unit9" / "outputs" / "unit9" / "analysis" / "C9_RESULTS.json", "unit9_c9_results"),
        entry(HANDOFF / "unit9" / "outputs" / "unit9" / "extract" / "llama_instruct" / "EXTRACTION_RECEIPT.json", "unit9_llama_instruct_competence_receipt"),
        entry(HANDOFF / "unit9" / "outputs" / "unit9" / "extract" / "gemma_instruct" / "EXTRACTION_RECEIPT.json", "unit9_gemma_instruct_competence_receipt"),
        entry(PROJECTS / "geometry-relation-representation-v2-family" / "reports" / "direction_family_decomposition_v1" / "reporting_repair_v2" / "external_review_packet_v2.json", "unit4_g3_packet_v2"),
        entry(LOOP158 / "program_summary" / "PROGRAM_SUMMARY_G1_G5.json", "program_summary"),
        entry(LOOP158 / "REVIEW_RESPONSE_2026-09-02.md", "review_response"),
        entry(LOOP158 / "SPEC.md", "relation_program_v2_spec"),
        entry(Path("external-artifacts"), "study_goal_text"),
    ]
    probes = {}
    for actor, (zp, expected) in PROBE_BUNDLES.items():
        e = entry(zp, f"m1_probe_bundle_{actor}")
        if e["sha256"] != expected:
            raise RuntimeError(f"probe bundle hash mismatch for {actor}")
        with zipfile.ZipFile(zp) as z:
            payload = z.read("m1_probe_parameters.npz")
        arr = np.load(io.BytesIO(payload), allow_pickle=False)
        direction = np.asarray(arr["difference_in_means_direction"], dtype=np.float32)
        dsha = C.sha256_bytes(np.ascontiguousarray(direction).tobytes())
        if dsha != C.ACTORS[actor]["raw_relation_dim_sha256"]:
            raise RuntimeError(f"probe direction is not the frozen raw relation DIM for {actor}")
        if int(arr["selected_layer"]) != C.ACTORS[actor]["primary_layer"] or str(arr["model_revision"]) != C.ACTORS[actor]["model_revision"]:
            raise RuntimeError(f"probe layer/revision mismatch for {actor}")
        probes[actor] = {"bundle_sha256": e["sha256"], "member": "m1_probe_parameters.npz", "member_sha256": C.sha256_bytes(payload), "direction_sha256": dsha, "difference_in_means_midpoint": float(arr["difference_in_means_midpoint"]), "orientation": "score = activation . direction - midpoint; positive = Supports (aligned_difference_in_means = relation_sign * score)", "selected_layer": int(arr["selected_layer"]), "model_revision": str(arr["model_revision"]), "logistic_intercept": float(np.asarray(arr["logistic_intercept"]).reshape(-1)[0]), "logistic_coef_sha256": C.sha256_bytes(np.ascontiguousarray(np.asarray(arr["logistic_coef"], dtype=np.float32)).tobytes())}
        inputs.append(e)
    manifest = {
        "schema_version": f"{C.SCHEMA_PREFIX}_INPUT_MANIFEST", "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(), "study_root_commit": git_head(C.ROOT), "base_commit": "81b37e2", "inputs": inputs,
        "source_calibration": {"raw_relation_dim": probes, "jointly_residualized_relation": "SOURCE_CALIBRATION_MISSING (relation-direction-identity postmortem stores no midpoint/threshold for residual_dim; level A is not computed for it)"},
        "retained_activations": {"unit6_llama": {"nfs_root": f"{NFS}/outputs/unit6/llama", "archive_sha256": C.read_json(d6 / "llama" / "technical_receipt.json")["archive"]["sha256"], "rows": 3744, "per_row_sha256": "full_activation_artifact_sha256 in the completed-jobs ledger (verified at load)"}, "unit7_gemma": {"nfs_root": f"{NFS}/outputs/unit7/gemma", "archive_sha256": C.read_json(d7 / "gemma" / "technical_receipt.json")["archive"]["sha256"], "rows": 3744, "per_row_sha256": "full_activation_artifact_sha256 in the completed-jobs ledger (verified at load)"}},
        "text_encoder": MINILM,
        "naturalistic_population": cell_summary,
        "forbidden_inputs": ["V3 paths or outcomes", "the sealed 60-board human-confirmation set", "confirmatory splits", "sealed Llama V3 selection"],
    }
    C.write_create_only(root / "INPUT_MANIFEST.json", C.canonical_json(manifest))

    plan = analysis_plan()
    C.write_create_only(root / "ANALYSIS_PLAN.json", C.canonical_json(plan))
    C.write_create_only(root / "TERMINAL_CLASSIFICATIONS.json", C.canonical_json(terminal_classifications(plan)))
    C.write_create_only(root / "CLAIM_BOUNDARY.md", CLAIM_BOUNDARY_MD.encode("utf-8"))

    contract = {
        "schema_version": f"{C.SCHEMA_PREFIX}_STUDY_CONTRACT", "study_id": C.STUDY_ID, "status": "FROZEN_BEFORE_ANY_NEW_OUTCOME_COMPUTATION", "recorded_at": manifest["recorded_at"],
        "question": "Is the Track A hidden-readout / native-answer gap caused by (A) a broken zero-shot answer format, (B) a valid internal-readout / behavioral-report dissociation on multi-entity relations, (C) prompt sensitivity of the internal representation, or (D) an actor-specific mixture?",
        "fixed_facts": ["G1 terminal BOUND_RELATIONAL_REPRESENTATION (decoding sense) both actors", "G2 READOUT_IDENTICAL_TO_REPORT (frozen DIM strongly report-aligned; logistic less so)", "G3 exploratory two-cell association with adverse Llama diagnostic", "G4 CLARITY_NARROW_CLAIM_STANDS", "G5 ENDPOINT_AT_FLOOR", "C9 FEATURE_PRESENT_IN_BASE", "V3 instrument terminal invalid (untouched)"],
        "actors": C.ACTORS, "activation_site": C.ACTIVATION_POSITION, "execution": {"dtype": "exact bfloat16, no quantization or offload", "batch_size": 1, "scorer": "frozen candidate-sequence scorer (teacher-forced log-probability sum, float32 log-softmax)", "one_model_resident": True},
        "splits": {"track_a": {"fit": "residual_probe_train (72 worlds)", "development": "development (24 worlds)", "evaluation": "heldout_evaluation (48 worlds)", "source": "RELATION_BINDING_FACTORIAL_V2_SPLITS.json"}, "naturalistic": {"population": "open human-reviewed 125 boards / 500 cells (Unit 2)", "folds": "NATURALISTIC_FOLD_ASSIGNMENTS.json"}},
        "arms": {"A": "no-new-inference characterization (4.1-4.6), label PROSPECTIVELY_FROZEN_POST_HOC_DIAGNOSTICS", "B": "prospective Track A re-elicitation (F1, F2)", "C": "naturalistic Q-REPORT V2 extraction and frozen probe analysis"},
        "thresholds": {"practical_gate": C.PRACTICAL_GATE, "direct_competence": C.DIRECT_COMPETENCE_GATE, "direct_competence_ci_low": C.DIRECT_COMPETENCE_CI_LOW, "agreement": C.AGREEMENT_GATE, "role_order_transfer": 0.60, "primary_margin_joint": 0.05, "four_class_chance": 0.25, "margin_reproduction_tolerance": 0.002},
        "no_retry_rules": ["no adaptive prompt tuning or selection after outcomes", "no third prompt format", "no threshold relaxation", "no model/layer/dtype/direction substitution", "zero scientific retries after any valid scientific outcome is written", "development worlds may not alter prompts or select the reported format"],
        "source_of_truth_order": ["frozen raw result artifacts, exact analysis code, hashes", "this freeze"],
        "boundaries": ["sealed confirmation set untouched", "no quantization or substitute checkpoint"],
        "files": ["STUDY_CONTRACT.json", "STUDY_CONTRACT.md", "INPUT_MANIFEST.json", "ANALYSIS_PLAN.json", "TRACK_A_REELICITATION_PROMPT_FREEZE.json", "TRACK_A_DEMONSTRATION_FREEZE.jsonl", "TRACK_A_REELICITATION_PROMPTS.jsonl", "NATURALISTIC_CELL_INPUTS.jsonl", "NATURALISTIC_FOLD_ASSIGNMENTS.json", "TERMINAL_CLASSIFICATIONS.json", "CLAIM_BOUNDARY.md", "artifact_manifest.json"],
    }
    C.write_create_only(root / "STUDY_CONTRACT.json", C.canonical_json(contract))
    md = ["# Study contract — " + C.STUDY_ID, "", f"Status: **{contract['status']}** ({contract['recorded_at']})", "", "## Question", "", contract["question"], "", "## Fixed facts (not rerun)", ""] + [f"- {x}" for x in contract["fixed_facts"]] + ["", "## Actors", ""] + [f"- {a}: `{c['model_id']}` @ `{c['model_revision']}`, primary layer {c['primary_layer']}, band {c['band'][0]}–{c['band'][-1]}, site: {C.ACTIVATION_POSITION}" for a, c in C.ACTORS.items()] + ["", "## Execution", ""] + [f"- {k}: {v}" for k, v in contract["execution"].items()] + ["", "## Arms", ""] + [f"- Arm {k}: {v}" for k, v in contract["arms"].items()] + ["", "## Thresholds", ""] + [f"- {k}: {v}" for k, v in contract["thresholds"].items()] + ["", "## No-retry rules", ""] + [f"- {x}" for x in contract["no_retry_rules"]] + ["", "## Boundaries", ""] + [f"- {x}" for x in contract["boundaries"]] + ["", "## Demonstrations (frozen, identical for every prompt/actor/format)", "", "| position | slot | example | family | entailment |", "|---|---|---|---|---|"] + [f"| {d['presentation_position']} | {d['slot']} | {d['example_id']} | {d['variant_family']} | {d['entailment_label']} |" for d in demos] + ["", "## Prompt counts", ""] + [f"- {k}: {v}" for k, v in prompt_freeze["prompt_counts"].items()] + ["", "## Naturalistic population", "", f"- {cell_summary}", "", "See ANALYSIS_PLAN.json for every metric, threshold, uncertainty procedure and the result-to-interpretation map; TERMINAL_CLASSIFICATIONS.json for the decision rules; CLAIM_BOUNDARY.md for licensed and prohibited wording.", ""]
    C.write_create_only(root / "STUDY_CONTRACT.md", "\n".join(md).encode("utf-8"))
    C.write_create_only(root / "artifact_manifest.json", C.canonical_json({"schema_version": f"{C.SCHEMA_PREFIX}_FREEZE_MANIFEST", "study_id": C.STUDY_ID, "outcome_values_parsed_or_computed": False, "files": C.manifest_for(root)}))
    print(f"FREEZE_WRITTEN {root} demos={[d['example_id'] for d in demos]} prompts={len(corpus)} cells={len(cells)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
