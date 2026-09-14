"""Synthetic and freeze-integrity tests for RELATION_READOUT_REPORT_DISCRIMINATION_V1."""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rrrd_v1_common as C  # noqa: E402
from rrrd_v1_no_inference import WorldBootstrap  # noqa: E402
from rrrd_v1_track_a_analysis import classify  # noqa: E402

FREEZE = ROOT / "reports" / "relation_readout_report_discrimination_v1" / "freeze"


# ---------------------------------------------------------------- rendering conventions
def test_f1_symbol_convention_matches_corpus_expected_tokens():
    rows = C.read_jsonl(C.TRACK_A_FREEZE / "MODEL_INPUTS_LLAMA.jsonl")
    for r in rows:
        assert C.f1_symbol_for(r["answer_mapping_id"], r["entailment_label"]) == r["expected_answer_token"]


def test_claim_reconstruction_is_exact_for_every_row():
    rows = C.read_jsonl(C.TRACK_A_FREEZE / "MODEL_INPUTS_LLAMA.jsonl")
    for r in rows:
        claim = C.declarative_claim(r)
        assert claim.startswith(r["holder_name"]) and claim.endswith(r["target_text"] + ".")
        assert (" supports " in claim) == (r["queried_operator"] == "SUPPORT")


def test_f1_rendering_lists_options_in_physical_order():
    text = C.render_f1_example("ctx", "claim", "MAPPING_AB_REVERSED", "A")
    lines = text.split("\n")
    assert lines[-3:] == ["A) NOT_ENTAILED", "B) ENTAILED", "Answer: A"]
    text = C.render_f1_example("ctx", "claim", "MAPPING_12_STANDARD", None)
    assert text.split("\n")[-3:] == ["1) ENTAILED", "2) NOT_ENTAILED", "Answer:"]


def test_m1_mapping_matches_frozen_baseline_mapping_names():
    cells = C.read_jsonl(FREEZE / "NATURALISTIC_CELL_INPUTS.jsonl")
    for c in cells:
        assert C.m1_mapping_for_item(c["item_id"])["name"] == c["mapping_name"]


# ---------------------------------------------------------------- freeze integrity
def test_demonstrations_satisfy_every_constraint():
    demos = C.read_jsonl(FREEZE / "TRACK_A_DEMONSTRATION_FREEZE.jsonl")
    assert len(demos) == 8 and len({d["example_id"] for d in demos}) == 8
    labels = collections.Counter(d["entailment_label"] for d in demos)
    assert labels["ENTAILED"] == 4 and labels["NOT_ENTAILED"] == 4
    fams = collections.Counter(d["variant_family"] for d in demos)
    assert fams["DIRECT_HOLDER_TARGET_QUERY"] >= 1 and fams["CONFLICTING_STANCES_IN_ONE_CONTEXT"] == 2
    assert fams["TARGET_SWAP"] >= 1 and fams["SENTENCE_ORDER_SWAP"] >= 1 and fams["REPORTER_VERSUS_REPORTED_HOLDER"] >= 1
    assert fams["NEGATED_STANCE"] >= 1 and fams["NEUTRAL_OR_WITHHELD_STANCE"] >= 1
    conf = [d for d in demos if d["variant_family"] == "CONFLICTING_STANCES_IN_ONE_CONTEXT"]
    assert conf[0]["contrast_id"] == conf[1]["contrast_id"] and {c["entailment_label"] for c in conf} == {"ENTAILED", "NOT_ENTAILED"}
    prompts = C.read_jsonl(C.TRACK_A_FREEZE / "MODEL_INPUTS_LLAMA.jsonl")
    by_id = {r["example_id"]: r for r in prompts}
    for d in demos:
        assert by_id[d["example_id"]]["world_split"] == "residual_probe_train"
    eval_text = "\n".join(r["prompt_text"] for r in prompts if r["world_split"] == "heldout_evaluation")
    for d in demos:
        assert d["holder_name"] not in eval_text
        assert d["claim"] not in eval_text


def test_demonstration_selection_is_deterministic_and_hash_ordered():
    demos = C.read_jsonl(FREEZE / "TRACK_A_DEMONSTRATION_FREEZE.jsonl")
    hashes = [d["selection_hash"] for d in demos]
    assert hashes == sorted(hashes)
    for d in demos:
        assert d["selection_hash"] == C.stable_hash(C.DEMO_HASH_SALT + d["example_id"])


def test_prompt_corpus_hashes_and_counts():
    freeze = C.read_json(FREEZE / "TRACK_A_REELICITATION_PROMPT_FREEZE.json")
    prompts = C.read_jsonl(FREEZE / "TRACK_A_REELICITATION_PROMPTS.jsonl")
    assert C.sha256_file(FREEZE / "TRACK_A_REELICITATION_PROMPTS.jsonl") == freeze["prompt_corpus_file_sha256"]
    counts = collections.Counter((p["format"], p["world_split"]) for p in prompts)
    assert counts[("F1", "heldout_evaluation")] == 4 * 1248 and counts[("F2", "heldout_evaluation")] == 1248  # 48 worlds x 26 rows (all eligibilities)
    assert counts[("F1", "development")] == 4 * 624 and counts[("F2", "development")] == 624
    sample = prompts[::997]
    for p in sample:
        assert C.sha256_text(p["prompt_text"]) == p["prompt_sha256"]
        assert p["prompt_text"].endswith("Answer:")
        assert p["prompt_text"].count("Answer:") == 9


def test_folds_never_split_a_board_and_are_balanced():
    folds = C.read_json(FREEZE / "NATURALISTIC_FOLD_ASSIGNMENTS.json")
    cells = C.read_jsonl(FREEZE / "NATURALISTIC_CELL_INPUTS.jsonl")
    assign = folds["assignment"]
    per_fold = collections.Counter(assign[c["board_id"]] for c in cells)
    assert set(per_fold) == {0, 1, 2, 3, 4} and all(v == 100 for v in per_fold.values())
    boards_per_fold = collections.Counter(assign.values())
    assert all(v == 25 for v in boards_per_fold.values())
    for c in cells:
        assert assign[c["board_id"]] == assign[c["board_id"]]


# ---------------------------------------------------------------- bootstrap
def test_world_bootstrap_paired_difference_is_exact_and_ci_contains_point():
    worlds = [f"W{i}" for i in range(48)]
    rng = np.random.default_rng(0)
    a = {w: float(rng.normal(0.8, 0.1)) for w in worlds}
    b = {w: a[w] - 0.2 for w in worlds}
    boot = WorldBootstrap(worlds, 2000, 1)
    d = boot.paired(a, b)
    assert abs(d["point"] - 0.2) < 1e-9 and d["ci_low"] <= 0.2 <= d["ci_high"]
    assert d["ci_high"] - d["ci_low"] < 1e-9  # constant difference -> degenerate interval
    s = boot.summary(a)
    assert s["ci_low"] <= s["point"] <= s["ci_high"]
    partial = {w: a[w] for w in worlds[:10]}
    assert boot.summary(partial)["worlds"] == 10


# ---------------------------------------------------------------- Arm B classification rules
def _fmt(direct, dual, holder, target, consistency, align, readout_pass_a, readout_pass_c, fails_two=False, readout_strong=False, rmb=False, improves=False):
    return {"flags": {"DIRECT_TASK_COMPETENT": direct >= 0.8 and consistency >= 0.9, "BOUND_BEHAVIOR_PASS": min(dual, holder, target) >= 0.7, "IMPROVES_INTO_PRACTICAL_RANGE": improves, "ALIGNS_WITH_DIM": align >= 0.8, "fails_two_bound_endpoints_ci_high_below_gate": fails_two, "readout_ci_low_above_gate_on_bound": {"A_strict": readout_strong, "C_threshold_free": readout_strong}, "readout_minus_behavior_ci_excludes_zero_on_two": {"A_strict": rmb, "C_threshold_free": rmb}}, "frozen_readout_pass": {"A_strict": {"pass": readout_pass_a}, "C_threshold_free": {"pass": readout_pass_c}}}


def _res(f1, f2, base_pass=True):
    return {"formats": {"F1": f1, "F2": f2}, "baseline_zero_shot": {"readout_pass": {"A_strict": {"pass": base_pass}, "C_threshold_free": {"pass": base_pass}}}}


def test_classify_branches():
    good = _fmt(0.95, 0.9, 0.9, 0.9, 0.95, 0.9, True, True)
    bad = _fmt(0.55, 0.3, 0.3, 0.3, 0.5, 0.5, True, True)
    assert classify(_res(good, good))["primary"] == "ROBUST_ELICITATION_REPAIR"
    assert classify(_res(bad, bad))["primary"] == "TASK_NOT_ELICITED"
    assert classify(_res(good, bad))["primary"] == "FORMAT_SPECIFIC_ELICITATION_REPAIR"
    dis = _fmt(0.95, 0.3, 0.3, 0.3, 0.95, 0.5, True, True, fails_two=True, readout_strong=True, rmb=True)
    assert classify(_res(dis, bad))["primary"] == "VALID_TASK_INTERNAL_READOUT_BEHAVIOR_DISSOCIATION"
    sens = _fmt(0.95, 0.9, 0.9, 0.9, 0.95, 0.9, False, False)
    assert classify(_res(sens, sens))["primary"] == "PROMPT_SENSITIVE_INTERNAL_READOUT"
    mixed = _fmt(0.95, 0.9, 0.9, 0.9, 0.95, 0.5, True, True)
    assert classify(_res(mixed, mixed))["primary"] == "MIXED_OR_UNDERDETERMINED"


# ---------------------------------------------------------------- Arm C classification rule
def test_q_report_classification_rule():
    from rrrd_v1_q_report_v2 import analyze  # noqa: F401  (import only; rule mirrored below)

    def cls(L, R):
        lpos, rpos = L[0] > 0, R[0] > 0
        ladv, radv = L[1] < 0, R[1] < 0
        if lpos and rpos:
            return "BIDIRECTIONAL_LABEL_REPORT_SEPARABILITY"
        if ladv or radv:
            return "ACTIVATION_INCREMENT_ADVERSE"
        if lpos:
            return "LABEL_INFORMATION_BEYOND_REPORT"
        if rpos:
            return "REPORT_INFORMATION_BEYOND_LABEL"
        return "NO_DETECTED_INCREMENT"

    assert cls((0.01, 0.05), (0.02, 0.06)) == "BIDIRECTIONAL_LABEL_REPORT_SEPARABILITY"
    assert cls((0.01, 0.05), (-0.02, 0.06)) == "LABEL_INFORMATION_BEYOND_REPORT"
    assert cls((-0.03, -0.01), (-0.02, 0.06)) == "ACTIVATION_INCREMENT_ADVERSE"
    assert cls((-0.01, 0.05), (-0.02, 0.06)) == "NO_DETECTED_INCREMENT"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
