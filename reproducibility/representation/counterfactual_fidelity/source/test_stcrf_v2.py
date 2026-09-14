"""Unit tests for STRONG_TARGET_COUNTERFACTUAL_RELATION_FIDELITY_V2 on synthetic states and signatures.

Required by the study contract before any model inference:
  - the frozen root rule finds the minimum-norm signed multiple reaching a target and reports
    unreachable / saturated / numerically invalid cases;
  - the target definition is exactly margin_base + tau * (margin_cf - margin_base);
  - Q1 never enters any fidelity score and the CSR mathematics is CRIF v1's, unchanged;
  - dose validity requires reach, tolerance, hard answer, margin, no saturation and the norm budget;
  - the corpus splits are disjoint, balanced and free of prior names, targets and templates;
  - direction qualification is world-grouped and refuses a fit that only tracks the answer label;
  - the classification precedence and the strong-control / fidelity criteria behave as frozen.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import crif_v1_common as V1  # noqa: E402
import stcrf_v2_analysis as A  # noqa: E402
import stcrf_v2_build_corpus as B  # noqa: E402
import stcrf_v2_common as C  # noqa: E402
import stcrf_v2_fit as F  # noqa: E402

RNG = np.random.default_rng(11)


def synthetic_natural(n: int = 40) -> np.ndarray:
    N = np.zeros((n, 5))
    N[:, 0] = -10 + RNG.normal(0, 1.5, n)
    N[:, 1] = -9 + RNG.normal(0, 1.5, n)
    N[:, 2] = 11 + RNG.normal(0, 1.5, n)
    N[:, 3] = RNG.normal(0, 0.3, n)
    N[:, 4] = RNG.normal(0, 0.3, n)
    return N


# ---------------------------------------------------------------- endpoint reuse
def test_csr_is_v1_mathematics_unchanged():
    N = synthetic_natural()
    s = C.robust_scales(N)
    assert C.csr is V1.csr and C.robust_scales is V1.robust_scales and C.squared_distance is V1.squared_distance
    assert np.allclose(C.csr(N.copy(), N, s)["csr"], 1.0)
    assert np.allclose(C.csr(np.zeros_like(N), N, s)["csr"], 0.0)
    r = C.csr(N.copy(), N, s)
    assert np.allclose(r["csr_focal"] + r["csr_invariance"] - 1.0, r["csr"])
    I = N.copy()
    I[:, 3] += 3.0
    assert np.all(C.csr(I, N, s)["csr_invariance"] < 1.0)
    assert np.all(C.csr(-N, N, s)["csr"] < 0.0)


def test_bootstrap_is_world_grouped():
    idx = C.bootstrap_indices(12, replicates=50)
    assert idx.shape == (50, 12) and idx.min() >= 0 and idx.max() < 12


# ---------------------------------------------------------------- root rule and targets
def test_target_definition():
    base, cf = -8.0, 12.0
    targets = {t: base + t * (cf - base) for t in C.TAUS}
    assert targets[1.00] == pytest.approx(cf) and targets[0.50] == pytest.approx(2.0)
    assert targets[0.75] == pytest.approx(7.0)


def test_root_rule_interpolates_minimum_norm_crossing():
    grid = [0.5, 1.0, 1.5, 2.0]
    margins = [-3.0, -1.0, 1.0, 3.0]  # linear in the multiple
    r = C.select_multiple(grid, margins, base_margin=-5.0, target=0.0)
    assert r["status"] == "REACHED"
    assert r["multiple"] == pytest.approx(1.25)
    assert r["bracket"] == [1.0, 1.5]
    # the first crossing is taken, never a later one
    r2 = C.select_multiple(grid, margins, base_margin=-5.0, target=-1.0)
    assert r2["multiple"] == pytest.approx(1.0)


def test_root_rule_reports_unreachable_and_saturation():
    grid = [0.5, 1.0]
    r = C.select_multiple(grid, [-3.0, -2.0], base_margin=-5.0, target=4.0)
    assert r["status"] == "UNREACHABLE_IN_GRID" and r["multiple"] is None
    r2 = C.select_multiple([0.5, 1.0], [-4.0, 9.0], base_margin=-5.0, target=0.0)
    assert r2["status"] == "REACHED" and r2["saturated"] is True  # jump > 5 nats
    r3 = C.select_multiple([0.5, 1.0], [float("nan"), 1.0], base_margin=-5.0, target=0.0)
    assert r3["status"] == "NUMERICALLY_INVALID"


def test_signed_root_rule_scans_both_rays_and_takes_the_smaller_multiple():
    grid = [0.5, 1.0, 1.5, 2.0]
    plus = [-6.0, -7.0, -8.0, -9.0]  # runs away from the target along +u
    minus = [-3.0, -1.0, 1.0, 3.0]
    r = C.select_signed(grid, plus, minus, base_margin=-5.0, target=0.0, natural_sign=1.0)
    assert r["status"] == "REACHED" and r["sign"] == -1.0 and r["multiple"] == pytest.approx(1.25)
    assert r["sign_agrees_with_natural"] is False
    # both rays reach: the smaller multiple wins; an exact tie goes to the natural-projection ray
    r2 = C.select_signed(grid, [-3.0, 1.0, 3.0, 5.0], minus, base_margin=-5.0, target=0.0, natural_sign=-1.0)
    assert r2["sign"] == 1.0 and r2["multiple"] == pytest.approx(0.875)
    r3 = C.select_signed(grid, minus, minus, base_margin=-5.0, target=0.0, natural_sign=-1.0)
    assert r3["sign"] == -1.0
    r4 = C.select_signed(grid, plus, [-6.0, -7.0, -8.0, -9.0], base_margin=-5.0, target=0.0, natural_sign=1.0)
    assert r4["status"] == "UNREACHABLE_IN_GRID" and r4["multiple"] is None


def test_achieved_tolerance_rule():
    assert C.achieved_ok(7.3, 7.0, natural_change=20.0)  # within 0.5 nat
    assert C.achieved_ok(8.5, 7.0, natural_change=20.0)  # within 10% of the natural change
    assert not C.achieved_ok(10.0, 7.0, natural_change=20.0)


# ---------------------------------------------------------------- signatures and dose
def make_summary(wid: str, base: dict, cf: dict, arms: dict, r_l2: float = 10.0, direction: str = "SUPPORT_TO_OPPOSE", multiples: dict | None = None) -> dict:
    rows = []
    for fmt in ("F2", "F1"):
        for real in (("A", "B") if fmt == "F2" else ("A",)):
            for q in C.QUERIES:
                for state, m in (("base", base), ("counterfactual", cf)):
                    rows.append({"format": fmt, "realization": real, "state": state, "query": q, "intervention": "NO_INTERVENTION", "tau": None, "semantic_margin": m[q], "label": "ENTAILED" if m[q] > 0 else "NOT_ENTAILED"})
                for name, per in arms.items():
                    if fmt != "F2":
                        continue
                    rows.append({"format": fmt, "realization": real, "state": "base", "query": q, "intervention": name, "tau": C.PRIMARY_TAU, "semantic_margin": per[q], "delta_margin": per[q] - base[q]})
    nat = {}
    for fmt in ("F2", "F1"):
        nat[fmt] = {"base_q1_A": base["Q1"], "cf_q1_A": cf["Q1"], "natural_q1_change_A": cf["Q1"] - base["Q1"], "targets": {str(t): base["Q1"] + t * (cf["Q1"] - base["Q1"]) for t in C.TAUS}, "natural_full_state_l2_q1_A": r_l2}
    mults = multiples or {name: 1.0 for name in arms}
    applied = {}
    for name, per in arms.items():
        ach = per["Q1"]
        target = nat["F2"]["targets"][str(C.PRIMARY_TAU)]
        applied[name] = {"arm": name, "format": "F2", "sign": 1.0, "r_l2": r_l2, "roots": {str(C.PRIMARY_TAU): {"status": "REACHED", "multiple": mults[name], "saturated": False}}, "applied": {str(C.PRIMARY_TAU): {"status": "APPLIED", "multiple": mults[name], "norm_ratio": mults[name], "achieved_q1_A": ach, "target_q1_A": target, "within_tolerance": C.achieved_ok(ach, target, nat["F2"]["natural_q1_change_A"]), "hard_answer_counterfactual": bool(ach > 0), "signed_margin_ge_1": bool(ach >= 1.0), "saturated": False, "finite": True}}}
    return {"world_id": wid, "protocol": "P1", "split": "final", "direction": direction, "natural": nat, "arms": {"F2": applied, "F1": {}}, "rows": rows}


def test_q1_never_enters_the_signature():
    base = {q: -5.0 for q in C.QUERIES}
    cf = {"Q1": 12.0, "Q2": -6.0, "Q3": 6.0, "Q4": -6.0, "Q5": -5.0, "Q6": -5.0}
    arm = {q: cf[q] for q in C.QUERIES}
    s1 = A.world_signatures(make_summary("W1", base, cf, {"REGIME_MATCHED_DIM": arm}), "F2", "A")
    arm2 = dict(arm)
    arm2["Q1"] = 400.0
    s2 = A.world_signatures(make_summary("W1", base, cf, {"REGIME_MATCHED_DIM": arm2}), "F2", "A")
    k = ("REGIME_MATCHED_DIM", str(C.PRIMARY_TAU))
    assert np.array_equal(s1["I"][k], s2["I"][k])
    assert s1["N"].shape == (5,)
    assert s2["dQ1"][k] != s1["dQ1"][k]
    assert C.csr(s2["I"][k][None, :], s2["N"][None, :], np.ones(5))["csr"][0] == pytest.approx(1.0)


def test_logit_control_signature_is_the_constant_target_bias():
    base = {q: -4.0 for q in C.QUERIES}
    cf = {q: 8.0 for q in C.QUERIES}
    s = A.world_signatures(make_summary("W", base, cf, {}), "F2", "A")
    k = (C.LOGIT_CONTROL, str(C.PRIMARY_TAU))
    assert np.allclose(s["I"][k], 0.75 * 12.0)
    assert s["dQ1"][k] == pytest.approx(9.0)


def test_dose_validity_requires_every_frozen_condition_and_the_norm_budget():
    base = {q: -4.0 for q in C.QUERIES}
    cf = {"Q1": 12.0, "Q2": -6.0, "Q3": 6.0, "Q4": -6.0, "Q5": -4.0, "Q6": -4.0}
    good = {q: cf[q] for q in C.QUERIES}
    good["Q1"] = 8.0  # target = -4 + 0.75*16 = 8
    ss = [make_summary(f"W{i}", base, cf, {"REGIME_MATCHED_DIM": good}, multiples={"REGIME_MATCHED_DIM": 1.5}) for i in range(20)]
    d = A.dose_report(ss, "F2", "REGIME_MATCHED_DIM", str(C.PRIMARY_TAU))
    assert d["fraction_dose_valid"] == 1.0 and d["dose_valid"] and d["norm_ratio"]["median"] == pytest.approx(1.5)
    # a wrong hard answer invalidates the dose even when the multiple is small
    bad = dict(good)
    bad["Q1"] = -0.4
    ss_bad = [make_summary(f"W{i}", base, cf, {"REGIME_MATCHED_DIM": bad}) for i in range(20)]
    d2 = A.dose_report(ss_bad, "F2", "REGIME_MATCHED_DIM", str(C.PRIMARY_TAU))
    assert d2["fraction_dose_valid"] == 0.0 and not d2["dose_valid"] and d2["criteria"]["hard_answer_counterfactual"] == 0.0
    # reaching the target only with an extreme displacement fails the norm budget, not the reach test
    ss_big = [make_summary(f"W{i}", base, cf, {"REGIME_MATCHED_DIM": good}, multiples={"REGIME_MATCHED_DIM": 9.0}) for i in range(20)]
    d3 = A.dose_report(ss_big, "F2", "REGIME_MATCHED_DIM", str(C.PRIMARY_TAU))
    assert d3["reach_ok"] and not d3["norm_budget_ok"] and not d3["dose_valid"]


def test_unreachable_worlds_are_excluded_from_csr_but_counted():
    base = {q: -4.0 for q in C.QUERIES}
    cf = {"Q1": 12.0, "Q2": -6.0, "Q3": 6.0, "Q4": -6.0, "Q5": -4.0, "Q6": -4.0}
    good = {q: cf[q] for q in C.QUERIES}
    good["Q1"] = 8.0
    ss = [make_summary(f"W{i}", base, cf, {"REGIME_MATCHED_DIM": good}) for i in range(10)]
    for s in ss[:4]:
        s["arms"]["F2"]["REGIME_MATCHED_DIM"]["applied"][str(C.PRIMARY_TAU)] = {"status": "UNREACHABLE_IN_GRID"}
    m = A.applied_mask(ss, "F2", "REGIME_MATCHED_DIM", str(C.PRIMARY_TAU))
    assert m.sum() == 6
    d = A.dose_report(ss, "F2", "REGIME_MATCHED_DIM", str(C.PRIMARY_TAU))
    assert d["statuses"]["UNREACHABLE_IN_GRID"] == 4 and d["fraction_dose_valid"] == pytest.approx(0.6) and not d["reach_ok"]


# ---------------------------------------------------------------- corpus
def test_corpus_splits_balance_and_disjointness():
    worlds = B.build_worlds()
    assert len(worlds) == C.WORLD_COUNT
    for s, k in C.SPLIT_SIZES.items():
        assert sum(w["split"] == s for w in worlds) == k
    ids = [w["world_id"] for w in worlds]
    assert len(set(ids)) == len(ids)
    balance = B.balance_audit(worlds)
    assert balance["all_factors_balanced_within_each_split"] and balance["split_sizes_ok"]
    fresh = B.build_demos()
    audit = B.overlap_audit(worlds, fresh)
    assert audit["disjoint"], audit
    # fit worlds share no world index with selection or final worlds
    by_split = {s: {w["world_index"] for w in worlds if w["split"] == s} for s in C.SPLITS}
    assert not (by_split["fit"] & by_split["final"]) and not (by_split["selection"] & by_split["final"])


def test_world_semantics_match_v1_structure():
    for w in B.build_worlds():
        f = w["facts"]["focal"]
        assert f["base"] != f["counterfactual"]
        for k in ("holder_control", "target_control", "diagonal"):
            assert w["facts"][k]["base"] == w["facts"][k]["counterfactual"]
        assert w["queries"]["Q1"]["label_base"] == "NOT_ENTAILED" and w["queries"]["Q1"]["label_counterfactual"] == "ENTAILED"
        assert w["queries"]["Q2"]["label_base"] == "ENTAILED"
        for q in ("Q5", "Q6"):
            assert w["queries"][q]["label_base"] == w["queries"][q]["label_counterfactual"]
        d = w["factors"]["direction"]
        assert w["queries"]["Q3"]["expected_sign"] == (-1 if d == "SUPPORT_TO_OPPOSE" else 1)
        assert w["queries"]["Q4"]["expected_sign"] == (1 if d == "SUPPORT_TO_OPPOSE" else -1)
        assert w["realizations"]["A"]["template_family"] != w["realizations"]["B"]["template_family"]
        assert w["realizations"]["A"]["sentence_order"] != w["realizations"]["B"]["sentence_order"]
        assert w["realizations"]["A"]["context_base"] != w["realizations"]["A"]["context_counterfactual"]


def test_protocol_prompts_differ_and_use_their_own_demonstrations():
    v1 = C.load_v1_demonstrations()
    fresh = B.build_demos()
    ctx, claim = "X affirms the view that y.", "X supports the proposition that y."
    t1, c1 = C.render_prompt("P1", v1, "F2", ctx, claim, None)
    t2, _ = C.render_prompt("P2", fresh, "F2", ctx, claim, None)
    t3, _ = C.render_prompt("P3", fresh, "F2", ctx, claim, None)
    assert c1 == [" ENTAILED", " NOT_ENTAILED"]
    assert t1 != t2 != t3 and t1 != t3
    assert t1.count("Context:") == 9 and t2.count("Context:") == 9 and t3.count("Statement:") == 9
    assert t1.endswith("Answer:") and t3.startswith(C.P3_F2_TASK_LINE)
    assert "protocol p-6a8289ff" in t1 and "protocol p-6a8289ff" not in t2  # P2/P3 use the fresh demonstrations only
    tf1, cf1 = C.render_prompt("P3", fresh, "F1", ctx, claim, "MAPPING_AB_REVERSED")
    assert cf1 == [" B", " A"] and "A) NOT_ENTAILED" in tf1


# ---------------------------------------------------------------- protocol selection and directions
def test_protocol_selection_rule_is_lexicographic_and_natural_only():
    per = {
        "P1": {"natural_gates_passed": 4, "ctnc": 0.9, "ba": 0.99},
        "P2": {"natural_gates_passed": 5, "ctnc": 0.62, "ba": 0.93},
        "P3": {"natural_gates_passed": 5, "ctnc": 0.71, "ba": 0.91},
    }
    order = sorted(per, key=lambda p: (-per[p]["natural_gates_passed"], -per[p]["ctnc"], -per[p]["ba"], p))
    assert order[0] == "P3"  # gates first, then cross-template consistency, then balanced accuracy
    assert per[order[0]]["natural_gates_passed"] == len(C.NATURAL_ONLY_GATES)


def synth_rows(n_worlds: int, dim: int, answer_only: bool) -> list[dict]:
    """Fit rows whose stance is (or is not) linearly encoded, plus a strong answer-label component."""
    rel = np.zeros(dim)
    rel[0] = 1.0
    ans = np.zeros(dim)
    ans[1] = 1.0
    rows = []
    for w in range(n_worlds):
        for real in ("A", "B"):
            for fmt in ("F2", "F1"):
                for q in F.FIT_QUERIES:
                    for state in ("base", "counterfactual"):
                        stance = "SUPPORT" if (state == "base") == (w % 2 == 0) else "OPPOSE"
                        label = "ENTAILED" if (q in ("Q1", "Q3")) == (stance == "SUPPORT") else "NOT_ENTAILED"
                        x = RNG.normal(0, 0.25, dim)
                        x += 4.0 * ans * (1 if label == "ENTAILED" else -1)
                        if not answer_only:
                            x += 3.0 * rel * (1 if stance == "SUPPORT" else -1)
                        rows.append({"world_id": f"W{w}", "world_index": w, "format": fmt, "realization": real, "state": state, "query": q, "stance": stance, "label": label, "x": x})
    return rows


def test_direction_qualification_passes_a_real_relation_code_and_fails_an_answer_only_code():
    ok = F.qualify_direction(synth_rows(24, 12, answer_only=False), folds=8)
    assert ok["qualified"] and ok["checks"]["cv_relation_sign_balanced_accuracy"]["value"] >= C.DIRECTION_GATES["cv_relation_sign_balanced_accuracy_min"]
    bad = F.qualify_direction(synth_rows(24, 12, answer_only=True), folds=8)
    assert not bad["qualified"]


def test_answer_residual_direction_removes_the_answer_component():
    rows = [r for r in synth_rows(16, 12, answer_only=False) if r["format"] == "F2"]
    relation = F.dim(rows, "stance", "SUPPORT")
    answer = F.dim(rows, "label", "ENTAILED")
    a_hat = C.unit(answer)
    residual = relation - float(relation @ a_hat) * a_hat
    assert abs(float(C.unit(residual) @ a_hat)) < 1e-9
    assert float(C.unit(relation) @ C.unit(residual)) > 0.5


def test_witness_projection_removes_the_relation_directions():
    d = np.zeros((6, 2))
    d[0, 0] = 1.0
    d[1, 1] = 1.0
    X = RNG.normal(0, 1, (20, 6))
    Xp = F.project_out(X, [d[:, 0], d[:, 1]])
    assert np.allclose(Xp @ d[:, 0], 0, atol=1e-9) and np.allclose(Xp @ d[:, 1], 0, atol=1e-9)


# ---------------------------------------------------------------- classification
def base_result(csr_mean: float, ci: tuple[float, float], focal: tuple[float, float], inv: tuple[float, float], interp_low: float, beats: bool, ctnc: float = 0.8) -> dict:
    def bs(mean, low, high):
        return {"mean": mean, "ci_low": low, "ci_high": high, "bootstrap_p_two_sided": 0.0001, "n": 96}

    prim = {"csr": bs(csr_mean, *ci), "csr_focal": bs(focal[0], focal[0], focal[1]), "csr_invariance": bs(inv[0], inv[0], inv[1]), "n": 96}
    panel = {
        f"{A.PRIMARY_ARM}@{A.TAU_KEY}": {"applied": prim, "dose_valid": prim},
        f"{C.NATURAL_INTERPOLATION}@{A.TAU_KEY}": {"applied": {"csr": bs(0.9, interp_low, 0.95)}},
        f"{A.TRANSPORT_ARM}@{A.TAU_KEY}": {"applied": {"csr": bs(-0.1, -0.2, -0.02)}},
    }
    pc = {c: {"mean": 0.3 if beats else 0.0, "ci_low": 0.1 if beats else -0.1, "ci_high": 0.5 if beats else 0.1, "bootstrap_p_two_sided": 0.0001 if beats else 0.9, "holm": {"reject": beats, "p_holm": 0.0001 if beats else 0.9}} for c in A.COMPARATORS}
    return {
        "reference": {"all_pass": True, "checks": {"cross_template_consistency": {"value": {"mean": ctnc}, "pass": True}}},
        "direction": {"status": "DIRECTION_QUALIFIED", "qualification": {"checks": {}}},
        "dose": {A.PRIMARY_ARM: {A.TAU_KEY: {"reach_ok": True, "norm_budget_ok": True, "dose_valid": True, "fraction_dose_valid": 0.95, "norm_ratio": {"median": 1.2, "p90": 2.0}}}},
        "csr": panel, "paired": pc, "pairwise_matched": {}, "comparators": {c: {"valid": True, "coverage": 0.95} for c in A.COMPARATORS},
        "by_direction": {f"{A.PRIMARY_ARM}@{A.TAU_KEY}": {d: {"mean": csr_mean} for d in C.DIRECTIONS}},
    }


def test_classification_precedence_and_criteria():
    r = base_result(0.6, (0.45, 0.75), (0.5, 0.7), (0.8, 0.95), 0.85, beats=True)
    assert A.classify(r)["class"] == "COUNTERFACTUAL_FIDELITY_SUPPORTED"
    r2 = base_result(0.0, (-0.05, 0.05), (-0.05, 0.05), (0.85, 0.95), 0.85, beats=False)
    c2 = A.classify(r2)
    assert c2["class"] == "STRONG_CONTROL_WITHOUT_COUNTERFACTUAL_FIDELITY"
    # an unreachable strong target outranks any fidelity reading
    r3 = base_result(0.6, (0.45, 0.75), (0.5, 0.7), (0.8, 0.95), 0.85, beats=True)
    r3["dose"][A.PRIMARY_ARM][A.TAU_KEY]["reach_ok"] = False
    r3["dose"][A.PRIMARY_ARM][A.TAU_KEY]["fraction_dose_valid"] = 0.5
    assert A.classify(r3)["class"] == "STRONG_TARGET_UNREACHABLE"
    r4 = base_result(0.6, (0.45, 0.75), (0.5, 0.7), (0.8, 0.95), 0.85, beats=True)
    r4["dose"][A.PRIMARY_ARM][A.TAU_KEY]["norm_budget_ok"] = False
    assert A.classify(r4)["class"] == "TARGET_REQUIRES_EXTREME_DISPLACEMENT"
    r5 = base_result(0.6, (0.45, 0.75), (0.5, 0.7), (0.8, 0.95), 0.85, beats=True)
    r5["reference"]["all_pass"] = False
    r5["reference"]["checks"]["cross_template_consistency"]["pass"] = False
    assert A.classify(r5)["class"] == "REFERENCE_ASSAY_INVALID"
    r6 = base_result(0.6, (0.45, 0.75), (0.5, 0.7), (0.8, 0.95), 0.85, beats=True)
    r6["direction"]["status"] = "DIRECTION_NOT_QUALIFIED"
    r6["direction"]["qualification"]["checks"] = {"cv_relation_sign_balanced_accuracy": {"pass": False}}
    assert A.classify(r6)["class"] == "DIRECTION_NOT_QUALIFIED"


def test_fidelity_requires_half_the_cross_template_ceiling_and_beating_every_comparator():
    weak = base_result(0.3, (0.2, 0.4), (0.3, 0.5), (0.8, 0.95), 0.85, beats=True, ctnc=0.9)  # 0.2 < 0.45
    assert A.classify(weak)["class"] != "COUNTERFACTUAL_FIDELITY_SUPPORTED"
    nobeat = base_result(0.6, (0.45, 0.75), (0.5, 0.7), (0.8, 0.95), 0.85, beats=False)
    out = A.classify(nobeat)
    assert out["class"] != "COUNTERFACTUAL_FIDELITY_SUPPORTED"
    assert any(f.startswith("SHARED_OR_NONUNIQUE_CONTROL_SIGNATURE") for f in out["flags"])


def test_strong_control_requires_a_working_natural_interpolation_control():
    r = base_result(0.0, (-0.05, 0.05), (-0.05, 0.05), (0.85, 0.95), 0.40, beats=False)  # interpolation control fails
    assert A.classify(r)["class"] != "STRONG_CONTROL_WITHOUT_COUNTERFACTUAL_FIDELITY"


def test_format_shift_flag_only_when_regime_direction_succeeds_and_original_does_not():
    r = base_result(0.6, (0.45, 0.75), (0.5, 0.7), (0.8, 0.95), 0.85, beats=True)
    out = A.classify(r)
    assert out["class"] == "COUNTERFACTUAL_FIDELITY_SUPPORTED" and "FORMAT_SHIFT_EXPLAINS_V1" in out["flags"]


def test_transport_disposition_uses_the_same_criteria_and_never_changes_the_terminal_class():
    r = base_result(0.0, (-0.05, 0.05), (-0.05, 0.05), (0.85, 0.95), 0.85, beats=False)
    r["by_direction"][f"{A.TRANSPORT_ARM}@{A.TAU_KEY}"] = {d: {"mean": -0.02} for d in C.DIRECTIONS}
    r["dose"][A.TRANSPORT_ARM] = {A.TAU_KEY: dict(r["dose"][A.PRIMARY_ARM][A.TAU_KEY])}
    r["csr"][f"{A.TRANSPORT_ARM}@{A.TAU_KEY}"]["dose_valid"] = r["csr"][f"{A.TRANSPORT_ARM}@{A.TAU_KEY}"]["applied"]
    r["csr"][f"{A.TRANSPORT_ARM}@{A.TAU_KEY}"]["applied"].update({"csr_focal": {"mean": -0.02, "ci_low": -0.06, "ci_high": 0.02}, "csr_invariance": {"mean": 0.9, "ci_low": 0.85, "ci_high": 0.95}})
    t = A.criteria_for(r, A.TRANSPORT_ARM, {c: {"mean": 0.0, "ci_low": -0.1, "ci_high": 0.1, "holm": {"reject": False}} for c in A.COMPARATORS})
    assert t["available"] and t["disposition"] == "STRONG_CONTROL_WITHOUT_COUNTERFACTUAL_FIDELITY"
    # an unqualified primary direction still yields DIRECTION_NOT_QUALIFIED, with the transport arm carried in the reason
    r["direction"]["status"] = "DIRECTION_NOT_QUALIFIED"
    r["direction"]["qualification"]["checks"] = {"cv_relation_sign_balanced_accuracy": {"pass": False}}
    r["transport_disposition"] = t
    out = A.classify(r)
    assert out["class"] == "DIRECTION_NOT_QUALIFIED"
    assert out["transport_disposition"] == "STRONG_CONTROL_WITHOUT_COUNTERFACTUAL_FIDELITY"
    assert "regime-matched arm only" in out["scope"]


def test_joint_class_never_averages_an_invalid_actor():
    assert A.joint_class({"llama": "REFERENCE_ASSAY_INVALID", "gemma": "STRONG_CONTROL_WITHOUT_COUNTERFACTUAL_FIDELITY"}) == "SINGLE_VALID_ACTOR"
    assert A.joint_class({"llama": "COUNTERFACTUAL_FIDELITY_SUPPORTED", "gemma": "STRONG_CONTROL_WITHOUT_COUNTERFACTUAL_FIDELITY"}) == "MODEL_DEPENDENT"
    assert A.joint_class({"llama": "ASSAY_NOT_QUALIFIED", "gemma": "REFERENCE_ASSAY_INVALID"}) == "NO_VALID_ACTOR"
    # an unqualified primary direction is an arm outcome, not an invalid assay
    assert A.joint_class({"llama": "ASSAY_NOT_QUALIFIED", "gemma": "DIRECTION_NOT_QUALIFIED"}) == "SINGLE_VALID_ACTOR"
    assert A.joint_class({"llama": "STRONG_CONTROL_WITHOUT_COUNTERFACTUAL_FIDELITY", "gemma": "STRONG_CONTROL_WITHOUT_COUNTERFACTUAL_FIDELITY"}).startswith("REPLICATED")
