#!/usr/bin/env python3
"""Final analysis and closeout for STRONG_TARGET_COUNTERFACTUAL_RELATION_FIDELITY_V2.

Modes
  final   per actor: reference gates (five natural gates + full-state patch CSR), dose validity and
          norm ratios per arm and tau, the CSR panel (world, focal, invariance, cross-template, mapped
          format) per arm and tau, Holm-corrected paired differences and pairwise underidentification
          at the primary tau, the template-matched witness panel, and the terminal classification.
  joint   both actors: STRONG_TARGET_RESULTS.json, JOINT_REPORT.md, MANUSCRIPT_CLAIM_MATRIX.md, the
          external review packet, the dose-response and CSR figures and the study manifest.

No prompt, endpoint, threshold, direction or calibration rule is chosen here; all come from
stcrf_v2_common and the freeze packages. The CSR mathematics is CRIF v1's, imported unchanged.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stcrf_v2_common as C  # noqa: E402
import stcrf_v2_fit as F  # noqa: E402

PRIMARY_ARM = "REGIME_MATCHED_DIM"
TRANSPORT_ARM = "ORIGINAL_FROZEN_DIM"
TAU_KEY = str(C.PRIMARY_TAU)
COMPARATORS = (C.LOGIT_CONTROL, "MATCHED_SENTIMENT_DIRECTION", "MATCHED_ANSWER_TOKEN_DIRECTION", "MATCHED_TRUTH_DIRECTION")


# ---------------------------------------------------------------- signatures
def world_signatures(summary: dict[str, Any], fmt: str, realization: str) -> dict[str, Any]:
    """Natural signature N (Q2..Q6) and, per (arm, tau), the intervention signature I and the Q1 change."""
    nat = F.natural_signature(summary, fmt, realization)
    base = nat["base"]
    rows = [r for r in summary["rows"] if r["format"] == fmt and r["realization"] == realization and "semantic_margin" in r]
    I: dict[tuple[str, str], np.ndarray] = {}
    dq1: dict[tuple[str, str], float] = {}
    achieved: dict[tuple[str, str], float] = {}
    for r in rows:
        if r["intervention"] in ("NO_INTERVENTION",):
            continue
        key = (r["intervention"], "" if r["tau"] is None else str(r["tau"]))
        I.setdefault(key, {})
        I[key][r["query"]] = r["semantic_margin"] - base[r["query"]]
    out: dict[tuple[str, str], np.ndarray] = {}
    for key, per_q in I.items():
        if not all(q in per_q for q in C.QUERIES):
            raise ValueError(f"{summary['world_id']} {fmt} {realization} {key}: incomplete query panel")
        out[key] = np.array([per_q[q] for q in C.CONSEQUENCE_QUERIES])
        dq1[key] = per_q["Q1"]
        achieved[key] = base["Q1"] + per_q["Q1"]
    # analytic final-logit control at every tau
    natrec = summary["natural"][fmt]
    for t in C.TAUS:
        b = natrec["targets"][str(t)] - natrec["base_q1_A"]
        out[(C.LOGIT_CONTROL, str(t))] = np.full(len(C.CONSEQUENCE_QUERIES), b)
        dq1[(C.LOGIT_CONTROL, str(t))] = b
        achieved[(C.LOGIT_CONTROL, str(t))] = natrec["base_q1_A"] + b
    out[("NO_INTERVENTION", "")] = np.zeros(len(C.CONSEQUENCE_QUERIES))
    dq1[("NO_INTERVENTION", "")] = 0.0
    return {"N": nat["N"], "N_Q1": nat["N_Q1"], "base": base, "cf": nat["cf"], "native": nat["native"], "I": out, "dQ1": dq1, "achieved_Q1": achieved}


def arm_keys(sigs: list[dict[str, Any]]) -> list[tuple[str, str]]:
    keys = sorted({k for s in sigs for k in s["I"]}, key=lambda k: (k[0], k[1]))
    return [k for k in keys if k[0] != "NO_INTERVENTION"]


def applied_mask(summaries: list[dict[str, Any]], fmt: str, arm: str, tau: str) -> np.ndarray:
    if arm == C.LOGIT_CONTROL:
        return np.ones(len(summaries), dtype=bool)
    out = []
    for s in summaries:
        rec = s["arms"].get(fmt, {}).get(arm, {})
        ap = rec.get("applied", {}).get(tau, {})
        out.append(ap.get("status") == "APPLIED" and bool(ap.get("finite", True)))
    return np.array(out, dtype=bool)


def dose_report(summaries: list[dict[str, Any]], fmt: str, arm: str, tau: str) -> dict[str, Any]:
    n = len(summaries)
    if arm == C.LOGIT_CONTROL:
        return {"arm": arm, "tau": tau, "analytic": True, "n": n, "fraction_applied": 1.0, "fraction_dose_valid": 1.0, "dose_valid": True}
    recs = [s["arms"].get(fmt, {}).get(arm, {}) for s in summaries]
    ap = [r.get("applied", {}).get(tau, {}) for r in recs]
    applied = [a for a in ap if a.get("status") == "APPLIED"]
    valid = [a for a in applied if a.get("within_tolerance") and a.get("hard_answer_counterfactual") and a.get("signed_margin_ge_1") and a.get("finite", True) and not a.get("saturated")]
    ratios = np.array([a.get("norm_ratio") for a in applied if a.get("norm_ratio") is not None]) if applied else np.array([])
    med = float(np.median(ratios)) if ratios.size else None
    p90 = float(np.percentile(ratios, 90)) if ratios.size else None
    frac_valid = len(valid) / n
    budget_ok = bool(med is not None and med <= C.DOSE_VALIDITY["norm_ratio_median_max"] and p90 is not None and p90 <= C.DOSE_VALIDITY["norm_ratio_p90_max"])
    reach_ok = bool(frac_valid >= C.DOSE_VALIDITY["world_fraction_min"])
    return {
        "arm": arm, "tau": tau, "n": n, "fraction_applied": len(applied) / n, "fraction_dose_valid": frac_valid,
        "criteria": {"within_tolerance": float(np.mean([bool(a.get("within_tolerance")) for a in applied])) if applied else 0.0, "hard_answer_counterfactual": float(np.mean([bool(a.get("hard_answer_counterfactual")) for a in applied])) if applied else 0.0, "signed_margin_ge_1": float(np.mean([bool(a.get("signed_margin_ge_1")) for a in applied])) if applied else 0.0, "not_saturated": float(np.mean([not a.get("saturated") for a in applied])) if applied else 0.0},
        "statuses": {k: sum(a.get("status") == k for a in ap) for k in ("APPLIED", "UNREACHABLE_IN_GRID", "NUMERICALLY_INVALID")},
        "norm_ratio": {"median": med, "p90": p90, "max": float(ratios.max()) if ratios.size else None, "mean": float(ratios.mean()) if ratios.size else None},
        "achieved_q1_mean": float(np.mean([a["achieved_q1_A"] for a in applied])) if applied else None,
        "target_q1_mean": float(np.mean([a["target_q1_A"] for a in applied])) if applied else None,
        "reach_ok": reach_ok, "norm_budget_ok": budget_ok, "dose_valid": bool(reach_ok and budget_ok),
    }


def dose_valid_mask(summaries: list[dict[str, Any]], fmt: str, arm: str, tau: str) -> np.ndarray:
    if arm == C.LOGIT_CONTROL:
        return np.ones(len(summaries), dtype=bool)
    out = []
    for s in summaries:
        a = s["arms"].get(fmt, {}).get(arm, {}).get("applied", {}).get(tau, {})
        out.append(a.get("status") == "APPLIED" and bool(a.get("within_tolerance")) and bool(a.get("hard_answer_counterfactual")) and bool(a.get("signed_margin_ge_1")) and bool(a.get("finite", True)) and not a.get("saturated"))
    return np.array(out, dtype=bool)


def csr_for(sigs: list[dict[str, Any]], key: tuple[str, str], scales: np.ndarray, mask: np.ndarray, idx_cache: dict[int, np.ndarray]) -> dict[str, Any] | None:
    sel = [s for s, m in zip(sigs, mask) if m and key in s["I"]]
    if len(sel) < 5:
        return None
    N = np.stack([s["N"] for s in sel])
    I = np.stack([s["I"][key] for s in sel])
    r = C.csr(I, N, scales)
    idx = idx_cache.setdefault(len(sel), C.bootstrap_indices(len(sel)))
    out = {k: C.bootstrap_mean(v, idx) for k, v in r.items()}
    out["n"] = len(sel)
    out["mean_signature"] = {q: float(I[:, i].mean()) for i, q in enumerate(C.CONSEQUENCE_QUERIES)}
    out["mean_standardized_signature"] = {q: float((I[:, i] / scales[i]).mean()) for i, q in enumerate(C.CONSEQUENCE_QUERIES)}
    out["per_world_csr"] = r["csr"].tolist()
    out["world_ids"] = [s["world_id"] for s in sel]
    return out


def aligned_signature(sigs: list[dict[str, Any]], key: tuple[str, str] | None, scales: np.ndarray, directions: list[str], mask: np.ndarray) -> dict[str, float]:
    sel = [(s, d) for s, d, m in zip(sigs, directions, mask) if m and (key is None or key in s["I"])]
    if not sel:
        return {q: float("nan") for q in C.CONSEQUENCE_QUERIES}
    V = np.stack([s["N"] if key is None else s["I"][key] for s, _ in sel])
    align = np.array([[float(C.expected_sign(d, q)) if q in C.FOCAL_QUERIES else 1.0 for q in C.CONSEQUENCE_QUERIES] for _, d in sel])
    return {q: float((align[:, i] * V[:, i] / scales[i]).mean()) for i, q in enumerate(C.CONSEQUENCE_QUERIES)}


def paired(sigs: list[dict[str, Any]], a: tuple[str, str], b: tuple[str, str], scales: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray, idx_cache: dict[int, np.ndarray]) -> dict[str, Any] | None:
    m = mask_a & mask_b
    sel = [s for s, x in zip(sigs, m) if x and a in s["I"] and b in s["I"]]
    if len(sel) < 5:
        return None
    N = np.stack([s["N"] for s in sel])
    ca = C.csr(np.stack([s["I"][a] for s in sel]), N, scales)["csr"]
    cb = C.csr(np.stack([s["I"][b] for s in sel]), N, scales)["csr"]
    idx = idx_cache.setdefault(len(sel), C.bootstrap_indices(len(sel)))
    return C.bootstrap_mean(ca - cb, idx) | {"n": len(sel)}


# ---------------------------------------------------------------- witness
def witness_panel(gpu_root: Path, summaries: list[dict[str, Any]], worlds: dict[str, dict[str, Any]], witness: dict[str, Any], keys: list[tuple[str, str]]) -> dict[str, Any]:
    pred, truth, pred_b, truth_b = [], [], [], []
    wsr: dict[str, list[float]] = {}
    inv_move: dict[str, list[float]] = {}
    inv_nat: list[float] = []
    for s in summaries:
        acts = F.load_world_acts(gpu_root, s["world_id"], "witness")
        w = worlds[s["world_id"]]
        stance = {"base": w["facts"]["focal"]["base"], "counterfactual": w["facts"]["focal"]["counterfactual"]}
        for real, P, T in (("A", pred, truth), ("B", pred_b, truth_b)):
            for state in ("base", "counterfactual"):
                for q in ("Q1", "Q2"):
                    k = F.act_key("F2", real, state, q)
                    if k not in acts:
                        continue
                    z = float(F.witness_logit(witness, acts[k][None, :])[0])
                    P.append(int(z > 0))
                    T.append(int(stance[state] == "SUPPORT"))
        zb = {q: float(F.witness_logit(witness, acts[F.act_key("F2", "A", "base", q)][None, :])[0]) for q in C.QUERIES}
        zc = {q: float(F.witness_logit(witness, acts[F.act_key("F2", "A", "counterfactual", q)][None, :])[0]) for q in C.QUERIES}
        inv_nat.append(float(np.mean([abs(zc[q] - zb[q]) for q in C.INVARIANCE_QUERIES])))
        for arm, tau in keys:
            name = f"{arm}@{tau}"
            ks = {q: F.act_key("F2", "A", "base", q, arm, float(tau) if tau else None) for q in C.QUERIES}
            if not all(k in acts for k in ks.values()):
                continue
            zi = {q: float(F.witness_logit(witness, acts[ks[q]][None, :])[0]) for q in C.QUERIES}
            num = sum((zi[q] - zc[q]) ** 2 for q in ("Q1",) + C.FOCAL_QUERIES)
            den = sum((zc[q] - zb[q]) ** 2 for q in ("Q1",) + C.FOCAL_QUERIES)
            wsr.setdefault(name, []).append(1.0 - num / den if den > 0 else float("nan"))
            inv_move.setdefault(name, []).append(float(np.mean([abs(zi[q] - zb[q]) for q in C.INVARIANCE_QUERIES])))
    ba, ba_b = C.balanced_accuracy(pred, truth), C.balanced_accuracy(pred_b, truth_b)
    g = C.WITNESS_GATES
    gate = {"balanced_accuracy": {"value": ba, "min": g["balanced_accuracy_min"], "pass": bool(ba >= g["balanced_accuracy_min"]), "n": len(pred)}, "cross_template_balanced_accuracy": {"value": ba_b, "min": g["cross_template_balanced_accuracy_min"], "pass": bool(ba_b >= g["cross_template_balanced_accuracy_min"]), "n": len(pred_b)}}
    passed = all(v["pass"] for v in gate.values())
    out: dict[str, Any] = {"gate": gate, "status": "WITNESS_VALID" if passed else "WITNESS_GATE_FAILED", "natural_invariance_witness_movement_mean": float(np.mean(inv_nat)), "interventions": {}}
    for name, v in wsr.items():
        arr = np.array(v)
        idx = C.bootstrap_indices(len(arr))
        out["interventions"][name] = {"witness_signature_recovery": C.bootstrap_mean(np.nan_to_num(arr, nan=0.0), idx), "n": int(arr.size), "nan_worlds": int(np.isnan(arr).sum()), "invariance_witness_movement_mean": float(np.mean(inv_move[name]))}
    if not passed:
        out["note"] = "witness gate failed: preserved as a failure; no internal-state claim is made and the behavioral CSR result is unaffected"
    return out


# ---------------------------------------------------------------- criteria (one code path for the primary and transport arms)
def criteria_for(res: dict[str, Any], arm: str, comparator_paired: dict[str, Any]) -> dict[str, Any]:
    """The frozen fidelity / strong-control criteria evaluated for one relation arm at the primary tau.

    Used for the primary inferential object (REGIME_MATCHED_DIM) and, with identical thresholds, for the
    transport/replication arm (ORIGINAL_FROZEN_DIM). Reporting only: the terminal class comes from classify().
    """
    panel = res["csr"]
    entry = panel.get(f"{arm}@{TAU_KEY}", {})
    prim = entry.get("dose_valid") or entry.get("applied")
    dose = res["dose"].get(arm, {}).get(TAU_KEY)
    if prim is None or dose is None:
        return {"arm": arm, "available": False}
    interp = panel.get(f"{C.NATURAL_INTERPOLATION}@{TAU_KEY}", {}).get("applied")
    ctnc = res["reference"]["checks"]["cross_template_consistency"]["value"]["mean"]
    valid = [c for c in COMPARATORS if res["comparators"][c]["valid"]]
    beats = {c: bool(comparator_paired.get(c) and comparator_paired[c]["holm"]["reject"] and comparator_paired[c]["mean"] > 0) for c in valid}
    per_dir = res["by_direction"].get(f"{arm}@{TAU_KEY}", {})
    xt = entry.get("cross_template")
    fidelity = bool(dose["dose_valid"] and prim["csr"]["ci_low"] >= C.FIDELITY_CI_LOW_FRACTION_OF_CTNC * ctnc and prim["csr_focal"]["ci_low"] > 0 and prim["csr_invariance"]["ci_low"] > 0 and beats and all(beats.values()) and per_dir and all(v["mean"] > 0 for v in per_dir.values()))
    strong_control = bool(dose["dose_valid"] and interp is not None and interp["csr"]["ci_low"] >= C.STRONG_CONTROL_INTERPOLATION_CI_LOW_MIN and prim["csr"]["ci_high"] <= C.STRONG_CONTROL_RELATION_CI_HIGH_MAX and prim["csr_focal"]["ci_high"] <= C.STRONG_CONTROL_RELATION_CI_HIGH_MAX and per_dir and all(v["mean"] <= C.STRONG_CONTROL_RELATION_CI_HIGH_MAX for v in per_dir.values()) and (xt is None or xt["csr"]["mean"] <= C.STRONG_CONTROL_RELATION_CI_HIGH_MAX))
    disposition = "COUNTERFACTUAL_FIDELITY_SUPPORTED" if fidelity else ("STRONG_CONTROL_WITHOUT_COUNTERFACTUAL_FIDELITY" if strong_control else ("STRONG_TARGET_UNREACHABLE" if not dose["reach_ok"] else ("TARGET_REQUIRES_EXTREME_DISPLACEMENT" if not dose["norm_budget_ok"] else "PARTIAL_OR_UNDERDETERMINED")))
    return {"arm": arm, "available": True, "disposition": disposition, "dose": dose, "csr": prim["csr"], "csr_focal": prim["csr_focal"], "csr_invariance": prim["csr_invariance"], "cross_template_csr": xt["csr"] if xt else None, "half_ctnc_threshold": C.FIDELITY_CI_LOW_FRACTION_OF_CTNC * ctnc, "interpolation_csr": interp["csr"] if interp else None, "beats_comparators_holm": beats, "valid_comparators": valid, "per_direction": {k: v["mean"] for k, v in per_dir.items()}, "licensed_wording": C.LICENSED_WORDING.get(disposition, "no positive claim licensed; report the disposition")}


# ---------------------------------------------------------------- classification
def classify(res: dict[str, Any]) -> dict[str, Any]:
    flags: list[str] = []
    if not res["reference"]["all_pass"]:
        return {"class": "REFERENCE_ASSAY_INVALID", "flags": flags, "reason": "natural-reference gate failure: " + ", ".join(k for k, v in res["reference"]["checks"].items() if v["pass"] is False)}
    if res["direction"]["status"] != "DIRECTION_QUALIFIED":
        t = res.get("transport_disposition", {})
        return {"class": "DIRECTION_NOT_QUALIFIED", "flags": flags, "reason": "the primary inferential object (REGIME_MATCHED_DIM) failed its frozen fit-population gates: " + ", ".join(k for k, v in res["direction"]["qualification"]["checks"].items() if not v["pass"]), "scope": "applies to the regime-matched arm only; the transport/replication arm ORIGINAL_FROZEN_DIM is a frozen upstream direction and its panel is reported separately as a secondary disposition", "transport_disposition": t.get("disposition") if t.get("available") else None}
    dose = res["dose"][PRIMARY_ARM][TAU_KEY]
    if not dose["reach_ok"]:
        return {"class": "STRONG_TARGET_UNREACHABLE", "flags": flags, "reason": f"tau=0.75 dose-valid on {dose['fraction_dose_valid']:.3f} of final worlds (< {C.DOSE_VALIDITY['world_fraction_min']})", "detail": dose}
    if not dose["norm_budget_ok"]:
        return {"class": "TARGET_REQUIRES_EXTREME_DISPLACEMENT", "flags": flags, "reason": f"norm ratio median {dose['norm_ratio']['median']}, p90 {dose['norm_ratio']['p90']} exceed the frozen budget", "detail": dose}
    panel = res["csr"]
    prim = panel[f"{PRIMARY_ARM}@{TAU_KEY}"]["dose_valid"]
    interp = panel.get(f"{C.NATURAL_INTERPOLATION}@{TAU_KEY}", {}).get("applied")
    ctnc = res["reference"]["checks"]["cross_template_consistency"]["value"]["mean"]
    pc = res["paired"]
    valid_comparators = [c for c in COMPARATORS if res["comparators"][c]["valid"]]
    beats = {c: bool(pc.get(c) and pc[c]["holm"]["reject"] and pc[c]["mean"] > 0) for c in valid_comparators}
    per_dir = res["by_direction"][f"{PRIMARY_ARM}@{TAU_KEY}"]
    xt = panel[f"{PRIMARY_ARM}@{TAU_KEY}"].get("cross_template")
    fidelity = bool(prim["csr"]["ci_low"] >= C.FIDELITY_CI_LOW_FRACTION_OF_CTNC * ctnc and prim["csr_focal"]["ci_low"] > 0 and prim["csr_invariance"]["ci_low"] > 0 and all(beats.values()) and all(v["mean"] > 0 for v in per_dir.values()))
    strong_control = bool(interp is not None and interp["csr"]["ci_low"] >= C.STRONG_CONTROL_INTERPOLATION_CI_LOW_MIN and prim["csr"]["ci_high"] <= C.STRONG_CONTROL_RELATION_CI_HIGH_MAX and prim["csr_focal"]["ci_high"] <= C.STRONG_CONTROL_RELATION_CI_HIGH_MAX and all(v["mean"] <= C.STRONG_CONTROL_RELATION_CI_HIGH_MAX for v in per_dir.values()) and (xt is None or xt["csr"]["mean"] <= C.STRONG_CONTROL_RELATION_CI_HIGH_MAX))
    # secondary flags
    underid = any(v["holm"]["reject"] for v in res["pairwise_matched"].values())
    shared = [c for c in valid_comparators if c != C.LOGIT_CONTROL and pc.get(c) and not (pc[c]["ci_low"] > 0 or pc[c]["ci_high"] < 0)]
    if underid:
        flags.append("TARGET_EFFECT_CAUSALLY_UNDERIDENTIFIED")
    if shared:
        flags.append("SHARED_OR_NONUNIQUE_CONTROL_SIGNATURE:" + ",".join(shared))
    transport = panel.get(f"{TRANSPORT_ARM}@{TAU_KEY}", {}).get("applied")
    if fidelity and transport is not None and not (transport["csr"]["ci_low"] >= C.FIDELITY_CI_LOW_FRACTION_OF_CTNC * ctnc):
        flags.append("FORMAT_SHIFT_EXPLAINS_V1")
    detail = {"primary_csr": prim["csr"], "primary_focal": prim["csr_focal"], "primary_invariance": prim["csr_invariance"], "half_ctnc_threshold": C.FIDELITY_CI_LOW_FRACTION_OF_CTNC * ctnc, "interpolation_csr": interp["csr"] if interp else None, "beats_comparators_holm": beats, "valid_comparators": valid_comparators, "per_direction": {k: v["mean"] for k, v in per_dir.items()}, "cross_template_csr": xt["csr"] if xt else None, "dose": dose}
    if fidelity:
        return {"class": "COUNTERFACTUAL_FIDELITY_SUPPORTED", "flags": flags, "detail": detail}
    if strong_control:
        return {"class": "STRONG_CONTROL_WITHOUT_COUNTERFACTUAL_FIDELITY", "flags": flags, "detail": detail}
    if underid:
        return {"class": "TARGET_EFFECT_CAUSALLY_UNDERIDENTIFIED", "flags": flags, "detail": detail}
    return {"class": "PARTIAL_OR_UNDERDETERMINED", "flags": flags, "detail": detail}


# ---------------------------------------------------------------- actor analysis
def analyze_actor(actor: str, gpu_root: Path, freeze_root: Path, final_freeze_path: Path, split: str = "final") -> dict[str, Any]:
    fz = C.read_json(final_freeze_path)
    a = fz["actors"][actor]
    scales = np.array(a["scales_Q2_to_Q6"], dtype=np.float64)
    worlds = {w["world_id"]: w for w in C.read_jsonl(freeze_root / "CORPUS.jsonl")}
    summaries = sorted([s for s in C.read_jsonl(gpu_root / "world_summaries.jsonl") if s["split"] == split], key=lambda s: s["world_id"])
    if {s["protocol"] for s in summaries} != {a["protocol"]}:
        raise RuntimeError("final outputs use a protocol other than the frozen one")
    n = len(summaries)
    directions = [s["direction"] for s in summaries]
    sigs = {(fmt, real): [world_signatures(s, fmt, real) | {"world_id": s["world_id"]} for s in summaries] for fmt in ("F2", "F1") for real in (C.REALIZATIONS if fmt == "F2" else ("A",))}
    idx_cache: dict[int, np.ndarray] = {}
    idx = C.bootstrap_indices(n)
    sA = sigs[("F2", "A")]
    NA = np.stack([s["N"] for s in sA])
    # reference gates (patch CSR from the final worlds' own full-state patch)
    patch_key = (C.FULL_PATCH, "")
    patch = csr_for(sA, patch_key, scales, np.ones(n, dtype=bool), idx_cache)
    gates = F.natural_gates(summaries, scales, idx, patch["csr"] if patch else None)
    res: dict[str, Any] = {"schema_version": f"{C.SCHEMA_PREFIX}_ACTOR_RESULT", "actor": actor, "protocol": a["protocol"], "split": split, "worlds": n, "world_ids": [s["world_id"] for s in summaries], "scales": scales.tolist(), "final_analysis_freeze_sha256": C.sha256_file(final_freeze_path), "gpu_root": str(gpu_root), "gpu_manifest_sha256": C.sha256_file(gpu_root / "artifact_manifest.json")}
    res["reference"] = gates
    res["direction"] = {"status": a.get("direction_status"), "qualification": a.get("direction_qualification"), "directions_file_sha256": a.get("directions_file_sha256")}
    keys = arm_keys(sA)
    res["dose"] = {}
    for arm, tau in keys:
        if arm == C.FULL_PATCH:
            continue
        res["dose"].setdefault(arm, {})[tau] = dose_report(summaries, "F2", arm, tau)
    for t in C.TAUS:
        res["dose"].setdefault(C.LOGIT_CONTROL, {})[str(t)] = dose_report(summaries, "F2", C.LOGIT_CONTROL, str(t))
    # CSR panel
    panel: dict[str, Any] = {}
    by_direction: dict[str, Any] = {}
    for arm, tau in [(C.FULL_PATCH, ""), ("NO_INTERVENTION", "")] + keys + [(C.LOGIT_CONTROL, str(t)) for t in C.TAUS]:
        name = f"{arm}@{tau}" if tau else arm
        if name in panel:
            continue
        m_app = applied_mask(summaries, "F2", arm, tau) if arm not in (C.FULL_PATCH, "NO_INTERVENTION") else np.ones(n, dtype=bool)
        entry: dict[str, Any] = {}
        app = csr_for(sA, (arm, tau), scales, m_app, idx_cache)
        if app is None:
            panel[name] = {"applied": None, "note": "fewer than five worlds applied"}
            continue
        entry["applied"] = app
        m_valid = dose_valid_mask(summaries, "F2", arm, tau) if arm not in (C.FULL_PATCH, "NO_INTERVENTION") else np.ones(n, dtype=bool)
        dv = csr_for(sA, (arm, tau), scales, m_valid, idx_cache)
        entry["dose_valid"] = dv if dv is not None else app
        xt = csr_for(sigs[("F2", "B")], (arm, tau), scales, m_app, idx_cache)
        if xt is not None:
            entry["cross_template"] = xt
        f1 = csr_for(sigs[("F1", "A")], (arm, tau), scales, np.ones(n, dtype=bool), idx_cache) if any((arm, tau) in s["I"] for s in sigs[("F1", "A")]) else None
        if f1 is not None:
            entry["mapped_format"] = f1
        entry["mean_aligned_standardized_signature"] = aligned_signature(sA, (arm, tau), scales, directions, m_app)
        entry["mean_achieved_q1"] = float(np.mean([s["achieved_Q1"][(arm, tau)] for s, mm in zip(sA, m_app) if mm and (arm, tau) in s.get("achieved_Q1", {})])) if arm != "NO_INTERVENTION" else None
        entry["mean_dq1"] = float(np.mean([s["dQ1"][(arm, tau)] for s, mm in zip(sA, m_app) if mm and (arm, tau) in s["dQ1"]])) if arm != "NO_INTERVENTION" else 0.0
        panel[name] = entry
        pd = {}
        for d in C.DIRECTIONS:
            md = m_app & np.array([dd == d for dd in directions])
            r = csr_for(sA, (arm, tau), scales, md, idx_cache)
            if r is not None:
                pd[d] = r["csr"] | {"n": r["n"]}
        by_direction[name] = pd
    panel["NATURAL"] = {"mean_aligned_standardized_signature": aligned_signature(sA, None, scales, directions, np.ones(n, dtype=bool)), "mean_signature": {q: float(NA[:, i].mean()) for i, q in enumerate(C.CONSEQUENCE_QUERIES)}, "natural_q1_change_mean": float(np.mean([s["N_Q1"] for s in sA]))}
    res["csr"] = panel
    res["by_direction"] = by_direction
    # comparator validity and paired differences at the primary tau
    res["comparators"] = {}
    for c in COMPARATORS:
        d = res["dose"].get(c, {}).get(TAU_KEY)
        cov = d["fraction_dose_valid"] if d else 0.0
        minimum = C.COVERAGE_MIN_FOR_TRUTH if c == "MATCHED_TRUTH_DIRECTION" else C.COMPARATOR_COVERAGE_MIN
        res["comparators"][c] = {"coverage": cov, "min": minimum, "valid": bool(cov >= minimum), "note": "the matched truth direction enters the panel only at >= 80% matching coverage" if c == "MATCHED_TRUTH_DIRECTION" else None}
    mp = applied_mask(summaries, "F2", PRIMARY_ARM, TAU_KEY)
    diffs: dict[str, Any] = {}
    for c in COMPARATORS:
        if not res["comparators"][c]["valid"]:
            continue
        mc = applied_mask(summaries, "F2", c, TAU_KEY)
        r = paired(sA, (PRIMARY_ARM, TAU_KEY), (c, TAU_KEY), scales, mp, mc, idx_cache)
        if r is not None:
            diffs[c] = r
    h = C.holm({c: v["bootstrap_p_two_sided"] for c, v in diffs.items()}) if diffs else {}
    for c in diffs:
        diffs[c] |= {"holm": h[c], "primary_exceeds_comparator_holm": bool(h[c]["reject"] and diffs[c]["mean"] > 0)}
    res["paired"] = diffs
    # transport comparison and pairwise underidentification among matched interventions
    extra = {}
    r = paired(sA, (PRIMARY_ARM, TAU_KEY), (TRANSPORT_ARM, TAU_KEY), scales, mp, applied_mask(summaries, "F2", TRANSPORT_ARM, TAU_KEY), idx_cache)
    if r is not None:
        extra["REGIME_MATCHED_minus_ORIGINAL_FROZEN"] = r
    r = paired(sA, (PRIMARY_ARM, TAU_KEY), (C.NATURAL_INTERPOLATION, TAU_KEY), scales, mp, applied_mask(summaries, "F2", C.NATURAL_INTERPOLATION, TAU_KEY), idx_cache)
    if r is not None:
        extra["REGIME_MATCHED_minus_NATURAL_INTERPOLATION"] = r
    r = paired(sA, (PRIMARY_ARM, TAU_KEY), ("REGIME_ANSWER_RESIDUAL_DIM", TAU_KEY), scales, mp, applied_mask(summaries, "F2", "REGIME_ANSWER_RESIDUAL_DIM", TAU_KEY), idx_cache)
    if r is not None:
        extra["REGIME_MATCHED_minus_ANSWER_RESIDUAL"] = r
    res["additional_paired"] = extra
    members = [PRIMARY_ARM] + [c for c in COMPARATORS if res["comparators"][c]["valid"]]
    pw = {}
    for x, y in itertools.combinations(members, 2):
        r = paired(sA, (x, TAU_KEY), (y, TAU_KEY), scales, applied_mask(summaries, "F2", x, TAU_KEY), applied_mask(summaries, "F2", y, TAU_KEY), idx_cache)
        if r is not None:
            pw[f"{x}|{y}"] = r
    hpw = C.holm({k: v["bootstrap_p_two_sided"] for k, v in pw.items()}) if pw else {}
    for k in pw:
        pw[k]["holm"] = hpw[k]
    res["pairwise_matched"] = pw
    # transport-arm comparator differences and dispositions (same thresholds, same code path; reporting only)
    mt = applied_mask(summaries, "F2", TRANSPORT_ARM, TAU_KEY)
    tdiffs: dict[str, Any] = {}
    for c in COMPARATORS:
        if not res["comparators"][c]["valid"]:
            continue
        r = paired(sA, (TRANSPORT_ARM, TAU_KEY), (c, TAU_KEY), scales, mt, applied_mask(summaries, "F2", c, TAU_KEY), idx_cache)
        if r is not None:
            tdiffs[c] = r
    ht = C.holm({c: v["bootstrap_p_two_sided"] for c, v in tdiffs.items()}) if tdiffs else {}
    for c in tdiffs:
        tdiffs[c] |= {"holm": ht[c], "primary_exceeds_comparator_holm": bool(ht[c]["reject"] and tdiffs[c]["mean"] > 0)}
    res["paired_transport"] = tdiffs
    res["primary_disposition"] = criteria_for(res, PRIMARY_ARM, res["paired"])
    res["transport_disposition"] = criteria_for(res, TRANSPORT_ARM, tdiffs)
    # dose-response
    res["dose_response"] = {arm: {str(t): {"csr": (panel.get(f"{arm}@{t}") or {}).get("applied", {}).get("csr") if (panel.get(f"{arm}@{t}") or {}).get("applied") else None, "achieved_q1": (panel.get(f"{arm}@{t}") or {}).get("mean_achieved_q1"), "dq1": (panel.get(f"{arm}@{t}") or {}).get("mean_dq1"), "dose": res["dose"].get(arm, {}).get(str(t))} for t in C.TAUS} for arm in list(C.ALL_TAU_ARMS) + [C.LOGIT_CONTROL]}
    # witness
    wpath = final_freeze_path.parent / a["witness_weights"]
    if C.sha256_file(wpath) != a["witness_weights_sha256"]:
        raise RuntimeError("witness weights hash mismatch")
    res["witness"] = witness_panel(gpu_root, summaries, worlds, F.load_witness(wpath), [k for k in keys if k[0] in (PRIMARY_ARM, TRANSPORT_ARM, C.NATURAL_INTERPOLATION, C.FULL_PATCH) or k[1] == TAU_KEY])
    res["classification"] = classify(res)
    res["licensed_wording"] = C.LICENSED_WORDING.get(res["classification"]["class"], "no positive claim licensed; report the classification")
    res["per_world"] = {"N_A": NA.tolist(), "N_B": np.stack([s["N"] for s in sigs[("F2", "B")]]).tolist(), "N_Q1_A": [s["N_Q1"] for s in sA], "directions": directions, "achieved_q1": {f"{k[0]}@{k[1]}": [s["achieved_Q1"].get(k) for s in sA] for k in keys if k[0] != C.FULL_PATCH}, "norm_ratio": {f"{arm}@{tau}": [s["arms"].get("F2", {}).get(arm, {}).get("applied", {}).get(tau, {}).get("norm_ratio") for s in summaries] for arm, tau in keys if arm != C.FULL_PATCH}}
    return res


def actor_markdown(res: dict[str, Any]) -> str:
    def bs(b: dict[str, Any] | None) -> str:
        return "—" if not b else f"{b['mean']:.3f} [{b['ci_low']:.3f}, {b['ci_high']:.3f}]"

    L = [f"# {res['actor']} — {res['split']} worlds (n={res['worlds']}), protocol {res['protocol']}", "", f"Classification: **{res['classification']['class']}**" + (f" (flags: {', '.join(res['classification']['flags'])})" if res["classification"]["flags"] else ""), "", f"Licensed wording: {res['licensed_wording']}", "", "## Natural reference gates", "", "| gate | value | threshold | pass |", "|---|---|---|---|"]
    for k, v in res["reference"]["checks"].items():
        val = v["value"]
        val = f"{val['mean']:.3f} [{val['ci_low']:.3f}, {val['ci_high']:.3f}]" if isinstance(val, dict) else (f"{val:.3f}" if isinstance(val, float) else str(val))
        thr = v.get("min", v.get("max", f"mean>={v.get('mean_min')} & ci_low>={v.get('ci_low_min')}"))
        L.append(f"| {k} | {val} | {thr} | {v['pass']} |")
    L += ["", f"Direction: **{res['direction']['status']}**" + ("" if not res["direction"].get("qualification") else " — " + ", ".join(f"{k} {v['value']:.3f} (min {v['min']})" for k, v in res["direction"]["qualification"]["checks"].items())), ""]
    for tag, key in (("Primary inferential object", "primary_disposition"), ("Transport / replication arm", "transport_disposition")):
        d = res.get(key, {})
        if d.get("available"):
            L += ["", f"### {tag}: {d['arm']} at tau={C.PRIMARY_TAU} — **{d['disposition']}**", "", f"- CSR {bs(d['csr'])}, focal {bs(d['csr_focal'])}, invariance {bs(d['csr_invariance'])}, cross-template {bs(d['cross_template_csr'])}", f"- dose valid: {d['dose']['dose_valid']} (dose-valid worlds {d['dose']['fraction_dose_valid']:.2f}, norm ratio median {d['dose']['norm_ratio']['median']}, p90 {d['dose']['norm_ratio']['p90']})", f"- natural full-state interpolation control: {bs(d['interpolation_csr'])}; half cross-template ceiling threshold {d['half_ctnc_threshold']:.3f}", f"- beats comparators (Holm): {d['beats_comparators_holm']}", f"- licensed wording: {d['licensed_wording']}", ""]
        elif key in res:
            L += ["", f"### {tag}: not available (arm not applied on enough worlds)", ""]
    L += ["## Dose (Q1 calibration only)", "", "| arm | tau | applied | dose-valid | achieved Q1 | target Q1 | norm ratio median | p90 | valid |", "|---|---|---|---|---|---|---|---|---|"]
    for arm, per in res["dose"].items():
        for tau, d in sorted(per.items(), key=lambda kv: float(kv[0])):
            nr = d.get("norm_ratio", {})
            L.append(f"| {arm} | {tau} | {d['fraction_applied']:.2f} | {d['fraction_dose_valid']:.2f} | {'' if d.get('achieved_q1_mean') is None else f'{d['achieved_q1_mean']:.2f}'} | {'' if d.get('target_q1_mean') is None else f'{d['target_q1_mean']:.2f}'} | {'' if nr.get('median') is None else f'{nr['median']:.2f}'} | {'' if nr.get('p90') is None else f'{nr['p90']:.2f}'} | {d.get('dose_valid')} |")
    L += ["", "## CSR panel (realization A, primary format; applied-world set)", "", "| arm@tau | n | CSR | focal | invariance | cross-template | mapped format | mean dQ1 |", "|---|---|---|---|---|---|---|---|"]
    for name, e in res["csr"].items():
        if name == "NATURAL" or not e.get("applied"):
            continue
        a_ = e["applied"]
        L.append(f"| {name} | {a_['n']} | {bs(a_['csr'])} | {bs(a_['csr_focal'])} | {bs(a_['csr_invariance'])} | {bs(e.get('cross_template', {}).get('csr') if e.get('cross_template') else None)} | {bs(e.get('mapped_format', {}).get('csr') if e.get('mapped_format') else None)} | {0.0 if e.get('mean_dq1') is None else e['mean_dq1']:.2f} |")
    L += ["", f"## Paired CSR differences at tau={C.PRIMARY_TAU} (primary {PRIMARY_ARM} minus comparator; Holm)", "", "| comparator | difference | Holm p | primary exceeds | comparator valid (coverage) |", "|---|---|---|---|---|"]
    for c in COMPARATORS:
        d = res["paired"].get(c)
        v = res["comparators"][c]
        L.append(f"| {c} | {bs(d)} | {'' if not d else f'{d['holm']['p_holm']:.4f}'} | {'' if not d else d['primary_exceeds_comparator_holm']} | {v['valid']} ({v['coverage']:.2f}) |")
    if res["additional_paired"]:
        L += ["", "| additional paired difference | value |", "|---|---|"] + [f"| {k} | {bs(v)} |" for k, v in res["additional_paired"].items()]
    L += ["", "## Dose-response", "", "| arm | tau | mean achieved Q1 | CSR |", "|---|---|---|---|"]
    for arm, per in res["dose_response"].items():
        for t, v in per.items():
            if v["csr"]:
                L.append(f"| {arm} | {t} | {'' if v['achieved_q1'] is None else f'{v['achieved_q1']:.2f}'} | {bs(v['csr'])} |")
    w = res["witness"]
    L += ["", f"## Template-matched witness: {w['status']} (BA {w['gate']['balanced_accuracy']['value']:.3f}, cross-template BA {w['gate']['cross_template_balanced_accuracy']['value']:.3f})", ""]
    if w["status"] == "WITNESS_VALID":
        L += ["| intervention | witness signature recovery (Q1-Q4) | invariance movement (natural %.3f) |" % w["natural_invariance_witness_movement_mean"], "|---|---|---|"] + [f"| {k} | {bs(v['witness_signature_recovery'])} | {v['invariance_witness_movement_mean']:.3f} |" for k, v in w["interventions"].items()]
    else:
        L.append(w.get("note", ""))
    L += ["", "## Mean standardized signatures (focal dims aligned so the expected natural direction is positive)", "", "| condition | " + " | ".join(C.CONSEQUENCE_QUERIES) + " |", "|---|" + "---|" * 5]
    for name in ["NATURAL"] + [k for k in res["csr"] if k != "NATURAL" and res["csr"][k].get("applied")]:
        s = res["csr"][name]["mean_aligned_standardized_signature"]
        L.append(f"| {name} | " + " | ".join(f"{s[q]:.2f}" for q in C.CONSEQUENCE_QUERIES) + " |")
    return "\n".join(L) + "\n"


def final(actor: str, gpu_root: Path, freeze_root: Path, final_freeze_path: Path, output_root: Path, split: str) -> dict[str, Any]:
    res = analyze_actor(actor, gpu_root, freeze_root, final_freeze_path, split)
    output_root.mkdir(parents=True, exist_ok=True)
    C.write_json(output_root / f"ACTOR_RESULT_{actor}.json", res)
    (output_root / f"ACTOR_RESULT_{actor}.md").write_text(actor_markdown(res), encoding="utf-8")
    p = res["csr"].get(f"{PRIMARY_ARM}@{TAU_KEY}", {}).get("dose_valid", {})
    print(json.dumps({"actor": actor, "class": res["classification"]["class"], "flags": res["classification"]["flags"], "reference_all_pass": res["reference"]["all_pass"], "dose": {k: v["dose_valid"] for k, v in res["dose"].get(PRIMARY_ARM, {}).items()}, "primary_csr": p.get("csr"), "witness": res["witness"]["status"]}, indent=2, default=str))
    return res


# ---------------------------------------------------------------- joint
def figures(results: dict[str, dict[str, Any]], out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    actors = list(results)
    written: list[str] = []
    # 1. dose-response: CSR vs achieved Q1 control
    fig, axes = plt.subplots(1, len(actors), figsize=(6 * len(actors), 4.5), squeeze=False)
    for ax, actor in zip(axes[0], actors):
        r = results[actor]
        for arm, per in r["dose_response"].items():
            xs, ys, lo, hi = [], [], [], []
            for t, v in per.items():
                if v["csr"] and v["achieved_q1"] is not None:
                    xs.append(v["achieved_q1"])
                    ys.append(v["csr"]["mean"])
                    lo.append(v["csr"]["mean"] - v["csr"]["ci_low"])
                    hi.append(v["csr"]["ci_high"] - v["csr"]["mean"])
            if xs:
                ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, label=arm)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("mean achieved Q1 margin (nats)")
        ax.set_ylabel("CSR")
        ax.set_title(f"{actor}: dose-response")
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = out / "fig1_dose_response.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p.name)
    # 2. CSR by arm at the primary tau
    fig, ax = plt.subplots(figsize=(12, 4.5))
    names = [n for n in results[actors[0]]["csr"] if n.endswith(f"@{TAU_KEY}") or n == C.FULL_PATCH]
    x = np.arange(len(names))
    wdt = 0.8 / len(actors)
    for k, actor in enumerate(actors):
        panel = results[actor]["csr"]
        means = [panel.get(n, {}).get("applied", {}).get("csr", {}).get("mean", np.nan) if panel.get(n, {}).get("applied") else np.nan for n in names]
        lo = [panel[n]["applied"]["csr"]["mean"] - panel[n]["applied"]["csr"]["ci_low"] if panel.get(n, {}).get("applied") else 0 for n in names]
        hi = [panel[n]["applied"]["csr"]["ci_high"] - panel[n]["applied"]["csr"]["mean"] if panel.get(n, {}).get("applied") else 0 for n in names]
        ax.bar(x + k * wdt, means, wdt, yerr=[lo, hi], capsize=3, label=actor)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x + wdt * (len(actors) - 1) / 2)
    ax.set_xticklabels([n.replace(f"@{TAU_KEY}", "") for n in names], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("CSR (world bootstrap 95%)")
    ax.legend()
    fig.tight_layout()
    p = out / "fig2_csr_by_arm.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p.name)
    # 3. focal vs invariance at the primary tau
    fig, axes = plt.subplots(1, len(actors), figsize=(5.5 * len(actors), 4.5), squeeze=False)
    for ax, actor in zip(axes[0], actors):
        panel = results[actor]["csr"]
        for n in names:
            e = panel.get(n, {}).get("applied")
            if not e:
                continue
            ax.errorbar(e["csr_focal"]["mean"], e["csr_invariance"]["mean"], xerr=[[e["csr_focal"]["mean"] - e["csr_focal"]["ci_low"]], [e["csr_focal"]["ci_high"] - e["csr_focal"]["mean"]]], yerr=[[e["csr_invariance"]["mean"] - e["csr_invariance"]["ci_low"]], [e["csr_invariance"]["ci_high"] - e["csr_invariance"]["mean"]]], fmt="o", label=n.replace(f"@{TAU_KEY}", ""))
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("focal CSR")
        ax.set_ylabel("invariance CSR")
        ax.set_title(actor)
        ax.legend(fontsize=6)
    fig.tight_layout()
    p = out / "fig3_focal_vs_invariance.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p.name)
    # 4. signature heatmap
    fig, axes = plt.subplots(1, len(actors), figsize=(6 * len(actors), 4.5), squeeze=False)
    for ax, actor in zip(axes[0], actors):
        panel = results[actor]["csr"]
        conds = ["NATURAL"] + [n for n in names if panel.get(n, {}).get("applied")]
        M = np.array([[panel[c]["mean_aligned_standardized_signature"][q] for q in C.CONSEQUENCE_QUERIES] for c in conds])
        vmax = float(np.max(np.abs(M))) or 1.0
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(5))
        ax.set_xticklabels(C.CONSEQUENCE_QUERIES)
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels([c.replace(f"@{TAU_KEY}", "") for c in conds], fontsize=6)
        ax.set_title(f"{actor}: mean standardized signature")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    p = out / "fig4_signature_heatmap.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p.name)
    # 5. norm ratio distribution
    fig, axes = plt.subplots(1, len(actors), figsize=(5.5 * len(actors), 4), squeeze=False)
    for ax, actor in zip(axes[0], actors):
        nr = results[actor]["per_world"]["norm_ratio"]
        keys = [k for k in nr if k.endswith(f"@{TAU_KEY}")]
        data = [[v for v in nr[k] if v is not None] for k in keys]
        data = [d if d else [np.nan] for d in data]
        ax.boxplot(data, tick_labels=[k.replace(f"@{TAU_KEY}", "") for k in keys])
        ax.axhline(2.0, color="r", lw=0.5, ls="--")
        ax.axhline(4.0, color="r", lw=0.5, ls=":")
        ax.set_ylabel("intervention L2 / natural full-state L2")
        ax.set_title(f"{actor}: norm ratio at tau={C.PRIMARY_TAU}")
        ax.tick_params(axis="x", rotation=40, labelsize=6)
    fig.tight_layout()
    p = out / "fig5_norm_ratios.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p.name)
    return written


def joint_class(classes: dict[str, str]) -> str:
    """Joint labels key on assay validity, as the goal defines them ('both assays are valid but actor
    classifications differ'). DIRECTION_NOT_QUALIFIED is an outcome for the primary direction, not an
    invalid assay, so it counts as a valid actor here and is reported with its own disposition."""
    valid = {a: c for a, c in classes.items() if c not in ("REFERENCE_ASSAY_INVALID", "ASSAY_NOT_QUALIFIED")}
    if len(valid) == 2:
        return "MODEL_DEPENDENT" if len(set(valid.values())) == 2 else f"REPLICATED_{list(valid.values())[0]}"
    if len(valid) == 1:
        return "SINGLE_VALID_ACTOR"
    return "NO_VALID_ACTOR"


def joint(analysis_root: Path, freeze_root: Path, final_freeze_path: Path, output_root: Path, dev_root: Path, execution_notes: Path | None) -> dict[str, Any]:
    results = {a: C.read_json(analysis_root / f"ACTOR_RESULT_{a}.json") for a in C.ACTORS if (analysis_root / f"ACTOR_RESULT_{a}.json").exists()}
    fz = C.read_json(final_freeze_path)
    for a, e in fz["actors"].items():
        if not e["assay_qualified"] and a not in results:
            results[a] = {"actor": a, "protocol": e["protocol"], "classification": {"class": "ASSAY_NOT_QUALIFIED", "flags": [], "reason": "no predeclared protocol passed the five natural gates on the selection worlds"}, "licensed_wording": "no positive claim licensed; report the classification", "protocol_selection_sha256": e["protocol_selection_sha256"], "worlds": 0}
    output_root.mkdir(parents=True, exist_ok=True)
    classes = {a: r["classification"]["class"] for a, r in results.items()}
    jc = joint_class(classes)
    figs = figures({a: r for a, r in results.items() if "csr" in r}, output_root / "figures")
    notes = C.read_json(execution_notes) if execution_notes else {}
    headline = {}
    for a, r in results.items():
        if "csr" not in r:
            headline[a] = {"status": r["classification"]["class"]}
            continue
        panel = r["csr"]
        headline[a] = {
            "protocol": r["protocol"], "worlds": r["worlds"], "reference_all_pass": r["reference"]["all_pass"],
            "reference_checks": {k: v["value"] for k, v in r["reference"]["checks"].items()},
            "direction": r["direction"], "dose": r["dose"], "comparators": r["comparators"],
            "csr": {n: {k: e["applied"][k] for k in ("csr", "csr_focal", "csr_invariance", "n")} for n, e in panel.items() if n != "NATURAL" and e.get("applied")},
            "primary_dose_valid_csr": panel.get(f"{PRIMARY_ARM}@{TAU_KEY}", {}).get("dose_valid", {}).get("csr"),
            "cross_template": {n: e["cross_template"]["csr"] for n, e in panel.items() if n != "NATURAL" and e.get("cross_template")},
            "mapped_format": {n: e["mapped_format"]["csr"] for n, e in panel.items() if n != "NATURAL" and e.get("mapped_format")},
            "by_direction": r["by_direction"], "paired": r["paired"], "paired_transport": r.get("paired_transport", {}), "additional_paired": r["additional_paired"], "pairwise_matched": r["pairwise_matched"],
            "primary_disposition": r.get("primary_disposition"), "transport_disposition": r.get("transport_disposition"),
            "dose_response": r["dose_response"], "witness": {"status": r["witness"]["status"], "gate": r["witness"]["gate"], "recovery": {k: v["witness_signature_recovery"] for k, v in r["witness"]["interventions"].items()}},
            "natural": panel["NATURAL"],
        }
    summary = {
        "schema_version": f"{C.SCHEMA_PREFIX}_STRONG_TARGET_RESULTS", "study_id": C.STUDY_ID, "terminal_status": f"{C.STUDY_ID}_COMPLETE",
        "closed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "actor_classifications": {a: r["classification"] for a, r in results.items()}, "joint_classification": jc,
        "licensed_wording": {a: r.get("licensed_wording", "no positive claim licensed; report the classification") for a, r in results.items()},
        "headline": headline,
        "freeze": {"corpus_freeze_manifest_sha256": C.sha256_file(freeze_root / "artifact_manifest.json"), "final_analysis_freeze_sha256": C.sha256_file(final_freeze_path), "protocol_selection": {a: e.get("protocol_selection_sha256") for a, e in fz["actors"].items()}, "directions": {a: e.get("directions_file_sha256") for a, e in fz["actors"].items()}, "witness": {a: e.get("witness_weights_sha256") for a, e in fz["actors"].items()}},
        "figures": figs, "execution_notes": notes,
    }
    C.write_json(output_root / "STRONG_TARGET_RESULTS.json", summary)
    (output_root / "JOINT_REPORT.md").write_text(joint_report_md(summary, results), encoding="utf-8")
    (output_root / "MANUSCRIPT_CLAIM_MATRIX.md").write_text(claim_matrix_md(summary, results), encoding="utf-8")
    packet = review_packet(summary, results, freeze_root, final_freeze_path, notes)
    C.write_json(output_root / "external_review_packet_v1.json", packet)
    (output_root / "external_review_packet_v1.md").write_text(packet_md(packet), encoding="utf-8")
    print(json.dumps({"actor_classes": classes, "joint": jc}, indent=2))
    return summary


def fmt_bs(b: dict[str, Any] | None) -> str:
    return "—" if not b else f"{b['mean']:.3f} [{b['ci_low']:.3f}, {b['ci_high']:.3f}]"


def joint_report_md(s: dict[str, Any], results: dict[str, Any]) -> str:
    L = [f"# Joint report — {C.STUDY_ID}", "", f"Terminal status: **{s['terminal_status']}** ({s['closed_at']})", "", f"Joint classification: **{s['joint_classification']}**", "", "## Actor-level terminal classifications", ""]
    for a, c in s["actor_classifications"].items():
        L.append(f"- **{a}**: `{c['class']}`" + (f" — flags: {', '.join(c['flags'])}" if c.get("flags") else "") + (f"; reason: {c.get('reason')}" if c.get("reason") else ""))
    L += ["", "## Licensed wording", ""] + [f"- {a}: {w}" for a, w in s["licensed_wording"].items()] + ["", "## Headline numbers (final worlds, realization A, primary format)", ""]
    for a, h in s["headline"].items():
        if "csr" not in h:
            L += [f"### {a}", "", f"- {h['status']}", ""]
            continue
        L += [f"### {a} (protocol {h['protocol']}, n={h['worlds']})", "", "- reference gates all pass: " + str(h["reference_all_pass"]) + "; " + "; ".join(f"{k} = {v if not isinstance(v, dict) else round(v['mean'], 3)}" for k, v in h["reference_checks"].items()), f"- direction: {h['direction']['status']}", "", "| arm@tau | n | CSR | focal | invariance | cross-template | mapped |", "|---|---|---|---|---|---|---|"]
        for n, v in h["csr"].items():
            L.append(f"| {n} | {v['n']} | {fmt_bs(v['csr'])} | {fmt_bs(v['csr_focal'])} | {fmt_bs(v['csr_invariance'])} | {fmt_bs(h['cross_template'].get(n))} | {fmt_bs(h['mapped_format'].get(n))} |")
        L += ["", f"- primary object ({PRIMARY_ARM} at tau={C.PRIMARY_TAU}) on its dose-valid subset: {fmt_bs(h['primary_dose_valid_csr'])}", "", "| dose | tau | applied | dose-valid | norm ratio median / p90 | valid |", "|---|---|---|---|---|---|"]
        for arm, per in h["dose"].items():
            for t, d in sorted(per.items(), key=lambda kv: float(kv[0])):
                nr = d.get("norm_ratio", {})
                L.append(f"| {arm} | {t} | {d['fraction_applied']:.2f} | {d['fraction_dose_valid']:.2f} | {'' if nr.get('median') is None else round(nr['median'], 2)} / {'' if nr.get('p90') is None else round(nr['p90'], 2)} | {d.get('dose_valid')} |")
        L += ["", "| comparator | primary minus comparator | Holm p | primary exceeds | valid |", "|---|---|---|---|---|"]
        for c in COMPARATORS:
            d = h["paired"].get(c)
            L.append(f"| {c} | {fmt_bs(d)} | {'' if not d else round(d['holm']['p_holm'], 4)} | {'' if not d else d['primary_exceeds_comparator_holm']} | {h['comparators'][c]['valid']} ({h['comparators'][c]['coverage']:.2f}) |")
        if h["additional_paired"]:
            L += ["", "| additional paired | value |", "|---|---|"] + [f"| {k} | {fmt_bs(v)} |" for k, v in h["additional_paired"].items()]
        for tag, key in (("primary inferential object", "primary_disposition"), ("transport / replication arm", "transport_disposition")):
            d = h.get(key) or {}
            if d.get("available"):
                L += ["", f"- {tag} ({d['arm']}, tau={C.PRIMARY_TAU}): **{d['disposition']}** — CSR {fmt_bs(d['csr'])}, focal {fmt_bs(d['csr_focal'])}, invariance {fmt_bs(d['csr_invariance'])}, cross-template {fmt_bs(d['cross_template_csr'])}, dose valid {d['dose']['dose_valid']} (norm ratio median {d['dose']['norm_ratio']['median']}, p90 {d['dose']['norm_ratio']['p90']}); natural interpolation control {fmt_bs(d['interpolation_csr'])}"]
        L += ["", "- CSR by flip direction at the primary tau: " + ", ".join(f"{d}: {v['mean']:.3f}" for d, v in h["by_direction"].get(f"{PRIMARY_ARM}@{TAU_KEY}", {}).items()), f"- witness: {h['witness']['status']} (BA {h['witness']['gate']['balanced_accuracy']['value']:.3f}, cross-template {h['witness']['gate']['cross_template_balanced_accuracy']['value']:.3f})", ""]
    L += ["## Figures", ""] + [f"- figures/{f}" for f in s["figures"]] + ["", "## Execution notes", "", "```json", json.dumps(s["execution_notes"], indent=2), "```", ""]
    L += ["## Not licensed", "", "- 'mind versus mouth', belief editing, hidden knowledge, or truth readings", "- that a vector is the concept itself, that the selected layer is the unique mechanism, or a global 'more concepts than control knobs' claim", "- any reinterpretation of CRIF v1, relation-program v2, V3, Claim 2, C9 or RRRD verdicts", "- converting an invalid actor into a negative scientific result", "", "## Artifact pickup", "", f"- corpus freeze manifest SHA-256 `{s['freeze']['corpus_freeze_manifest_sha256']}`", f"- final-analysis freeze SHA-256 `{s['freeze']['final_analysis_freeze_sha256']}`", ""]
    return "\n".join(L)


def claim_matrix_md(s: dict[str, Any], results: dict[str, Any]) -> str:
    rows = [(k, C.LICENSED_WORDING.get(k, C.INTERPRETATIONS.get(k, ""))) for k in ("COUNTERFACTUAL_FIDELITY_SUPPORTED", "STRONG_CONTROL_WITHOUT_COUNTERFACTUAL_FIDELITY", "TARGET_EFFECT_CAUSALLY_UNDERIDENTIFIED", "TARGET_REQUIRES_EXTREME_DISPLACEMENT", "STRONG_TARGET_UNREACHABLE", "FORMAT_SHIFT_EXPLAINS_V1", "REFERENCE_ASSAY_INVALID", "ASSAY_NOT_QUALIFIED", "DIRECTION_NOT_QUALIFIED", "PARTIAL_OR_UNDERDETERMINED")]
    actors = list(results)
    L = [f"# Manuscript claim matrix — {C.STUDY_ID}", "", f"Joint classification: **{s['joint_classification']}**", "", "| classification | licensed wording / meaning | " + " | ".join(actors) + " |", "|---|---|" + "---|" * len(actors)]
    for cls, wording in rows:
        cells = []
        for a in actors:
            c = results[a]["classification"]
            cells.append("TRIGGERED" if c["class"] == cls else ("FLAG" if any(f.startswith(cls) for f in c.get("flags", [])) else "—"))
        L.append(f"| {cls} | {wording} | " + " | ".join(cells) + " |")
    L += ["", "## Prohibited stronger wording", "", "- 'the model's mind did not change' / 'steering only hacks the mouth'", "- 'the vector is the concept itself'; 'the selected layer is the unique mechanism'; global 'more concepts than control knobs'", "- the external relation label as truth, belief or hidden knowledge; any multi-agent or contagion claim", "- any claim that an invalid or unqualified actor produced a negative scientific result", ""]
    return "\n".join(L)


def review_packet(s: dict[str, Any], results: dict[str, Any], freeze_root: Path, final_freeze_path: Path, notes: dict[str, Any]) -> dict[str, Any]:
    contract = C.read_json(freeze_root / "STUDY_CONTRACT.json")
    per = {}
    for a, r in results.items():
        if "csr" not in r:
            per[a] = {"class": r["classification"]}
            continue
        per[a] = {"class": r["classification"], "protocol": r["protocol"], "reference": r["reference"]["checks"], "direction": r["direction"], "dose": r["dose"], "comparators": r["comparators"], "csr": {n: {k: e["applied"][k] for k in ("csr", "csr_focal", "csr_invariance", "n")} for n, e in r["csr"].items() if n != "NATURAL" and e.get("applied")}, "primary_dose_valid": r["csr"].get(f"{PRIMARY_ARM}@{TAU_KEY}", {}).get("dose_valid", {}).get("csr"), "cross_template": {n: e["cross_template"]["csr"] for n, e in r["csr"].items() if n != "NATURAL" and e.get("cross_template")}, "mapped_format": {n: e["mapped_format"]["csr"] for n, e in r["csr"].items() if n != "NATURAL" and e.get("mapped_format")}, "paired": r["paired"], "paired_transport": r.get("paired_transport", {}), "primary_disposition": r.get("primary_disposition"), "transport_disposition": r.get("transport_disposition"), "additional_paired": r["additional_paired"], "pairwise_matched": r["pairwise_matched"], "by_direction": r["by_direction"], "dose_response": r["dose_response"], "witness": r["witness"], "scales": r["scales"], "worlds": r["worlds"]}
    return {
        "schema_version": f"{C.SCHEMA_PREFIX}_EXTERNAL_REVIEW_PACKET_V1", "study_id": C.STUDY_ID, "terminal_status": "EXTERNAL_REVIEW_PACKET_READY",
        "1_research_question": {"question": contract["question"], "hypothesis": "when rank-1 relation steering converts the direct answer with counterfactual-like strength (tau=0.75 of the natural Q1 change, correct hard answer, margin >= 1 nat, within the frozen norm budget), it reproduces the held-out Q2-Q6 consequences of changing the relation", "target_variable": "COUNTERFACTUAL_SIGNATURE_RECOVERY (CSR) over Q2-Q6, CRIF v1 definition unchanged", "expected_positive": contract["interpretations"]["COUNTERFACTUAL_FIDELITY_SUPPORTED"], "expected_falsifying": contract["interpretations"]["STRONG_CONTROL_WITHOUT_COUNTERFACTUAL_FIDELITY"], "frozen_thresholds": {"reference": C.REFERENCE_GATES, "direction": C.DIRECTION_GATES, "dose": C.DOSE_VALIDITY, "witness": C.WITNESS_GATES, "taus": list(C.TAUS), "primary_tau": C.PRIMARY_TAU}},
        "2_capability_delta": {"new_runnable_behavior": ["scripts/stcrf_v2_build_corpus.py: fresh 192-world corpus (selection/fit/final) with two realizations and three 8-shot protocols", "scripts/stcrf_v2_gpu_run.py: strong-target alpha-grid calibration and tau-controlled steering at the frozen site, natural full-state interpolation arm, dual-layer activation capture, durable per-world checkpoints", "scripts/stcrf_v2_fit.py: protocol qualification, regime-matched direction fit and cross-validated qualification, template-matched witness, calibration and final-analysis freeze", "scripts/stcrf_v2_analysis.py: dose validity, CSR panel per arm and tau, dose-response, Holm-corrected comparisons, witness panel, classifications, figures, packet"], "tests": "tests/test_stcrf_v2.py", "infrastructure": "reused unchanged: CRIF v1 CSR mathematics, Lambda A100 runtime, NFS layout, systemd launcher, verify_and_promote"},
        "3_evidence_topology": {"independent_actors": len(results), "independent_units": "world_id (96 final worlds per actor; 32 selection and 64 fit worlds never enter a final estimate)", "splits": C.SPLIT_SIZES, "realizations": 2, "formats": {"primary": "F2", "robustness": "F1 mapped symbol"}, "clustering": "all queries, arms and taus of a world are one cluster; every interval is a world bootstrap (10,000 replicates)", "training_exposure": "regime-matched direction and witness fit only on the 64 fit worlds; protocol chosen only on the 32 selection worlds; no final-world fitting", "invalid_or_censored": notes.get("invalid_or_censored", "none recorded")},
        "4_exact_empirical_result": per,
        "5_negative_and_surprising_evidence": notes.get("negative_and_surprising", []),
        "6_held_fixed": {"target": "semantic margin logp(ENTAILED) - logp(NOT_ENTAILED)", "context": "frozen corpus and prompt manifest", "model_parameters": {a: C.ACTORS[a]["model_revision"] for a in C.ACTORS}, "site": "primary layer final prompt position", "action_space": "add along a unit direction, or set the full state (patch / interpolation), at one position", "calibration": "Q1 realization A only", "history_cutoff": "corpus freeze precedes all inference", "timing": "selection before fit before final-analysis freeze before final"},
        "7_competing_explanations": notes.get("competing_explanations", []), "8_barrier_audit": notes.get("barrier_audit", []), "9_hypothesis_space": notes.get("hypothesis_space", {}), "10_process_diagnosis": notes.get("process", {}),
        "11_artifact_pickup": {"corpus_freeze_manifest_sha256": s["freeze"]["corpus_freeze_manifest_sha256"], "final_analysis_freeze_sha256": s["freeze"]["final_analysis_freeze_sha256"], "protocol_selection_sha256": s["freeze"]["protocol_selection"], "directions_sha256": s["freeze"]["directions"], "witness_sha256": s["freeze"]["witness"], "gpu_roots": {a: r.get("gpu_root") for a, r in results.items()}, "gpu_manifests": {a: r.get("gpu_manifest_sha256") for a, r in results.items()}} | notes.get("pickup", {}),
        "12_reviewer_questions": notes.get("reviewer_questions", []), "candidate_branches": notes.get("candidate_branches", []),
        "actor_classifications": s["actor_classifications"], "joint_classification": s["joint_classification"],
    }


def packet_md(p: dict[str, Any]) -> str:
    L = [f"# External review packet v1 — {p['study_id']}", "", f"Terminal status: **{p['terminal_status']}**", "", "Actor classifications: " + ", ".join(f"{a}: `{c['class']}`" for a, c in p["actor_classifications"].items()), f"Joint: `{p['joint_classification']}`", ""]
    for key in ("1_research_question", "2_capability_delta", "3_evidence_topology", "5_negative_and_surprising_evidence", "6_held_fixed", "7_competing_explanations", "8_barrier_audit", "9_hypothesis_space", "10_process_diagnosis", "11_artifact_pickup", "12_reviewer_questions", "candidate_branches"):
        L += [f"## {key}", "", "```json", json.dumps(p[key], indent=2, default=str), "```", ""]
    L += ["## 4_exact_empirical_result", "", "See STRONG_TARGET_RESULTS.json and ACTOR_RESULT_<actor>.json (full panels, per-world arrays).", ""]
    for a, r in p["4_exact_empirical_result"].items():
        L += [f"### {a}", ""]
        if "csr" not in r:
            L += [f"- {r['class']['class']}", ""]
            continue
        L += ["| arm@tau | n | CSR | focal | invariance |", "|---|---|---|---|---|"]
        for n, v in r["csr"].items():
            L.append(f"| {n} | {v['n']} | {fmt_bs(v['csr'])} | {fmt_bs(v['csr_focal'])} | {fmt_bs(v['csr_invariance'])} |")
        L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["final", "joint"])
    ap.add_argument("--actor", choices=sorted(C.ACTORS))
    ap.add_argument("--gpu-root", type=Path)
    ap.add_argument("--freeze-root", type=Path, default=C.FREEZE_ROOT)
    ap.add_argument("--final-freeze", type=Path)
    ap.add_argument("--output-root", type=Path)
    ap.add_argument("--analysis-root", type=Path)
    ap.add_argument("--dev-root", type=Path)
    ap.add_argument("--execution-notes", type=Path)
    ap.add_argument("--split", choices=["fit", "final"], default="final")
    args = ap.parse_args(argv)
    if args.mode == "final":
        final(args.actor, args.gpu_root, args.freeze_root, args.final_freeze, args.output_root, args.split)
    else:
        joint(args.analysis_root, args.freeze_root, args.final_freeze, args.output_root, args.dev_root, args.execution_notes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
