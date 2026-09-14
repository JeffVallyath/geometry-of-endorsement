#!/usr/bin/env python3
"""Arm A of RELATION_READOUT_REPORT_DISCRIMINATION_V1: no-new-inference characterization.

PROSPECTIVELY_FROZEN_POST_HOC_DIAGNOSTICS on the retained Unit 6/7 Track A
forwards (scores, full selected-layer activations, frozen direction bundles,
world metadata, frozen splits). No forward pass. Reuses the frozen Track A
analysis code (relation_binding_analysis_v2) for every calibrated readout.

  analyze --actor A   -> <out>/<actor>/ARM_A_<actor>.json  (all five diagnostics)
  assemble            -> the required FROZEN_DIRECTION_TRANSFER_LADDER,
                         PAIRED_SPECIFICITY_RESULTS, RELATION_PRESENCE_POLARITY_RESULTS,
                         MAPPING_REVERSAL_MECHANISM_RESULTS, ADDITIVE_HOLDER_SIGN_RESULTS
                         {md,json,csv}, G3_REPORTING_AMENDMENT.md, external_review_packet_v1,
                         artifact_manifest.json

Every per-world statistic is bootstrapped with one shared 10,000-replicate
multinomial resampling of the 48 evaluation worlds so paired differences are
exact world-paired differences.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import relation_binding_analysis_v2 as RBA  # noqa: E402
import rrrd_v1_common as C  # noqa: E402

CORE = ("same_context_dual_query_reversal", "holder_swap_correct", "target_swap_correct")
TF_ENDPOINTS = ("operator_sign_world_auroc", "same_context_dual_query_reversal", "holder_swap_correct", "target_swap_correct", "role_permutation_transfer", "order_crossed_transfer", "abs_context_conditioned_effect")
SEMANTIC = ("raw_relation_dim", "jointly_residualized_relation", "sentiment", "factual", "answer_token")
COMPARATORS = ("sentiment", "factual", "answer_token")
FAMILIES = {"negated": ("NEGATED_STANCE",), "withheld": ("NEUTRAL_OR_WITHHELD_STANCE",), "attribution": ("REPORTER_VERSUS_REPORTED_HOLDER", "QUOTER_VERSUS_QUOTED_HOLDER"), "lexical": ("EXPLICIT_LEXICAL_REALIZATION", "LEXICAL_FREE_OR_IMPLICIT_REALIZATION"), "discourse": ("QUOTER_VERSUS_QUOTED_HOLDER",), "mapping": ("ANSWER_MAPPING_REVERSAL",)}
LADDER_METRICS = {"sign_balanced_accuracy": "operator_sign_balanced_accuracy", "same_context_dual_query_reversal": "same_context_two_query_sign_flip_correct_rate", "holder_swap_correct": "holder_swap_correct_rate", "target_swap_correct": "target_swap_correct_rate", "role_permutation_transfer": "fit_canonical_supports_evaluate_alternate_supports_balanced_accuracy", "order_crossed_transfer": "fit_order_canonical_evaluate_order_reversed_sign_balanced_accuracy", "mapping_reversal_semantic_agreement": "mapping_reversal_semantic_agreement"}


# ---------------------------------------------------------------- bootstrap
class WorldBootstrap:
    def __init__(self, worlds: list[str], replicates: int, seed: int):
        self.worlds = list(worlds)
        self.index = {w: i for i, w in enumerate(self.worlds)}
        rng = np.random.default_rng(seed)
        n = len(self.worlds)
        self.counts = rng.multinomial(n, np.full(n, 1.0 / n), size=replicates).astype(np.float64)

    def vec(self, per_world: dict[str, float]) -> np.ndarray:
        v = np.full(len(self.worlds), np.nan)
        for w, x in per_world.items():
            if w in self.index and x is not None and np.isfinite(x):
                v[self.index[w]] = x
        return v

    def draws(self, per_world: dict[str, float]) -> np.ndarray:
        v = self.vec(per_world)
        avail = np.isfinite(v).astype(float)
        vv = np.where(avail > 0, v, 0.0)
        den = self.counts @ avail
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, (self.counts @ (vv * avail)) / den, np.nan)

    def summary(self, per_world: dict[str, float], gate: float | None = None, kind: str = "min") -> dict[str, Any]:
        v = self.vec(per_world)
        if not np.isfinite(v).any():
            return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "worlds": 0}
        d = self.draws(per_world)
        out = {"point": float(np.nanmean(v)), "ci_low": float(np.nanquantile(d, 0.025)), "ci_high": float(np.nanquantile(d, 0.975)), "worlds": int(np.isfinite(v).sum())}
        if gate is not None:
            side = (d < gate) if kind == "min" else (d > gate)
            out["gate"], out["p_vs_gate"] = gate, float((np.nansum(side) + 1) / (np.isfinite(d).sum() + 1))
        return out

    def paired(self, a: dict[str, float], b: dict[str, float]) -> dict[str, Any]:
        va, vb = self.vec(a), self.vec(b)
        both = np.isfinite(va) & np.isfinite(vb)
        diff = {w: float(va[i] - vb[i]) for w, i in self.index.items() if both[i]}
        s = self.summary(diff)
        d = self.draws(diff)
        s["p_le_zero"] = float((np.nansum(d <= 0) + 1) / (np.isfinite(d).sum() + 1))
        s["p_ge_zero"] = float((np.nansum(d >= 0) + 1) / (np.isfinite(d).sum() + 1))
        s["p_two_sided"] = float(min(1.0, 2 * min(s["p_le_zero"], s["p_ge_zero"])))
        return s


# ---------------------------------------------------------------- per-world metric builders
def per_world_mean(values: dict[str, list[float]]) -> dict[str, float]:
    return {w: float(np.mean(v)) for w, v in values.items() if v}


class Metrics:
    """Threshold-free and rule-based per-world statistics on evaluation worlds."""

    def __init__(self, t: RBA.Table):
        self.t = t
        self.ev = t.eval
        self.eval_all = t.split == "heldout_evaluation"
        self.by_contrast: dict[str, list[int]] = collections.defaultdict(list)
        for i in np.where(self.eval_all)[0]:
            self.by_contrast[t.contrast[i]].append(int(i))
        self.pairs_by_family: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        for cid, idx in self.by_contrast.items():
            if len(idx) == 2:
                m0, m1 = sorted(idx, key=lambda i: t.member[i])
                self.pairs_by_family[t.family[m0]].append((m0, m1))
        self.world_rows: dict[str, np.ndarray] = {w: np.where(self.eval_all & (t.world == w))[0] for w in np.unique(t.world[self.eval_all])}

    # -- threshold-free
    def opposite_pair_correct(self, proj: np.ndarray, families: tuple[str, ...], restrict: Callable[[int, int], bool] | None = None) -> dict[str, float]:
        t = self.t
        vals: dict[str, list[float]] = collections.defaultdict(list)
        for fam in families:
            for m0, m1 in self.pairs_by_family.get(fam, []):
                if not (t.signed[m0] and t.signed[m1]) or t.sign[m0] == t.sign[m1] or not (t.elig[m0] == "FINAL_EVALUATION"):
                    continue
                if restrict is not None and not restrict(m0, m1):
                    continue
                s, o = (m0, m1) if t.sign[m0] == 1 else (m1, m0)
                vals[t.world[m0]].append(1.0 if proj[s] > proj[o] else (0.5 if proj[s] == proj[o] else 0.0))
        return per_world_mean(vals)

    def world_auroc(self, proj: np.ndarray, row_mask: np.ndarray | None = None) -> dict[str, float]:
        t = self.t
        out = {}
        for w, rows in self.world_rows.items():
            rows = rows[(t.elig[rows] == "FINAL_EVALUATION") & t.signed[rows]]
            if row_mask is not None:
                pos = rows[row_mask[rows] & (t.sign[rows] == 1)]
                neg_all = rows[t.sign[rows] == 0]
                neg = neg_all
                pos_all = rows[t.sign[rows] == 1]
                # rows in the mask are compared against every opposite-label signed row of the world
                masked = rows[row_mask[rows]]
                if len(masked) == 0:
                    continue
                score = []
                for i in masked:
                    opp = neg_all if t.sign[i] == 1 else pos_all
                    if len(opp) == 0:
                        continue
                    if t.sign[i] == 1:
                        score.append(float(np.mean((proj[i] > proj[opp]) + 0.5 * (proj[i] == proj[opp]))))
                    else:
                        score.append(float(np.mean((proj[i] < proj[opp]) + 0.5 * (proj[i] == proj[opp]))))
                if score:
                    out[w] = float(np.mean(score))
                continue
            pos, neg = rows[t.sign[rows] == 1], rows[t.sign[rows] == 0]
            if len(pos) == 0 or len(neg) == 0:
                continue
            diff = proj[pos][:, None] - proj[neg][None, :]
            out[w] = float(np.mean((diff > 0) + 0.5 * (diff == 0)))
        return out

    def abs_context_effect(self, proj: np.ndarray) -> dict[str, float]:
        t = self.t
        hs, qo = {}, {}
        for i, r in enumerate(t.records):
            if r["variant_family"] == "HOLDER_SWAP":
                hs.setdefault(r["world_id"], {})[int(r["contrast_member"])] = proj[i]
            elif r["variant_family"] == "QUERY_ONLY":
                qo.setdefault(r["world_id"], {})[int(r["paired_reference_member"])] = proj[i]
        out = {}
        for w in self.world_rows:
            if w in hs and w in qo and len(hs[w]) == 2 and len(qo[w]) == 2:
                out[w] = float(abs((hs[w][0] - hs[w][1]) - (qo[w][0] - qo[w][1])))
        return out

    def threshold_free(self, proj: np.ndarray) -> dict[str, dict[str, float]]:
        t = self.t
        order_mask = (t.family == "SENTENCE_ORDER_SWAP") & (t.member == 1)
        return {
            "operator_sign_world_auroc": self.world_auroc(proj),
            "same_context_dual_query_reversal": self.opposite_pair_correct(proj, ("CONFLICTING_STANCES_IN_ONE_CONTEXT",)),
            "holder_swap_correct": self.opposite_pair_correct(proj, ("HOLDER_SWAP",)),
            "target_swap_correct": self.opposite_pair_correct(proj, ("TARGET_SWAP",)),
            "role_permutation_transfer": self.opposite_pair_correct(proj, ("HOLDER_SWAP", "CONFLICTING_STANCES_IN_ONE_CONTEXT"), restrict=lambda m0, m1: not t.canonical_supports[m0]),
            "order_crossed_transfer": self.world_auroc(proj, order_mask),
            "abs_context_conditioned_effect": self.abs_context_effect(proj),
        }

    # -- rule-based (strict source calibration) readout
    def strict_rule(self, score: np.ndarray) -> Callable[[str, np.ndarray], np.ndarray]:
        t = self.t
        sign = (score > 0).astype(int)
        op_support = np.array([r["queried_operator"] == "SUPPORT" for r in t.records])
        entail = np.where(op_support, sign, 1 - sign)

        def rule(task: str, mask: np.ndarray) -> np.ndarray:
            if task in ("sign", "role_perm", "order"):
                return sign[mask]
            if task == "entail":
                return entail[mask]
            if task == "state":
                return np.where(sign[mask] == 1, "SUPPORT", "OPPOSE").astype(object)
            return np.array([f"{'C' if h == 'CANONICAL_HOLDER' else 'A'}_{'S' if s else 'O'}" for h, s in zip(t.holder_role[mask], sign[mask])], dtype=object)

        return rule


def readout_metrics(ro: RBA.Readout) -> dict[str, dict[str, float]]:
    return ro.world_stats()


def calibrated_sign_pred(t: RBA.Table, proj: np.ndarray) -> np.ndarray:
    """The frozen competitor-battery statistic: 1-D logistic (C=1) on fit rows."""
    fit = t.fit & t.signed
    model = RBA.fit_logistic(proj[fit][:, None], t.sign[fit], 1.0)
    return RBA.predict(model, proj[:, None])


def per_world_ba(t: RBA.Table, y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    m = mask
    return RBA.macro_world_ba(y[m], pred[m], t.world[m]) if m.sum() else {}


# ---------------------------------------------------------------- analyses
def analyze(actor: str, completed_jobs: Path, artifact_root: Path, freeze_root: Path, output_root: Path, replicates: int) -> dict[str, Any]:
    records = RBA.read_jsonl(completed_jobs)
    validation = RBA.validate_completed(records, actor)
    t = RBA.Table(records)
    x = RBA.load_activations(records, artifact_root, verify_hashes=True)
    dim_ledger = np.array([float(r["dim_score"]) for r in records])
    manifest = RBA.json.loads((C.TRACK_A_FREEZE / "TRACK_A_COMPETITOR_DIRECTIONS.json").read_text(encoding="utf-8"))
    entry = next(a for a in manifest["actors"] if a["actor"] == actor)
    bundle = C.TRACK_A_FREEZE / "runtime" / actor / "competitor_directions_v2.npz"
    if C.sha256_file(bundle) != entry["bundle"]["sha256"]:
        raise RuntimeError("direction bundle hash mismatch")
    z = np.load(bundle, allow_pickle=False)
    dirs = {k: np.asarray(z[k], dtype=np.float64) for k in SEMANTIC}
    for k in SEMANTIC:
        if C.sha256_bytes(np.ascontiguousarray(z[k]).tobytes()) != entry["directions"][k]["sha256"]:
            raise RuntimeError(f"direction hash mismatch {k}")
    rand = {k: np.asarray(z[f"matched_norm_random__{k}"], dtype=np.float64) for k in SEMANTIC}
    input_manifest = C.read_json(freeze_root / "INPUT_MANIFEST.json")
    midpoint = float(input_manifest["source_calibration"]["raw_relation_dim"][actor]["difference_in_means_midpoint"])
    proj = {k: x @ v for k, v in dirs.items()}
    ledger_delta = float(np.max(np.abs(proj["raw_relation_dim"] - dim_ledger)))
    if ledger_delta > 1e-3:
        raise RuntimeError(f"recomputed raw DIM projection differs from the ledger dim_score: {ledger_delta}")
    ev_worlds = sorted(set(t.world[t.split == "heldout_evaluation"]))
    boot = WorldBootstrap(ev_worlds, replicates, C.BOOTSTRAP_SEED)
    M = Metrics(t)
    nuis_all, nuis_no_dim = RBA.nuisance_matrix(records, True), RBA.nuisance_matrix(records, False)
    resid = np.zeros_like(x)
    rf, ro_ = RBA.residualize(x[t.fit], x[~t.fit], nuis_all[t.fit], nuis_all[~t.fit])
    resid[t.fit], resid[~t.fit] = rf, ro_
    out: dict[str, Any] = {"schema_version": f"{C.SCHEMA_PREFIX}_ARM_A", "actor": actor, "label": "PROSPECTIVELY_FROZEN_POST_HOC_DIAGNOSTICS", "validation": validation, "evaluation_worlds": len(ev_worlds), "ledger_projection_max_abs_delta": ledger_delta, "source_midpoint": midpoint}

    # ---------------- 4.1 ladder
    ladder: dict[str, Any] = {}
    for dname in ("raw_relation_dim", "jointly_residualized_relation"):
        p = proj[dname]
        levels: dict[str, Any] = {}
        if dname == "raw_relation_dim":
            strict = RBA.RuleReadout("strict", t, M.strict_rule(p - midpoint))
            ws = readout_metrics(strict)
            levels["A_STRICT_SOURCE_CALIBRATION"] = {"score": "activation . direction - midpoint", "midpoint": midpoint, "metrics": {k: boot.summary(ws[v]) for k, v in LADDER_METRICS.items()}}
        else:
            levels["A_STRICT_SOURCE_CALIBRATION"] = {"status": "SOURCE_CALIBRATION_MISSING", "reason": "no authoritative source midpoint/threshold exists for the jointly residualized direction; zero was not substituted"}
        cal = RBA.Readout(f"cal_{dname}", t, p[:, None])
        ws = readout_metrics(cal)
        levels["B_TARGET_SCALAR_CALIBRATION"] = {"calibration": "frozen Track A one-dimensional logistic on fit rows, C on development worlds", "chosen_C": cal.chosen_c, "metrics": {k: boot.summary(ws[v]) for k, v in LADDER_METRICS.items()}}
        tf = M.threshold_free(p)
        levels["C_THRESHOLD_FREE_PAIRED_TRANSFER"] = {"metrics": {k: boot.summary(v) for k, v in tf.items()}, "mapping_reversal_semantic_agreement": "NOT_DEFINED_AT_THIS_LEVEL"}
        ladder[dname] = levels
    full = RBA.Readout("full_activation_logistic", t, x)
    ws_full = readout_metrics(full)
    ladder["target_trained_high_dimensional_decoding_comparator"] = {"readout": "full_activation_logistic (frozen code path)", "metrics": {k: boot.summary(ws_full[v]) for k, v in LADDER_METRICS.items()}}
    out["transfer_ladder"] = ladder

    # ---------------- 4.2 paired specificity
    def endpoint_set(p: np.ndarray) -> dict[str, dict[str, float]]:
        d = M.threshold_free(p)
        d["operator_sign_calibrated_ba"] = per_world_ba(t, t.sign, calibrated_sign_pred(t, p), t.eval & t.signed)
        return d

    sem_eps = {}
    for k in SEMANTIC:
        unit = dirs[k] / np.linalg.norm(dirs[k])
        sem_eps[k] = endpoint_set(x @ unit)
    rand_eps: dict[str, dict[str, list[float]]] = {}
    rand_points: dict[str, dict[str, list[float]]] = {}
    for ref in SEMANTIC:
        rand_points[ref] = collections.defaultdict(list)
        for j in range(rand[ref].shape[0]):
            unit = rand[ref][j] / np.linalg.norm(rand[ref][j])
            eps = endpoint_set(x @ unit)
            for ep, pw in eps.items():
                rand_points[ref][ep].append(float(np.mean(list(pw.values()))) if pw else float("nan"))
    bands = {ref: {ep: {"q025": float(np.nanquantile(v, 0.025)), "q975": float(np.nanquantile(v, 0.975)), "max": float(np.nanmax(v)), "count": len(v), "discriminating_at_0.70": bool(np.nanquantile(v, 0.975) < C.PRACTICAL_GATE) if ep != "abs_context_conditioned_effect" else None} for ep, v in eps.items()} for ref, eps in rand_points.items()}
    points = {k: {ep: boot.summary(pw) for ep, pw in eps.items()} for k, eps in sem_eps.items()}
    rel = sem_eps["jointly_residualized_relation"]
    primary, secondary = {}, {}
    for c in COMPARATORS:
        core_diff: dict[str, list[float]] = collections.defaultdict(list)
        for ep in CORE:
            a, b = rel[ep], sem_eps[c][ep]
            for w in a:
                if w in b:
                    core_diff[w].append(a[w] - b[w])
        primary[c] = boot.summary({w: float(np.mean(v)) for w, v in core_diff.items()})
        d = boot.draws({w: float(np.mean(v)) for w, v in core_diff.items()})
        primary[c]["p_le_zero"] = float((np.nansum(d <= 0) + 1) / (np.isfinite(d).sum() + 1))
        primary[c]["p_ge_zero"] = float((np.nansum(d >= 0) + 1) / (np.isfinite(d).sum() + 1))
        secondary[c] = {ep: boot.paired(rel[ep], sem_eps[c][ep]) for ep in TF_ENDPOINTS + ("operator_sign_calibrated_ba",)}
    holm_pos = RBA.holm({c: primary[c]["p_le_zero"] for c in COMPARATORS})
    holm_neg = RBA.holm({c: primary[c]["p_ge_zero"] for c in COMPARATORS})
    for c in COMPARATORS:
        primary[c]["holm_p_positive"], primary[c]["holm_p_negative"] = holm_pos[c], holm_neg[c]
        primary[c]["significant_positive"] = bool(holm_pos[c] < 0.05 and primary[c]["point"] > 0)
        primary[c]["significant_negative"] = bool(holm_neg[c] < 0.05 and primary[c]["point"] < 0)
    raw_vs_res = {ep: boot.paired(sem_eps["raw_relation_dim"][ep], rel[ep]) for ep in TF_ENDPOINTS}
    nondisc = all(not bands["jointly_residualized_relation"][ep]["discriminating_at_0.70"] for ep in CORE)
    if nondisc:
        spec_class = "RANDOM_NULL_NONDISCRIMINATING"
    elif all(primary[c]["significant_positive"] for c in COMPARATORS):
        spec_class = "RELATION_SPECIFICITY_FAVORED"
    elif not any(primary[c]["significant_positive"] for c in COMPARATORS) and any(primary[c]["significant_negative"] for c in COMPARATORS):
        spec_class = "RELATION_SPECIFICITY_NOT_FAVORED"
    elif any(primary[c]["significant_positive"] for c in COMPARATORS):
        spec_class = "MIXED_SPECIFICITY"
    else:
        spec_class = "UNDERDETERMINED"
    out["paired_specificity"] = {"classification": spec_class, "primary": primary, "secondary_per_endpoint": secondary, "raw_minus_residualized": raw_vs_res, "points": points, "random_bands": bands, "random_calibrated_sign_band_note": "the calibrated sign statistic band is reported; it is non-discriminating when its q97.5 >= 0.70", "direction_normalization": "unit L2 norm for every direction (continuous effect comparability); sign/pair statistics are scale-invariant"}

    # ---------------- 4.3 presence vs polarity
    presence = np.where(t.signed, 1, 0)
    state3 = t.state.copy()
    state3[state3 == "UNSIGNED"] = "ABSENT_OR_WITHHELD"
    fit, dev, ev_all = t.fit, t.dev, (t.split == "heldout_evaluation")

    def fit_task(features: np.ndarray, y: np.ndarray, fit_mask: np.ndarray, dev_mask: np.ndarray):
        if len(np.unique(y[fit_mask])) < 2:
            return None, None
        c, scores = RBA.select_c(features[fit_mask], y[fit_mask], features[dev_mask], y[dev_mask], t.world[dev_mask]) if dev_mask.sum() and len(np.unique(y[dev_mask])) > 1 else (1.0, {})
        return RBA.fit_logistic(features[fit_mask], y[fit_mask], c), c

    praw = proj["raw_relation_dim"]
    feats = {"1_dim_polarity_alone": np.column_stack([praw, np.abs(praw)]), "2_presence_full_activation": x, "3_presence_after_dim_and_nuisance_removal": resid}
    models, chosen = {}, {}
    for name, f in feats.items():
        models[name], chosen[name] = fit_task(f, presence, fit, dev)
    preds = {name: RBA.predict(models[name], feats[name]) for name in feats}
    sign_model, sign_c = fit_task(praw[:, None], t.sign, fit & t.signed, dev & t.signed)
    sign_pred = RBA.predict(sign_model, praw[:, None])
    two_factor = np.where(preds["2_presence_full_activation"] == 1, np.where(sign_pred == 1, "SUPPORT", "OPPOSE"), "ABSENT_OR_WITHHELD").astype(object)
    dim_two_factor = np.where(preds["1_dim_polarity_alone"] == 1, np.where(sign_pred == 1, "SUPPORT", "OPPOSE"), "ABSENT_OR_WITHHELD").astype(object)
    m5, c5 = fit_task(x, state3, fit, dev)
    direct3 = RBA.predict(m5, x)
    fam_masks = {f: np.isin(t.family, fams) for f, fams in FAMILIES.items()}
    pres_res: dict[str, Any] = {"chosen_C": {**chosen, "sign": sign_c, "5_direct_three_state": c5}}
    for name in feats:
        pres_res[name] = {"presence_ba_eval": boot.summary(per_world_ba(t, presence, preds[name], t.eval)), "presence_ba_eval_all_rows": boot.summary(per_world_ba(t, presence, preds[name], ev_all)), "by_family": {f: boot.summary(per_world_ba(t, presence, preds[name], ev_all & fam_masks[f])) for f in FAMILIES}}
    pres_res["4_two_factor_frozen_polarity_plus_learned_presence"] = {"three_state_ba_eval": boot.summary(per_world_ba(t, state3, two_factor, t.eval)), "by_family": {f: boot.summary(per_world_ba(t, state3, two_factor, ev_all & fam_masks[f])) for f in FAMILIES}}
    pres_res["4b_two_factor_dim_only"] = {"three_state_ba_eval": boot.summary(per_world_ba(t, state3, dim_two_factor, t.eval))}
    pres_res["5_direct_three_state"] = {"three_state_ba_eval": boot.summary(per_world_ba(t, state3, direct3, t.eval)), "by_family": {f: boot.summary(per_world_ba(t, state3, direct3, ev_all & fam_masks[f])) for f in FAMILIES}}
    pres_res["paired_model2_minus_model1_presence"] = boot.paired(per_world_ba(t, presence, preds["2_presence_full_activation"], t.eval), per_world_ba(t, presence, preds["1_dim_polarity_alone"], t.eval))
    pres_res["paired_model5_minus_model4_three_state"] = boot.paired(per_world_ba(t, state3, direct3, t.eval), per_world_ba(t, state3, two_factor, t.eval))
    lofo = {}
    direct_eval = ev_all & (t.family == "DIRECT_HOLDER_TARGET_QUERY")
    for f in ("negated", "withheld", "attribution", "lexical"):
        fm = fit & ~fam_masks[f]
        m, c = fit_task(x, presence, fm, dev & ~fam_masks[f])
        pr = RBA.predict(m, x)
        union = (ev_all & fam_masks[f]) | direct_eval
        lofo[f] = {"chosen_C": c, "presence_ba_family_union_direct": boot.summary(per_world_ba(t, presence, pr, union)), "family_absent_recall": boot.summary(per_world_ba(t, presence, pr, ev_all & fam_masks[f] & (presence == 0))) if (ev_all & fam_masks[f] & (presence == 0)).sum() else None}
    pres_res["leave_one_family_out_model2"] = lofo
    p2 = pres_res["2_presence_full_activation"]["presence_ba_eval"]
    lofo_ok = {f: lofo[f]["presence_ba_family_union_direct"]["point"] >= 0.60 for f in ("negated", "withheld")}
    if p2["point"] < 0.60 or p2["ci_low"] <= 0.50:
        pres_class = "BIPOLAR_POLARITY_ONLY"
    elif p2["point"] >= C.PRACTICAL_GATE and p2["ci_low"] > 0.60 and all(lofo_ok.values()) and pres_res["paired_model2_minus_model1_presence"]["ci_low"] > 0:
        pres_class = "PRESENCE_PLUS_POLARITY_SUPPORTED"
    elif p2["point"] >= C.PRACTICAL_GATE and not any(lofo_ok.values()):
        pres_class = "PRESENCE_TEMPLATE_SPECIFIC"
    elif p2["point"] >= C.PRACTICAL_GATE:
        pres_class = "MIXED_PRESENCE_RESULT"
    else:
        pres_class = "UNDERDETERMINED"
    pres_res["classification"] = pres_class
    pres_res["boundary"] = "richer-than-DIM decodability only; not a causal relation operator"
    out["presence_polarity"] = pres_res

    # ---------------- 4.4 mapping reversal mechanism
    mapping_res: dict[str, Any] = {}
    pairs = M.pairs_by_family.get("ANSWER_MAPPING_REVERSAL", [])
    pairs = [(a, b) for a, b in pairs if t.elig[a] == "FINAL_EVALUATION"]
    mapping_ids = np.array([r["answer_mapping_id"] for r in records], dtype=object)
    for dname in ("raw_relation_dim", "jointly_residualized_relation"):
        p = proj[dname]
        sf = fit & t.signed
        mu = {1: float(p[sf & (t.sign == 1)].mean()), 0: float(p[sf & (t.sign == 0)].mean())}
        centered = p - np.where(t.sign == 1, mu[1], mu[0])
        offsets = {m: float(centered[sf & (mapping_ids == m)].mean()) for m in C.MAPPINGS}
        corrected_int = p - np.array([offsets[m] for m in mapping_ids])
        a_pool, b_pool = np.polyfit(np.where(t.sign[sf] == 1, 1.0, -1.0), p[sf], 1)[::-1]
        affine = {}
        for m in C.MAPPINGS:
            sel = sf & (mapping_ids == m)
            slope, icpt = np.polyfit(np.where(t.sign[sel] == 1, 1.0, -1.0), p[sel], 1)
            affine[m] = {"a": float(icpt), "b": float(slope)}
        corrected_aff = np.array([(p[i] - affine[m]["a"]) / affine[m]["b"] * b_pool + a_pool for i, m in enumerate(mapping_ids)])
        variants = {"before": p, "intercept_corrected": corrected_int, "affine_corrected": corrected_aff}
        rows = {}
        for vname, pv in variants.items():
            m0 = np.array([pv[a] for a, b in pairs]); m1 = np.array([pv[b] for a, b in pairs])
            entry_v: dict[str, Any] = {"paired_raw_score_correlation": float(np.corrcoef(m0, m1)[0, 1]) if len(pairs) > 1 else float("nan"), "mean_abs_pair_gap": float(np.mean(np.abs(m0 - m1))) if pairs else float("nan"), "mean_mapping_shift_from_pooled": {m: float(pv[sf & (mapping_ids == m)].mean() - pv[sf].mean()) for m in C.MAPPINGS}}
            tf = M.threshold_free(pv)
            entry_v["threshold_free"] = {k: boot.summary(tf[k]) for k in ("same_context_dual_query_reversal", "holder_swap_correct", "target_swap_correct")}
            cal = RBA.Readout(f"map_{dname}_{vname}", t, pv[:, None])
            ws = readout_metrics(cal)
            entry_v["level_B_calibration_transfer"] = {k: boot.summary(ws[v]) for k, v in LADDER_METRICS.items()}
            entry_v["level_B_mapping_agreement"] = entry_v["level_B_calibration_transfer"]["mapping_reversal_semantic_agreement"]
            if dname == "raw_relation_dim":
                strict = RBA.RuleReadout("strict", t, M.strict_rule(pv - midpoint))
                entry_v["level_A_mapping_agreement"] = boot.summary(readout_metrics(strict)["mapping_reversal_semantic_agreement"])
            rows[vname] = entry_v
        agree_int = rows["intercept_corrected"]["level_B_mapping_agreement"]["point"]
        agree_aff = rows["affine_corrected"]["level_B_mapping_agreement"]["point"]
        if len(pairs) < 20:
            mclass = "UNDERDETERMINED"
        elif agree_int >= C.AGREEMENT_GATE:
            mclass = "MAPPING_OFFSET_EXPLAINS_INSTABILITY"
        elif agree_aff >= C.AGREEMENT_GATE:
            mclass = "MAPPING_AFFINE_SHIFT_EXPLAINS_INSTABILITY"
        elif max(agree_int, agree_aff) < 0.75:
            mclass = "DEEPER_MAPPING_DEPENDENCE"
        else:
            mclass = "MIXED_MAPPING_RESULT"
        mapping_res[dname] = {"classification": mclass, "evaluable_pairs": len(pairs), "intercept_offsets": offsets, "affine": affine, "pooled_affine": {"a": float(a_pool), "b": float(b_pool)}, "variants": rows}
    mapping_res["full_activation_comparator"] = boot.summary(ws_full["mapping_reversal_semantic_agreement"])
    mapping_res["classification"] = mapping_res["raw_relation_dim"]["classification"]
    out["mapping_reversal"] = mapping_res

    # ---------------- 4.5 additive holder + sign
    qo = t.family == "QUERY_ONLY"
    slot = np.where(t.holder_role == "CANONICAL_HOLDER", 1, 0)
    hm, hc = fit_task(x, slot, (t.split == "residual_probe_train") & qo, (t.split == "development") & qo)
    scaler_h, clf_h = hm
    p_holder = clf_h.predict_proba(scaler_h.transform(x))  # columns ordered by classes_
    holder_classes = list(clf_h.classes_)
    scaler_s, clf_s = sign_model
    p_sign = clf_s.predict_proba(scaler_s.transform(praw[:, None]))
    sign_classes = list(clf_s.classes_)
    labels = ["C_S", "C_O", "A_S", "A_O"]
    joint_prob = np.zeros((len(records), 4))
    for j, lab in enumerate(labels):
        hs, ss = (1 if lab[0] == "C" else 0), (1 if lab[2] == "S" else 0)
        joint_prob[:, j] = p_holder[:, holder_classes.index(hs)] * p_sign[:, sign_classes.index(ss)]
    additive = np.array([labels[i] for i in joint_prob.argmax(1)], dtype=object)
    strict_sign = (praw - midpoint > 0).astype(int)
    additive_strict = np.array([f"{'C' if p_holder[i, holder_classes.index(1)] >= 0.5 else 'A'}_{'S' if strict_sign[i] else 'O'}" for i in range(len(records))], dtype=object)
    res_ro = RBA.Readout("residual_activation_logistic", t, resid)
    ws_res = readout_metrics(res_ro)
    pm = "macro_world_joint_binding_balanced_accuracy"
    add_pw = per_world_ba(t, t.joint, additive, t.eval & t.two_actor)
    add_strict_pw = per_world_ba(t, t.joint, additive_strict, t.eval & t.two_actor)
    holder_only = per_world_ba(t, slot, np.where(p_holder[:, holder_classes.index(1)] >= 0.5, 1, 0), t.eval & t.two_actor)
    add_res = {"holder_decoder": {"chosen_C": hc, "training_rows": int(((t.split == "residual_probe_train") & qo).sum()), "eval_two_actor_slot_ba": boot.summary(holder_only)}, "additive_level_B": boot.summary(add_pw), "additive_level_A_strict": boot.summary(add_strict_pw), "full_activation_joint": boot.summary(ws_full[pm]), "residual_activation_joint": boot.summary(ws_res[pm]), "dim_scalar_joint_frozen_reference": boot.summary(readout_metrics(RBA.Readout("dim_scalar", t, dim_ledger[:, None]))[pm]), "paired_full_minus_additive": boot.paired(ws_full[pm], add_pw), "paired_residual_minus_additive": boot.paired(ws_res[pm], add_pw)}
    verdicts = []
    for key in ("paired_full_minus_additive", "paired_residual_minus_additive"):
        d = add_res[key]
        if d["ci_low"] > 0.05:
            verdicts.append("JOINT")
        elif d["ci_high"] < 0.05 and add_res["additive_level_B"]["ci_low"] > 0.25:
            verdicts.append("ADDITIVE")
        else:
            verdicts.append("UNDET")
    if verdicts == ["JOINT", "JOINT"]:
        add_class = "JOINT_INTERACTION_ADDS_INFORMATION"
    elif verdicts == ["ADDITIVE", "ADDITIVE"]:
        add_class = "ADDITIVE_HOLDER_PLUS_SIGN_SUFFICIENT"
    elif "UNDET" in verdicts and len(set(verdicts)) == 1:
        add_class = "UNDERDETERMINED"
    elif len(set(verdicts)) > 1 and "UNDET" not in verdicts:
        add_class = "MIXED_ADDITIVE_RESULT"
    else:
        add_class = "UNDERDETERMINED"
    add_res["classification"] = add_class
    add_res["G1_preserved"] = "BOUND_RELATIONAL_REPRESENTATION (frozen; not re-decided here)"
    out["additive_holder_sign"] = add_res
    out["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return out


# ---------------------------------------------------------------- assemble
def fmt(s: dict[str, Any] | None) -> str:
    if not s or s.get("point") is None or not np.isfinite(s.get("point", float("nan"))):
        return "n/a"
    return f"{s['point']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]"


def assemble(output_root: Path, g3_packet: Path | None) -> None:
    per = {a: C.read_json(output_root / a / f"ARM_A_{a}.json") for a in C.ACTORS if (output_root / a / f"ARM_A_{a}.json").is_file()}
    if not per:
        raise SystemExit("no per-actor Arm A results found")
    actors = list(per)
    # ---- ladder
    ladder_json = {a: per[a]["transfer_ladder"] for a in actors}
    md = ["# Frozen-direction transfer ladder (Arm A 4.1)", "", "Label: PROSPECTIVELY_FROZEN_POST_HOC_DIAGNOSTICS. The vector is frozen at every level; only the scalar calibration status differs. Never call all three levels zero-shot.", ""]
    csv_rows = []
    for a in actors:
        for dname, levels in per[a]["transfer_ladder"].items():
            if dname.startswith("target_trained"):
                md += [f"## {a} — target-trained high-dimensional decoding comparator", "", "| metric | value |", "|---|---|"] + [f"| {k} | {fmt(v)} |" for k, v in levels["metrics"].items()] + [""]
                for k, v in levels["metrics"].items():
                    csv_rows.append({"actor": a, "direction": "full_activation_logistic", "level": "TARGET_TRAINED", "metric": k, **{kk: v.get(kk) for kk in ("point", "ci_low", "ci_high", "worlds")}})
                continue
            md += [f"## {a} — {dname}", ""]
            for lname, lv in levels.items():
                if "status" in lv:
                    md += [f"- {lname}: **{lv['status']}** — {lv['reason']}", ""]
                    csv_rows.append({"actor": a, "direction": dname, "level": lname, "metric": "status", "point": lv["status"]})
                    continue
                md += [f"### {lname}", "", "| metric | value |", "|---|---|"]
                for k, v in lv["metrics"].items():
                    md.append(f"| {k} | {fmt(v)} |")
                    csv_rows.append({"actor": a, "direction": dname, "level": lname, "metric": k, **{kk: v.get(kk) for kk in ("point", "ci_low", "ci_high", "worlds")}})
                if lname.startswith("C_"):
                    md.append(f"| mapping_reversal_semantic_agreement | {lv['mapping_reversal_semantic_agreement']} |")
                md.append("")
    write_triplet(output_root, "FROZEN_DIRECTION_TRANSFER_LADDER", ladder_json, md, csv_rows, ["actor", "direction", "level", "metric", "point", "ci_low", "ci_high", "worlds"])
    # ---- specificity
    spec = {a: per[a]["paired_specificity"] for a in actors}
    md = ["# Paired semantic-specificity comparisons (Arm A 4.2)", ""]
    csv_rows = []
    for a in actors:
        s = spec[a]
        md += [f"## {a}: **{s['classification']}**", "", "Primary (jointly residualized relation minus comparator, mean over the three core threshold-free endpoints; Holm within actor):", "", "| comparator | diff [95% CI] | Holm p (>0) | Holm p (<0) |", "|---|---|---|---|"]
        for c, v in s["primary"].items():
            md.append(f"| {c} | {fmt(v)} | {v['holm_p_positive']:.4f} | {v['holm_p_negative']:.4f} |")
            csv_rows.append({"actor": a, "kind": "primary", "comparator": c, "endpoint": "core_mean", "point": v["point"], "ci_low": v["ci_low"], "ci_high": v["ci_high"], "holm_p_positive": v["holm_p_positive"], "holm_p_negative": v["holm_p_negative"]})
        md += ["", "Per-endpoint points (direction; random band q2.5–q97.5 of the matched-norm random directions):", "", "| endpoint | " + " | ".join(SEMANTIC) + " | random band (residual ref) | discriminating |", "|---|" + "---|" * (len(SEMANTIC) + 2)]
        for ep in TF_ENDPOINTS + ("operator_sign_calibrated_ba",):
            band = s["random_bands"]["jointly_residualized_relation"][ep]
            md.append(f"| {ep} | " + " | ".join(fmt(s["points"][k][ep]) for k in SEMANTIC) + f" | {band['q025']:.3f}–{band['q975']:.3f} | {band['discriminating_at_0.70']} |")
            for k in SEMANTIC:
                v = s["points"][k][ep]
                csv_rows.append({"actor": a, "kind": "point", "comparator": k, "endpoint": ep, "point": v["point"], "ci_low": v["ci_low"], "ci_high": v["ci_high"]})
            for c in COMPARATORS:
                v = s["secondary_per_endpoint"][c][ep]
                csv_rows.append({"actor": a, "kind": "paired_diff_residual_minus", "comparator": c, "endpoint": ep, "point": v["point"], "ci_low": v["ci_low"], "ci_high": v["ci_high"], "p_two_sided": v["p_two_sided"]})
        md.append("")
    write_triplet(output_root, "PAIRED_SPECIFICITY_RESULTS", spec, md, csv_rows, ["actor", "kind", "comparator", "endpoint", "point", "ci_low", "ci_high", "holm_p_positive", "holm_p_negative", "p_two_sided"])
    # ---- presence
    pres = {a: per[a]["presence_polarity"] for a in actors}
    md = ["# Relation presence versus bipolar polarity (Arm A 4.3)", ""]
    csv_rows = []
    for a in actors:
        p = pres[a]
        md += [f"## {a}: **{p['classification']}**", "", "| model | presence BA (eval, FINAL_EVALUATION rows) | three-state BA |", "|---|---|---|"]
        for name in ("1_dim_polarity_alone", "2_presence_full_activation", "3_presence_after_dim_and_nuisance_removal"):
            md.append(f"| {name} | {fmt(p[name]['presence_ba_eval'])} | — |")
            csv_rows.append({"actor": a, "model": name, "metric": "presence_ba_eval", **{k: p[name]["presence_ba_eval"].get(k) for k in ("point", "ci_low", "ci_high")}})
        for name in ("4_two_factor_frozen_polarity_plus_learned_presence", "4b_two_factor_dim_only", "5_direct_three_state"):
            md.append(f"| {name} | — | {fmt(p[name]['three_state_ba_eval'])} |")
            csv_rows.append({"actor": a, "model": name, "metric": "three_state_ba_eval", **{k: p[name]["three_state_ba_eval"].get(k) for k in ("point", "ci_low", "ci_high")}})
        md += ["", f"Paired model 2 − model 1 presence BA: {fmt(p['paired_model2_minus_model1_presence'])}; model 5 − model 4 three-state: {fmt(p['paired_model5_minus_model4_three_state'])}", "", "Per family (all evaluation-world rows incl. diagnostic-only):", "", "| family | model 2 presence | model 3 presence | model 4 three-state | model 5 three-state |", "|---|---|---|---|---|"]
        for f in FAMILIES:
            md.append(f"| {f} | {fmt(p['2_presence_full_activation']['by_family'][f])} | {fmt(p['3_presence_after_dim_and_nuisance_removal']['by_family'][f])} | {fmt(p['4_two_factor_frozen_polarity_plus_learned_presence']['by_family'][f])} | {fmt(p['5_direct_three_state']['by_family'][f])} |")
            for name in ("2_presence_full_activation", "3_presence_after_dim_and_nuisance_removal"):
                csv_rows.append({"actor": a, "model": name, "metric": f"family_{f}", **{k: p[name]["by_family"][f].get(k) for k in ("point", "ci_low", "ci_high")}})
        md += ["", "Leave-one-family-out (model 2 refit without the family; evaluated on that family's evaluation rows ∪ DIRECT evaluation rows):", "", "| family | presence BA | absent-class recall |", "|---|---|---|"]
        for f, v in p["leave_one_family_out_model2"].items():
            md.append(f"| {f} | {fmt(v['presence_ba_family_union_direct'])} | {fmt(v['family_absent_recall'])} |")
            csv_rows.append({"actor": a, "model": "2_lofo", "metric": f"lofo_{f}", **{k: v["presence_ba_family_union_direct"].get(k) for k in ("point", "ci_low", "ci_high")}})
        md.append("")
    write_triplet(output_root, "RELATION_PRESENCE_POLARITY_RESULTS", pres, md, csv_rows, ["actor", "model", "metric", "point", "ci_low", "ci_high"])
    # ---- mapping
    mp = {a: per[a]["mapping_reversal"] for a in actors}
    md = ["# Mapping-reversal mechanism (Arm A 4.4)", ""]
    csv_rows = []
    for a in actors:
        m = mp[a]
        md += [f"## {a}: **{m['classification']}** (raw DIM; full-activation comparator agreement {fmt(m['full_activation_comparator'])})", ""]
        for dname in ("raw_relation_dim", "jointly_residualized_relation"):
            d = m[dname]
            md += [f"### {dname} — {d['classification']} ({d['evaluable_pairs']} evaluation pairs)", "", f"Intercept offsets: {d['intercept_offsets']}", "", "| variant | pair corr | mean |gap| | level-A agreement | level-B agreement | dual-query (TF) | holder swap (TF) | target swap (TF) | level-B sign BA |", "|---|---|---|---|---|---|---|---|---|"]
            for vname, v in d["variants"].items():
                md.append(f"| {vname} | {v['paired_raw_score_correlation']:.3f} | {v['mean_abs_pair_gap']:.4f} | {fmt(v.get('level_A_mapping_agreement'))} | {fmt(v['level_B_mapping_agreement'])} | {fmt(v['threshold_free']['same_context_dual_query_reversal'])} | {fmt(v['threshold_free']['holder_swap_correct'])} | {fmt(v['threshold_free']['target_swap_correct'])} | {fmt(v['level_B_calibration_transfer']['sign_balanced_accuracy'])} |")
                csv_rows.append({"actor": a, "direction": dname, "variant": vname, "pair_correlation": v["paired_raw_score_correlation"], "level_B_agreement": v["level_B_mapping_agreement"]["point"], "level_B_agreement_ci_low": v["level_B_mapping_agreement"]["ci_low"], "level_B_agreement_ci_high": v["level_B_mapping_agreement"]["ci_high"], "level_A_agreement": (v.get("level_A_mapping_agreement") or {}).get("point")})
            md.append("")
    write_triplet(output_root, "MAPPING_REVERSAL_MECHANISM_RESULTS", mp, md, csv_rows, ["actor", "direction", "variant", "pair_correlation", "level_B_agreement", "level_B_agreement_ci_low", "level_B_agreement_ci_high", "level_A_agreement"])
    # ---- additive
    ad = {a: per[a]["additive_holder_sign"] for a in actors}
    md = ["# Additive holder-plus-sign baseline (Arm A 4.5)", "", "The frozen G1 decoding verdict is preserved regardless of this diagnostic.", ""]
    csv_rows = []
    for a in actors:
        d = ad[a]
        md += [f"## {a}: **{d['classification']}**", "", "| readout | macro-world joint BA |", "|---|---|"]
        for k in ("additive_level_B", "additive_level_A_strict", "full_activation_joint", "residual_activation_joint", "dim_scalar_joint_frozen_reference"):
            md.append(f"| {k} | {fmt(d[k])} |")
            csv_rows.append({"actor": a, "metric": k, **{kk: d[k].get(kk) for kk in ("point", "ci_low", "ci_high")}})
        md += [f"| holder decoder slot BA (QUERY_ONLY-trained, eval two-actor rows) | {fmt(d['holder_decoder']['eval_two_actor_slot_ba'])} |", "", f"Paired full − additive: {fmt(d['paired_full_minus_additive'])}; residual − additive: {fmt(d['paired_residual_minus_additive'])} (interaction margin 0.05).", ""]
        for k in ("paired_full_minus_additive", "paired_residual_minus_additive"):
            csv_rows.append({"actor": a, "metric": k, **{kk: d[k].get(kk) for kk in ("point", "ci_low", "ci_high")}})
    write_triplet(output_root, "ADDITIVE_HOLDER_SIGN_RESULTS", ad, md, csv_rows, ["actor", "metric", "point", "ci_low", "ci_high"])
    # ---- G3 amendment
    g3 = g3_amendment(g3_packet)
    C.write_create_only(output_root / "G3_REPORTING_AMENDMENT.md", g3.encode("utf-8"))
    # ---- packet
    classes = {a: {"4.2_specificity": spec[a]["classification"], "4.3_presence": pres[a]["classification"], "4.4_mapping": mp[a]["classification"], "4.5_additive": ad[a]["classification"]} for a in actors}
    packet = {
        "schema_version": "EXTERNAL_REVIEW_PACKET_V1", "unit": f"{C.STUDY_ID} / Arm A no-inference characterization", "evidence_status": "PROSPECTIVELY_FROZEN_POST_HOC_DIAGNOSTICS",
        "research_question": {"question": "Which of the frozen-direction transfer, semantic specificity, presence-vs-polarity, mapping-offset, and additive-vs-joint questions can be settled from the retained Track A forwards without new inference?", "hypothesis": "see ANALYSIS_PLAN.json arm_a", "target_variables": ["ladder metrics per calibration level", "world-paired specificity differences", "presence BA", "mapping agreement after prospectively fixed correction", "joint minus additive BA"], "frozen_practical_threshold": {"gate": C.PRACTICAL_GATE, "role_order": 0.60, "agreement": C.AGREEMENT_GATE, "interaction_margin": 0.05}},
        "capability_delta": {"new_runnable_behavior": "scripts/rrrd_v1_no_inference.py: shared world-bootstrap paired inference over threshold-free, strict-source, and target-calibrated readouts on the retained activations"},
        "evidence_topology": {"independent_unit": "world_id", "evaluation_worlds": 48, "fit_worlds": 72, "development_worlds": 24, "rows_per_actor": 3744, "clustering": "rows nest in worlds; contrast pairs share a world; every CI resamples worlds with one shared multinomial draw", "training_exposure": "fit worlds for coefficients; development worlds for C; evaluation worlds seen once per readout"},
        "exact_empirical_result": {"classifications": classes, "ladder": ladder_json, "specificity_primary": {a: spec[a]["primary"] for a in actors}, "presence_key": {a: {k: pres[a][k] for k in ("2_presence_full_activation", "paired_model2_minus_model1_presence", "leave_one_family_out_model2")} for a in actors}, "mapping_key": {a: {d: {v: mp[a][d]["variants"][v]["level_B_mapping_agreement"] for v in mp[a][d]["variants"]} for d in ("raw_relation_dim", "jointly_residualized_relation")} for a in actors}, "additive_key": {a: {k: ad[a][k] for k in ("additive_level_B", "full_activation_joint", "residual_activation_joint", "paired_full_minus_additive", "paired_residual_minus_additive")} for a in actors}},
        "negative_and_surprising_evidence": negative_notes(per),
        "what_was_held_fixed": ["frozen directions and hashes", "frozen splits", "frozen nuisance covariates and residualization", "C grid and development selection", "10,000 world bootstraps, seed " + str(C.BOOTSTRAP_SEED), "Holm within actor and family", "source midpoint from the M1 probe bundle"],
        "competing_explanations": [{"name": "relation-specific linear coordinate", "evidence_for": "specificity favored on threshold-free endpoints", "evidence_against": "random band non-discriminating or comparator ties", "what_current_data_cannot_distinguish": "decodability from causal use", "smallest_discriminating_test": "activation patching on paired worlds (out of scope)"}, {"name": "sign-plus-holder additive code", "evidence_for": "additive matches joint", "evidence_against": "joint exceeds additive by > 0.05", "what_current_data_cannot_distinguish": "linear interaction from nonlinear composition", "smallest_discriminating_test": "held-out-world interaction probe with matched capacity"}],
        "barrier_and_impossibility_audit": [{"item": "level A for the jointly residualized direction", "class": "GROUND_TRUTH_UNAVAILABLE", "evidence": "no source midpoint artifact exists", "more_data_helps": False, "minimum_new_data": "the postmortem fitting population projections", "falsifier": "an authoritative midpoint artifact", "confidence": "high"}],
        "state_of_hypothesis_space": {"TERMINAL_BRANCHES": ["G1 decoding verdict (frozen)"], "FORBIDDEN_RETRIES": ["refitting the relation vector", "rerunning Track A forwards"], "OPEN_BRANCHES": ["causal use of the presence coordinate", "cross-format readout stability (Arm B)"], "UNSUPPORTED_BUT_POSSIBLE_BRANCHES": ["nonlinear holder-relation composition"], "NEW_INFORMATION_REQUIRED": ["Arm B/C outcomes for the manuscript wording"]},
        "deviations": ["none from ANALYSIS_PLAN.json; the leave-one-family-out evaluation set unions the held-out family with DIRECT evaluation rows because negated rows are single-class"],
        "claim_boundary": "see freeze/CLAIM_BOUNDARY.md; Arm A results are post hoc diagnostics of already-observed forwards",
        "artifact_pickup": {"results": [f"{n}.json" for n in ("FROZEN_DIRECTION_TRANSFER_LADDER", "PAIRED_SPECIFICITY_RESULTS", "RELATION_PRESENCE_POLARITY_RESULTS", "MAPPING_REVERSAL_MECHANISM_RESULTS", "ADDITIVE_HOLDER_SIGN_RESULTS")], "per_actor": {a: f"{a}/ARM_A_{a}.json" for a in actors}},
    }
    C.write_create_only(output_root / "external_review_packet_v1.json", C.canonical_json(packet))
    lines = ["# External review packet — Arm A (no-inference characterization)", "", f"Study: {C.STUDY_ID}. Evidence status: PROSPECTIVELY_FROZEN_POST_HOC_DIAGNOSTICS.", "", "## Terminal classifications", "", "| actor | 4.2 specificity | 4.3 presence | 4.4 mapping | 4.5 additive |", "|---|---|---|---|---|"] + [f"| {a} | {c['4.2_specificity']} | {c['4.3_presence']} | {c['4.4_mapping']} | {c['4.5_additive']} |" for a, c in classes.items()] + ["", "## Negative and surprising evidence", ""] + [f"- {n}" for n in packet["negative_and_surprising_evidence"]] + ["", "## Files", ""] + [f"- {n}.{{md,json,csv}}" for n in ("FROZEN_DIRECTION_TRANSFER_LADDER", "PAIRED_SPECIFICITY_RESULTS", "RELATION_PRESENCE_POLARITY_RESULTS", "MAPPING_REVERSAL_MECHANISM_RESULTS", "ADDITIVE_HOLDER_SIGN_RESULTS")] + ["- G3_REPORTING_AMENDMENT.md", ""]
    C.write_create_only(output_root / "external_review_packet_v1.md", "\n".join(lines).encode("utf-8"))
    C.write_create_only(output_root / "artifact_manifest.json", C.canonical_json({"schema_version": f"{C.SCHEMA_PREFIX}_ARM_A_MANIFEST", "classifications": classes, "files": C.manifest_for(output_root)}))
    print(RBA.json.dumps(classes, indent=2))


def negative_notes(per: dict[str, Any]) -> list[str]:
    notes = []
    for a, r in per.items():
        lad = r["transfer_ladder"]["raw_relation_dim"]
        for lname, lv in lad.items():
            if "metrics" not in lv:
                continue
            for k, v in lv["metrics"].items():
                gate = {"sign_balanced_accuracy": 0.7, "same_context_dual_query_reversal": 0.7, "holder_swap_correct": 0.7, "target_swap_correct": 0.7, "role_permutation_transfer": 0.6, "order_crossed_transfer": 0.6, "mapping_reversal_semantic_agreement": 0.9, "operator_sign_world_auroc": 0.7}.get(k)
                if gate is not None and np.isfinite(v.get("point", float("nan"))) and v["point"] < gate:
                    notes.append(f"{a} raw DIM {lname} {k} = {v['point']:.3f} below the reference gate {gate}")
        s = r["paired_specificity"]
        for c, v in s["primary"].items():
            if not v["significant_positive"]:
                notes.append(f"{a} specificity: residualized relation minus {c} not significantly positive ({v['point']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}])")
        for ep in CORE:
            b = s["random_bands"]["jointly_residualized_relation"][ep]
            if not b["discriminating_at_0.70"]:
                notes.append(f"{a} random band non-discriminating on {ep} (q97.5 {b['q975']:.3f})")
        b = s["random_bands"]["jointly_residualized_relation"]["operator_sign_calibrated_ba"]
        notes.append(f"{a} calibrated sign random band q97.5 = {b['q975']:.3f} ({'non-' if not b['discriminating_at_0.70'] else ''}discriminating at 0.70)")
        p = r["presence_polarity"]
        for f, v in p["leave_one_family_out_model2"].items():
            if v["presence_ba_family_union_direct"]["point"] < 0.60:
                notes.append(f"{a} presence readout fails held-out family {f} ({v['presence_ba_family_union_direct']['point']:.3f})")
        m = r["mapping_reversal"]
        for d in ("raw_relation_dim", "jointly_residualized_relation"):
            notes.append(f"{a} {d} mapping agreement before/intercept/affine = " + "/".join(f"{m[d]['variants'][v]['level_B_mapping_agreement']['point']:.3f}" for v in ("before", "intercept_corrected", "affine_corrected")))
        ad = r["additive_holder_sign"]
        notes.append(f"{a} additive joint BA {ad['additive_level_B']['point']:.3f} vs full {ad['full_activation_joint']['point']:.3f} / residual {ad['residual_activation_joint']['point']:.3f}")
    return notes


def g3_amendment(packet_path: Path | None) -> str:
    if packet_path is None or not packet_path.is_file():
        return "# G3 reporting amendment\n\nUnit 4 packet not available at assembly time; amendment deferred.\n"
    d = C.read_json(packet_path)
    res = d["exact_empirical_result"]
    lines = ["# G3 (Q-FAMILY) reporting amendment — factual, no recomputation", "", "The frozen Unit 4 result artifact is unchanged. This amendment states what the frozen numbers can and cannot support and enters the study's claim matrix.", "", f"- Frozen program classification: **{res['program_disposition']}**.", ""]
    for a, r in res["actors"].items():
        lines.append(f"- {a}: slope {r['slope']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]; disposition {r['disposition']}; independent dataset cells = {r['independent_dataset_cells']}; distinct bootstrap slope values = {r['distinct_bootstrap_estimates']}; regression rows = {r['regression_rows']}.")
    ll = res["actors"]["llama"]["llama_sentiment_relation_diagnostic"]
    lines += ["", f"- Only two unique dataset cells (AMPERE++, ValuePrism) support the association; the grouped bootstrap therefore has only three distinct resampled slope values, so the reported intervals do not carry the usual meaning of a 5,000-replicate CI.", f"- The preregistered Llama sentiment-minus-relation diagnostic has the opposite sign: loading difference {ll['loading_difference']:.3f} versus mean efficacy difference {ll['mean_efficacy_difference']:.3f} (same_nonzero_sign = {ll['same_nonzero_sign']}).", "- Q-FAMILY remains mechanism-unresolved; the manuscript should describe G3 as a suggestive two-dataset association with an adverse preregistered diagnostic, not as an established family-structured compression result.", "- Source: Unit 4 external_review_packet_v2.json (SHA-256 in freeze/INPUT_MANIFEST.json, role unit4_g3_packet_v2).", ""]
    return "\n".join(lines)


def write_triplet(root: Path, name: str, payload: Any, md_lines: list[str], csv_rows: list[dict[str, Any]], columns: list[str]) -> None:
    C.write_create_only(root / f"{name}.json", C.canonical_json(payload))
    C.write_create_only(root / f"{name}.md", ("\n".join(md_lines) + "\n").encode("utf-8"))
    C.write_create_only(root / f"{name}.csv", C.csv_bytes(csv_rows, columns))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--actor", required=True, choices=sorted(C.ACTORS))
    a.add_argument("--completed-jobs", type=Path, required=True)
    a.add_argument("--artifact-root", type=Path, required=True)
    a.add_argument("--freeze-root", type=Path, default=C.FREEZE_ROOT)
    a.add_argument("--output-root", type=Path, required=True)
    a.add_argument("--replicates", type=int, default=C.BOOTSTRAP_REPLICATES)
    s = sub.add_parser("assemble")
    s.add_argument("--output-root", type=Path, required=True)
    s.add_argument("--g3-packet", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "analyze":
        out_dir = args.output_root / args.actor
        if out_dir.exists() and any(out_dir.iterdir()):
            raise SystemExit(f"output not empty: {out_dir}")
        res = analyze(args.actor, args.completed_jobs, args.artifact_root, args.freeze_root, args.output_root, args.replicates)
        res["source"] = {"completed_jobs_sha256": C.sha256_file(args.completed_jobs), "script_sha256": C.sha256_file(Path(__file__).resolve()), "rba_sha256": C.sha256_file(Path(RBA.__file__).resolve()), "replicates": args.replicates}
        C.write_create_only(out_dir / f"ARM_A_{args.actor}.json", C.canonical_json(res))
        C.write_create_only(out_dir / "artifact_manifest.json", C.canonical_json({"files": C.manifest_for(out_dir)}))
        print(RBA.json.dumps({k: res[k]["classification"] for k in ("paired_specificity", "presence_polarity", "mapping_reversal", "additive_holder_sign")}, indent=2))
    else:
        assemble(args.output_root, args.g3_packet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
