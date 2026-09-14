#!/usr/bin/env python3
"""Arm B analysis of RELATION_READOUT_REPORT_DISCRIMINATION_V1: Track A re-elicitation.

Inputs per actor: the promoted GPU output (F1/F2 scores, generated labels, band
activations), the frozen zero-shot baseline (Unit 6/7 completed jobs and
activations), the frozen direction bundle and the source midpoint from the
freeze. Every endpoint is a per-world statistic bootstrapped with the shared
10,000-replicate world resampling; prompt rows, mappings and templates are
repeated measures inside worlds.

  analyze --actor A  -> <out>/<actor>/TRACK_A_REELICITATION_<actor>.json
  assemble           -> TRACK_A_REELICITATION_RESULTS.{md,json,csv},
                        TRACK_A_REELICITATION_INTERPRETATION.md,
                        external_review_packet_v1.{md,json}, artifact_manifest.json
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import relation_binding_analysis_v2 as RBA  # noqa: E402
import rrrd_v1_common as C  # noqa: E402
from rrrd_v1_no_inference import Metrics, WorldBootstrap, fmt  # noqa: E402

FORMATS = ("F1", "F2")
BOUND = ("same_context_dual_query_reversal", "holder_swap_correct", "target_swap_correct")
RULE_METRICS = {"operator_sign_ba": "operator_sign_balanced_accuracy", "same_context_dual_query_reversal": "same_context_two_query_sign_flip_correct_rate", "holder_swap_correct": "holder_swap_correct_rate", "target_swap_correct": "target_swap_correct_rate", "role_permutation_transfer": "fit_canonical_supports_evaluate_alternate_supports_balanced_accuracy", "order_crossed_transfer": "fit_order_canonical_evaluate_order_reversed_sign_balanced_accuracy", "reporter_quoter_attribution": "reporter_quoter_attribution_correct_rate", "heldout_lexical_recall": "heldout_lexical_balanced_accuracy", "heldout_discourse_ba": "heldout_discourse_balanced_accuracy", "heldout_domain_ba": "heldout_domain_balanced_accuracy", "mapping_reversal_semantic_agreement": "mapping_reversal_semantic_agreement", "false_binary_opposite_rate": "false_binary_opposite_rate"}


def semantic_rule(t: RBA.Table, entailed: np.ndarray):
    """The frozen native_answer rule with an arbitrary per-row entailment prediction (1/0; -1 = unscored)."""
    op_support = np.array([r["queried_operator"] == "SUPPORT" for r in t.records])
    sign = np.where(op_support, entailed, 1 - entailed)

    def rule(task: str, mask: np.ndarray) -> np.ndarray:
        if task in ("sign", "role_perm", "order"):
            return sign[mask]
        if task == "entail":
            return entailed[mask]
        if task == "state":
            return np.where(sign[mask] == 1, "SUPPORT", "OPPOSE").astype(object)
        return np.array([f"{'C' if h == 'CANONICAL_HOLDER' else 'A'}_{'S' if s else 'O'}" for h, s in zip(t.holder_role[mask], sign[mask])], dtype=object)

    return rule


def load_gpu(actor: str, gpu_root: Path) -> dict[str, Any]:
    man = C.read_json(gpu_root / "artifact_manifest.json")
    bad = [f["path"] for f in man["files"] if C.sha256_file(gpu_root / f["path"]) != f["sha256"]]
    if bad:
        raise RuntimeError(f"GPU output hash mismatch: {bad[:5]}")
    out = {"manifest": man, "receipt": C.read_json(gpu_root / "RUN_RECEIPT.json"), "environment": C.read_json(gpu_root / "environment.json")}
    for fmt_ in FORMATS:
        out[fmt_] = C.read_jsonl(gpu_root / f"track_a_{fmt_.lower()}_scores.jsonl")
        with np.load(gpu_root / f"track_a_{fmt_.lower()}_activations.npz", allow_pickle=False) as z:
            out[f"{fmt_}_act"] = {"prompt_ids": [str(v) for v in z["prompt_ids"]], "primary": np.asarray(z[f"layer_{C.ACTORS[actor]['primary_layer']}"], dtype=np.float64)}
    return out


def analyze(actor: str, gpu_root: Path, completed_jobs: Path, artifact_root: Path, freeze_root: Path, replicates: int) -> dict[str, Any]:
    g = load_gpu(actor, gpu_root)
    records = RBA.read_jsonl(completed_jobs)
    RBA.validate_completed(records, actor)
    t = RBA.Table(records)
    idx = {r["example_id"]: i for i, r in enumerate(records)}
    x_base = RBA.load_activations(records, artifact_root, verify_hashes=True)
    freeze_inputs = C.read_json(freeze_root / "INPUT_MANIFEST.json")
    midpoint = float(freeze_inputs["source_calibration"]["raw_relation_dim"][actor]["difference_in_means_midpoint"])
    bundle = C.TRACK_A_FREEZE / "runtime" / actor / "competitor_directions_v2.npz"
    z = np.load(bundle, allow_pickle=False)
    dirs = {k: np.asarray(z[k], dtype=np.float64) for k in ("raw_relation_dim", "jointly_residualized_relation")}
    ev_worlds = sorted(set(t.world[t.split == "heldout_evaluation"]))
    boot = WorldBootstrap(ev_worlds, replicates, C.BOOTSTRAP_SEED)
    M = Metrics(t)
    scored_mask = np.isin(t.split, ["development", "heldout_evaluation"])
    direct = t.family == "DIRECT_HOLDER_TARGET_QUERY"
    negwith = np.isin(t.family, ["NEGATED_STANCE", "NEUTRAL_OR_WITHHELD_STANCE"])
    res: dict[str, Any] = {"schema_version": f"{C.SCHEMA_PREFIX}_TRACK_A_REELICITATION", "actor": actor, "label": "PROSPECTIVE_FOLLOW_UP_EVIDENCE", "gpu_receipt": g["receipt"], "environment": g["environment"], "source_midpoint": midpoint, "evaluation_worlds": len(ev_worlds), "formats": {}}

    # ---- baseline (frozen zero-shot native answer) and its readouts
    base_rule = RBA.RuleReadout("baseline_native", t, semantic_rule(t, t.native_semantic))
    base_ws = base_rule.world_stats()
    base_metrics = {k: base_ws[v] for k, v in RULE_METRICS.items()}
    base_metrics["direct_single_holder_entailment_ba"] = RBA.macro_world_ba(t.entailed[t.eval & direct], t.native_semantic[t.eval & direct], t.world[t.eval & direct])
    base_metrics["negated_withheld_entailment_ba"] = RBA.macro_world_ba(t.entailed[t.eval & negwith], t.native_semantic[t.eval & negwith], t.world[t.eval & negwith])
    base_proj_raw = x_base @ dirs["raw_relation_dim"]
    base_strict_sign = (base_proj_raw - midpoint > 0).astype(int)
    strict_ws = RBA.RuleReadout("s", t, M.strict_rule(base_proj_raw - midpoint)).world_stats()
    base_ladder: dict[str, Any] = {"C_threshold_free": M.threshold_free(base_proj_raw)}
    base_ladder["A_strict"] = {"sign_balanced_accuracy": strict_ws["operator_sign_balanced_accuracy"], "same_context_dual_query_reversal": strict_ws["same_context_two_query_sign_flip_correct_rate"], "holder_swap_correct": strict_ws["holder_swap_correct_rate"], "target_swap_correct": strict_ws["target_swap_correct_rate"]}
    res["baseline_zero_shot"] = {"native": {k: boot.summary(v) for k, v in base_metrics.items()}, "frozen_dim_ladder": {lvl: {k: boot.summary(v) for k, v in d.items()} for lvl, d in base_ladder.items()}, "readout_pass": {lvl: readout_pass(base_ladder[lvl], lvl) for lvl in base_ladder}}

    signed_eval = t.eval & t.signed
    for fmt_ in FORMATS:
        rows = g[fmt_]
        own = [r for r in rows if r["own_mapping"]]
        ent = np.full(len(records), -1, dtype=int)
        margin_own = np.full(len(records), np.nan)
        for r in own:
            ent[idx[r["example_id"]]] = int(r["native_entailed"])
            margin_own[idx[r["example_id"]]] = r["margin_entailed_minus_not"]
        if (ent[scored_mask] < 0).any():
            raise RuntimeError(f"{fmt_}: unscored development/evaluation rows")
        rule = RBA.RuleReadout(f"{fmt_}_native", t, semantic_rule(t, np.clip(ent, 0, 1)))
        ws = rule.world_stats()
        metrics = {k: ws[v] for k, v in RULE_METRICS.items()}
        metrics["direct_single_holder_entailment_ba"] = RBA.macro_world_ba(t.entailed[t.eval & direct], ent[t.eval & direct], t.world[t.eval & direct])
        metrics["negated_withheld_entailment_ba"] = RBA.macro_world_ba(t.entailed[t.eval & negwith], ent[t.eval & negwith], t.world[t.eval & negwith])
        metrics["entailment_ba_all_eval_rows"] = RBA.macro_world_ba(t.entailed[t.eval], ent[t.eval], t.world[t.eval])
        extra: dict[str, Any] = {}
        if fmt_ == "F1":
            by_ex: dict[str, list[int]] = collections.defaultdict(list)
            for r in rows:
                by_ex[r["example_id"]].append(int(r["native_entailed"]))
            agree = {}
            mean_margin = {}
            for ex, votes in by_ex.items():
                i = idx[ex]
                if t.eval[i]:
                    agree.setdefault(t.world[i], []).append(1.0 if len(set(votes)) == 1 and len(votes) == 4 else 0.0)
            metrics["F1_mapping_semantic_agreement"] = {w: float(np.mean(v)) for w, v in agree.items()}
            mm = collections.defaultdict(list)
            for r in rows:
                mm[r["example_id"]].append(r["margin_entailed_minus_not"])
            ent_mean = np.full(len(records), -1, dtype=int)
            for ex, v in mm.items():
                ent_mean[idx[ex]] = int(np.mean(v) > 0)
            metrics["F1_secondary_mean_margin_direct_ba"] = RBA.macro_world_ba(t.entailed[t.eval & direct], ent_mean[t.eval & direct], t.world[t.eval & direct])
            extra["per_mapping_direct_ba"] = {}
            for mapping in C.MAPPINGS:
                e_m = np.full(len(records), -1, dtype=int)
                for r in rows:
                    if r["mapping_id"] == mapping:
                        e_m[idx[r["example_id"]]] = int(r["native_entailed"])
                extra["per_mapping_direct_ba"][mapping] = boot.summary(RBA.macro_world_ba(t.entailed[t.eval & direct], e_m[t.eval & direct], t.world[t.eval & direct]))
        else:
            ga, parsed = {}, {}
            for r in own:
                i = idx[r["example_id"]]
                if t.eval[i]:
                    ga.setdefault(t.world[i], []).append(1.0 if r.get("generated_matches_candidate") == 1 else 0.0)
                    parsed.setdefault(t.world[i], []).append(1.0 if r.get("generated_label") is not None else 0.0)
            metrics["F2_candidate_vs_generated_agreement"] = {w: float(np.mean(v)) for w, v in ga.items()}
            metrics["F2_generation_parse_rate"] = {w: float(np.mean(v)) for w, v in parsed.items()}
            gen_ent = np.full(len(records), -1, dtype=int)
            for r in own:
                if r.get("generated_label") is not None:
                    gen_ent[idx[r["example_id"]]] = int(r["generated_label"] == "ENTAILED")
            gm = t.eval & direct & (gen_ent >= 0)
            metrics["F2_generated_label_direct_ba"] = RBA.macro_world_ba(t.entailed[gm], gen_ent[gm], t.world[gm])
            extra["generated_label_counts"] = dict(collections.Counter(str(r.get("generated_label")) for r in own if t.eval[idx[r["example_id"]]]))
            extra["generated_text_examples"] = [r["generated_text"] for r in own[:5]]
        # ---- frozen directions under the new format's activations
        act = g[f"{fmt_}_act"]
        row_of = {pid: k for k, pid in enumerate(act["prompt_ids"])}
        proj_new = {k: np.full(len(records), np.nan) for k in dirs}
        for r in own:
            k = row_of[r["prompt_id"]]
            for dname in dirs:
                proj_new[dname][idx[r["example_id"]]] = act["primary"][k] @ dirs[dname]
        ladder = {}
        for dname in dirs:
            p = np.where(np.isfinite(proj_new[dname]), proj_new[dname], 0.0)
            lv: dict[str, Any] = {}
            if dname == "raw_relation_dim":
                sw = RBA.RuleReadout("s", t, M.strict_rule(p - midpoint)).world_stats()
                lv["A_strict"] = {"sign_balanced_accuracy": sw["operator_sign_balanced_accuracy"], "same_context_dual_query_reversal": sw["same_context_two_query_sign_flip_correct_rate"], "holder_swap_correct": sw["holder_swap_correct_rate"], "target_swap_correct": sw["target_swap_correct_rate"], "mapping_reversal_semantic_agreement": sw["mapping_reversal_semantic_agreement"]}
            dev_signed = t.dev & t.signed
            model = RBA.fit_logistic(p[dev_signed][:, None], t.sign[dev_signed], 1.0)
            pred = RBA.predict(model, p[:, None])
            lv["B_development_calibrated"] = {"sign_balanced_accuracy": RBA.macro_world_ba(t.sign[signed_eval], pred[signed_eval], t.world[signed_eval]), "note": "scalar calibrated on development worlds only (fit worlds were not re-elicited); reported separately, never gating"}
            lv["C_threshold_free"] = M.threshold_free(p)
            ladder[dname] = lv
        native_sign = np.where(np.array([r["queried_operator"] == "SUPPORT" for r in records]), np.clip(ent, 0, 1), 1 - np.clip(ent, 0, 1))
        new_strict_sign = (np.where(np.isfinite(proj_new["raw_relation_dim"]), proj_new["raw_relation_dim"], 0.0) - midpoint > 0).astype(int)
        align_new = {w: float(np.mean(native_sign[signed_eval & (t.world == w)] == new_strict_sign[signed_eval & (t.world == w)])) for w in ev_worlds}
        align_base = {w: float(np.mean(native_sign[signed_eval & (t.world == w)] == base_strict_sign[signed_eval & (t.world == w)])) for w in ev_worlds}
        metrics["native_vs_frozen_dim_sign_agreement_new_activations"] = align_new
        metrics["native_vs_frozen_dim_sign_agreement_baseline_activations"] = align_base
        summ = {k: boot.summary(v) for k, v in metrics.items()}
        improvement = {k: boot.paired(metrics[k], base_metrics[k]) for k in base_metrics if k in metrics}
        ladder_summ = {d: {lvl: ({k: boot.summary(v) for k, v in vals.items() if k != "note"} | ({"note": vals["note"]} if "note" in vals else {})) for lvl, vals in lv.items()} for d, lv in ladder.items()}
        raw_l = ladder["raw_relation_dim"]
        rpass = {lvl: readout_pass(raw_l[lvl], lvl) for lvl in ("A_strict", "C_threshold_free")}
        rmb = {}
        for lvl in ("A_strict", "C_threshold_free"):
            rmb[lvl] = {ep: boot.paired(raw_l[lvl][ep], metrics[ep]) for ep in BOUND}
        direct_s = summ["direct_single_holder_entailment_ba"]
        consistency = summ["F1_mapping_semantic_agreement"] if fmt_ == "F1" else summ["F2_candidate_vs_generated_agreement"]
        competent = bool(direct_s["point"] >= C.DIRECT_COMPETENCE_GATE and direct_s["ci_low"] > C.DIRECT_COMPETENCE_CI_LOW and consistency["point"] >= C.AGREEMENT_GATE)
        bound_pass = all(summ[ep]["point"] >= C.PRACTICAL_GATE for ep in BOUND)
        improves = all(summ[ep]["point"] >= 0.60 and improvement[ep]["ci_low"] > 0 for ep in BOUND)
        aligns = bool(summ["native_vs_frozen_dim_sign_agreement_new_activations"]["point"] >= 0.80)
        fails_two = sum(summ[ep]["ci_high"] < C.PRACTICAL_GATE for ep in BOUND) >= 2
        readout_strong = {lvl: all(boot.summary(raw_l[lvl][ep])["ci_low"] > C.PRACTICAL_GATE for ep in BOUND) for lvl in ("A_strict", "C_threshold_free")}
        rmb_excl = {lvl: sum(1 for ep in BOUND if rmb[lvl][ep]["ci_low"] > 0 or rmb[lvl][ep]["ci_high"] < 0) >= 2 for lvl in rmb}
        res["formats"][fmt_] = {"rows_scored": len(rows), "own_mapping_rows": len(own), "metrics": summ, "paired_improvement_over_zero_shot": improvement, "frozen_dim_ladder_under_format": ladder_summ, "frozen_readout_pass": rpass, "readout_minus_behavior_paired": rmb, "flags": {"DIRECT_TASK_COMPETENT": competent, "BOUND_BEHAVIOR_PASS": bound_pass, "IMPROVES_INTO_PRACTICAL_RANGE": improves, "ALIGNS_WITH_DIM": aligns, "fails_two_bound_endpoints_ci_high_below_gate": fails_two, "readout_ci_low_above_gate_on_bound": readout_strong, "readout_minus_behavior_ci_excludes_zero_on_two": rmb_excl}, "extra": extra}
    res["classification"] = classify(res)
    res["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return res


def readout_pass(level_metrics: dict[str, dict[str, float]], lvl: str) -> dict[str, Any]:
    sign_key = "sign_balanced_accuracy" if lvl.startswith("A") else "operator_sign_world_auroc"
    dual = float(np.mean(list(level_metrics["same_context_dual_query_reversal"].values()))) if level_metrics.get("same_context_dual_query_reversal") else float("nan")
    sign = float(np.mean(list(level_metrics[sign_key].values()))) if level_metrics.get(sign_key) else float("nan")
    return {"dual_query": dual, "sign": sign, "pass": bool(dual >= C.PRACTICAL_GATE and sign >= C.PRACTICAL_GATE)}


def classify(res: dict[str, Any]) -> dict[str, Any]:
    F = res["formats"]
    comp = [f for f in FORMATS if F[f]["flags"]["DIRECT_TASK_COMPETENT"]]
    base_pass = res["baseline_zero_shot"]["readout_pass"]
    reasons = []
    if not comp:
        return {"primary": "TASK_NOT_ELICITED", "competent_formats": comp, "reasons": ["no format reaches direct-task competence"]}
    for f in comp:
        fp = F[f]["frozen_readout_pass"]
        for lvl in ("A_strict", "C_threshold_free"):
            if base_pass[lvl]["pass"] and not fp[lvl]["pass"]:
                reasons.append(f"{f}: frozen readout fails at {lvl} under the new format while the zero-shot activations pass")
    if reasons and all(not F[f]["frozen_readout_pass"]["C_threshold_free"]["pass"] and not F[f]["frozen_readout_pass"]["A_strict"]["pass"] for f in comp):
        return {"primary": "PROMPT_SENSITIVE_INTERNAL_READOUT", "competent_formats": comp, "reasons": reasons}
    ok = lambda f: (F[f]["flags"]["BOUND_BEHAVIOR_PASS"] or F[f]["flags"]["IMPROVES_INTO_PRACTICAL_RANGE"]) and F[f]["flags"]["ALIGNS_WITH_DIM"]  # noqa: E731
    if len(comp) == 2 and all(ok(f) for f in comp):
        return {"primary": "ROBUST_ELICITATION_REPAIR", "competent_formats": comp, "reasons": ["both formats competent, bound endpoints pass or improve into range, aligned with DIM"] + reasons}
    for f in comp:
        fl = F[f]["flags"]
        if fl["fails_two_bound_endpoints_ci_high_below_gate"] and any(fl["readout_ci_low_above_gate_on_bound"].values()) and any(fl["readout_minus_behavior_ci_excludes_zero_on_two"].values()):
            return {"primary": "VALID_TASK_INTERNAL_READOUT_BEHAVIOR_DISSOCIATION", "competent_formats": comp, "dissociating_format": f, "reasons": [f"{f}: competent, fails >=2 bound endpoints (CI high < 0.70), frozen readout passes them (CI low > 0.70), paired readout-minus-behavior excludes zero"] + reasons}
    if len(comp) == 1 and ok(comp[0]):
        return {"primary": "FORMAT_SPECIFIC_ELICITATION_REPAIR", "competent_formats": comp, "reasons": [f"only {comp[0]} is competent and it repairs/aligns"] + reasons}
    return {"primary": "MIXED_OR_UNDERDETERMINED", "competent_formats": comp, "reasons": reasons or ["no classification conjunction satisfied cleanly"]}


def assemble(output_root: Path, gpu_roots: dict[str, Path]) -> None:
    per = {a: C.read_json(output_root / a / f"TRACK_A_REELICITATION_{a}.json") for a in C.ACTORS if (output_root / a / f"TRACK_A_REELICITATION_{a}.json").is_file()}
    actors = list(per)
    for a in actors:
        dst = output_root / a / "gpu_scores"
        if not dst.exists():
            dst.mkdir(parents=True)
            for name in ("track_a_f1_scores.jsonl", "track_a_f2_scores.jsonl", "artifact_manifest.json", "RUN_RECEIPT.json", "environment.json", "PROMOTION_RECEIPT.json"):
                shutil.copy2(gpu_roots[a] / name, dst / name)
    joint_pairs = {a: per[a]["classification"]["primary"] for a in actors}
    joint = "SAME_CLASS:" + list(joint_pairs.values())[0] if len(set(joint_pairs.values())) == 1 and len(actors) == 2 else "ACTOR_DIVERGENT:" + "/".join(f"{a}={c}" for a, c in joint_pairs.items())
    results = {"schema_version": f"{C.SCHEMA_PREFIX}_TRACK_A_REELICITATION_RESULTS", "label": "PROSPECTIVE_FOLLOW_UP_EVIDENCE", "classification": joint_pairs, "joint_two_actor": joint, "actors": per}
    keys = ["direct_single_holder_entailment_ba", "operator_sign_ba", "same_context_dual_query_reversal", "holder_swap_correct", "target_swap_correct", "role_permutation_transfer", "order_crossed_transfer", "reporter_quoter_attribution", "negated_withheld_entailment_ba", "mapping_reversal_semantic_agreement", "F1_mapping_semantic_agreement", "F2_candidate_vs_generated_agreement", "F2_generation_parse_rate", "F2_generated_label_direct_ba", "native_vs_frozen_dim_sign_agreement_new_activations", "native_vs_frozen_dim_sign_agreement_baseline_activations", "heldout_lexical_recall", "heldout_discourse_ba", "heldout_domain_ba", "false_binary_opposite_rate"]
    md = ["# Track A re-elicitation results (Arm B)", "", f"Joint two-actor classification: **{joint}**", ""]
    csv_rows = []
    for a in actors:
        r = per[a]
        md += [f"## {a}: **{r['classification']['primary']}**", "", "Reasons: " + "; ".join(r["classification"]["reasons"]), "", "| endpoint | zero-shot baseline | F1 | F1 − baseline | F2 | F2 − baseline |", "|---|---|---|---|---|---|"]
        for k in keys:
            b = r["baseline_zero_shot"]["native"].get(k)
            cells = [fmt(b)]
            for f in FORMATS:
                m = r["formats"][f]["metrics"].get(k)
                d = r["formats"][f]["paired_improvement_over_zero_shot"].get(k)
                cells += [fmt(m), fmt(d)]
                if m:
                    csv_rows.append({"actor": a, "format": f, "endpoint": k, **{kk: m.get(kk) for kk in ("point", "ci_low", "ci_high", "worlds")}, "improvement_point": (d or {}).get("point"), "improvement_ci_low": (d or {}).get("ci_low"), "improvement_ci_high": (d or {}).get("ci_high")})
            if b:
                csv_rows.append({"actor": a, "format": "zero_shot_baseline", "endpoint": k, **{kk: b.get(kk) for kk in ("point", "ci_low", "ci_high", "worlds")}})
            md.append(f"| {k} | " + " | ".join(cells) + " |")
        md += ["", "Flags:", ""]
        for f in FORMATS:
            md.append(f"- {f}: " + ", ".join(f"{k}={v}" for k, v in r["formats"][f]["flags"].items() if not isinstance(v, dict)))
        md += ["", "Frozen raw DIM ladder under each format's activations (sign: BA at A, world AUROC at C):", "", "| level | endpoint | zero-shot activations | F1 activations | F2 activations |", "|---|---|---|---|---|"]
        for lvl in ("A_strict", "C_threshold_free"):
            eps = ["sign_balanced_accuracy" if lvl == "A_strict" else "operator_sign_world_auroc", "same_context_dual_query_reversal", "holder_swap_correct", "target_swap_correct", "role_permutation_transfer", "order_crossed_transfer"]
            for ep in eps:
                b = r["baseline_zero_shot"]["frozen_dim_ladder"][lvl].get(ep)
                cells = [fmt(b)] + [fmt(r["formats"][f]["frozen_dim_ladder_under_format"]["raw_relation_dim"][lvl].get(ep)) for f in FORMATS]
                md.append(f"| {lvl} | {ep} | " + " | ".join(cells) + " |")
                for f in FORMATS:
                    m = r["formats"][f]["frozen_dim_ladder_under_format"]["raw_relation_dim"][lvl].get(ep)
                    if m:
                        csv_rows.append({"actor": a, "format": f"{f}_dim_{lvl}", "endpoint": ep, **{kk: m.get(kk) for kk in ("point", "ci_low", "ci_high", "worlds")}})
        md += ["", f"Frozen readout pass (dual-query ≥ 0.70 and sign ≥ 0.70): zero-shot {r['baseline_zero_shot']['readout_pass']}; " + "; ".join(f"{f} {r['formats'][f]['frozen_readout_pass']}" for f in FORMATS), ""]
        for f in FORMATS:
            ex = r["formats"][f]["extra"]
            if ex:
                md.append(f"- {f} extra: {ex}")
        md.append("")
    C.write_create_only(output_root / "TRACK_A_REELICITATION_RESULTS.json", C.canonical_json(results))
    C.write_create_only(output_root / "TRACK_A_REELICITATION_RESULTS.md", ("\n".join(md) + "\n").encode("utf-8"))
    C.write_create_only(output_root / "TRACK_A_REELICITATION_RESULTS.csv", C.csv_bytes(csv_rows, ["actor", "format", "endpoint", "point", "ci_low", "ci_high", "worlds", "improvement_point", "improvement_ci_low", "improvement_ci_high"]))
    interp = ["# Track A re-elicitation — interpretation (Arm B)", "", "Evidence status: prospective follow-up evidence on the frozen Track A v2 corpus; the zero-shot native results are the frozen baseline and were not rerun.", ""]
    for a in actors:
        c = per[a]["classification"]["primary"]
        case = {"ROBUST_ELICITATION_REPAIR": "Case 1: the old Track A native failure was primarily an output-elicitation failure; Track A does not establish a representation-report dissociation.", "FORMAT_SPECIFIC_ELICITATION_REPAIR": "Case 1 (format-specific): one competent protocol repairs behavior and aligns with DIM; the other protocol remains poor.", "VALID_TASK_INTERNAL_READOUT_BEHAVIOR_DISSOCIATION": "Case 2: licensed wording only — \"The linear internal readout tracks the constructed queried relation on cases where a separately competent output protocol fails to report it.\" Not hidden knowledge, not belief, not natural use.", "PROMPT_SENSITIVE_INTERNAL_READOUT": "Case 3: the earlier internal transfer is prompt-format dependent; weaken the general query-bound claim.", "TASK_NOT_ELICITED": "Case 4: native Track A behavior remains uninterpretable; preserve the activation result; do not use the native endpoint to argue dissociation.", "MIXED_OR_UNDERDETERMINED": "No clean conjunction; report all flags; no dissociation or repair claim."}[c]
        interp += [f"## {a}: {c}", "", case, "", "Flags per format: " + "; ".join(f"{f}: " + ", ".join(f"{k}={v}" for k, v in per[a]["formats"][f]["flags"].items() if not isinstance(v, dict)) for f in FORMATS), ""]
    interp += [f"Joint two-actor classification: {joint}. Actors are not forced to agree.", ""]
    C.write_create_only(output_root / "TRACK_A_REELICITATION_INTERPRETATION.md", "\n".join(interp).encode("utf-8"))
    packet = {"schema_version": "EXTERNAL_REVIEW_PACKET_V1", "unit": f"{C.STUDY_ID} / Arm B Track A re-elicitation", "evidence_status": "PROSPECTIVE_FOLLOW_UP_EVIDENCE", "research_question": {"question": "Is the Track A hidden-readout / native-answer gap an elicitation failure, a valid readout/report dissociation, prompt sensitivity of the readout, or an actor-specific mixture?", "hypothesis": "co-primary formats F1 and F2; classification rules in freeze/ANALYSIS_PLAN.json arm_b", "expected_positive_result": "at least one competent format with either repaired bound behavior (Case 1) or a validated dissociation (Case 2)", "expected_falsifying_result": "no competent format (Case 4) or readout destroyed under competent elicitation (Case 3)", "frozen_practical_threshold": {"direct": C.DIRECT_COMPETENCE_GATE, "bound": C.PRACTICAL_GATE, "agreement": C.AGREEMENT_GATE}}, "capability_delta": {"new_runnable_behavior": "scripts/rrrd_v1_gpu_run.py (few-shot re-elicitation with band activations) and scripts/rrrd_v1_track_a_analysis.py (world-paired behavior/readout comparison against the frozen baseline)"}, "evidence_topology": {"independent_unit": "world_id", "evaluation_worlds": 48, "development_worlds": 24, "rows_per_format_per_actor": {a: {f: per[a]["formats"][f]["rows_scored"] for f in FORMATS} for a in actors}, "repeated_measures": "prompt rows, mappings, templates nest in worlds"}, "exact_empirical_result": {"classification": joint_pairs, "joint": joint, "per_actor": {a: {"flags": {f: per[a]["formats"][f]["flags"] for f in FORMATS}, "key_metrics": {f: {k: per[a]["formats"][f]["metrics"].get(k) for k in ("direct_single_holder_entailment_ba", "same_context_dual_query_reversal", "holder_swap_correct", "target_swap_correct", "F1_mapping_semantic_agreement", "F2_candidate_vs_generated_agreement", "native_vs_frozen_dim_sign_agreement_new_activations")} for f in FORMATS}, "readout_pass": {"baseline": per[a]["baseline_zero_shot"]["readout_pass"], **{f: per[a]["formats"][f]["frozen_readout_pass"] for f in FORMATS}}} for a in actors}}, "negative_and_surprising_evidence": [f"{a} {f}: {k} = {per[a]['formats'][f]['metrics'][k]['point']:.3f}" for a in actors for f in FORMATS for k in ("direct_single_holder_entailment_ba",) + BOUND if per[a]["formats"][f]["metrics"][k]["point"] < C.PRACTICAL_GATE] + [f"{a} {f}: frozen readout fails at {lvl}" for a in actors for f in FORMATS for lvl in ("A_strict", "C_threshold_free") if not per[a]["formats"][f]["frozen_readout_pass"][lvl]["pass"]], "what_was_held_fixed": ["eight demonstrations and both templates (freeze)", "row's own frozen mapping for the primary F1 prediction", "candidate-sequence scorer", "greedy generation with 8 new tokens", "frozen directions and source midpoint", "10,000 world bootstraps"], "deviations": ["F1/F2 are raw-text few-shot prompts (the C9 scaffold) while the frozen zero-shot baseline used the chat template; this is the protocol contrast under test, not a confound to remove"], "claim_boundary": "see freeze/CLAIM_BOUNDARY.md", "artifact_pickup": {"results": "TRACK_A_REELICITATION_RESULTS.json", "per_row_scores": {a: f"{a}/gpu_scores/" for a in actors}, "activations": "NFS outputs/rrrd_v1/<actor>/track_a_{f1,f2}_activations.npz (hashes in each gpu_scores/artifact_manifest.json)"}}
    C.write_create_only(output_root / "external_review_packet_v1.json", C.canonical_json(packet))
    C.write_create_only(output_root / "external_review_packet_v1.md", ("\n".join(["# External review packet — Arm B (Track A re-elicitation)", "", f"Classification: {joint_pairs}; joint: {joint}", "", "## Negative and surprising evidence", ""] + [f"- {n}" for n in packet["negative_and_surprising_evidence"]] + ["", "See TRACK_A_REELICITATION_RESULTS.md and TRACK_A_REELICITATION_INTERPRETATION.md.", ""])).encode("utf-8"))
    C.write_create_only(output_root / "artifact_manifest.json", C.canonical_json({"schema_version": f"{C.SCHEMA_PREFIX}_ARM_B_MANIFEST", "classification": joint_pairs, "files": C.manifest_for(output_root)}))
    print(RBA.json.dumps({"classification": joint_pairs, "joint": joint}, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--actor", required=True, choices=sorted(C.ACTORS))
    a.add_argument("--gpu-root", type=Path, required=True)
    a.add_argument("--completed-jobs", type=Path, required=True)
    a.add_argument("--artifact-root", type=Path, required=True)
    a.add_argument("--freeze-root", type=Path, default=C.FREEZE_ROOT)
    a.add_argument("--output-root", type=Path, required=True)
    a.add_argument("--replicates", type=int, default=C.BOOTSTRAP_REPLICATES)
    s = sub.add_parser("assemble")
    s.add_argument("--output-root", type=Path, required=True)
    s.add_argument("--gpu-root-llama", type=Path, required=True)
    s.add_argument("--gpu-root-gemma", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "analyze":
        out_dir = args.output_root / args.actor
        if out_dir.exists() and any(out_dir.iterdir()):
            raise SystemExit(f"output not empty: {out_dir}")
        res = analyze(args.actor, args.gpu_root, args.completed_jobs, args.artifact_root, args.freeze_root, args.replicates)
        res["source"] = {"gpu_manifest_sha256": C.sha256_file(args.gpu_root / "artifact_manifest.json"), "completed_jobs_sha256": C.sha256_file(args.completed_jobs), "script_sha256": C.sha256_file(Path(__file__).resolve()), "replicates": args.replicates}
        C.write_create_only(out_dir / f"TRACK_A_REELICITATION_{args.actor}.json", C.canonical_json(res))
        C.write_create_only(out_dir / "artifact_manifest.json", C.canonical_json({"files": C.manifest_for(out_dir)}))
        print(RBA.json.dumps(res["classification"], indent=2))
    else:
        assemble(args.output_root, {"llama": args.gpu_root_llama, "gemma": args.gpu_root_gemma})
    return 0


if __name__ == "__main__":
    sys.exit(main())
