#!/usr/bin/env python3
"""Frozen Track A v2 analysis (Unit 6/7 of RELATION_REPRESENTATION_PROGRAM_V2).

Implements TRACK_A_ANALYSIS_FREEZE.json for one actor's completed forward jobs:
six readouts, fold-local residualization on fit worlds only, development-world
C selection, the frozen primary/secondary endpoints, the eight nulls, the
first-class competitor battery with the per-direction effect decomposition,
10,000-replicate world-cluster bootstrap, Holm within model and endpoint
family, and mandated negative reporting.

Label operationalization (the freeze names the endpoints; this file fixes the
row sets and label constructions and is committed before any outcome is read):

* joint binding label (primary, four classes, chance 0.25): queried holder slot
  {CANONICAL, ALTERNATE} x relation {SUPPORT, OPPOSE} on the signed two-actor
  families DIRECT_HOLDER_TARGET_QUERY, HOLDER_SWAP, CONFLICTING_STANCES_IN_ONE_CONTEXT.
  The role permutation makes every cell occur. A stance-only scalar can recover
  the relation but not the holder coordinate (N_A_GLOBAL_VALENCE); an identity
  heuristic recovers the holder but not the relation.
* macro-world: balanced accuracy computed within each evaluation world over the
  classes present there, then averaged over worlds (world_id is the unit).
* H_A_JOINT_BINDING (secondary): the same four-class label restricted to the
  multi-holder contexts HOLDER_SWAP + CONFLICTING_STANCES (both stances present).
* sign tasks use relation_label in {SUPPORT, OPPOSE}; the relation-state task
  uses {SUPPORT, OPPOSE, UNSIGNED} where UNSIGNED covers WITHHELD,
  DOES_NOT_SUPPORT, DOES_NOT_OPPOSE (never recoded as binary opposites);
  entailment uses entailment_label.
* fit rows: world_split residual_probe_train & readout_eligibility FIT;
  tuning rows: development & DEVELOPMENT; evaluation rows: heldout_evaluation &
  FINAL_EVALUATION. QUERY_ONLY rows are diagnostic-only (decomposition term).
* G1 conjuncts: the primary increment is the residual readout over the DIM
  scalar; the role-permutation and order-crossed transfer gates are read from the
  full-activation readout (the residual readout has the sign scalar removed by
  construction, so its sign transfer is reported but not gated). All readouts
  report every endpoint.
* Features for every logistic readout are standardized on fit rows only.
* The corpus's held-out lexical templates realize SUPPORT only, so the held-out
  lexical rows form a single class; the metric is reported with a
  ``degenerate_single_class`` flag and N_A_LEXICAL_MEMORIZATION is marked not
  evaluable rather than counted as behaving or misbehaving.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

SCHEMA_VERSION = "RELATION_BINDING_ANALYSIS_V2"
CORPUS_SCHEMA = "RELATION_BINDING_FACTORIAL_V2"
COMPLETE_STATE = "FORWARD_COMPLETE_ANALYSIS_PENDING"
READOUTS = ("native_answer", "dim_scalar", "nuisance_only_logistic", "dim_plus_nuisance_logistic", "full_activation_logistic", "residual_activation_logistic")
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
MAX_ITER = 4000
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 15914427
TWO_ACTOR_FAMILIES = ("DIRECT_HOLDER_TARGET_QUERY", "HOLDER_SWAP", "CONFLICTING_STANCES_IN_ONE_CONTEXT")
MULTI_HOLDER_FAMILIES = ("HOLDER_SWAP", "CONFLICTING_STANCES_IN_ONE_CONTEXT")
MAPPINGS = ("MAPPING_AB_STANDARD", "MAPPING_AB_REVERSED", "MAPPING_12_STANDARD", "MAPPING_12_REVERSED")
LEXICAL_FAMILIES = ("EXPLICIT_RELATION", "LEXICALLY_ABLATED_RELATION")
OBSERVED_DIRECTIONS = ("raw_relation_dim", "jointly_residualized_relation", "sentiment", "factual", "answer_token")
JOINT_CHANCE = 0.25
PRIMARY_MARGIN = 0.05
SECONDARY_GATES = {
    "operator_sign_balanced_accuracy": ("min", 0.7, "H_A_OPERATOR"),
    "joint_holder_target_relation_balanced_accuracy": ("min", 0.65, "H_A_JOINT_BINDING"),
    "holder_swap_correct_rate": ("min", 0.7, "H_A_ROLE_EQUIVARIANCE"),
    "target_swap_correct_rate": ("min", 0.7, "H_A_TARGET_EQUIVARIANCE"),
    "reporter_quoter_attribution_correct_rate": ("min", 0.7, "H_A_ATTRIBUTION"),
    "heldout_lexical_balanced_accuracy": ("min", 0.6, "H_A_LEXICAL_TRANSFER"),
    "heldout_discourse_balanced_accuracy": ("min", 0.6, "H_A_DISCOURSE_TRANSFER"),
    "heldout_domain_balanced_accuracy": ("min", 0.6, "H_A_DOMAIN_TRANSFER"),
    "mapping_reversal_semantic_agreement": ("min", 0.9, "H_A_MAPPING_INVARIANCE"),
    "false_binary_opposite_rate": ("max", 0.1, "H_A_NEGATION_POLICY"),
    "fit_canonical_supports_evaluate_alternate_supports_balanced_accuracy": ("min", 0.6, "H_A_ROLE_PERMUTATION_TRANSFER"),
    "fit_order_canonical_evaluate_order_reversed_sign_balanced_accuracy": ("min", 0.6, "H_A_ORDER_CROSSED_TRANSFER"),
    "same_context_two_query_sign_flip_correct_rate": ("min", 0.7, "H_A_DUAL_QUERY_FLIP"),
}
NULL_EXPECTATIONS = {
    "N_A_WORLD_SHUFFLE": ("macro_world_joint_binding_balanced_accuracy", "ci_includes_chance"),
    "N_A_GLOBAL_VALENCE": ("joint_holder_target_relation_balanced_accuracy", "below", 0.65),
    "N_A_BAG_OF_ENTITY": ("holder_swap_correct_rate", "below", 0.7),
    "N_A_TARGET_FREE": ("target_swap_correct_rate", "below", 0.7),
    "N_A_QUOTATION_HEURISTIC": ("reporter_quoter_attribution_correct_rate", "below", 0.7),
    "N_A_LEXICAL_MEMORIZATION": ("heldout_lexical_balanced_accuracy", "below", 0.6),
    "N_A_ANSWER_TOKEN": ("mapping_reversal_semantic_agreement", "below", 0.9),
    "N_A_BINARY_NEGATION": ("false_binary_opposite_rate", "above", 0.1),
}
SEMANTIC_MAPPINGS = {"MAPPING_AB_STANDARD": {"A": 1, "B": 0}, "MAPPING_AB_REVERSED": {"B": 1, "A": 0}, "MAPPING_12_STANDARD": {"1": 1, "2": 0}, "MAPPING_12_REVERSED": {"2": 1, "1": 0}}


# --------------------------------------------------------------------------- io
def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=float) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


# ------------------------------------------------------------------ loading
def validate_completed(records: list[dict[str, Any]], actor: str) -> dict[str, Any]:
    errors = []
    seen: set[str] = set()
    for r in records:
        jid = r.get("job_id")
        if jid in seen:
            errors.append(f"duplicate job {jid}")
        seen.add(jid)
        if r.get("schema_version") != CORPUS_SCHEMA:
            errors.append(f"schema {jid}")
        if r.get("model_key") != actor:
            errors.append(f"actor {jid}")
        if r.get("outcome_state") != COMPLETE_STATE:
            errors.append(f"state {jid}: {r.get('outcome_state')}")
        for f in ("native_answer", "native_probability", "dim_score", "full_activation_artifact", "full_activation_artifact_sha256"):
            if r.get(f) is None:
                errors.append(f"missing {f} {jid}")
        try:
            semantic_native(r)
        except KeyError:
            errors.append(f"native mapping {jid}")
    if errors:
        raise RuntimeError("completed-job validation failed: " + "; ".join(errors[:20]))
    worlds = {r["world_id"] for r in records}
    return {"jobs": len(records), "worlds": len(worlds), "actor": actor}


def semantic_native(r: dict[str, Any]) -> int:
    return SEMANTIC_MAPPINGS[r["answer_mapping_id"]][str(r["native_answer"]).strip()]


def load_activations(records: list[dict[str, Any]], artifact_root: Path, verify_hashes: bool = True) -> np.ndarray:
    rows = []
    for r in records:
        path = artifact_root / r["full_activation_artifact"]
        if verify_hashes and sha256_file(path) != r["full_activation_artifact_sha256"]:
            raise RuntimeError(f"activation hash mismatch: {r['job_id']}")
        with np.load(path) as z:
            rows.append(np.asarray(z["selected_layer_activation"], dtype=np.float64).reshape(-1))
    x = np.stack(rows)
    if not np.isfinite(x).all():
        raise RuntimeError("non-finite activation values")
    return x


# ------------------------------------------------------------------ features
def context_sentence_count(prompt: str) -> int:
    lines = prompt.split("\n")
    try:
        start = lines.index("Context:") + 1
    except ValueError:
        return 0
    count = 0
    for line in lines[start:]:
        if not line.strip():
            break
        count += 1
    return count


def nuisance_matrix(records: list[dict[str, Any]], include_dim: bool) -> np.ndarray:
    cols = []
    if include_dim:
        cols.append([float(r["dim_score"]) for r in records])
    cols.append([float(r["native_probability"]) for r in records])
    cols.append([float(len(r["prompt_text"])) for r in records])
    cols.append([float(context_sentence_count(r["prompt_text"])) for r in records])
    for m in MAPPINGS:
        cols.append([1.0 if r["answer_mapping_id"] == m else 0.0 for r in records])
    for lf in LEXICAL_FAMILIES:
        cols.append([1.0 if r["lexical_family"] == lf else 0.0 for r in records])
    return np.column_stack(cols)


def residualize(x_fit: np.ndarray, x_other: np.ndarray, z_fit: np.ndarray, z_other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fit_design = np.column_stack([np.ones(len(z_fit)), z_fit])
    other_design = np.column_stack([np.ones(len(z_other)), z_other])
    coef = np.linalg.pinv(fit_design) @ x_fit
    return x_fit - fit_design @ coef, x_other - other_design @ coef


class Table:
    """Column-oriented view of the records with the frozen label constructions."""

    def __init__(self, records: list[dict[str, Any]]):
        self.records = records
        g = lambda k: np.array([r[k] for r in records], dtype=object)  # noqa: E731
        self.world = g("world_id")
        self.split = g("world_split")
        self.elig = g("readout_eligibility")
        self.family = g("variant_family")
        self.member = np.array([int(r["contrast_member"]) for r in records])
        self.contrast = g("contrast_id")
        self.holder_role = g("holder_role")
        self.target_slot = g("target_slot")
        self.relation = g("relation_label")
        self.entailed = np.array([1 if r["entailment_label"] == "ENTAILED" else 0 for r in records])
        self.signed = np.isin(self.relation, ["SUPPORT", "OPPOSE"])
        self.sign = np.where(self.relation == "SUPPORT", 1, 0)
        self.state = np.where(self.relation == "SUPPORT", "SUPPORT", np.where(self.relation == "OPPOSE", "OPPOSE", "UNSIGNED")).astype(object)
        slot = np.where(self.holder_role == "CANONICAL_HOLDER", "C", np.where(self.holder_role == "ALTERNATE_HOLDER", "A", "X")).astype(object)
        self.joint = np.array([f"{s}_{'S' if sg else 'O'}" for s, sg in zip(slot, self.sign)], dtype=object)
        self.two_actor = np.isin(self.family, TWO_ACTOR_FAMILIES) & self.signed & np.isin(slot, ["C", "A"])
        self.multi_holder = np.isin(self.family, MULTI_HOLDER_FAMILIES) & self.two_actor
        self.canonical_supports = ((self.holder_role == "CANONICAL_HOLDER") & (self.relation == "SUPPORT")) | ((self.holder_role == "ALTERNATE_HOLDER") & (self.relation == "OPPOSE"))
        self.lexical_holdout = np.array([bool(r["lexical_holdout"]) for r in records])
        self.discourse_holdout = np.array([bool(r["discourse_holdout"]) for r in records])
        self.domain_holdout = np.array([bool(r["domain_holdout"]) for r in records])
        self.discourse = g("discourse_construction")
        self.template = g("lexical_template_id")
        self.native_token = np.array([str(r["native_answer"]).strip() for r in records], dtype=object)
        self.native_semantic = np.array([semantic_native(r) for r in records])
        self.fit = (self.split == "residual_probe_train") & (self.elig == "FIT")
        self.dev = (self.split == "development") & (self.elig == "DEVELOPMENT")
        self.eval = (self.split == "heldout_evaluation") & (self.elig == "FINAL_EVALUATION")
        self.query_only = self.family == "QUERY_ONLY"


# ------------------------------------------------------------ classifiers
def fit_logistic(x: np.ndarray, y: np.ndarray, c: float, seed: int = 0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(C=c, penalty="l2", class_weight="balanced", solver="lbfgs", max_iter=MAX_ITER, random_state=seed).fit(scaler.transform(x), y)
    return scaler, clf


def predict(model, x: np.ndarray) -> np.ndarray:
    scaler, clf = model
    return clf.predict(scaler.transform(x))


def balanced_accuracy(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    if len(y) == 0:
        return float("nan")
    w = np.ones(len(y)) if w is None else w
    recalls = []
    for c in np.unique(y):
        m = y == c
        if w[m].sum() <= 0:
            continue
        recalls.append(float((w[m] * (p[m] == c)).sum() / w[m].sum()))
    return float(np.mean(recalls)) if recalls else float("nan")


def macro_world_ba(y: np.ndarray, p: np.ndarray, worlds: np.ndarray) -> dict[str, float]:
    return {w: balanced_accuracy(y[worlds == w], p[worlds == w]) for w in np.unique(worlds)}


def select_c(x_fit: np.ndarray, y_fit: np.ndarray, x_dev: np.ndarray, y_dev: np.ndarray, w_dev: np.ndarray) -> tuple[float, dict[str, float]]:
    scores = {}
    for c in C_GRID:
        model = fit_logistic(x_fit, y_fit, c)
        per_world = macro_world_ba(y_dev, predict(model, x_dev), w_dev)
        scores[str(c)] = float(np.nanmean(list(per_world.values())))
    best = max(scores.values())
    chosen = min(c for c in C_GRID if scores[str(c)] >= best - 1e-12)
    return chosen, scores


class Readout:
    """Fits the frozen task classifiers for one feature family; native_answer is rule-based."""

    def __init__(self, name: str, t: Table, features: np.ndarray | None):
        self.name = name
        self.t = t
        self.x = features
        self.chosen_c: dict[str, float] = {}
        self.dev_scores: dict[str, dict[str, float]] = {}
        self.models: dict[str, Any] = {}
        if name != "native_answer":
            self._fit_task("sign", t.fit & t.signed, t.sign)
            self._fit_task("joint", t.fit & t.two_actor, t.joint)
            self._fit_task("entail", t.fit, t.entailed)
            self._fit_task("state", t.fit, t.state)
            self._fit_task("role_perm", t.fit & t.two_actor & t.canonical_supports, t.sign)
            self._fit_task("order", (t.split == "residual_probe_train") & (t.family == "SENTENCE_ORDER_SWAP") & (t.member == 0), t.sign)

    def _fit_task(self, task: str, fit_mask: np.ndarray, y: np.ndarray) -> None:
        t = self.t
        dev_mask = t.dev & (fit_mask | t.dev) if task in ("entail", "state") else t.dev & self._task_scope(task)
        if fit_mask.sum() == 0 or len(np.unique(y[fit_mask])) < 2:
            self.models[task] = None
            return
        if dev_mask.sum() > 0 and len(np.unique(y[dev_mask])) > 1:
            c, scores = select_c(self.x[fit_mask], y[fit_mask], self.x[dev_mask], y[dev_mask], t.world[dev_mask])
        else:
            c, scores = 1.0, {}
        self.chosen_c[task], self.dev_scores[task] = c, scores
        self.models[task] = fit_logistic(self.x[fit_mask], y[fit_mask], c)

    def _task_scope(self, task: str) -> np.ndarray:
        t = self.t
        return {"sign": t.signed, "joint": t.two_actor, "role_perm": t.two_actor & ~t.canonical_supports, "order": (t.family == "SENTENCE_ORDER_SWAP") & (t.member == 1)}.get(task, np.ones(len(t.records), bool))

    def _pred(self, task: str, mask: np.ndarray) -> np.ndarray:
        t = self.t
        if self.name == "native_answer":
            if task in ("sign", "role_perm", "order"):
                # signed rows are queried with operator SUPPORT except the negation family
                return np.where(t.native_semantic[mask] == 1, 1, 0)
            if task == "entail":
                return t.native_semantic[mask]
            if task == "state":
                return np.where(t.native_semantic[mask] == 1, "SUPPORT", "OPPOSE").astype(object)
            if task == "joint":
                return np.array([f"{'C' if h == 'CANONICAL_HOLDER' else 'A'}_{'S' if s else 'O'}" for h, s in zip(t.holder_role[mask], t.native_semantic[mask])], dtype=object)
        model = self.models.get(task)
        if model is None:
            return np.full(mask.sum(), np.nan if task in ("sign", "entail", "role_perm", "order") else "NONE", dtype=object)
        return predict(model, self.x[mask])

    # ---- per-world statistics for every endpoint (bootstrap resamples worlds)
    def world_stats(self) -> dict[str, dict[str, float]]:
        t = self.t
        ev = t.eval
        stats: dict[str, dict[str, float]] = {}

        self.degenerate: dict[str, bool] = getattr(self, "degenerate", {})

        def per_world_ba(mask: np.ndarray, y: np.ndarray, task: str, metric: str = "") -> dict[str, float]:
            m = ev & mask
            if m.sum() == 0:
                return {}
            if metric:
                self.degenerate[metric] = bool(len(np.unique(y[m])) < 2)
            p = self._pred(task, m)
            return macro_world_ba(y[m], p, t.world[m])

        def per_world_pooled(mask: np.ndarray, y: np.ndarray, task: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            m = ev & mask
            p = self._pred(task, m)
            out = {}
            for w in np.unique(t.world[m]):
                sel = t.world[m] == w
                out[w] = (y[m][sel], p[sel])
            return out

        def pair_rate(mask: np.ndarray, task: str, y: np.ndarray) -> dict[str, float]:
            m = ev & mask
            if m.sum() == 0:
                return {}
            p = self._pred(task, m)
            correct = (p == y[m])
            by_pair: dict[str, list[bool]] = defaultdict(list)
            for cid, ok in zip(t.contrast[m], correct):
                by_pair[cid].append(bool(ok))
            by_world: dict[str, list[float]] = defaultdict(list)
            for cid, oks in by_pair.items():
                by_world[cid.split("__")[0]].append(1.0 if len(oks) == 2 and all(oks) else 0.0)
            return {w: float(np.mean(v)) for w, v in by_world.items()}

        stats["macro_world_joint_binding_balanced_accuracy"] = per_world_ba(t.two_actor, t.joint, "joint")
        stats["joint_holder_target_relation_balanced_accuracy"] = per_world_ba(t.multi_holder, t.joint, "joint")
        stats["operator_sign_balanced_accuracy"] = per_world_ba(t.signed, t.sign, "sign")
        stats["holder_swap_correct_rate"] = pair_rate(t.family == "HOLDER_SWAP", "sign", t.sign)
        stats["target_swap_correct_rate"] = pair_rate(t.family == "TARGET_SWAP", "sign", t.sign)
        stats["same_context_two_query_sign_flip_correct_rate"] = pair_rate(t.family == "CONFLICTING_STANCES_IN_ONE_CONTEXT", "sign", t.sign)
        stats["reporter_quoter_attribution_correct_rate"] = pair_rate(np.isin(t.family, ["REPORTER_VERSUS_REPORTED_HOLDER", "QUOTER_VERSUS_QUOTED_HOLDER"]), "entail", t.entailed)
        stats["heldout_lexical_balanced_accuracy"] = per_world_ba(t.lexical_holdout & t.signed, t.sign, "sign", "heldout_lexical_balanced_accuracy")
        stats["heldout_discourse_balanced_accuracy"] = per_world_ba(t.discourse_holdout, t.entailed, "entail", "heldout_discourse_balanced_accuracy")
        stats["heldout_domain_balanced_accuracy"] = per_world_ba(t.domain_holdout & t.signed, t.sign, "sign", "heldout_domain_balanced_accuracy")
        # mapping reversal: the two members share semantics; agreement of semantic entailment predictions
        m = ev & (t.family == "ANSWER_MAPPING_REVERSAL")
        agree: dict[str, list[float]] = defaultdict(list)
        if m.sum():
            p = self._pred("entail", m)
            by_pair: dict[str, list[Any]] = defaultdict(list)
            for cid, v in zip(t.contrast[m], p):
                by_pair[cid].append(v)
            for cid, vals in by_pair.items():
                agree[cid.split("__")[0]].append(1.0 if len(vals) == 2 and vals[0] == vals[1] else 0.0)
        stats["mapping_reversal_semantic_agreement"] = {w: float(np.mean(v)) for w, v in agree.items()}
        # negation policy: relation-state readout on DOES_NOT_* rows predicting the binary opposite
        m = ev & np.isin(t.relation, ["DOES_NOT_SUPPORT", "DOES_NOT_OPPOSE"])
        fbo: dict[str, list[float]] = defaultdict(list)
        if m.sum():
            p = self._pred("state", m)
            for w, rel, pv in zip(t.world[m], t.relation[m], p):
                opposite = "OPPOSE" if rel == "DOES_NOT_SUPPORT" else "SUPPORT"
                fbo[w].append(1.0 if pv == opposite else 0.0)
        stats["false_binary_opposite_rate"] = {w: float(np.mean(v)) for w, v in fbo.items()}
        stats["fit_canonical_supports_evaluate_alternate_supports_balanced_accuracy"] = per_world_ba(t.two_actor & ~t.canonical_supports, t.sign, "role_perm")
        stats["fit_order_canonical_evaluate_order_reversed_sign_balanced_accuracy"] = per_world_ba((t.family == "SENTENCE_ORDER_SWAP") & (t.member == 1), t.sign, "order")
        return stats


# -------------------------------------------------------------- bootstrap
def bootstrap_world_stats(stats: dict[str, dict[str, float]], rng: np.random.Generator, replicates: int) -> dict[str, dict[str, Any]]:
    out = {}
    for metric, per_world in stats.items():
        worlds = sorted(per_world)
        vals = np.array([per_world[w] for w in worlds], dtype=float)
        vals = vals[np.isfinite(vals)] if len(vals) else vals
        if len(vals) == 0:
            out[metric] = {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "worlds": 0, "replicates": 0}
            continue
        point = float(vals.mean())
        counts = rng.multinomial(len(vals), np.full(len(vals), 1.0 / len(vals)), size=replicates)
        draws = (counts @ vals) / len(vals)
        out[metric] = {"point": point, "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)), "worlds": int(len(vals)), "replicates": int(replicates), "_draws": draws}
    return out


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    n = len(items)
    adjusted, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (n - i) * p))
        adjusted[k] = running
    return adjusted


def gate_secondaries(boot: dict[str, dict[str, Any]]) -> dict[str, Any]:
    raw_p = {}
    for metric, (kind, gate, hid) in SECONDARY_GATES.items():
        b = boot.get(metric)
        if b is None or "_draws" not in b:
            continue
        d = b["_draws"]
        raw_p[metric] = float(((d < gate).sum() if kind == "min" else (d > gate).sum()) + 1) / (len(d) + 1)
    adjusted = holm(raw_p)
    rows = {}
    for metric, (kind, gate, hid) in SECONDARY_GATES.items():
        b = boot.get(metric, {})
        point = b.get("point", float("nan"))
        passes_point = (point >= gate) if kind == "min" else (point <= gate)
        rows[metric] = {"hypothesis": hid, "gate": gate, "kind": kind, "point": point, "ci_low": b.get("ci_low"), "ci_high": b.get("ci_high"), "p_raw": raw_p.get(metric), "p_holm": adjusted.get(metric), "pass": bool(passes_point and adjusted.get(metric, 1.0) < 0.05)}
    return rows


# --------------------------------------------------------------- nulls
def null_readouts(t: Table, x: np.ndarray, nuis_no_dim: np.ndarray, dim: np.ndarray, rng: np.random.Generator) -> dict[str, Readout]:
    """Classifier-based nulls; the rule-based nulls are built in ``analyze``."""
    nulls: dict[str, Readout] = {}
    templates = sorted(set(t.template))
    # template-ID feature only: IID success permitted, held-out templates are all-zero rows
    nulls["N_A_LEXICAL_MEMORIZATION"] = Readout("N_A_LEXICAL_MEMORIZATION", t, np.column_stack([(t.template == tp).astype(float) for tp in templates]))
    return nulls


def pair_member0_rule(t: Table, families: tuple[str, ...]) -> Callable[[str, np.ndarray], np.ndarray]:
    """Heuristic with no coordinate for the queried slot: within the named swap
    families both contrast members receive member 0's answer (the same stance is
    read off regardless of which holder/target is queried); elsewhere the native
    semantic answer is used."""
    member0_sign: dict[str, int] = {}
    member0_entail: dict[str, int] = {}
    for i, r in enumerate(t.records):
        if r["variant_family"] in families and int(r["contrast_member"]) == 0:
            member0_sign[r["contrast_id"]] = int(t.sign[i])
            member0_entail[r["contrast_id"]] = int(t.entailed[i])

    def rule(task: str, mask: np.ndarray) -> np.ndarray:
        idx = np.where(mask)[0]
        sign = np.array([member0_sign.get(t.contrast[i], t.native_semantic[i]) if t.family[i] in families else t.native_semantic[i] for i in idx])
        ent = np.array([member0_entail.get(t.contrast[i], t.native_semantic[i]) if t.family[i] in families else t.native_semantic[i] for i in idx])
        if task in ("sign", "role_perm", "order"):
            return sign
        if task == "entail":
            return ent
        if task == "state":
            return np.where(sign == 1, "SUPPORT", "OPPOSE").astype(object)
        return np.array([f"{'C' if h == 'CANONICAL_HOLDER' else 'A'}_{'S' if s else 'O'}" for h, s in zip(t.holder_role[mask], sign)], dtype=object)

    return rule


def raw_token_rule(t: Table) -> Callable[[str, np.ndarray], np.ndarray]:
    """Candidate token identity without mapping normalization: A/1 read as yes."""
    raw = np.isin(t.native_token, ["A", "1"]).astype(int)

    def rule(task: str, mask: np.ndarray) -> np.ndarray:
        if task == "entail":
            return raw[mask]
        if task in ("sign", "role_perm", "order"):
            return raw[mask]
        if task == "state":
            return np.where(raw[mask] == 1, "SUPPORT", "OPPOSE").astype(object)
        return np.array([f"{'C' if h == 'CANONICAL_HOLDER' else 'A'}_{'S' if s else 'O'}" for h, s in zip(t.holder_role[mask], raw[mask])], dtype=object)

    return rule


class RuleReadout(Readout):
    """Rule-based nulls that replace the classifier predictions."""

    def __init__(self, name: str, t: Table, rule: Callable[[str, np.ndarray], np.ndarray]):
        self.name, self.t, self.x, self.rule = name, t, None, rule
        self.chosen_c, self.dev_scores, self.models = {}, {}, {}

    def _pred(self, task: str, mask: np.ndarray) -> np.ndarray:
        return self.rule(task, mask)


def quotation_rule(t: Table) -> Callable[[str, np.ndarray], np.ndarray]:
    marker = np.array([("reports that" in r["prompt_text"] or "quotes" in r["prompt_text"]) for r in t.records])

    def rule(task: str, mask: np.ndarray) -> np.ndarray:
        # assign the stance to whoever is queried whenever a report/quote marker appears
        if task == "entail":
            return np.where(marker[mask], 1, t.native_semantic[mask])
        if task in ("sign", "role_perm", "order"):
            return np.where(t.native_semantic[mask] == 1, 1, 0)
        if task == "state":
            return np.where(t.native_semantic[mask] == 1, "SUPPORT", "OPPOSE").astype(object)
        return np.array([f"{'C' if h == 'CANONICAL_HOLDER' else 'A'}_{'S' if s else 'O'}" for h, s in zip(t.holder_role[mask], t.native_semantic[mask])], dtype=object)

    return rule


def binary_negation_rule(t: Table) -> Callable[[str, np.ndarray], np.ndarray]:
    def rule(task: str, mask: np.ndarray) -> np.ndarray:
        rel = t.relation[mask]
        recoded = np.where(rel == "DOES_NOT_SUPPORT", "OPPOSE", np.where(rel == "DOES_NOT_OPPOSE", "SUPPORT", rel)).astype(object)
        if task == "state":
            return np.where(np.isin(recoded, ["SUPPORT", "OPPOSE"]), recoded, "UNSIGNED").astype(object)
        if task == "entail":
            return t.entailed[mask]
        if task in ("sign", "role_perm", "order"):
            return np.where(recoded == "SUPPORT", 1, 0)
        return np.array([f"{'C' if h == 'CANONICAL_HOLDER' else 'A'}_{'S' if r == 'SUPPORT' else 'O'}" for h, r in zip(t.holder_role[mask], recoded)], dtype=object)

    return rule


# ------------------------------------------------------- competitor battery
def competitor_battery(t: Table, x: np.ndarray, bundle: Path, manifest_entry: dict[str, Any]) -> dict[str, Any]:
    z = np.load(bundle)
    out: dict[str, Any] = {"directions": {}, "random_null_band": {}}
    ev_worlds = sorted(set(t.world[t.eval]))

    def sign_ba_from_projection(proj: np.ndarray) -> float:
        fit = t.fit & t.signed
        model = fit_logistic(proj[fit][:, None], t.sign[fit], 1.0)
        m = t.eval & t.signed
        return balanced_accuracy(t.sign[m], predict(model, proj[m][:, None]))

    def decomposition(proj: np.ndarray) -> dict[str, float]:
        hs, qo = {}, {}
        for i, r in enumerate(t.records):
            if r["variant_family"] == "HOLDER_SWAP":
                hs.setdefault(r["world_id"], {})[int(r["contrast_member"])] = proj[i]
            elif r["variant_family"] == "QUERY_ONLY":
                qo.setdefault(r["world_id"], {})[int(r["paired_reference_member"])] = proj[i]
        full, query = [], []
        for w in ev_worlds:
            if w in hs and w in qo and len(hs[w]) == 2 and len(qo[w]) == 2:
                full.append(hs[w][0] - hs[w][1])
                query.append(qo[w][0] - qo[w][1])
        full_a, query_a = np.array(full), np.array(query)
        ctx = full_a - query_a
        return {"worlds": int(len(full_a)), "full_context_paired_effect": float(full_a.mean()) if len(full_a) else float("nan"), "query_only_movement": float(query_a.mean()) if len(query_a) else float("nan"), "context_conditioned_effect": float(ctx.mean()) if len(ctx) else float("nan"), "context_conditioned_abs_effect": float(np.abs(ctx).mean()) if len(ctx) else float("nan")}

    for name in OBSERVED_DIRECTIONS:
        d = np.asarray(z[name], dtype=np.float64)
        expected = manifest_entry["directions"][name]["sha256"]
        actual = hashlib.sha256(np.ascontiguousarray(z[name]).tobytes()).hexdigest()
        proj = x @ d
        out["directions"][name] = {"sha256_pinned": expected, "sha256_bytes": actual, "hash_matches_manifest": actual == expected, "l2_norm": float(np.linalg.norm(d)), "sign_balanced_accuracy": sign_ba_from_projection(proj), **decomposition(proj)}
        rand = np.asarray(z[f"matched_norm_random__{name}"], dtype=np.float64)
        bas, ctxs = [], []
        for k in range(rand.shape[0]):
            p = x @ rand[k]
            bas.append(sign_ba_from_projection(p))
            ctxs.append(decomposition(p)["context_conditioned_abs_effect"])
        bas_a, ctx_a = np.array(bas), np.array(ctxs)
        out["random_null_band"][name] = {"count": int(rand.shape[0]), "sign_balanced_accuracy_q025": float(np.quantile(bas_a, 0.025)), "sign_balanced_accuracy_q975": float(np.quantile(bas_a, 0.975)), "sign_balanced_accuracy_max": float(bas_a.max()), "context_conditioned_abs_effect_q975": float(np.quantile(ctx_a, 0.975)), "observed_sign_ba_exceeds_random_q975": bool(out["directions"][name]["sign_balanced_accuracy"] > np.quantile(bas_a, 0.975)), "observed_context_effect_exceeds_random_q975": bool(out["directions"][name]["context_conditioned_abs_effect"] > np.quantile(ctx_a, 0.975))}
    return out


# ---------------------------------------------------------------- driver
def strip_draws(boot: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {k: {kk: vv for kk, vv in v.items() if kk != "_draws"} for k, v in boot.items()}


def analyze(actor: str, completed_jobs: Path, artifact_root: Path, direction_bundle: Path, competitor_manifest: Path, *, replicates: int = BOOTSTRAP_REPLICATES, verify_hashes: bool = True) -> dict[str, Any]:
    records = read_jsonl(completed_jobs)
    validation = validate_completed(records, actor)
    t = Table(records)
    x = load_activations(records, artifact_root, verify_hashes)
    dim = np.array([float(r["dim_score"]) for r in records])
    nuis_all = nuisance_matrix(records, include_dim=True)
    nuis_no_dim = nuisance_matrix(records, include_dim=False)
    resid = np.zeros_like(x)
    fit_rows = t.fit
    resid_fit, resid_other = residualize(x[fit_rows], x[~fit_rows], nuis_all[fit_rows], nuis_all[~fit_rows])
    resid[fit_rows], resid[~fit_rows] = resid_fit, resid_other
    features = {"native_answer": None, "dim_scalar": dim[:, None], "nuisance_only_logistic": nuis_no_dim, "dim_plus_nuisance_logistic": nuis_all, "full_activation_logistic": x, "residual_activation_logistic": resid}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    readouts = {name: Readout(name, t, features[name]) for name in READOUTS}
    results: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "actor": actor, "validation": validation, "rows": {"fit": int(t.fit.sum()), "development": int(t.dev.sum()), "evaluation": int(t.eval.sum()), "evaluation_worlds": int(len(set(t.world[t.eval])))}, "readouts": {}, "nulls": {}, "competitor_battery": {}}
    boots: dict[str, dict[str, Any]] = {}
    for name, ro in readouts.items():
        b = bootstrap_world_stats(ro.world_stats(), rng, replicates)
        boots[name] = b
        metrics = strip_draws(b)
        for metric, flag in ro.degenerate.items():
            if metric in metrics:
                metrics[metric]["degenerate_single_class"] = flag
        results["readouts"][name] = {"chosen_C": ro.chosen_c, "development_selection": ro.dev_scores, "metrics": metrics, "secondary_gates": gate_secondaries(b)}
        for metric, flag in ro.degenerate.items():
            if flag and metric in results["readouts"][name]["secondary_gates"]:
                results["readouts"][name]["secondary_gates"][metric]["degenerate_single_class"] = True
    # ---- primary gate (G1 core) for the residual readout vs the DIM baseline
    res_b, dim_b = boots["residual_activation_logistic"], boots["dim_scalar"]
    pm = "macro_world_joint_binding_balanced_accuracy"
    diff_draws = res_b[pm]["_draws"] - dim_b[pm]["_draws"]
    primary = {"metric": pm, "residual_point": res_b[pm]["point"], "residual_ci": [res_b[pm]["ci_low"], res_b[pm]["ci_high"]], "dim_point": dim_b[pm]["point"], "dim_ci": [dim_b[pm]["ci_low"], dim_b[pm]["ci_high"]], "residual_minus_dim": float(res_b[pm]["point"] - dim_b[pm]["point"]), "residual_minus_dim_ci": [float(np.quantile(diff_draws, 0.025)), float(np.quantile(diff_draws, 0.975))], "chance": JOINT_CHANCE, "margin_required": PRIMARY_MARGIN}
    primary["pass"] = bool(primary["residual_minus_dim"] >= PRIMARY_MARGIN and res_b[pm]["ci_low"] > JOINT_CHANCE)
    results["primary"] = primary
    # ---- nulls
    null_ros = null_readouts(t, x, nuis_no_dim, dim, rng)
    null_ros["N_A_QUOTATION_HEURISTIC"] = RuleReadout("N_A_QUOTATION_HEURISTIC", t, quotation_rule(t))
    null_ros["N_A_BINARY_NEGATION"] = RuleReadout("N_A_BINARY_NEGATION", t, binary_negation_rule(t))
    null_ros["N_A_BAG_OF_ENTITY"] = RuleReadout("N_A_BAG_OF_ENTITY", t, pair_member0_rule(t, ("HOLDER_SWAP", "CONFLICTING_STANCES_IN_ONE_CONTEXT")))
    null_ros["N_A_TARGET_FREE"] = RuleReadout("N_A_TARGET_FREE", t, pair_member0_rule(t, ("TARGET_SWAP",)))
    null_ros["N_A_ANSWER_TOKEN"] = RuleReadout("N_A_ANSWER_TOKEN", t, raw_token_rule(t))
    null_results = {}
    for nid, (metric, *rule) in NULL_EXPECTATIONS.items():
        if nid == "N_A_WORLD_SHUFFLE":
            # world-cluster label permutation: joint labels are permuted among the
            # two-actor rows of each world, in every split, so world composition and
            # per-world label marginals are preserved while row-label pairing is destroyed
            shuffled = Table(records)
            for w in sorted(set(t.world[t.two_actor])):
                idx = np.where((shuffled.world == w) & shuffled.two_actor)[0]
                shuffled.joint[idx] = shuffled.joint[idx][rng.permutation(len(idx))]
            ro = Readout("N_A_WORLD_SHUFFLE", shuffled, resid)
            b = bootstrap_world_stats(ro.world_stats(), rng, replicates)
            v = b[metric]
            null_results[nid] = {"metric": metric, "point": v["point"], "ci_low": v["ci_low"], "ci_high": v["ci_high"], "expected": "ci includes chance 0.25", "behaves": bool(v["ci_low"] <= JOINT_CHANCE <= v["ci_high"] or v["point"] < JOINT_CHANCE + 0.05)}
            continue
        if nid == "N_A_GLOBAL_VALENCE":
            v = boots["dim_scalar"][metric]
            degenerate = False
        else:
            ro = null_ros[nid]
            b = bootstrap_world_stats(ro.world_stats(), rng, replicates)
            v = b[metric]
            degenerate = bool(ro.degenerate.get(metric, False))
        kind, gate = rule
        behaves = (v["point"] < gate) if kind == "below" else (v["point"] > gate)
        row = {"metric": metric, "point": v["point"], "ci_low": v["ci_low"], "ci_high": v["ci_high"], "expected": f"{kind} {gate}", "behaves": bool(behaves)}
        if degenerate:
            row.update({"behaves": None, "not_evaluable": "evaluation rows for this metric contain a single class (corpus construction); the null cannot be tested here"})
        null_results[nid] = row
    results["nulls"] = null_results
    # ---- competitor battery + decomposition
    manifest = json.loads(competitor_manifest.read_text(encoding="utf-8"))
    entry = next(a for a in manifest["actors"] if a["actor"] == actor)
    results["competitor_battery"] = competitor_battery(t, x, direction_bundle, entry)
    # ---- G1 disposition
    full_gates = results["readouts"]["full_activation_logistic"]["secondary_gates"]
    role = full_gates["fit_canonical_supports_evaluate_alternate_supports_balanced_accuracy"]
    order = full_gates["fit_order_canonical_evaluate_order_reversed_sign_balanced_accuracy"]
    evaluable = {k: v for k, v in null_results.items() if v["behaves"] is not None}
    nulls_ok = all(v["behaves"] for v in evaluable.values())
    conjuncts = {"primary_residual_over_dim_and_above_chance": primary["pass"], "role_permutation_transfer_ge_0.60": bool(role["point"] >= 0.6), "order_crossed_transfer_ge_0.60": bool(order["point"] >= 0.6), "all_evaluable_nulls_behave": nulls_ok}
    results["G1"] = {"conjuncts": conjuncts, "transfer_readout": "full_activation_logistic", "nulls_evaluable": len(evaluable), "nulls_not_evaluable": [k for k, v in null_results.items() if v["behaves"] is None], "disposition": "BOUND_RELATIONAL_REPRESENTATION" if all(conjuncts.values()) else "HEURISTIC_OR_UNBOUND_REPRESENTATION", "failed_conjuncts": [k for k, v in conjuncts.items() if not v]}
    # ---- negative reporting
    negative = []
    for name in READOUTS:
        for metric, row in results["readouts"][name]["secondary_gates"].items():
            if not row["pass"]:
                negative.append(f"{name}: {row['hypothesis']} {metric} = {row['point']:.3f} (gate {row['kind']} {row['gate']}, Holm p {row['p_holm']})")
    for nid, v in null_results.items():
        if v["behaves"] is None:
            negative.append(f"null {nid} not evaluable: {v['not_evaluable']}")
        elif not v["behaves"]:
            negative.append(f"null {nid} did not behave: {v['metric']} = {v['point']:.3f}, expected {v['expected']}")
    for name in READOUTS:
        for metric, m in results["readouts"][name]["metrics"].items():
            if m.get("degenerate_single_class"):
                negative.append(f"{name}: {metric} evaluated on single-class rows (corpus construction); value is the recall of that class, not a balanced accuracy")
    simpler = results["readouts"]["dim_plus_nuisance_logistic"]["metrics"][pm]["point"]
    if simpler >= res_b[pm]["point"] - 1e-9:
        negative.append(f"simpler baseline tie/regression: dim_plus_nuisance_logistic joint BA {simpler:.3f} >= residual {res_b[pm]['point']:.3f}")
    results["negative_reporting"] = negative
    return results


def report_markdown(res: dict[str, Any]) -> str:
    pm = "macro_world_joint_binding_balanced_accuracy"
    lines = [f"# Track A v2 analysis — {res['actor']}", "", f"G1 disposition: **{res['G1']['disposition']}**  (failed: {res['G1']['failed_conjuncts'] or 'none'})", "", f"Rows: fit {res['rows']['fit']}, development {res['rows']['development']}, evaluation {res['rows']['evaluation']} ({res['rows']['evaluation_worlds']} worlds)", "", "## Primary", "", f"- residual readout {pm}: {res['primary']['residual_point']:.3f} CI [{res['primary']['residual_ci'][0]:.3f}, {res['primary']['residual_ci'][1]:.3f}]", f"- DIM baseline: {res['primary']['dim_point']:.3f} CI [{res['primary']['dim_ci'][0]:.3f}, {res['primary']['dim_ci'][1]:.3f}]", f"- residual − DIM: {res['primary']['residual_minus_dim']:.3f} CI [{res['primary']['residual_minus_dim_ci'][0]:.3f}, {res['primary']['residual_minus_dim_ci'][1]:.3f}] (needs ≥ 0.05, residual CI low > 0.25) → {'PASS' if res['primary']['pass'] else 'FAIL'}", "", "## Readouts × endpoints (point [95% CI])", "", "| endpoint | " + " | ".join(READOUTS) + " |", "|---|" + "---|" * len(READOUTS)]
    metrics = [pm] + list(SECONDARY_GATES)
    for metric in metrics:
        cells = []
        for name in READOUTS:
            m = res["readouts"][name]["metrics"].get(metric, {})
            cells.append(f"{m.get('point', float('nan')):.3f} [{m.get('ci_low', float('nan')):.3f}, {m.get('ci_high', float('nan')):.3f}]")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines += ["", "## Secondary gates (residual readout, Holm within model)", "", "| hypothesis | metric | point | gate | Holm p | pass |", "|---|---|---|---|---|---|"]
    for metric, row in res["readouts"]["residual_activation_logistic"]["secondary_gates"].items():
        lines.append(f"| {row['hypothesis']} | {metric} | {row['point']:.3f} | {row['kind']} {row['gate']} | {row['p_holm'] if row['p_holm'] is None else f'{row['p_holm']:.4f}'} | {row['pass']} |")
    lines += ["", "## Nulls", "", "| null | metric | point [CI] | expected | behaves |", "|---|---|---|---|---|"]
    for nid, v in res["nulls"].items():
        lines.append(f"| {nid} | {v['metric']} | {v['point']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}] | {v['expected']} | {v['behaves']} |")
    lines += ["", "## Competitor battery (evaluation worlds)", "", "| direction | hash ok | sign BA | random q97.5 | full-context paired | query-only | context-conditioned |abs| | random |abs| q97.5 |", "|---|---|---|---|---|---|---|---|"]
    for name, d in res["competitor_battery"]["directions"].items():
        band = res["competitor_battery"]["random_null_band"][name]
        lines.append(f"| {name} | {d['hash_matches_manifest']} | {d['sign_balanced_accuracy']:.3f} | {band['sign_balanced_accuracy_q975']:.3f} | {d['full_context_paired_effect']:.4f} | {d['query_only_movement']:.4f} | {d['context_conditioned_abs_effect']:.4f} | {band['context_conditioned_abs_effect_q975']:.4f} |")
    lines += ["", "## Negative reporting", ""] + [f"- {x}" for x in res["negative_reporting"]] if res["negative_reporting"] else ["", "## Negative reporting", "", "- none"]
    return "\n".join(lines) + "\n"


def external_review_packet(res: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    pm = "macro_world_joint_binding_balanced_accuracy"
    rg = res["readouts"]["residual_activation_logistic"]["secondary_gates"]
    return {
        "schema_version": "EXTERNAL_REVIEW_PACKET_V1",
        "unit": f"RELATION_REPRESENTATION_PROGRAM_V2 / Track A v2 / {res['actor']}",
        "research_question": {"question": "Q-BIND: is the cross-domain support/opposition readout a bound relational representation (who-supports-what) or a positional/identity/valence heuristic?", "hypothesis": "H_A_PRIMARY_RESIDUAL_BINDING", "target_variable": pm, "expected_positive_result": "residual readout ≥ DIM + 0.05 with world-bootstrap CI above 0.25, role-permutation and order-crossed transfer ≥ 0.60, all eight nulls behaving", "expected_falsifying_result": "any conjunct fails → frozen heuristic/unbound classification", "frozen_practical_threshold": {"primary_margin": PRIMARY_MARGIN, "chance": JOINT_CHANCE}},
        "capability_delta": {"new_runnable_behavior": "scripts/relation_binding_analysis_v2.py: frozen six-readout analysis with world-cluster bootstrap, Holm, eight nulls, competitor battery and effect decomposition on the v2 corpus forwards"},
        "evidence_topology": {"independent_unit": "world_id", "evaluation_worlds": res["rows"]["evaluation_worlds"], "evaluation_rows": res["rows"]["evaluation"], "fit_rows": res["rows"]["fit"], "development_rows": res["rows"]["development"], "clustering": "rows nest in worlds; paired contrast members share a world; bootstrap resamples worlds", "training_exposure": "fit worlds only for coefficients; development worlds only for C selection; evaluation worlds seen once"},
        "exact_empirical_result": {"G1": res["G1"], "primary": res["primary"], "residual_secondary_gates": rg, "all_readouts": {n: res["readouts"][n]["metrics"] for n in READOUTS}, "nulls": res["nulls"], "competitor_battery": res["competitor_battery"]},
        "negative_and_surprising_evidence": res["negative_reporting"] or ["no failed gate, null misbehaviour, or simpler-baseline tie"],
        "held_fixed": ["label operationalization in the module docstring", "C grid and development-world selection", "residualization predictors", "10,000 world bootstraps, seed 15914427", "Holm within model and endpoint family", "pinned direction hashes"],
        "competing_explanations": [
            {"name": "bound relational representation", "evidence_for": "primary pass plus transfer gates", "evidence_against": "a failed transfer gate or misbehaving null", "what_current_data_cannot_distinguish": "linear decodability from causal use of the binding", "smallest_discriminating_test": "Track B activation addendum steering (Unit 8)"},
            {"name": "identity/positional/valence heuristic", "evidence_for": "role-permutation transfer ≤ 0.4 or order-crossed transfer below chance", "evidence_against": "transfer gates ≥ 0.6", "what_current_data_cannot_distinguish": "mixtures of heuristics that jointly mimic binding on this corpus", "smallest_discriminating_test": "the eight nulls' designated failures"},
        ],
        "claim_boundary": "A passing result supports a model- and layer-specific linear residual binding claim on this controlled corpus; it does not establish a universal symbolic relation operator, behavioral causality, or robustness under arbitrary discourse.",
        "artifact_pickup": source,
    }


def csv_rows(res: dict[str, Any]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["readout", "metric", "point", "ci_low", "ci_high", "worlds"])
    for name in READOUTS:
        for metric, m in res["readouts"][name]["metrics"].items():
            w.writerow([name, metric, m["point"], m["ci_low"], m["ci_high"], m["worlds"]])
    return buf.getvalue().encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actor", required=True, choices=["llama", "gemma"])
    ap.add_argument("--completed-jobs", type=Path, required=True)
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--direction-bundle", type=Path, required=True)
    ap.add_argument("--competitor-manifest", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--source-commit", default="")
    ap.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = ap.parse_args(argv)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"output root not empty: {args.output_root}")
    res = analyze(args.actor, args.completed_jobs, args.artifact_root, args.direction_bundle, args.competitor_manifest, replicates=args.replicates)
    source = {"source_commit": args.source_commit, "completed_jobs_sha256": sha256_file(args.completed_jobs), "direction_bundle_sha256": sha256_file(args.direction_bundle), "script_sha256": sha256_file(Path(__file__).resolve()), "replicates": args.replicates}
    res["source"] = source
    packet = external_review_packet(res, source)
    A = args.actor.upper()
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_create_only(args.output_root / f"RELATION_BINDING_V2_RESULTS_{A}.json", canonical_json(res))
    write_create_only(args.output_root / f"RELATION_BINDING_V2_METRICS_{A}.csv", csv_rows(res))
    write_create_only(args.output_root / f"RELATION_BINDING_V2_REPORT_{A}.md", report_markdown(res).encode("utf-8"))
    write_create_only(args.output_root / f"external_review_packet_{args.actor}_v1.json", canonical_json(packet))
    files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(args.output_root.iterdir())]
    write_create_only(args.output_root / "artifact_manifest.json", canonical_json({"schema_version": "RELATION_BINDING_V2_RESULT_MANIFEST_V1", "actor": args.actor, "disposition": res["G1"]["disposition"], "files": files}))
    print(json.dumps({"actor": args.actor, "G1": res["G1"], "primary": {k: v for k, v in res["primary"].items() if k != "metric"}}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
