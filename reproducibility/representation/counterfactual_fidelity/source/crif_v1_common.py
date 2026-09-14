#!/usr/bin/env python3
"""Shared frozen constants and helpers for COUNTERFACTUAL_RELATION_INTERVENTION_FIDELITY_V1 (CRIF v1).

Every script of the study (corpus builder, GPU runner, analysis/closeout, tests)
imports its constants from here so the freeze, the runner and the analysis cannot
drift. Nothing in this module reads a model outcome.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rrrd_v1_common as R  # noqa: E402  frozen 8-shot formats, demonstrations, actor bindings

STUDY_ID = "COUNTERFACTUAL_RELATION_INTERVENTION_FIDELITY_V1"
SCHEMA_PREFIX = "CRIF_V1"
ROOT = R.ROOT
STUDY_ROOT = ROOT / "reports" / "counterfactual_relation_intervention_fidelity_v1"
FREEZE_ROOT = STUDY_ROOT / "freeze"
RRRD_FREEZE = R.FREEZE_ROOT
TRACK_A_FREEZE = R.TRACK_A_FREEZE
ACTORS = R.ACTORS  # model ids, revisions, primary layers, direction hashes (frozen upstream)

# ------------------------------------------------------------------ corpus
CORPUS_SEED = 20260903_171
WORLD_COUNT = 128
DEVELOPMENT_WORLDS = 32
FINAL_WORLDS = 96
FACTORS = ("direction", "holder_slot", "target_slot", "sentence_order", "distractor_sign", "target_type", "template_family")
DIRECTIONS = ("SUPPORT_TO_OPPOSE", "OPPOSE_TO_SUPPORT")
REALIZATIONS = ("A", "B")
QUERIES = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")
CONSEQUENCE_QUERIES = ("Q2", "Q3", "Q4", "Q5", "Q6")
FOCAL_QUERIES = ("Q2", "Q3", "Q4")
INVARIANCE_QUERIES = ("Q5", "Q6")
GROUP_WEIGHTS = {"Q2": 1 / 3, "Q3": 1 / 3, "Q4": 1 / 3, "Q5": 1 / 2, "Q6": 1 / 2}  # equal total weight per group
FORMATS = ("F2", "F1")  # F2 = DIRECT_SEMANTIC_LABEL_8SHOT (primary); F1 = C9_MAPPED_ENTAILMENT_8SHOT (robustness)
PRIMARY_FORMAT = "F2"

# Fresh name pools: disjoint from the Track A v2 GIVEN_NAMES / SURNAMES pools.
GIVEN_NAMES = (
    "Adair", "Bryn", "Corin", "Delia", "Emrys", "Fenna", "Gideon", "Halle",
    "Idris", "Juno", "Kestrel", "Liora", "Marlow", "Nell", "Osric", "Perrin",
    "Quilla", "Rowan", "Soren", "Tamsin", "Ulric", "Vesna", "Wilder", "Xanthe",
    "Yusra", "Zane", "Anouk", "Basil", "Cyra", "Dov", "Elowen", "Farid",
)
SURNAMES = (
    "Ashcombe", "Bellweather", "Calloway", "Dunmore", "Everly", "Fairbank", "Greaves", "Hollis",
    "Ingram", "Jessop", "Kirkwood", "Lindqvist", "Mortlake", "Nightingale", "Oakes", "Pemberton",
    "Quayle", "Rossiter", "Selwyn", "Thackeray", "Underhill", "Vance", "Whitlock", "Yardley",
    "Ziegler", "Abernathy", "Blakemore", "Crowther", "Danforth", "Ellsworth", "Fenwick", "Garland",
)
# Two domain-neutral target types (scheduling proposals, allocation plans); no Track A domain vocabulary.
TARGET_TYPES = ("SCHEDULE_PROPOSAL", "ALLOCATION_PLAN")
SCHEDULE_NOUNS = ("budget", "roster", "inventory", "audit", "maintenance", "onboarding", "vendor", "safety", "payroll", "catering", "parking", "signage")
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
ALLOCATION_RESOURCES = ("spare van", "second projector", "reserve generator", "mobile crane", "loaner laptop set", "backup printer", "portable stage", "courier account", "field kit", "extra freezer", "shared minibus", "relief shift")
ALLOCATION_PLACES = ("depot", "annex", "wing", "hub", "station", "yard", "floor", "unit")

# Context lexical template families (disjoint from Track A: supports/opposes the proposition that, does not support,
# considered ... withheld, reports that, called it the conclusion to accept/reject, said the evidence ...).
TEMPLATE_FAMILIES = {
    "ENDORSE_REJECT": {"SUPPORT": "{holder} endorses the claim that {target}.", "OPPOSE": "{holder} rejects the claim that {target}."},
    "BACK_DISPUTE": {"SUPPORT": "{holder} publicly backs the claim that {target}.", "OPPOSE": "{holder} publicly disputes the claim that {target}."},
}
TEMPLATE_FAMILY_ORDER = ("ENDORSE_REJECT", "BACK_DISPUTE")
# Claim (query) wording. Q1/Q2/Q5/Q6 use the frozen protocol's operator skeleton; Q3/Q4 are the paraphrases.
OPERATOR_CLAIM = {"SUPPORT": "{holder} supports the proposition that {target}.", "OPPOSE": "{holder} opposes the proposition that {target}."}
PARAPHRASE_CLAIM = {"SUPPORT": "{holder} is in favor of the proposition that {target}.", "OPPOSE": "{holder} is against the proposition that {target}."}
FACT_KEYS = ("focal", "holder_control", "target_control", "diagonal")
# Sentence orders for the four facts; realization B never repeats realization A's order.
ORDERS = {
    ("A", 0): ("focal", "holder_control", "target_control", "diagonal"),
    ("A", 1): ("diagonal", "target_control", "holder_control", "focal"),
    ("B", 0): ("holder_control", "focal", "diagonal", "target_control"),
    ("B", 1): ("target_control", "diagonal", "focal", "holder_control"),
}

# ------------------------------------------------------------------ interventions
INTERVENTIONS = ("NO_INTERVENTION", "FULL_STATE_SOURCE_PATCH", "RAW_RELATION_COORDINATE_INTERCHANGE", "RESIDUAL_RELATION_COORDINATE_INTERCHANGE", "MATCHED_SENTIMENT_CONTROL", "MATCHED_TRUTH_CONTROL", "MATCHED_ANSWER_TOKEN_CONTROL", "MATCHED_FINAL_LOGIT_CONTROL")
CONTROL_DIRECTIONS = {"MATCHED_SENTIMENT_CONTROL": "sentiment", "MATCHED_TRUTH_CONTROL": "factual", "MATCHED_ANSWER_TOKEN_CONTROL": "answer_token"}
RELATION_DIRECTIONS = {"RAW_RELATION_COORDINATE_INTERCHANGE": "raw_relation_dim", "RESIDUAL_RELATION_COORDINATE_INTERCHANGE": "jointly_residualized_relation"}
DIRECTION_KEYS = ("raw_relation_dim", "jointly_residualized_relation", "sentiment", "factual", "answer_token")
# Control magnitude grid, in multiples of the raw relation displacement L2 norm ||delta * d|| of that world (Q1, realization A).
INITIAL_CONTROL_GRID_MULTIPLES = (0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
GRID_EXTENSION_MULTIPLES = (24.0, 32.0)
GRID_EXTENSION_RULE = "if fewer than 90% of development worlds select a magnitude strictly inside the initial grid range for a control, the final grid appends the extension multiples (both signs); otherwise it is the initial grid unchanged"
MATCH_TOLERANCE_RELATIVE = 0.20
MATCH_TOLERANCE_ABSOLUTE_NATS = 0.25
COMPARATOR_VALIDITY_MIN_MATCH_FRACTION = 0.80
SELECTION_RULE = "argmin over the signed grid of |dQ1(control, m) - dQ1(raw relation)| on Q1 realization A; ties broken toward the smaller |m|, then toward the positive sign"

# ------------------------------------------------------------------ endpoint / gates
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260903_1710
SCALE_FLOOR_FRACTION_OF_MEDIAN_FOCAL_SCALE = 0.25
MAD_TO_SIGMA = 1.4826
REFERENCE_GATES = {
    "primary_format_balanced_accuracy_min": 0.90,
    "mapped_format_semantic_agreement_min": 0.95,
    "focal_consequence_expected_direction_min": 0.90,
    "control_stability_ratio_max": 0.25,  # median |N| on Q5-Q6 divided by median |N| on Q2-Q4 must be at most this
    "cross_template_consistency_mean_min": 0.50,
    "cross_template_consistency_ci_low_min": 0.25,
}
TARGET_EFFECT_GATES = {"sign_agreement_min": 0.75, "bootstrap_interval_above_zero": True}
WITNESS_GATES = {"no_intervention_balanced_accuracy_min": 0.85}
WITNESS_TRAINING_SPLITS = ("development",)  # old Track A v2 development worlds only (RRRD F2 activations)
WITNESS_RIDGE_C = 0.05
HOLM_ALPHA = 0.05

ACTOR_CLASSES = ("REFERENCE_ASSAY_INVALID", "NO_RELIABLE_CAUSAL_TARGET_EFFECT", "COUNTERFACTUAL_RELATION_FIDELITY_SUPPORTED", "BEHAVIORAL_CONTROL_WITHOUT_COUNTERFACTUAL_FIDELITY", "TARGET_EFFECT_CAUSALLY_UNDERIDENTIFIED", "SHARED_OR_NONUNIQUE_CONTROL_SIGNATURE", "MIXED_OR_UNDERDETERMINED")
JOINT_CLASSES = ("REPLICATED_FIDELITY", "REPLICATED_BEHAVIOR_WITHOUT_FIDELITY", "REPLICATED_TARGET_UNDERIDENTIFICATION", "MODEL_DEPENDENT", "MIXED_OR_UNDERDETERMINED", "INVALID_ASSAY")
LICENSED_WORDING = {
    "COUNTERFACTUAL_RELATION_FIDELITY_SUPPORTED": "Under the tested models, layers, and relation task, a one-dimensional relation intervention reproduces a structured portion of the counterfactual consequences of changing the queried relation, beyond a matched output intervention and semantically adjacent controls.",
    "BEHAVIORAL_CONTROL_WITHOUT_COUNTERFACTUAL_FIDELITY": "Steering reliably changed the scored relation answer without reproducing the broader counterfactual signature of changing the relation itself.",
    "TARGET_EFFECT_CAUSALLY_UNDERIDENTIFIED": "The target-answer effect alone does not identify which intervention or semantic change occurred.",
    "SHARED_OR_NONUNIQUE_CONTROL_SIGNATURE": "Under the measured endpoint panel, semantically distinct interventions access overlapping downstream control effects.",
}

# ------------------------------------------------------------------ helpers (re-exported)
canonical_json = R.canonical_json
sha256_bytes = R.sha256_bytes
sha256_file = R.sha256_file
sha256_text = R.sha256_text
read_json = R.read_json
read_jsonl = R.read_jsonl
jsonl_bytes = R.jsonl_bytes
write_create_only = R.write_create_only
manifest_for = R.manifest_for


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def stable_code(prefix: str, world_index: int, slot: str) -> str:
    h = hashlib.sha256(f"{STUDY_ID}|{CORPUS_SEED}|{world_index}|{slot}".encode()).hexdigest()
    return f"{prefix}-{h[:4]}"


def world_bits(world_index: int) -> dict[str, int]:
    return {f: (world_index >> k) & 1 for k, f in enumerate(FACTORS)}


def is_development(world_index: int) -> bool:
    b = world_bits(world_index)
    parity_all = sum(b.values()) % 2
    parity_sub = b["direction"] ^ b["target_slot"] ^ b["distractor_sign"]
    return parity_all == 0 and parity_sub == 0


def expected_sign(direction: str, query: str) -> int:
    """Expected sign of the natural semantic-margin change (counterfactual minus base) per query."""
    s = 1 if direction == "SUPPORT_TO_OPPOSE" else -1
    return {"Q1": +1, "Q2": -1, "Q3": -s, "Q4": +s, "Q5": 0, "Q6": 0}[query]


def load_demonstrations() -> list[dict[str, Any]]:
    demos = read_jsonl(RRRD_FREEZE / "TRACK_A_DEMONSTRATION_FREEZE.jsonl")
    if len(demos) != 8:
        raise RuntimeError("expected eight frozen demonstrations")
    return sorted(demos, key=lambda d: d["presentation_position"])


def render_prompt(demos: list[dict[str, Any]], fmt: str, context: str, claim: str, mapping_id: str | None) -> tuple[str, list[str]]:
    if fmt == "F2":
        return R.render_f2_prompt(demos, context, claim), [" ENTAILED", " NOT_ENTAILED"]
    if fmt == "F1":
        sym_e, sym_n = R.MAPPING_SYMBOLS[mapping_id]
        return R.render_f1_prompt(demos, context, claim, mapping_id), [f" {sym_e}", f" {sym_n}"]
    raise ValueError(fmt)


def load_direction_bundle(actor: str) -> dict[str, np.ndarray]:
    """Frozen competitor direction bundle (Track A v2 freeze), hash-verified; returned as float64 vectors."""
    bundle = TRACK_A_FREEZE / "runtime" / actor / "competitor_directions_v2.npz"
    manifest = read_json(TRACK_A_FREEZE / "TRACK_A_COMPETITOR_DIRECTIONS.json")
    ent = next(a for a in manifest["actors"] if a["actor"] == actor)
    if sha256_file(bundle) != ent["bundle"]["sha256"]:
        raise RuntimeError("competitor direction bundle hash mismatch")
    out: dict[str, np.ndarray] = {}
    with np.load(bundle, allow_pickle=False) as z:
        for key in DIRECTION_KEYS:
            arr = np.ascontiguousarray(z[key])
            if sha256_bytes(arr.tobytes()).lower() != ent["directions"][key]["sha256"]:
                raise RuntimeError(f"direction {key} hash mismatch")
            out[key] = np.asarray(arr, dtype=np.float64)
    if sha256_bytes(np.ascontiguousarray(out["raw_relation_dim"].astype(np.float32)).tobytes()) != ACTORS[actor]["raw_relation_dim_sha256"]:
        raise RuntimeError("raw relation DIM is not the frozen actor direction")
    return out


def direction_manifest_entry(actor: str) -> dict[str, Any]:
    manifest = read_json(TRACK_A_FREEZE / "TRACK_A_COMPETITOR_DIRECTIONS.json")
    ent = next(a for a in manifest["actors"] if a["actor"] == actor)
    return {"bundle": ent["bundle"], "directions": {k: ent["directions"][k] for k in DIRECTION_KEYS}, "dimension": ent["dimension"]}


# ------------------------------------------------------------------ endpoint mathematics (pure numpy; unit-tested)
def robust_scales(dev_natural: np.ndarray) -> np.ndarray:
    """dev_natural: (n_dev_worlds, 5) natural signature over Q2..Q6. Returns per-dimension scales."""
    med = np.median(dev_natural, axis=0)
    mad = np.median(np.abs(dev_natural - med), axis=0) * MAD_TO_SIGMA
    focal_idx = [CONSEQUENCE_QUERIES.index(q) for q in FOCAL_QUERIES]
    floor = SCALE_FLOOR_FRACTION_OF_MEDIAN_FOCAL_SCALE * float(np.median(mad[focal_idx]))
    return np.maximum(mad, floor)


def group_weights() -> np.ndarray:
    return np.array([GROUP_WEIGHTS[q] for q in CONSEQUENCE_QUERIES], dtype=np.float64)


def squared_distance(a: np.ndarray, b: np.ndarray, scales: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Frozen standardized, group-balanced squared Euclidean distance along the last axis."""
    w = group_weights()
    if mask is not None:
        w = w * mask
    z = (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) / scales
    return np.sum(w * z * z, axis=-1)


def csr(intervention_sig: np.ndarray, natural_sig: np.ndarray, scales: np.ndarray) -> dict[str, np.ndarray]:
    """COUNTERFACTUAL_SIGNATURE_RECOVERY per world plus its focal and invariance decomposition.

    CSR = 1 - D(I,N)^2 / D(0,N)^2. With D^2 = D_focal^2 + D_inv^2 the decomposition
    CSR_focal = 1 - D_focal(I,N)^2 / D(0,N)^2 and CSR_inv = 1 - D_inv(I,N)^2 / D(0,N)^2
    satisfies CSR = CSR_focal + CSR_inv - 1. Q1 never enters (signatures are Q2..Q6 only).
    """
    I = np.asarray(intervention_sig, dtype=np.float64)
    N = np.asarray(natural_sig, dtype=np.float64)
    if I.shape[-1] != len(CONSEQUENCE_QUERIES) or N.shape[-1] != len(CONSEQUENCE_QUERIES):
        raise ValueError("signatures must span exactly Q2..Q6")
    focal_mask = np.array([1.0 if q in FOCAL_QUERIES else 0.0 for q in CONSEQUENCE_QUERIES])
    inv_mask = 1.0 - focal_mask
    d0 = squared_distance(np.zeros_like(N), N, scales)
    total = 1.0 - squared_distance(I, N, scales) / d0
    focal = 1.0 - squared_distance(I, N, scales, focal_mask) / d0
    inv = 1.0 - squared_distance(I, N, scales, inv_mask) / d0
    return {"csr": total, "csr_focal": focal, "csr_invariance": inv}


def bootstrap_indices(n: int, replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(replicates, n))


def bootstrap_mean(values: np.ndarray, idx: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_two = 2 * min(float(np.mean(means <= 0)), float(np.mean(means >= 0)))
    return {"mean": float(v.mean()), "ci_low": float(lo), "ci_high": float(hi), "bootstrap_p_two_sided": min(1.0, p_two), "n": int(v.shape[0])}


def holm(pvalues: dict[str, float], alpha: float = HOLM_ALPHA) -> dict[str, dict[str, float | bool]]:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict[str, float | bool]] = {}
    running = 0.0
    for k, (name, p) in enumerate(items):
        adj = min(1.0, (m - k) * p)
        running = max(running, adj)
        out[name] = {"p_raw": float(p), "p_holm": float(running), "alpha_step": alpha / (m - k), "reject": bool(running < alpha)}
    return out
