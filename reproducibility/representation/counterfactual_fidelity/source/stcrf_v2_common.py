#!/usr/bin/env python3
"""Shared frozen constants and helpers for STRONG_TARGET_COUNTERFACTUAL_RELATION_FIDELITY_V2 (STCRF v2).

Successor of CRIF v1 (COUNTERFACTUAL_RELATION_INTERVENTION_FIDELITY_V1). The endpoint mathematics
(robust scales, standardized group-balanced distance, CSR and its focal/invariance decomposition,
world bootstrap, Holm) are imported unchanged from crif_v1_common so their meaning cannot drift.
Nothing in this module reads a model outcome.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crif_v1_common as V1  # noqa: E402  frozen endpoint mathematics, V1 pools (for disjointness audits)
import rrrd_v1_common as R  # noqa: E402  frozen 8-shot formats, demonstrations, actor bindings

STUDY_ID = "STRONG_TARGET_COUNTERFACTUAL_RELATION_FIDELITY_V2"
SCHEMA_PREFIX = "STCRF_V2"
ROOT = R.ROOT
STUDY_ROOT = ROOT / "reports" / "strong_target_counterfactual_relation_fidelity_v2"
FREEZE_ROOT = STUDY_ROOT / "freeze"
V1_STUDY_ROOT = V1.STUDY_ROOT
TRACK_A_FREEZE = R.TRACK_A_FREEZE
ACTORS = R.ACTORS
# Witness layer: the top of the frozen RRRD layer band (a later state than the primary intervention site).
WITNESS_LAYER = {"llama": 24, "gemma": 32}

# ------------------------------------------------------------------ corpus
CORPUS_SEED = 20260904_172
FACTORS = V1.FACTORS  # direction, holder_slot, target_slot, sentence_order, distractor_sign, target_type, template_family
DIRECTIONS = V1.DIRECTIONS
REALIZATIONS = V1.REALIZATIONS
QUERIES = V1.QUERIES
CONSEQUENCE_QUERIES = V1.CONSEQUENCE_QUERIES
FOCAL_QUERIES = V1.FOCAL_QUERIES
INVARIANCE_QUERIES = V1.INVARIANCE_QUERIES
SPLITS = ("selection", "fit", "final")
SPLIT_SIZES = {"selection": 32, "fit": 64, "final": 96}
WORLD_COUNT = 192

# Fresh name pools: disjoint from Track A v2 and from the CRIF v1 pools (audited at freeze time).
GIVEN_NAMES = (
    "Ansel", "Beatrix", "Caspian", "Dagny", "Elias", "Freya", "Gunnar", "Hattie",
    "Ingrid", "Jasper", "Kai", "Lorcan", "Maeve", "Niall", "Odette", "Piers",
    "Quentin", "Rosalind", "Silas", "Thea", "Ursula", "Viggo", "Wallis", "Xavier",
    "Yolanda", "Zora", "Amos", "Bettina", "Cormac", "Dahlia", "Enzo", "Fiona",
)
SURNAMES = (
    "Barrington", "Castellan", "Draycott", "Ellery", "Farnsworth", "Gallagher", "Harcourt", "Iverson",
    "Jarvis", "Kingsley", "Lockhart", "Marchetti", "Norwood", "Okonkwo", "Prescott", "Quintrell",
    "Ravenscroft", "Sinclair", "Tremaine", "Upton", "Villiers", "Wexford", "Yeats", "Zimmerman",
    "Ainsley", "Brightwater", "Copeland", "Delacroix", "Esterhazy", "Fairweather", "Godfrey", "Hartigan",
)
# Demonstration-only name pools (fresh protocols P2/P3); disjoint from every study world.
DEMO_GIVEN = ("Leander", "Marisol", "Tobias", "Winifred", "Rafael", "Sigrid", "Percival", "Annika")
DEMO_SURNAMES = ("Pendleton", "Vasquez", "Holloway", "Stirling", "Beaumont", "Calder", "Whitcombe", "Lindgren")

# Two fresh domain-neutral target types; vocabulary disjoint from Track A domain words and the CRIF v1 target vocabulary.
TARGET_TYPES = ("VENUE_BOOKING", "ROUTE_REVISION")
EVENTS = ("orientation", "rehearsal", "briefing", "showcase", "tasting", "screening", "seminar", "clinic", "forum", "gala", "recital", "debate")
ROOMS = ("east hall", "atrium", "pavilion", "loft", "terrace", "chapel", "gallery", "sunroom")
LINES = ("harbor", "campus", "ridge", "riverside", "market", "airport", "stadium", "museum", "valley", "lakeside", "downtown", "hillside")
STOPS = ("gate", "pier", "plaza", "junction", "bridge", "square", "crossing", "landing")

# Fresh context lexical template families (disjoint from CRIF v1 ENDORSE_REJECT / BACK_DISPUTE and from Track A wording).
TEMPLATE_FAMILIES = {
    "AFFIRM_DENY": {"SUPPORT": "{holder} affirms the view that {target}.", "OPPOSE": "{holder} denies the view that {target}."},
    "CHAMPION_CONTEST": {"SUPPORT": "{holder} openly champions the view that {target}.", "OPPOSE": "{holder} openly contests the view that {target}."},
}
TEMPLATE_FAMILY_ORDER = ("AFFIRM_DENY", "CHAMPION_CONTEST")
# Query (claim) wording is part of the validated V1 protocol and is kept verbatim.
OPERATOR_CLAIM = V1.OPERATOR_CLAIM
PARAPHRASE_CLAIM = V1.PARAPHRASE_CLAIM
FACT_KEYS = V1.FACT_KEYS
ORDERS = V1.ORDERS

# ------------------------------------------------------------------ protocols
# P1 = V1's validated F2 protocol (frozen RRRD demonstrations, Context/Claim/Answer frame). Gemma uses P1.
# P2 = same frame and task line, fresh template-matched demonstrations.
# P3 = fresh demonstrations, alternative query wording (Statement / "established by").
PROTOCOLS = ("P1", "P2", "P3")
GEMMA_PROTOCOL = "P1"
LLAMA_CANDIDATE_PROTOCOLS = ("P1", "P2", "P3")
P3_F2_TASK_LINE = (
    "Task: Decide whether the STATEMENT is established by the CONTEXT. Use only what the "
    "context states. Reply with exactly ENTAILED or NOT_ENTAILED."
)
P3_F1_TASK_LINE = (
    "Task: Decide whether the STATEMENT is established by the CONTEXT. Use only what the "
    "context states. Follow the answer mapping shown below and output only the answer symbol."
)
FORMATS = ("F2", "F1")  # F2 = direct semantic label (primary); F1 = mapped symbol (robustness / mapping agreement)
PRIMARY_FORMAT = "F2"
PROTOCOL_SELECTION_RULE = "lexicographic on the 32 selection worlds using unsteered natural base/counterfactual behavior only: (1) number of natural reference gates passed (of five), (2) cross-template consistency mean, (3) balanced accuracy; frozen before any fit or final inference; no protocol passing all five gates -> ASSAY_NOT_QUALIFIED"

# ------------------------------------------------------------------ interventions
TAUS = (0.25, 0.50, 0.75, 1.00)
PRIMARY_TAU = 0.75
DIRECTION_ARMS = ("ORIGINAL_FROZEN_DIM", "REGIME_MATCHED_DIM", "REGIME_ANSWER_RESIDUAL_DIM", "MATCHED_SENTIMENT_DIRECTION", "MATCHED_ANSWER_TOKEN_DIRECTION", "MATCHED_TRUTH_DIRECTION")
ALL_TAU_ARMS = ("ORIGINAL_FROZEN_DIM", "REGIME_MATCHED_DIM")  # run at every tau (plus the analytic final-logit control)
PRIMARY_TAU_ONLY_ARMS = ("REGIME_ANSWER_RESIDUAL_DIM", "MATCHED_SENTIMENT_DIRECTION", "MATCHED_ANSWER_TOKEN_DIRECTION", "MATCHED_TRUTH_DIRECTION")
NATURAL_INTERPOLATION = "NATURAL_FULL_STATE_INTERPOLATION"
LOGIT_CONTROL = "MATCHED_FINAL_LOGIT_CONTROL"
FULL_PATCH = "FULL_STATE_SOURCE_PATCH"
TRACK_A_CONTROL_KEYS = {"MATCHED_SENTIMENT_DIRECTION": "sentiment", "MATCHED_ANSWER_TOKEN_DIRECTION": "answer_token", "MATCHED_TRUTH_DIRECTION": "factual"}
# Dense alpha grid in multiples of the world's natural full-state L2 displacement r_w = ||h_cf - h_base|| (Q1, realization A).
ALPHA_GRID = tuple(round(0.05 * k, 4) for k in range(1, 81)) + tuple(round(4.0 + 0.25 * k, 4) for k in range(1, 17))  # 0.05..4.00 step 0.05, 4.25..8.00 step 0.25
INTERPOLATION_GRID = tuple(round(0.05 * k, 4) for k in range(1, 41)) + tuple(round(2.0 + 0.25 * k, 4) for k in range(1, 9))  # 0.05..2.00, 2.25..4.00
GRID_ROWS_PER_FORWARD = 32
SCAN_POINTS_PER_FORWARD = 8  # prefix-cached scan: 8 grid points x 2 candidates = 16 rows per forward (peak memory headroom on the 40 GB card)
SCAN_PATH_MAX_ABS_DELTA_NATS = 0.25  # the prefix-cached scan must reproduce the frozen path within this on the receipt check
ROOT_RULE = "signed minimum-norm (revision v2, fixed on the fit worlds before the final-analysis freeze): both rays +u and -u are scanned over the grid on Q1 realization A; along each ray the margin curve g(m) is read in increasing m and the root is the linear interpolation between the first grid point with g(m_k) >= target and its predecessor (m_0 = 0, g(0) = base margin); the selected intervention is the reached root with the smaller multiple (tie -> the ray agreeing with the natural projection sign s_nat = sign(u . (h_cf - h_base))); if neither ray reaches the target the world is UNREACHABLE_IN_GRID; the interpolated signed multiple is applied unchanged to Q1-Q6 in both realizations on the frozen scoring path and the achieved Q1 margin is re-measured there"
ROOT_RULE_V1_SUPERSEDED = "revision v1 (corpus freeze text): only the ray s_nat u was scanned; superseded on the fit worlds after the first five calibrate checkpoints showed the regime direction's curve running away from the target on every SUPPORT_TO_OPPOSE world and the answer-token direction unreachable on every world along that ray; no final world had been forwarded"
SCAN_PATH_RULE = "the grid scan runs on a prefix-cached path (the prompt prefix through its penultimate token is computed once with use_cache; each grid point forwards only the final prompt token and the candidate tokens against the batch-expanded cache with the intervention applied at that final position); every measured row (natural, patch, applied arms) stays on the frozen right-padded batch-of-two full-forward path; a receipt compares the two paths on the first world of every run and fails closed above SCAN_PATH_MAX_ABS_DELTA_NATS"
TARGET_RULE = "target(tau) = margin_base(Q1) + tau * (margin_counterfactual(Q1) - margin_base(Q1)), Q1 realization A, primary format"
ACHIEVED_TOLERANCE = "abs(achieved_Q1 - target) <= max(0.5 nat, 0.10 * abs(natural Q1 change))"
SATURATION_RULE = "saturation: the crossing bracket jumps by more than 5 nats between adjacent grid points, or the selected multiple is the grid maximum; numerical invalidity: any non-finite candidate log-probability"
DOSE_VALIDITY = {"world_fraction_min": 0.90, "achieved_within_tolerance": True, "hard_answer_equals_counterfactual": True, "signed_margin_min_nats": 1.0, "no_invalidity_or_saturation": True, "norm_ratio_median_max": 2.0, "norm_ratio_p90_max": 4.0}
COVERAGE_MIN_FOR_TRUTH = 0.80
COMPARATOR_COVERAGE_MIN = 0.80
CSR_WORLD_SET_RULE = "per arm and tau, CSR is computed over the final worlds where that arm was applied with finite margins (root status APPLIED); n is always reported. The primary inferential object (REGIME_MATCHED_DIM at tau=0.75) is additionally reported on its dose-valid subset (achieved within tolerance, hard answer equal to the counterfactual answer, signed margin >= 1 nat, no saturation or numerical invalidity), and the dose is valid only if that subset covers >= 90% of final worlds. Paired comparisons use the worlds where both arms were applied. No world is dropped on the basis of a Q2-Q6 outcome."
FIDELITY_CI_LOW_FRACTION_OF_CTNC = 0.5
STRONG_CONTROL_INTERPOLATION_CI_LOW_MIN = 0.75
STRONG_CONTROL_RELATION_CI_HIGH_MAX = 0.10

# ------------------------------------------------------------------ gates
REFERENCE_GATES = {
    "primary_format_balanced_accuracy_min": 0.90,
    "mapped_format_semantic_agreement_min": 0.95,
    "focal_consequence_expected_direction_min": 0.95,
    "control_stability_ratio_max": 0.15,
    "cross_template_consistency_mean_min": 0.60,
    "cross_template_consistency_ci_low_min": 0.40,
    "full_state_patch_csr_min": 0.90,
}
NATURAL_ONLY_GATES = ("primary_format_balanced_accuracy", "mapped_format_semantic_agreement", "focal_consequence_expected_direction", "control_stability_ratio", "cross_template_consistency")
DIRECTION_GATES = {"cv_relation_sign_balanced_accuracy_min": 0.85, "mapping_agreement_min": 0.95, "cross_realization_balanced_accuracy_min": 0.80, "folds": 8}
WITNESS_GATES = {"balanced_accuracy_min": 0.85, "cross_template_balanced_accuracy_min": 0.80}
WITNESS_RIDGE_C = 0.05
BOOTSTRAP_REPLICATES = V1.BOOTSTRAP_REPLICATES
BOOTSTRAP_SEED = 20260904_1720
HOLM_ALPHA = 0.05
INTERPRETATIONS = {
    "STRONG_CONTROL_WITHOUT_COUNTERFACTUAL_FIDELITY": "reference, direction and target gates pass; natural full-state interpolation CSR CI low >= 0.75; primary relation CSR at tau=0.75 CI high <= 0.10 and focal CSR CI high <= 0.10; mean CSR <= 0.10 separately for SUPPORT_TO_OPPOSE, OPPOSE_TO_SUPPORT and realization B",
    "COUNTERFACTUAL_FIDELITY_SUPPORTED": "reference, direction and target gates pass; primary relation CSR CI low >= 0.5 x natural cross-template consistency mean; focal and invariance CSR CI low > 0; primary relation exceeds the matched final-logit, sentiment and answer-token controls (and truth when covered) with Holm-corrected paired intervals above zero, in both flip directions",
    "FORMAT_SHIFT_EXPLAINS_V1": "flag: the regime-matched direction meets the fidelity criteria while the original frozen direction does not",
    "TARGET_EFFECT_CAUSALLY_UNDERIDENTIFIED": "valid interventions reach the same Q1 target but have significantly different Q2-Q6 signatures (Holm-corrected pairwise CSR differences), or a semantically distinct direction is indistinguishable from relation steering while both fall short of the natural signature",
    "TARGET_REQUIRES_EXTREME_DISPLACEMENT": "the tau=0.75 target is reached within tolerance on >= 90% of final worlds but the norm-ratio budget (median <= 2, p90 <= 4) fails",
    "STRONG_TARGET_UNREACHABLE": "the tau=0.75 target is not reached within tolerance (with a correct hard answer, margin >= 1 nat, no invalidity or saturation) on >= 90% of final worlds within the frozen grid",
    "REFERENCE_ASSAY_INVALID": "any natural-reference gate fails on the final worlds",
    "ASSAY_NOT_QUALIFIED": "no predeclared Llama protocol passes the five natural gates on the selection worlds",
    "DIRECTION_NOT_QUALIFIED": "the regime-matched direction fails its cross-validated fit-population gates",
    "PARTIAL_OR_UNDERDETERMINED": "every other valid outcome",
    "MODEL_DEPENDENT": "joint: both assays valid, actor classifications differ",
}
LICENSED_WORDING = {
    "STRONG_CONTROL_WITHOUT_COUNTERFACTUAL_FIDELITY": "Rank-1 relation steering converted the direct answer with counterfactual-like strength but did not reproduce the broader consequences of changing the relation.",
    "COUNTERFACTUAL_FIDELITY_SUPPORTED": "Under the tested model, layer and relation task, rank-1 relation steering that converts the direct answer with counterfactual-like strength reproduces a structured portion of the held-out consequences of changing the relation, beyond matched output, sentiment and answer-token controls.",
    "TARGET_EFFECT_CAUSALLY_UNDERIDENTIFIED": "Reaching the same direct-answer target does not identify which intervention or semantic change occurred.",
    "TARGET_REQUIRES_EXTREME_DISPLACEMENT": "The strong direct-answer target is reachable along the rank-1 direction only with displacements outside the frozen norm budget; no fidelity claim.",
    "STRONG_TARGET_UNREACHABLE": "The strong direct-answer target is not reachable along the rank-1 direction within the frozen grid; no fidelity claim.",
}
PRECEDENCE = ("REFERENCE_ASSAY_INVALID", "DIRECTION_NOT_QUALIFIED", "TARGET_REQUIRES_EXTREME_DISPLACEMENT", "STRONG_TARGET_UNREACHABLE", "STRONG_CONTROL_WITHOUT_COUNTERFACTUAL_FIDELITY", "COUNTERFACTUAL_FIDELITY_SUPPORTED", "TARGET_EFFECT_CAUSALLY_UNDERIDENTIFIED", "PARTIAL_OR_UNDERDETERMINED")

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
write_json = V1.write_json
robust_scales = V1.robust_scales
squared_distance = V1.squared_distance
csr = V1.csr
bootstrap_mean = V1.bootstrap_mean
holm = V1.holm
GROUP_WEIGHTS = V1.GROUP_WEIGHTS


def bootstrap_indices(n: int, replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    return V1.bootstrap_indices(n, replicates, seed)


def stable_code(prefix: str, world_index: int, slot: str) -> str:
    h = hashlib.sha256(f"{STUDY_ID}|{CORPUS_SEED}|{world_index}|{slot}".encode()).hexdigest()
    return f"{prefix}-{h[:4]}"


def world_bits(world_index: int) -> dict[str, int]:
    return {f: (world_index >> k) & 1 for k, f in enumerate(FACTORS)}


def split_of(world_index: int) -> str | None:
    """Worlds 0..127 form a full 2^7 factorial: the V1-style even-parity 32 are selection worlds, the other 96 are final.
    Worlds 128..255 (bits of index-128) contribute the even-total-parity half fraction (64 worlds) as fit worlds."""
    if world_index < 128:
        b = world_bits(world_index)
        parity_all = sum(b.values()) % 2
        parity_sub = b["direction"] ^ b["target_slot"] ^ b["distractor_sign"]
        return "selection" if (parity_all == 0 and parity_sub == 0) else "final"
    j = world_index - 128
    if j >= 128:
        return None
    return "fit" if sum(world_bits(j).values()) % 2 == 0 else None


def factor_index(world_index: int) -> int:
    return world_index if world_index < 128 else world_index - 128


def expected_sign(direction: str, query: str) -> int:
    return V1.expected_sign(direction, query)


def load_v1_demonstrations() -> list[dict[str, Any]]:
    return V1.load_demonstrations()


# ------------------------------------------------------------------ prompt rendering per protocol
def render_f2_example_p3(context: str, claim: str, answer_label: str | None) -> str:
    lines = ["Context:", context, "", "Statement:", claim, "", "Answer:"]
    text = "\n".join(lines)
    if answer_label is not None:
        text += f" {answer_label}"
    return text


def render_f1_example_p3(context: str, claim: str, mapping_id: str, answer_symbol: str | None) -> str:
    sym_e, sym_n = R.MAPPING_SYMBOLS[mapping_id]
    ordered = sorted([(sym_e, R.F1_SEMANTICS[0]), (sym_n, R.F1_SEMANTICS[1])], key=lambda kv: kv[0])
    lines = ["Context:", context, "", "Statement:", claim, ""] + [f"{s}) {sem}" for s, sem in ordered] + ["Answer:"]
    text = "\n".join(lines)
    if answer_symbol is not None:
        text += f" {answer_symbol}"
    return text


def render_prompt(protocol: str, demos: list[dict[str, Any]], fmt: str, context: str, claim: str, mapping_id: str | None) -> tuple[str, list[str]]:
    """demos: list of {context, claim, entailment_label} in presentation order (P1: frozen RRRD demos; P2/P3: fresh demos)."""
    sep = R.SHOT_SEPARATOR
    if fmt == "F2":
        cands = [" ENTAILED", " NOT_ENTAILED"]
        if protocol in ("P1", "P2"):
            return R.render_f2_prompt(demos, context, claim), cands
        shots = [render_f2_example_p3(d["context"], d["claim"], d["entailment_label"]) for d in demos]
        return P3_F2_TASK_LINE + sep + sep.join(shots) + sep + render_f2_example_p3(context, claim, None), cands
    if fmt == "F1":
        sym_e, sym_n = R.MAPPING_SYMBOLS[mapping_id]
        cands = [f" {sym_e}", f" {sym_n}"]
        if protocol in ("P1", "P2"):
            return R.render_f1_prompt(demos, context, claim, mapping_id), cands
        shots = [render_f1_example_p3(d["context"], d["claim"], mapping_id, R.f1_symbol_for(mapping_id, d["entailment_label"])) for d in demos]
        return P3_F1_TASK_LINE + sep + sep.join(shots) + sep + render_f1_example_p3(context, claim, mapping_id, None), cands
    raise ValueError(fmt)


def prompt_id(world_id: str, protocol: str, fmt: str, real: str, state: str, q: str) -> str:
    return f"{world_id}__{protocol}__{fmt}__{real}__{state}__{q}"


# ------------------------------------------------------------------ directions
def load_track_a_bundle(actor: str) -> dict[str, np.ndarray]:
    return V1.load_direction_bundle(actor)


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 0:
        raise ValueError("zero or non-finite direction")
    return v / n


def load_direction_file(path: Path, expected: dict[str, str] | None = None) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as z:
        for k in z.files:
            arr = np.ascontiguousarray(z[k]).astype(np.float32)
            if expected is not None and k in expected and sha256_bytes(arr.tobytes()) != expected[k]:
                raise RuntimeError(f"direction {k} hash mismatch")
            out[k] = arr.astype(np.float64)
    return out


# ------------------------------------------------------------------ root finding (pure numpy; unit-tested)
def select_multiple(grid: list[float], margins: list[float], base_margin: float, target: float) -> dict[str, Any]:
    """Frozen ROOT_RULE. grid ascending multiples; margins the scanned Q1 margins; returns the interpolated multiple."""
    if not np.all(np.isfinite(margins)):
        return {"status": "NUMERICALLY_INVALID", "multiple": None}
    prev_m, prev_g = 0.0, float(base_margin)
    for m, g in zip(grid, margins):
        if g >= target:
            if g == prev_g:
                mult = float(m)
            else:
                mult = float(prev_m + (target - prev_g) / (g - prev_g) * (m - prev_m))
            mult = float(min(max(mult, prev_m), m))
            jump = float(g - prev_g)
            saturated = bool(jump > 5.0 or m >= grid[-1])
            return {"status": "REACHED", "multiple": mult, "bracket": [prev_m, float(m)], "bracket_margins": [prev_g, float(g)], "bracket_jump": jump, "saturated": saturated}
        prev_m, prev_g = float(m), float(g)
    return {"status": "UNREACHABLE_IN_GRID", "multiple": None, "max_margin": float(max(margins)), "max_multiple": float(grid[-1])}


def select_signed(grid: list[float], margins_plus: list[float], margins_minus: list[float], base_margin: float, target: float, natural_sign: float) -> dict[str, Any]:
    """ROOT_RULE v2: both rays; the reached root with the smaller multiple; tie -> the natural-projection ray."""
    rp = select_multiple(grid, margins_plus, base_margin, target)
    rm = select_multiple(grid, margins_minus, base_margin, target)
    cands = [(1.0, rp), (-1.0, rm)]
    reached = [(s, r) for s, r in cands if r["status"] == "REACHED"]
    if not reached:
        status = "NUMERICALLY_INVALID" if any(r["status"] == "NUMERICALLY_INVALID" for _, r in cands) else "UNREACHABLE_IN_GRID"
        return {"status": status, "multiple": None, "sign": None, "natural_sign": natural_sign, "rays": {"+1": rp, "-1": rm}}
    reached.sort(key=lambda sr: (sr[1]["multiple"], 0.0 if sr[0] == natural_sign else 1.0))
    s, r = reached[0]
    return {"status": "REACHED", "multiple": r["multiple"], "sign": s, "natural_sign": natural_sign, "sign_agrees_with_natural": bool(s == natural_sign), "bracket": r["bracket"], "bracket_margins": r["bracket_margins"], "bracket_jump": r["bracket_jump"], "saturated": r["saturated"], "rays": {"+1": rp, "-1": rm}}


def achieved_ok(achieved: float, target: float, natural_change: float) -> bool:
    return abs(achieved - target) <= max(0.5, 0.10 * abs(natural_change))


def balanced_accuracy(pred: list[int], truth: list[int]) -> float:
    p, t = np.asarray(pred), np.asarray(truth)
    if len(t) == 0:
        return float("nan")
    tpr = np.mean(p[t == 1] == 1) if np.any(t == 1) else np.nan
    tnr = np.mean(p[t == 0] == 0) if np.any(t == 0) else np.nan
    return float(np.nanmean([tpr, tnr]))
