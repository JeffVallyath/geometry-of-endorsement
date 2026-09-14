#!/usr/bin/env python3
"""Corpus builder and pre-inference freeze for STRONG_TARGET_COUNTERFACTUAL_RELATION_FIDELITY_V2.

Deterministically builds the fresh 192-world counterfactual corpus (32 protocol-selection / 64
direction-and-witness-fit / 96 final held-out), the Q1-Q6 battery with V1's semantic structure, two
surface realizations per world, the fresh 8-shot demonstrations for the Llama candidate protocols
P2/P3, the prompt manifest (hash of every rendered prompt under every protocol and format; the GPU
runner re-renders and verifies), the study contract, the input/direction manifest and the freeze
artifact manifest. Create-only. Reads no model outcome.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crif_v1_common as V1  # noqa: E402
import rrrd_v1_common as R  # noqa: E402
import stcrf_v2_common as C  # noqa: E402

FLIP = {"SUPPORT": "OPPOSE", "OPPOSE": "SUPPORT"}


def holder_names(i: int) -> tuple[str, str]:
    a = f"{C.GIVEN_NAMES[(2 * i) % 32]} {C.SURNAMES[(3 * i) % 32]}"
    b = f"{C.GIVEN_NAMES[(2 * i + 1) % 32]} {C.SURNAMES[(3 * i + 7) % 32]}"
    if a == b:
        raise RuntimeError("holder names collide")
    return a, b


def targets(i: int, target_type: str) -> tuple[str, str]:
    if target_type == "VENUE_BOOKING":
        ta = f"booking {C.stable_code('b', i, 'TA')} relocates the {C.EVENTS[(2 * i) % 12]} session to the {C.ROOMS[i % 8]}"
        tb = f"booking {C.stable_code('b', i, 'TB')} relocates the {C.EVENTS[(2 * i + 1) % 12]} session to the {C.ROOMS[(i + 3) % 8]}"
    else:
        ta = f"revision {C.stable_code('v', i, 'TA')} sends the {C.LINES[(2 * i) % 12]} shuttle via {C.STOPS[i % 8]} {3 + (i % 17)}"
        tb = f"revision {C.stable_code('v', i, 'TB')} sends the {C.LINES[(2 * i + 1) % 12]} shuttle via {C.STOPS[(i + 3) % 8]} {21 + (i % 13)}"
    return ta, tb


def build_world(i: int, mapping_id: str, split: str) -> dict[str, Any]:
    b = C.world_bits(C.factor_index(i))
    direction = C.DIRECTIONS[b["direction"]]
    name_a, name_b = holder_names(i)
    target_type = C.TARGET_TYPES[b["target_type"]]
    t_a, t_b = targets(i, target_type)
    hf, ho = (name_a, name_b) if b["holder_slot"] == 0 else (name_b, name_a)
    tf, to = (t_a, t_b) if b["target_slot"] == 0 else (t_b, t_a)
    focal_base = "SUPPORT" if direction == "SUPPORT_TO_OPPOSE" else "OPPOSE"
    focal_cf = FLIP[focal_base]
    hc = "SUPPORT" if b["distractor_sign"] == 0 else "OPPOSE"
    tc = "SUPPORT" if (b["distractor_sign"] ^ b["direction"]) == 0 else "OPPOSE"
    dd = "SUPPORT" if (b["distractor_sign"] ^ b["holder_slot"]) == 0 else "OPPOSE"
    facts = {
        "focal": {"holder": hf, "target": tf, "base": focal_base, "counterfactual": focal_cf},
        "holder_control": {"holder": ho, "target": tf, "base": hc, "counterfactual": hc},
        "target_control": {"holder": hf, "target": to, "base": tc, "counterfactual": tc},
        "diagonal": {"holder": ho, "target": to, "base": dd, "counterfactual": dd},
    }
    q5_entailed = (b["sentence_order"] ^ b["target_type"]) == 0
    q6_entailed = (b["holder_slot"] ^ b["template_family"]) == 0
    queries = {
        "Q1": {"role": "calibration", "holder": hf, "target": tf, "wording": "operator", "operator": focal_cf, "label_base": "NOT_ENTAILED", "label_counterfactual": "ENTAILED"},
        "Q2": {"role": "complement", "holder": hf, "target": tf, "wording": "operator", "operator": focal_base, "label_base": "ENTAILED", "label_counterfactual": "NOT_ENTAILED"},
        "Q3": {"role": "paraphrase_support", "holder": hf, "target": tf, "wording": "paraphrase", "operator": "SUPPORT", "label_base": "ENTAILED" if focal_base == "SUPPORT" else "NOT_ENTAILED", "label_counterfactual": "ENTAILED" if focal_cf == "SUPPORT" else "NOT_ENTAILED"},
        "Q4": {"role": "paraphrase_opposition", "holder": hf, "target": tf, "wording": "paraphrase", "operator": "OPPOSE", "label_base": "ENTAILED" if focal_base == "OPPOSE" else "NOT_ENTAILED", "label_counterfactual": "ENTAILED" if focal_cf == "OPPOSE" else "NOT_ENTAILED"},
        "Q5": {"role": "holder_control", "holder": ho, "target": tf, "wording": "operator", "operator": hc if q5_entailed else FLIP[hc], "label_base": "ENTAILED" if q5_entailed else "NOT_ENTAILED", "label_counterfactual": "ENTAILED" if q5_entailed else "NOT_ENTAILED"},
        "Q6": {"role": "target_control", "holder": hf, "target": to, "wording": "operator", "operator": tc if q6_entailed else FLIP[tc], "label_base": "ENTAILED" if q6_entailed else "NOT_ENTAILED", "label_counterfactual": "ENTAILED" if q6_entailed else "NOT_ENTAILED"},
    }
    for q, spec in queries.items():
        tmpl = C.OPERATOR_CLAIM if spec["wording"] == "operator" else C.PARAPHRASE_CLAIM
        spec["claim"] = tmpl[spec["operator"]].format(holder=spec["holder"], target=spec["target"])
        spec["expected_sign"] = C.expected_sign(direction, q)
        # focal stance in each state for the queried pair (used by the direction fit and the witness); controls have fixed stance
        if q in ("Q1", "Q2", "Q3", "Q4"):
            spec["queried_stance"] = {"base": focal_base, "counterfactual": focal_cf}
        elif q == "Q5":
            spec["queried_stance"] = {"base": hc, "counterfactual": hc}
        else:
            spec["queried_stance"] = {"base": tc, "counterfactual": tc}
    fam_a = C.TEMPLATE_FAMILY_ORDER[b["template_family"]]
    fam_b = C.TEMPLATE_FAMILY_ORDER[1 - b["template_family"]]
    realizations = {}
    for real, fam in (("A", fam_a), ("B", fam_b)):
        order = C.ORDERS[(real, b["sentence_order"])]
        contexts = {}
        for state in ("base", "counterfactual"):
            sents = [C.TEMPLATE_FAMILIES[fam][facts[k][state]].format(holder=facts[k]["holder"], target=facts[k]["target"]) for k in order]
            contexts[state] = "\n".join(sents)
        realizations[real] = {"template_family": fam, "sentence_order": list(order), "context_base": contexts["base"], "context_counterfactual": contexts["counterfactual"]}
    return {
        "schema_version": f"{C.SCHEMA_PREFIX}_WORLD",
        "world_id": f"STW{i:03d}",
        "world_index": i,
        "split": split,
        "factors": {"direction": direction, "holder_slot": "FIRST_HOLDER" if b["holder_slot"] == 0 else "SECOND_HOLDER", "target_slot": "TARGET_A" if b["target_slot"] == 0 else "TARGET_B", "sentence_order": b["sentence_order"], "distractor_sign": hc, "target_type": target_type, "template_family_A": fam_a},
        "factor_bits": b,
        "holders": {"first": name_a, "second": name_b, "focal": hf, "other": ho},
        "targets": {"A": t_a, "B": t_b, "focal": tf, "other": to},
        "facts": facts,
        "queries": queries,
        "realizations": realizations,
        "f1_mapping_id": mapping_id,
        "generation_seed": C.CORPUS_SEED,
    }


def build_worlds() -> list[dict[str, Any]]:
    worlds = []
    rank: Counter = Counter()
    for i in range(256):
        split = C.split_of(i)
        if split is None:
            continue
        mapping = R.MAPPINGS[rank[split] % 4]
        rank[split] += 1
        worlds.append(build_world(i, mapping, split))
    return worlds


def build_demos() -> list[dict[str, Any]]:
    """Eight fresh template-matched demonstrations for P2/P3: two-sentence contexts in the study's own template
    families, operator claim wording, balanced labels (4 ENTAILED / 4 NOT_ENTAILED) and operators; names and
    targets disjoint from every study world."""
    fams = ("AFFIRM_DENY", "CHAMPION_CONTEST")
    demos = []
    for k in range(8):
        h1 = f"{C.DEMO_GIVEN[k]} {C.DEMO_SURNAMES[k]}"
        h2 = f"{C.DEMO_GIVEN[(k + 3) % 8]} {C.DEMO_SURNAMES[(k + 5) % 8]}"
        fam = fams[k % 2]
        if k % 4 < 2:
            t1 = f"booking {C.stable_code('d', k, 'T1')} relocates the {C.EVENTS[(k + 5) % 12]} session to the {C.ROOMS[(k + 2) % 8]}"
            t2 = f"booking {C.stable_code('d', k, 'T2')} relocates the {C.EVENTS[(k + 8) % 12]} session to the {C.ROOMS[(k + 6) % 8]}"
        else:
            t1 = f"revision {C.stable_code('d', k, 'T1')} sends the {C.LINES[(k + 5) % 12]} shuttle via {C.STOPS[(k + 2) % 8]} {40 + k}"
            t2 = f"revision {C.stable_code('d', k, 'T2')} sends the {C.LINES[(k + 8) % 12]} shuttle via {C.STOPS[(k + 6) % 8]} {50 + k}"
        stance1 = "SUPPORT" if k % 2 == 0 else "OPPOSE"
        stance2 = "OPPOSE" if (k // 2) % 2 == 0 else "SUPPORT"
        s1 = C.TEMPLATE_FAMILIES[fam][stance1].format(holder=h1, target=t1)
        s2 = C.TEMPLATE_FAMILIES[fam][stance2].format(holder=h2, target=t2)
        context = "\n".join([s1, s2] if k % 3 else [s2, s1])
        # claim about holder 1 (k even-half) or holder 2; operator chosen so labels balance 4/4 across k
        about = (h1, t1, stance1) if k % 4 in (0, 3) else (h2, t2, stance2)
        entailed = k in (0, 2, 5, 7)
        op = about[2] if entailed else FLIP[about[2]]
        claim = C.OPERATOR_CLAIM[op].format(holder=about[0], target=about[1])
        demos.append({"demo_id": f"DEMO_V2_{k}", "presentation_position": k, "template_family": fam, "context": context, "claim": claim, "entailment_label": "ENTAILED" if entailed else "NOT_ENTAILED", "claim_operator": op, "queried_stance": about[2]})
    labels = Counter(d["entailment_label"] for d in demos)
    ops = Counter(d["claim_operator"] for d in demos)
    if labels["ENTAILED"] != 4 or ops["SUPPORT"] != 4:
        raise RuntimeError(f"demo balance failure: {labels} {ops}")
    return demos


def protocol_demos(protocol: str, v1_demos: list[dict[str, Any]], fresh: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return v1_demos if protocol == "P1" else fresh


def build_prompt_manifest(worlds: list[dict[str, Any]], v1_demos: list[dict[str, Any]], fresh: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for w in worlds:
        for protocol in C.PROTOCOLS:
            demos = protocol_demos(protocol, v1_demos, fresh)
            for fmt in C.FORMATS:
                reals = C.REALIZATIONS if fmt == "F2" else ("A",)
                for real in reals:
                    for state in ("base", "counterfactual"):
                        context = w["realizations"][real][f"context_{state}"]
                        for q in C.QUERIES:
                            spec = w["queries"][q]
                            mapping = w["f1_mapping_id"] if fmt == "F1" else None
                            text, cands = C.render_prompt(protocol, demos, fmt, context, spec["claim"], mapping)
                            rows.append({"prompt_id": C.prompt_id(w["world_id"], protocol, fmt, real, state, q), "world_id": w["world_id"], "split": w["split"], "protocol": protocol, "format": fmt, "realization": real, "state": state, "query": q, "mapping_id": mapping, "candidates": cands, "label": spec[f"label_{state}"], "expected_sign": spec["expected_sign"], "queried_stance": spec["queried_stance"][state], "prompt_sha256": C.sha256_text(text), "prompt_chars": len(text)})
    return rows


def render_from_manifest(world: dict[str, Any], row: dict[str, Any], v1_demos: list[dict[str, Any]], fresh: list[dict[str, Any]]) -> tuple[str, list[str]]:
    demos = protocol_demos(row["protocol"], v1_demos, fresh)
    context = world["realizations"][row["realization"]][f"context_{row['state']}"]
    claim = world["queries"][row["query"]]["claim"]
    text, cands = C.render_prompt(row["protocol"], demos, row["format"], context, claim, row["mapping_id"])
    if C.sha256_text(text) != row["prompt_sha256"] or cands != row["candidates"]:
        raise RuntimeError(f"prompt drift: {row['prompt_id']}")
    return text, cands


def overlap_audit(worlds: list[dict[str, Any]], fresh: list[dict[str, Any]]) -> dict[str, Any]:
    old = C.read_jsonl(C.TRACK_A_FREEZE / "RELATION_BINDING_FACTORIAL_V2_WORLDS.jsonl")
    v1_worlds = C.read_jsonl(V1.FREEZE_ROOT / "CORPUS.jsonl")
    old_names, old_targets = set(), set()
    for w in old:
        for k in ("canonical_holder_name", "alternate_holder_name", "reporter_name", "quoter_name"):
            old_names.add(w[k])
        old_targets.add(w["proposition_p_text"])
        old_targets.add(w["proposition_q_text"])
    v1_names = {n for w in v1_worlds for n in (w["holders"]["first"], w["holders"]["second"])}
    v1_targets = {t for w in v1_worlds for t in (w["targets"]["A"], w["targets"]["B"])}
    prior_given = {n.split()[0] for n in old_names | v1_names} | set(V1.GIVEN_NAMES)
    prior_sur = {n.split()[-1] for n in old_names | v1_names} | set(V1.SURNAMES)
    new_names = {n for w in worlds for n in (w["holders"]["first"], w["holders"]["second"])}
    new_targets = {t for w in worlds for t in (w["targets"]["A"], w["targets"]["B"])}
    new_given = {n.split()[0] for n in new_names}
    new_sur = {n.split()[-1] for n in new_names}
    prior_vocab = {"sample", "controller", "clause", "measure", "protocol", "archive", "phase", "assembly", "checkpoint", "plot", "cohort", "period", "crystallizes", "oscillation", "seedling", "recovery", "settlement", "filings"}
    prior_vocab |= set(V1.SCHEDULE_NOUNS) | set(V1.WEEKDAYS) | set(V1.ALLOCATION_PLACES) | {w for r in V1.ALLOCATION_RESOURCES for w in r.split()} | {"proposal", "plan", "review", "assigns", "moves"}
    target_word_hits = sorted({wd for t in new_targets | {d["claim"] for d in fresh} for wd in t.replace(".", "").split() if wd in prior_vocab})
    prior_templates = ("supports the proposition that", "opposes the proposition that", "does not support", "does not oppose", "withheld a stance", "reports that", "called it the conclusion", "said the evidence", "the conclusion to accept", "conclusion to reject", "endorses the claim that", "rejects the claim that", "publicly backs the claim that", "publicly disputes the claim that")
    contexts = [w["realizations"][r][f"context_{s}"] for w in worlds for r in C.REALIZATIONS for s in ("base", "counterfactual")] + [d["context"] for d in fresh]
    template_hits = sorted({t for t in prior_templates for c in contexts if t in c})
    demo_given = set(C.DEMO_GIVEN)
    demo_sur = set(C.DEMO_SURNAMES)
    return {
        "track_a_worlds_file_sha256": C.sha256_file(C.TRACK_A_FREEZE / "RELATION_BINDING_FACTORIAL_V2_WORLDS.jsonl"),
        "v1_corpus_sha256": C.sha256_file(V1.FREEZE_ROOT / "CORPUS.jsonl"),
        "prior_worlds": {"track_a": len(old), "v1": len(v1_worlds)},
        "new_full_names": len(new_names), "new_target_count": len(new_targets),
        "shared_given_names_with_prior": sorted(new_given & prior_given), "shared_surnames_with_prior": sorted(new_sur & prior_sur),
        "shared_targets_with_prior": sorted(new_targets & (old_targets | v1_targets)),
        "prior_vocabulary_in_new_targets": target_word_hits, "prior_context_templates_in_new_contexts": template_hits,
        "demo_names_shared_with_study_worlds": sorted((demo_given & new_given) | (demo_sur & new_sur)),
        "demo_given_shared_with_prior": sorted(demo_given & prior_given), "demo_surnames_shared_with_prior": sorted(demo_sur & prior_sur),
        "targets_unique_per_world": all(w["targets"]["A"] != w["targets"]["B"] for w in worlds),
        "disjoint": not (new_given & prior_given) and not (new_sur & prior_sur) and not (new_targets & (old_targets | v1_targets)) and not target_word_hits and not template_hits and not ((demo_given & new_given) | (demo_sur & new_sur)) and not (demo_given & prior_given) and not (demo_sur & prior_sur),
    }


def balance_audit(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in C.SPLITS + ("all",):
        sub = [w for w in worlds if split == "all" or w["split"] == split]
        out[split] = {"worlds": len(sub)}
        for f in C.FACTORS:
            out[split][f] = dict(Counter(str(w["factor_bits"][f]) for w in sub))
        out[split]["q5_entailed"] = dict(Counter(w["queries"]["Q5"]["label_base"] for w in sub))
        out[split]["q6_entailed"] = dict(Counter(w["queries"]["Q6"]["label_base"] for w in sub))
        out[split]["f1_mapping"] = dict(Counter(w["f1_mapping_id"] for w in sub))
    balanced = all(len(set(out[s][f].values())) == 1 for s in C.SPLITS for f in C.FACTORS)
    out["all_factors_balanced_within_each_split"] = balanced
    out["split_sizes_ok"] = all(out[s]["worlds"] == C.SPLIT_SIZES[s] for s in C.SPLITS)
    return out


def study_contract(v1_demos: list[dict[str, Any]], fresh: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": f"{C.SCHEMA_PREFIX}_STUDY_CONTRACT",
        "study_id": C.STUDY_ID,
        "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "predecessor": {"study": V1.STUDY_ID, "corpus_freeze_manifest_sha256": C.sha256_file(V1.FREEZE_ROOT / "artifact_manifest.json"), "results_commit": "8f62d37", "rule": "V1 is never altered or reinterpreted; its CSR endpoint, Q1-Q6 semantic structure, actors, layers, scoring path, model revisions and Lambda environment are reused unchanged"},
        "question": "When rank-1 relation steering actually converts the direct answer with counterfactual-like strength, does it reproduce the held-out consequences of changing the relation?",
        "not_a": ["belief study", "hidden-knowledge study", "moral-truth study", "deception study"],
        "actors": {a: {"model_id": cfg["model_id"], "model_revision": cfg["model_revision"], "primary_layer": cfg["primary_layer"], "witness_layer": C.WITNESS_LAYER[a], "hidden_size": cfg["hidden_size"], "dtype": "bfloat16 exact", "original_frozen_dim_sha256": cfg["raw_relation_dim_sha256"]} for a, cfg in C.ACTORS.items()},
        "intervention_site": "output of the frozen primary decoder layer at the final prompt position; candidates scored as exact candidate-sequence log-probabilities on the frozen right-padded batch-of-two scoring path",
        "corpus": {
            "worlds": C.WORLD_COUNT, "splits": C.SPLIT_SIZES, "seed": C.CORPUS_SEED,
            "design": "worlds 0-127: full 2^7 factorial over " + ", ".join(C.FACTORS) + " (the V1-style even-parity 32 are selection worlds, the remaining 96 are final); fit worlds: the 64-world even-total-parity half fraction of a second 2^7 factorial (indices 128-255); every factor is balanced within every split",
            "world_content": "two named holders, two distinct domain-neutral targets, four explicit holder-target stance facts (focal, holder control, target control, diagonal); the focal relation flips between base and counterfactual; every other fact is held fixed (V1 semantic structure)",
            "realizations": "A and B: different lexical template family (AFFIRM_DENY vs CHAMPION_CONTEST) and different sentence order; two independently rendered realizations per world; B is the frozen cross-template condition",
            "name_pools": {"given": list(C.GIVEN_NAMES), "surnames": list(C.SURNAMES)}, "target_types": list(C.TARGET_TYPES), "template_families": C.TEMPLATE_FAMILIES,
            "disjointness": "no world, name, target or context lexical template is shared with V1 or Track A (audited in INPUT_AND_DIRECTION_MANIFEST.json)",
        },
        "query_battery": {"Q1": "calibration: focal relation with the operator that becomes true in the counterfactual (base NOT_ENTAILED, counterfactual ENTAILED); the only query that may choose intervention strength; excluded from every fidelity score", "Q2": "complement", "Q3": "paraphrase of the focal support claim", "Q4": "paraphrase of the focal opposition claim", "Q5": "holder control", "Q6": "target control", "consequence_panel": list(C.CONSEQUENCE_QUERIES), "claim_wording": {"operator": C.OPERATOR_CLAIM, "paraphrase": C.PARAPHRASE_CLAIM}, "expected_natural_signs": {d: {q: C.expected_sign(d, q) for q in C.QUERIES} for d in C.DIRECTIONS}},
        "protocols": {
            "P1": {"description": "V1 validated F2 protocol: frozen RRRD demonstrations, Context/Claim/Answer frame, task line " + R.F2_TASK_LINE, "demonstrations_sha256": C.sha256_file(V1.RRRD_FREEZE / "TRACK_A_DEMONSTRATION_FREEZE.jsonl"), "mapped_variant": "C9 mapped-symbol F1 with the same demonstrations"},
            "P2": {"description": "P1 frame and task line with eight fresh template-matched demonstrations (DEMOS_V2.jsonl)", "mapped_variant": "F1 frame with the fresh demonstrations"},
            "P3": {"description": "fresh demonstrations with alternative query wording: Statement frame, task line " + C.P3_F2_TASK_LINE, "mapped_variant": "Statement frame with the mapping lines, task line " + C.P3_F1_TASK_LINE},
            "gemma": C.GEMMA_PROTOCOL, "llama_candidates": list(C.LLAMA_CANDIDATE_PROTOCOLS), "llama_selection_rule": C.PROTOCOL_SELECTION_RULE,
            "semantic_margin": "logp(ENTAILED candidate sequence) - logp(NOT_ENTAILED candidate sequence) (F1: entailed symbol minus not-entailed symbol)",
        },
        "natural_reference": {"signature": "N_w = margin(counterfactual) - margin(base) over Q2..Q6 per world, realization and format", "gates": C.REFERENCE_GATES, "natural_only_gates_for_protocol_selection": list(C.NATURAL_ONLY_GATES), "gate_definitions": {"primary_format_balanced_accuracy": "balanced accuracy of the F2 native answer against the frozen label over all Q1-Q6 base and counterfactual prompts of the split (both realizations)", "mapped_format_semantic_agreement": "fraction of F1 prompts whose native semantic answer equals the F2 native semantic answer on the same world/state/query (realization A)", "focal_consequence_expected_direction": "fraction of (world, Q2-Q4) natural changes whose sign equals the expected sign (realization A)", "control_stability_ratio": "median |N| over Q5-Q6 divided by median |N| over Q2-Q4 (realization A)", "cross_template_consistency": "CTNC_w = 1 - D(N_B, N_A)^2 / D(0, N_A)^2 with the frozen standardized distance; mean and world-bootstrap interval", "full_state_patch_csr": "mean CSR of FULL_STATE_SOURCE_PATCH (realization A)"}, "failure_class": "REFERENCE_ASSAY_INVALID"},
        "directions": {
            "ORIGINAL_FROZEN_DIM": "the existing Track A v2 raw relation difference-in-means direction (transport/replication test)",
            "REGIME_MATCHED_DIM": "difference in means (SUPPORT minus OPPOSE queried focal stance) of primary-layer final-position activations over the 64 fit worlds' unsteered F2 prompts under the selected protocol: Q1-Q4, base and counterfactual states, both realizations; balanced by construction across relation sign, query operator, answer label, holder, target and realization; never fit on final worlds",
            "REGIME_ANSWER_RESIDUAL_DIM": "REGIME_MATCHED_DIM with its projection on the regime-matched answer-token direction (ENTAILED minus NOT_ENTAILED label difference in means over the same prompts) removed",
            "qualification": C.DIRECTION_GATES | {"definitions": "world-grouped 8-fold cross-validation on the fit worlds (fold = world_index mod 8): relation-sign balanced accuracy of the projection sign with the training-fold midpoint threshold; mapping agreement = fraction of held-out F1 realization-A prompts whose projected sign equals the matching F2 prompt's; cross-realization BA = fit on realization A, test on held-out realization B and vice versa; failure -> DIRECTION_NOT_QUALIFIED"},
            "matched_controls": {k: f"Track A v2 frozen '{v}' unit direction" for k, v in C.TRACK_A_CONTROL_KEYS.items()} | {"truth_rule": f"MATCHED_TRUTH_DIRECTION enters the comparator panel only if its tau=0.75 matching coverage on final worlds is >= {C.COVERAGE_MIN_FOR_TRUTH:.0%}"},
            "unit_vectors": "every direction is L2-normalized before use; the frozen files carry the unit vectors and their SHA-256",
        },
        "strong_target_calibration": {"taus": list(C.TAUS), "primary_tau": C.PRIMARY_TAU, "target": C.TARGET_RULE, "alpha_grid_multiples_of_natural_full_state_l2": list(C.ALPHA_GRID), "interpolation_grid": list(C.INTERPOLATION_GRID), "root_rule": C.ROOT_RULE, "achieved_tolerance": C.ACHIEVED_TOLERANCE, "saturation": C.SATURATION_RULE, "dose_validity": C.DOSE_VALIDITY, "norm_ratio": "intervention L2 / natural full-state L2 = the selected multiple m_w by construction", "prohibited": "Q2-Q6, realization B, activation witnesses and the CSR endpoint never choose or adjust a multiple; the grid is never extended after final outcomes"},
        "interventions": {
            "NO_INTERVENTION": "unmodified base response",
            C.FULL_PATCH: "positive control / reference gate: per query, the base prompt's complete primary-layer final-position activation replaced by the same query's natural-counterfactual activation (same realization)",
            "relation_arms_all_taus": list(C.ALL_TAU_ARMS), "primary_tau_only_arms": list(C.PRIMARY_TAU_ONLY_ARMS),
            C.LOGIT_CONTROL: "analytic output-forcing baseline: bias b = target(tau) - margin_base(Q1) added to every query's margin (I_w = b on Q2-Q6, both realizations)",
            C.NATURAL_INTERPOLATION: "Q1 realization A chooses the fraction f along h_base -> h_cf (interpolation grid, same root rule, tau=0.75); f is then applied to each query's own natural state difference in both realizations",
            "rule": "every comparator reaches the same Q1 target using Q1 alone; the selected magnitude is applied unchanged to Q2-Q6",
            "mapped_format_robustness": "F1 realization A: natural, full-state patch, ORIGINAL_FROZEN_DIM and REGIME_MATCHED_DIM at every tau with their own F1 Q1 calibration",
        },
        "primary_endpoint": {"csr": "reused unchanged from V1 (crif_v1_common.csr): CSR_w = 1 - D(I_w, N_w)^2 / D(0, N_w)^2 over Q2..Q6 with development (fit-world) robust scales, equal focal/invariance group weight; decomposition CSR = CSR_focal + CSR_inv - 1", "scales": "1.4826 x MAD of the fit-world natural signatures (realization A, primary format, selected protocol), floored at 0.25 x the median focal scale; frozen in the final-analysis freeze", "reports": ["world-level CSR, focal CSR, invariance CSR per arm and tau", "cross-template CSR (realization B, realization-A multiple)", "10,000 paired world-bootstrap intervals, one shared draw", "Holm-corrected paired differences primary relation arm minus every comparator", "dose-response: CSR against achieved Q1 control across tau", "norm ratios (median, p90) per arm and tau"], "primary_inferential_object": "REGIME_MATCHED_DIM at tau=0.75", "transport_test": "ORIGINAL_FROZEN_DIM at tau=0.75 (replication of V1's direction in the new regime)", "csr_world_set": C.CSR_WORLD_SET_RULE, "fidelity_criteria": {"csr_ci_low_min_fraction_of_ctnc": C.FIDELITY_CI_LOW_FRACTION_OF_CTNC, "strong_control_interpolation_ci_low_min": C.STRONG_CONTROL_INTERPOLATION_CI_LOW_MIN, "strong_control_relation_ci_high_max": C.STRONG_CONTROL_RELATION_CI_HIGH_MAX}, "Q1_exclusion": "Q1 never enters any fidelity score"},
        "witness": {"readout": "L2-regularized logistic regression on the witness-layer final-position activation with the regime-matched and original relation directions projected out before fitting; trained only on the 64 fit worlds' unsteered realization-A F2 prompts (Q1-Q4, base and counterfactual) under the selected protocol; frozen before final inference", "gates": C.WITNESS_GATES, "gate_definition": "balanced accuracy of the witness stance sign on unsteered final-world Q1/Q2 base and counterfactual activations, realization A (BA) and realization B (cross-template BA)", "if_pass": "report witness movement toward the natural counterfactual (WSR on Q1-Q4, invariance movement on Q5-Q6) under every arm at tau=0.75 and for the relation arms across tau", "if_fail": "preserve the failure; no internal-state claim; behavioral CSR remains primary"},
        "interpretations": C.INTERPRETATIONS, "precedence": list(C.PRECEDENCE), "licensed_wording": C.LICENSED_WORDING,
        "joint": "MODEL_DEPENDENT when both assays are valid and the actor classifications differ; otherwise the actor classes are reported side by side; an invalid actor is never converted into a negative scientific result",
        "execution_sequence": ["inventory and hash-bind inputs", "freeze corpus, demonstrations, prompt manifest, protocols, calibration and endpoint rules (this package)", "unit tests", "Llama protocol selection on the 32 selection worlds (natural only) -> frozen protocol receipt", "fit-world natural forwards (both actors)", "direction fit + qualification, witness fit, alpha-grid calibration on fit worlds -> final-analysis freeze", "final worlds once per actor", "analysis without changes", "reports, figures, claim matrix, packet, manifest"],
        "exclusions": ["no rank-k, layer/site, multi-agent or infrastructure side studies", "no tuning on Q2-Q6", "no fitting on final worlds", "no gate weakening after outcomes", "no 'mind versus mouth', belief editing, or global 'more concepts than control knobs' language", "no V1 reinterpretation"],
        "recovery": "operational resume from durable per-world checkpoints only; failed attempts preserved; a resumed world must reproduce its checkpoint margins; no scientific retry after a valid terminal result",
    }


def contract_markdown(contract: dict[str, Any]) -> str:
    def sec(title: str, obj: Any, level: int = 2) -> str:
        head = "#" * level + " " + title + "\n\n"
        if isinstance(obj, dict):
            body = "".join(f"- **{k}**: {v if not isinstance(v, (dict, list)) else ''}\n" + ("".join(f"  - {kk}: {vv}\n" for kk, vv in v.items()) if isinstance(v, dict) else "".join(f"  - {x}\n" for x in v) if isinstance(v, list) else "") for k, v in obj.items())
        elif isinstance(obj, list):
            body = "".join(f"- {x}\n" for x in obj)
        else:
            body = str(obj) + "\n"
        return head + body + "\n"

    out = f"# Study contract — {contract['study_id']}\n\nFrozen at {contract['frozen_at']} before any model inference.\n\n"
    out += sec("Question", contract["question"])
    for key in ("predecessor", "actors", "corpus", "query_battery", "protocols", "natural_reference", "directions", "strong_target_calibration", "interventions", "primary_endpoint", "witness", "interpretations", "precedence", "licensed_wording", "joint", "execution_sequence", "exclusions"):
        out += sec(key.replace("_", " ").title(), contract[key])
    out += sec("Intervention site", contract["intervention_site"])
    out += sec("Recovery", contract["recovery"])
    return out


def input_manifest(audit: dict[str, Any]) -> dict[str, Any]:
    files = [
        ("rrrd_demonstration_freeze", V1.RRRD_FREEZE / "TRACK_A_DEMONSTRATION_FREEZE.jsonl"),
        ("rrrd_common_code", C.ROOT / "scripts" / "rrrd_v1_common.py"),
        ("frozen_track_a_runner", C.ROOT / "scripts" / "run_relation_binding_factorial_v2_model.py"),
        ("v1_common_code", C.ROOT / "scripts" / "crif_v1_common.py"),
        ("v1_corpus", V1.FREEZE_ROOT / "CORPUS.jsonl"),
        ("v1_study_contract", V1.FREEZE_ROOT / "STUDY_CONTRACT.json"),
        ("v1_final_analysis_freeze", V1.STUDY_ROOT / "final_freeze" / "FINAL_ANALYSIS_FREEZE.json"),
        ("v1_results", V1.STUDY_ROOT / "joint_closeout" / "COUNTERFACTUAL_SIGNATURE_RESULTS.json"),
        ("v1_review_packet", V1.STUDY_ROOT / "joint_closeout" / "external_review_packet_v1.json"),
        ("track_a_worlds", C.TRACK_A_FREEZE / "RELATION_BINDING_FACTORIAL_V2_WORLDS.jsonl"),
        ("competitor_direction_manifest", C.TRACK_A_FREEZE / "TRACK_A_COMPETITOR_DIRECTIONS.json"),
        ("competitor_directions_llama", C.TRACK_A_FREEZE / "runtime" / "llama" / "competitor_directions_v2.npz"),
        ("competitor_directions_gemma", C.TRACK_A_FREEZE / "runtime" / "gemma" / "competitor_directions_v2.npz"),
        ("study_common_code", C.ROOT / "scripts" / "stcrf_v2_common.py"),
        ("study_corpus_builder", C.ROOT / "scripts" / "stcrf_v2_build_corpus.py"),
    ]
    entries = [{"role": role, "path": str(p.relative_to(C.ROOT)).replace("\\", "/"), "bytes": p.stat().st_size, "sha256": C.sha256_file(p)} for role, p in files]
    return {
        "schema_version": f"{C.SCHEMA_PREFIX}_INPUT_AND_DIRECTION_MANIFEST",
        "source_of_truth_hierarchy": ["frozen scientific artifacts and manifests", "current study reports"],
        "base_commit": "8f62d37",
        "inputs": entries,
        "track_a_directions": {a: V1.direction_manifest_entry(a) for a in C.ACTORS},
        "original_frozen_dim_sha256": {a: C.ACTORS[a]["raw_relation_dim_sha256"] for a in C.ACTORS},
        "forbidden_inputs": ["V3 paths or outcomes", "the sealed 60-board human-confirmation set", "confirmatory splits", "any final-world outcome before the final-analysis freeze", "any final-world activation in a direction or witness fit"],
        "corpus_overlap_audit": audit,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze-root", type=Path, default=C.FREEZE_ROOT)
    args = ap.parse_args(argv)
    root: Path = args.freeze_root
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"freeze root exists and is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    v1_demos = C.load_v1_demonstrations()
    fresh = build_demos()
    worlds = build_worlds()
    audit = overlap_audit(worlds, fresh)
    if not audit["disjoint"]:
        raise SystemExit(f"corpus overlaps prior content: {audit}")
    balance = balance_audit(worlds)
    if not (balance["all_factors_balanced_within_each_split"] and balance["split_sizes_ok"]):
        raise SystemExit(f"balance failure: {balance}")
    manifest = build_prompt_manifest(worlds, v1_demos, fresh)
    ids = [p["prompt_id"] for p in manifest]
    if len(ids) != len(set(ids)) or len({p["prompt_sha256"] for p in manifest}) != len(manifest):
        raise SystemExit("duplicate prompt ids or texts")
    contract = study_contract(v1_demos, fresh)
    C.write_create_only(root / "CORPUS.jsonl", C.jsonl_bytes(worlds))
    C.write_create_only(root / "DEMOS_V2.jsonl", C.jsonl_bytes(fresh))
    C.write_create_only(root / "PROMPT_MANIFEST.jsonl", C.jsonl_bytes(manifest))
    C.write_create_only(root / "STUDY_CONTRACT.json", C.canonical_json(contract))
    C.write_create_only(root / "STUDY_CONTRACT.md", contract_markdown(contract).encode("utf-8"))
    C.write_create_only(root / "CALIBRATION_AND_ENDPOINT_FREEZE.json", C.canonical_json({"schema_version": f"{C.SCHEMA_PREFIX}_CALIBRATION_AND_ENDPOINT_FREEZE", "strong_target_calibration": contract["strong_target_calibration"], "interventions": contract["interventions"], "primary_endpoint": contract["primary_endpoint"], "natural_reference": contract["natural_reference"], "directions": contract["directions"], "witness": contract["witness"], "interpretations": contract["interpretations"], "precedence": contract["precedence"], "prompt_counts": dict(Counter(f"{p['protocol']}/{p['format']}/{p['split']}/{p['realization']}" for p in manifest)), "corpus_sha256": C.sha256_file(root / "CORPUS.jsonl"), "prompt_manifest_sha256": C.sha256_file(root / "PROMPT_MANIFEST.jsonl"), "balance_audit": balance, "bootstrap": {"replicates": C.BOOTSTRAP_REPLICATES, "seed": C.BOOTSTRAP_SEED, "unit": "world_id"}}))
    C.write_create_only(root / "INPUT_AND_DIRECTION_MANIFEST.json", C.canonical_json(input_manifest(audit)))
    files = C.manifest_for(root)
    C.write_create_only(root / "artifact_manifest.json", C.canonical_json({"schema_version": f"{C.SCHEMA_PREFIX}_FREEZE_MANIFEST", "study_id": C.STUDY_ID, "files": files}))
    print(f"worlds={len(worlds)} splits={ {s: sum(w['split'] == s for w in worlds) for s in C.SPLITS} } prompts={len(manifest)} disjoint={audit['disjoint']} balanced={balance['all_factors_balanced_within_each_split']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
