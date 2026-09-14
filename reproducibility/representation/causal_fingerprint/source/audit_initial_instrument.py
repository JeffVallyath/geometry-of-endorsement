from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geometry_endorsement.causal_fingerprint_v2.archive_audit import sha256_file, verify_v1_archive
from geometry_endorsement.causal_fingerprint_v2.mapping_factorial import PAIR_IDS, correlation, factorial_components
from geometry_endorsement.causal_fingerprint_v2.numerical_resolution import (
    diagnostics,
    estimate_resolution_floor,
    most_common_nonzero_increments,
    resolution_bins,
    subset_diagnostics,
    symmetric_margin_change,
)
from geometry_endorsement.causal_fingerprint_v2.reliability_v2 import stable_gradient_random_direction_fixture


FAMILY_SLICES = {
    "relation": slice(0, 8),
    "language": slice(8, 12),
    "sentiment": slice(12, 20),
    "truth": slice(20, 28),
    "token": slice(28, 32),
    "random": slice(32, 40),
}
DIRECTION_FAMILY_ORDER = ("relation", "truth", "sentiment", "language", "token", "random")
ENDPOINT_ORDER = ("language_eng_fra", "language_eng_spa", "relation", "sentiment", "truth")
EXPECTED_HEADLINES = {
    "llama": {"correlation": 0.8059667269, "sign_agreement": 0.8166666667, "median_absolute_relative_error": 0.5127568320, "saturation_fraction": 0.0, "mapping_reversal": -0.6628059086, "random_reliability": 0.9960649648},
    "gemma": {"correlation": 0.9474513967, "sign_agreement": 0.9166666667, "median_absolute_relative_error": 0.2803680245, "saturation_fraction": 0.0, "mapping_reversal": 0.6090684986, "random_reliability": 0.9913529492},
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_npz(archive: zipfile.ZipFile, name: str) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def pilot_members(archive: zipfile.ZipFile, actor: str) -> list[str]:
    return sorted(name for name in archive.namelist() if name.startswith(f"checkpoints/{actor}/pilot/") and name.endswith(".npz") and not name.endswith("pilot_fd_validation.npz"))


def replay_actor(archive: zipfile.ZipFile, actor: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    validation = load_npz(archive, f"checkpoints/{actor}/pilot/pilot_fd_validation.npz")
    fractions = validation["epsilon_fractions"].astype(np.float64)
    epsilon_rows = []
    for fraction in sorted(set(fractions.tolist())):
        mask = fractions == fraction
        epsilon_rows.append(diagnostics(validation["jvp"][mask], validation["finite_difference"][mask], validation["saturation"][mask], float(fraction)).to_dict())
    effects, groups, mappings, keys, records = [], [], [], [], []
    candidate_lengths: list[tuple[int, int]] = []
    for member in pilot_members(archive, actor):
        data = load_npz(archive, member)
        effect = data["effects"].astype(np.float64)
        dataset, group = str(data["dataset_id"]), str(data["group_id"])
        mapping, item, template = str(data["mapping"]), str(data["item_id"]), str(data["template"])
        effects.append(effect)
        groups.append(f"{dataset}:{group}")
        mappings.append(mapping)
        keys.append((item, template))
        candidate_lengths.append((len(data["positive_candidate_ids"]), len(data["negative_candidate_ids"])))
        records.append({"member": member, "actor": actor, "effect": effect, "dataset": dataset, "family": str(data["family"]), "group_id": group, "item_id": item, "template": template, "mapping": mapping, "semantic_margin": float(data["semantic_margin"]), "physical_token_margin": float(data["physical_token_margin"]), "candidate_lengths": candidate_lengths[-1]})
    matrix = np.stack(effects)
    unique_groups = sorted(set(groups))
    left_groups = set(unique_groups[::2])
    left = matrix[np.asarray([group in left_groups for group in groups])].mean(axis=0)
    right = matrix[np.asarray([group not in left_groups for group in groups])].mean(axis=0)
    family_reliability = {}
    for family in ("relation", "truth", "sentiment", "language"):
        part = FAMILY_SLICES[family]
        family_reliability[family] = float(np.corrcoef(left[part], right[part])[0, 1]) if np.std(left[part]) and np.std(right[part]) else 0.0
    paired: dict[tuple[str, str], dict[str, np.ndarray]] = defaultdict(dict)
    for effect, mapping, key in zip(effects, mappings, keys, strict=True):
        paired[key][mapping] = effect
    reversal = []
    semantic_indices = np.r_[0:28]
    for values in paired.values():
        for _symbol, standard, reversed_id in PAIR_IDS:
            if standard in values and reversed_id in values:
                value = correlation(values[standard][semantic_indices], values[reversed_id][semantic_indices])
                if value is not None:
                    reversal.append(value)
    random_reliability = correlation(left[FAMILY_SLICES["random"]], right[FAMILY_SLICES["random"]]) or 0.0
    replay = {
        "epsilon_diagnostics": epsilon_rows,
        "selected_epsilon_fraction": None,
        "selected_epsilon": None,
        "fingerprint_split_half_reliability": min(family_reliability.values()),
        "family_fingerprint_split_half_reliability": family_reliability,
        "mapping_reversal_median_fingerprint_correlation": float(np.median(reversal)),
        "random_direction_split_half_reliability": random_reliability,
        "all_candidates_single_token": all(positive == 1 and negative == 1 for positive, negative in candidate_lengths),
        "pilot_npz_records": len(records),
        "valid": False,
    }
    return replay, records, validation


def exact_replay(archive_path: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, dict[str, np.ndarray]]]:
    actor_records: dict[str, list[dict[str, Any]]] = {}
    validations: dict[str, dict[str, np.ndarray]] = {}
    actors: dict[str, Any] = {}
    with zipfile.ZipFile(archive_path) as archive:
        gate = json.loads(archive.read("checkpoints/A100_PILOT_GATE.json"))
        for actor in ("llama", "gemma"):
            replay, records, validation = replay_actor(archive, actor)
            actors[actor] = replay
            actor_records[actor] = records
            validations[actor] = validation
            recorded = gate["actors"][actor]
            for key in ("fingerprint_split_half_reliability", "mapping_reversal_median_fingerprint_correlation", "random_direction_split_half_reliability"):
                if not np.isclose(float(replay[key]), float(recorded[key]), rtol=0.0, atol=1e-12):
                    raise RuntimeError(f"V1_PILOT_REPLAY_MISMATCH:{actor}:{key}")
            for observed, expected in zip(replay["epsilon_diagnostics"], recorded["epsilon_diagnostics"], strict=True):
                for key in ("correlation", "sign_agreement", "median_absolute_relative_error", "saturation_fraction"):
                    if not np.isclose(float(observed[key]), float(expected[key]), rtol=0.0, atol=1e-12):
                        raise RuntimeError(f"V1_PILOT_REPLAY_MISMATCH:{actor}:{observed['epsilon_fraction']}:{key}")
            headline = next(value for value in replay["epsilon_diagnostics"] if value["epsilon_fraction"] == 0.01)
            expected_headline = EXPECTED_HEADLINES[actor]
            for key in ("correlation", "sign_agreement", "median_absolute_relative_error", "saturation_fraction"):
                if not np.isclose(float(headline[key]), expected_headline[key], rtol=0.0, atol=5e-10):
                    raise RuntimeError(f"V1_PILOT_REPLAY_MISMATCH:{actor}:headline:{key}")
        status = "CAUSAL_FINGERPRINT_MEASUREMENT_VALID" if all(value["valid"] for value in actors.values()) else "CAUSAL_FINGERPRINT_MEASUREMENT_INVALID"
        if status != gate["status"]:
            raise RuntimeError("V1_PILOT_REPLAY_MISMATCH:status")
    return {"schema_version": 1, "status": status, "actors": actors, "exact_replay_pass": True, "numerical_tolerance": {"recorded_values_atol": 1e-12, "headline_atol": 5e-10}}, actor_records, validations


def numerical_forensics(validations: dict[str, dict[str, np.ndarray]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    report: dict[str, Any] = {}
    quantization_rows: list[dict[str, Any]] = []
    epsilon_rows: list[dict[str, Any]] = []
    for actor, data in validations.items():
        fractions = data["epsilon_fractions"].astype(np.float64)
        median_norm = float(data["median_activation_norm"])
        epsilon = fractions * median_norm
        changes = symmetric_margin_change(data["finite_difference"], epsilon)
        q = estimate_resolution_floor(changes)
        bins = resolution_bins(changes, q)
        actor_report = {"median_activation_norm": median_norm, "estimated_resolution_floor": q, "most_common_nonzero_increments": most_common_nonzero_increments(changes), "absolute_displacements": {str(fraction): float(fraction * median_norm) for fraction in sorted(set(fractions.tolist()))}, "fractions": {}}
        for fraction in sorted(set(fractions.tolist())):
            mask = fractions == fraction
            local_changes = changes[mask]
            local_bins = resolution_bins(local_changes, q)
            expected = np.abs(data["jvp"][mask].astype(np.float64)) * (2.0 * epsilon[mask])
            observed = np.abs(local_changes)
            unresolved = observed <= q
            marginal = (observed > q) & (observed <= 4.0 * q)
            clear = observed > 4.0 * q
            summary = {
                "n": int(mask.sum()),
                "epsilon_absolute_l2": float(fraction * median_norm),
                "epsilon_fraction_of_activation_norm": float(fraction),
                "expected_first_order_margin_movement_median": float(np.median(expected)),
                "observed_symmetric_margin_movement_median": float(np.median(observed)),
                "signal_to_resolution_median": float(np.median(observed / max(q, 1e-30))),
                "exact_zero_fraction": float(np.mean(local_bins["exact_zero"])),
                "within_one_step_fraction": float(np.mean(local_bins["within_one_step"])),
                "within_two_steps_fraction": float(np.mean(local_bins["within_two_steps"])),
                "above_four_steps_fraction": float(np.mean(local_bins["above_four_steps"])),
                "unresolved_diagnostics": subset_diagnostics(data["jvp"][mask], data["finite_difference"][mask], unresolved),
                "marginal_diagnostics": subset_diagnostics(data["jvp"][mask], data["finite_difference"][mask], marginal),
                "clearly_resolved_diagnostics": subset_diagnostics(data["jvp"][mask], data["finite_difference"][mask], clear),
            }
            actor_report["fractions"][str(fraction)] = summary
            epsilon_rows.append({"actor": actor, "epsilon_fraction": fraction, **{key: value for key, value in summary.items() if not isinstance(value, dict)}})
        correlations = [actor_report["fractions"][str(value)]["clearly_resolved_diagnostics"]["correlation"] for value in sorted(set(fractions.tolist()))]
        original = [next(row for row in [diagnostics(data["jvp"][fractions == f], data["finite_difference"][fractions == f], data["saturation"][fractions == f], f).to_dict()] if row) for f in sorted(set(fractions.tolist()))]
        actor_report["performance_monotonic_with_epsilon"] = all(original[index]["correlation"] <= original[index + 1]["correlation"] for index in range(len(original) - 1))
        actor_report["clearly_resolved_correlation_sequence"] = correlations
        report[actor] = actor_report
        unique_ids = list(dict.fromkeys(data["direction_ids"].astype(str).tolist()))
        direction_family = {direction_id: DIRECTION_FAMILY_ORDER[index // 2] for index, direction_id in enumerate(unique_ids)}
        for index, (fraction, direction_id, change) in enumerate(zip(fractions, data["direction_ids"].astype(str), changes, strict=True)):
            prompt_block = index // (12 * 4)
            quantization_rows.append({"actor": actor, "endpoint_family": ENDPOINT_ORDER[prompt_block // 2], "endpoint_record_within_family": prompt_block % 2, "template": "first_lexicographic_template_for_endpoint", "mapping": "MAPPING_AB_STANDARD", "direction_family": direction_family[direction_id], "direction_id": direction_id, "epsilon_fraction": fraction, "epsilon_absolute_l2": float(fraction * median_norm), "symmetric_margin_change": float(change), "estimated_resolution_floor": q, "signal_to_resolution": float(abs(change) / max(q, 1e-30))})
    return report, quantization_rows, epsilon_rows


def mapping_forensics(records_by_actor: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    for actor, records in records_by_actor.items():
        index: dict[tuple[str, str, str, str, str], dict[str, np.ndarray]] = defaultdict(dict)
        for record in records:
            key = (record["item_id"], record["group_id"], record["dataset"], record["template"], record["family"])
            index[key][record["mapping"]] = record["effect"]
        raw_semantic_correlations, physical_correlations = [], []
        semantic_components: dict[str, list[np.ndarray]] = defaultdict(list)
        token_components: dict[str, list[np.ndarray]] = defaultdict(list)
        for key, values in index.items():
            for symbol, standard, reversed_id in PAIR_IDS:
                if standard not in values or reversed_id not in values:
                    continue
                semantic, token = factorial_components(values[standard], values[reversed_id])
                raw = correlation(values[standard][:28], values[reversed_id][:28])
                physical = correlation(values[standard][:28], -values[reversed_id][:28])
                if raw is not None:
                    raw_semantic_correlations.append(raw)
                if physical is not None:
                    physical_correlations.append(physical)
                semantic_components[symbol].append(semantic)
                token_components[symbol].append(token)
                item_id, group_id, dataset, template, family = key
                for direction_index in range(40):
                    pair_rows.append({"actor": actor, "item_id": item_id, "group_id": group_id, "dataset": dataset, "family": family, "template": template, "symbol_family": symbol, "direction_index": direction_index, "direction_family": next(name for name, part in FAMILY_SLICES.items() if direction_index in range(*part.indices(40))), "effect_standard_semantic": float(values[standard][direction_index]), "effect_reversed_semantic": float(values[reversed_id][direction_index]), "semantic_pair_effect": float(semantic[direction_index]), "token_contrast_effect": float(token[direction_index])})
        family_summary: dict[str, Any] = {}
        for symbol in ("AB", "12"):
            sem = np.stack(semantic_components[symbol])
            tok = np.stack(token_components[symbol])
            family_summary[symbol] = {"pairs": len(sem), "semantic_component_median_l2": float(np.median(np.linalg.norm(sem[:, :28], axis=1))), "token_component_median_l2": float(np.median(np.linalg.norm(tok[:, :28], axis=1))), "semantic_token_component_correlation": correlation(sem[:, :28].ravel(), tok[:, :28].ravel())}
            summaries.append({"actor": actor, "symbol_family": symbol, **family_summary[symbol]})
        cross_symbol_semantic = correlation(np.mean(np.stack(semantic_components["AB"]), axis=0)[:28], np.mean(np.stack(semantic_components["12"]), axis=0)[:28])
        report[actor] = {"median_raw_semantic_mapping_reversal_correlation": float(np.median(raw_semantic_correlations)), "median_physical_token_oriented_reversal_correlation": float(np.median(physical_correlations)), "mapping_paired_semantic_cross_symbol_correlation": cross_symbol_semantic, "symbol_family_summaries": family_summary, "negative_semantic_becomes_positive_physical": bool(np.median(raw_semantic_correlations) < 0 < np.median(physical_correlations))}
    return report, pair_rows, summaries


def dtype_audit(repo: Path) -> dict[str, Any]:
    overlay = Path(r"external-artifacts")
    runtime = repo / "src" / "geometry_endorsement" / "semantic_control_geometry" / "runtime_stage.py"
    gradient = repo / "src" / "geometry_endorsement" / "semantic_control_geometry" / "activation_extraction.py"
    return {
        "schema_version": 1,
        "audit_basis": {"runtime_stage_sha256": sha256_file(runtime), "activation_gradient_sha256": sha256_file(gradient), "executed_dependency_overlay_sha256": sha256_file(overlay)},
        "chain": [
            {"stage": "intervention_layer_activation", "dtype": "bfloat16", "status": "source-established", "basis": "BF16 actor plus hook hidden tensor"},
            {"stage": "requested_perturbation", "dtype": "float32", "status": "source-established", "basis": "float32 direction multiplied then torch.as_tensor(..., dtype=float32)"},
            {"stage": "perturbation_after_addition", "dtype": "bfloat16", "status": "source-established", "basis": "delta_tensor.to(hidden.device, dtype=hidden.dtype) before in-place add"},
            {"stage": "downstream_residual_stream", "dtype": "bfloat16", "status": "source-inferred-not-runtime-saved", "basis": "changed hidden retains hidden.dtype"},
            {"stage": "final_normalization", "dtype": "unknown_not_saved", "status": "not-recoverable-from-V1", "basis": "model-internal dtype not recorded"},
            {"stage": "lm_head_input", "dtype": "unknown_not_saved_likely_bfloat16", "status": "not-recoverable-from-V1", "basis": "pre-head tensor not recorded"},
            {"stage": "lm_head_weight", "dtype": "unknown_not_saved_likely_bfloat16", "status": "not-recoverable-from-V1", "basis": "weight dtype not recorded in receipt"},
            {"stage": "output_logit_after_explicit_cast", "dtype": "float32", "status": "source-established", "basis": "model(...).logits.float()"},
            {"stage": "log_softmax", "dtype": "float32", "status": "source-established", "basis": "called on explicitly float32 logits"},
            {"stage": "candidate_log_probability", "dtype": "float32", "status": "source-established", "basis": "float32 scalar indexing"},
            {"stage": "candidate_margin_accumulation", "dtype": "float32", "status": "source-established", "basis": "torch.zeros(... dtype=float32) and in-place additions"},
            {"stage": "saved_margin", "dtype": "float32", "status": "artifact-established", "basis": "semantic_margin arrays are float32"},
            {"stage": "activation_gradient", "dtype": "float32", "status": "source-and-artifact-established", "basis": "gradient.float().cpu().numpy() and saved float32"},
        ],
        "realized_perturbation_saved": False,
        "realized_perturbation_disposition": "REALIZED_PERTURBATION_NOT_RECOVERABLE_FROM_V1",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    audit = verify_v1_archive(args.archive)
    replay, records, validations = exact_replay(args.archive)
    write_json(output / "V1_PILOT_EXACT_REPLAY.json", {"archive_audit": audit.to_dict(), **replay})
    numerical, quantization_rows, epsilon_rows = numerical_forensics(validations)
    write_csv(output / "margin_quantization_audit.csv", quantization_rows, list(quantization_rows[0]))
    write_csv(output / "epsilon_resolution_audit.csv", epsilon_rows, list(epsilon_rows[0]))
    dtype = dtype_audit(REPO)
    write_json(output / "dtype_execution_audit.json", dtype)
    mapping, pair_rows, summaries = mapping_forensics(records)
    write_csv(output / "mapping_pair_effects.csv", pair_rows, list(pair_rows[0]))
    write_csv(output / "semantic_vs_token_effect_summary.csv", summaries, list(summaries[0]))
    numerical_md = ["# Numerical resolution forensic report", "", "V1 is retained as `CAUSAL_FINGERPRINT_MEASUREMENT_INVALID`. Saved evidence is diagnostic only.", ""]
    for actor, value in numerical.items():
        numerical_md += [f"## {actor}", "", f"Estimated symmetric-margin resolution floor: `{value['estimated_resolution_floor']:.12g}`. Median activation norm: `{value['median_activation_norm']:.12g}`. Performance monotonic with increasing epsilon: `{value['performance_monotonic_with_epsilon']}`.", "", "The raw plus/minus margins and post-cast realized activations were not persisted. Symmetric changes are reconstructed exactly from the saved float32 finite-difference slope and frozen epsilon; endpoint/template/mapping strata absent from the validation artifact cannot be invented.", ""]
    numerical_md += ["## Diagnosis", "", "The epsilon curves improve strongly and monotonically for both actors as absolute displacement grows, while the smallest steps contain lattice-like or near-zero symmetric changes. This supports numerical-resolution limitation as a material contributor, but does not establish it as the sole cause because V1 omitted post-cast activations and raw plus/minus margins.", ""]
    (output / "NUMERICAL_RESOLUTION_FORENSIC_REPORT.md").write_text("\n".join(numerical_md), encoding="utf-8")
    mapping_md = ["# Mapping factorial forensic report", "", "All summaries are post-hoc diagnosis of permanently invalid V1 and cannot count as V2 success.", ""]
    for actor, value in mapping.items():
        mapping_md += [f"## {actor}", "", f"Median semantically oriented standard/reversed correlation: `{value['median_raw_semantic_mapping_reversal_correlation']:.12g}`. Median physical-token-oriented correlation after reversing the second sign: `{value['median_physical_token_oriented_reversal_correlation']:.12g}`. Cross-symbol correlation of mapping-paired semantic means: `{value['mapping_paired_semantic_cross_symbol_correlation']}`.", "", f"Negative semantic correlation becomes positive physical orientation: `{value['negative_semantic_becomes_positive_physical']}`.", ""]
    mapping_md += ["## Interpretation boundary", "", "The standard and reversed prompts are crossed environments, not identical activation states. The average and contrast are factorial summaries, not perfectly isolated biological mechanisms.", ""]
    (output / "MAPPING_FACTORIAL_FORENSIC_REPORT.md").write_text("\n".join(mapping_md), encoding="utf-8")
    fixture = stable_gradient_random_direction_fixture()
    reliability = {
        "schema_version": 1,
        "v1_statistic": "Pearson correlation across fixed directions between group-half mean-gradient dot direction values.",
        "algebra": "For fixed directions d_k and half-mean gradients g_1,g_2, V1 correlates the vectors [g_1 dot d_k]_k and [g_2 dot d_k]_k. If g_1 approximately equals g_2, arbitrary fixed random directions are expected to be highly reliable.",
        "synthetic_fixture": fixture,
        "historical_v1_rule_retained": "random split-half reliability < 0.60",
        "v2_rule": "Random-direction reliability is reported as an instrument control and is not a failure condition. Primary reproducibility is per-direction correlation of multi-endpoint fingerprints across disjoint prompt halves plus upper-triangle causal-Gram correlation.",
    }
    write_json(output / "RELIABILITY_METRIC_FORENSIC_REPORT.json", reliability)
    (output / "RELIABILITY_METRIC_FORENSIC_REPORT.md").write_text("# Reliability metric forensic report\n\n" + reliability["algebra"] + "\n\nThe frozen V1 `<0.60` random-direction rule remains historical failed-contract logic. It is mathematically mis-specified as an invalidity test because stable gradients make fixed arbitrary projections reproducible.\n", encoding="utf-8")
    (output / "V2_RELIABILITY_DEFINITION.md").write_text("# V2 reliability definition\n\nFor every direction, build the fixed-order vector of mapping-paired semantic effects across at least 24 frozen endpoint environments independently on each group-disjoint prompt half. Primary reliability is the Pearson correlation between the two vectors. Report the median across directions and every family minimum. Independently construct row-cosine causal Gram matrices on each half and correlate their upper triangles. Random directions are instrument controls only; semantic specificity remains assigned to the frozen isotropic and geometry-matched downstream nulls.\n", encoding="utf-8")
    forensic_summary = {"archive_audit": audit.to_dict(), "numerical": numerical, "mapping": mapping, "dtype": dtype, "reliability": reliability, "all_candidates_single_token": all(replay["actors"][actor]["all_candidates_single_token"] for actor in ("llama", "gemma"))}
    write_json(output / "FORENSIC_SUMMARY.json", forensic_summary)
    print("CPU_ONLY_V2_FORENSIC_PREFLIGHT_PASS")
    print("V1_PILOT_EXACT_REPLAY_PASS")
    print("REALIZED_PERTURBATION_NOT_RECOVERABLE_FROM_V1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
