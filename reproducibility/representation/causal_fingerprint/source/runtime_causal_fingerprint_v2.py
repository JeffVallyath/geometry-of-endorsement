from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geometry_endorsement.causal_fingerprint_v2.archive_audit import sha256_file
from geometry_endorsement.causal_fingerprint_v2.pilot_gate_v2 import decide
from geometry_endorsement.causal_fingerprint_v2.numerical_resolution import diagnostics
from geometry_endorsement.causal_fingerprint_v2.perturbation_calibration import CalibrationThresholds
from geometry_endorsement.causal_fingerprint_v2.reliability_v2 import causal_gram_reliability, per_direction_fingerprint_similarity
from geometry_endorsement.causal_fingerprint_v2.reporting import receipt, write_json
from geometry_endorsement.semantic_control_geometry.prompt_contracts import MAPPINGS, TEMPLATES, render


MODELS = {
    "llama": {"id": "meta-llama/Meta-Llama-3.1-8B-Instruct", "revision": "0e9e39f249a16976918f6564b8830bc894c89659", "layer": 19, "hidden_size": 4096},
    "gemma": {"id": "google/gemma-2-9b-it", "revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819", "layer": 27, "hidden_size": 3584},
}
FAMILY_SLICES = {"relation": slice(0, 8), "language": slice(8, 12), "sentiment": slice(12, 20), "truth": slice(20, 28), "token": slice(28, 32), "random": slice(32, 40)}
DIRECTION_ARTIFACT_SHA256 = {
    "llama": "d12262ae74c56a56418f83e6ca97a6501bebb731d7ca2dd1f26449a283634fb9",
    "gemma": "de4e02ab71a7c88dbfdd7fe11cca2b82d4f549e0752c6b54d3e22ee0b319c529",
}
CALIBRATION_THRESHOLDS = CalibrationThresholds()


def _family(record: dict[str, Any]) -> str:
    return str(record["family"])


def _layer_stack(model: Any) -> Any:
    for path in (("model", "layers"), ("model", "model", "layers")):
        value = model
        try:
            for component in path:
                value = getattr(value, component)
            return value
        except AttributeError:
            continue
    raise RuntimeError("UNSUPPORTED_ACTOR_LAYER_TOPOLOGY")


def _lm_head(model: Any) -> Any:
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise RuntimeError("LM_HEAD_NOT_FOUND")


def _prompt_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    values = [int(value) for value in encoded]
    if not values:
        raise RuntimeError("EMPTY_PROMPT")
    return values


def _candidate_id(tokenizer: Any, candidate: str) -> int:
    values = list(tokenizer.encode(candidate, add_special_tokens=False))
    if len(values) != 1:
        raise RuntimeError(f"V2_CANDIDATE_NOT_SINGLE_TOKEN:{candidate}:{values}")
    return int(values[0])


def load_actor(actor: str, cache_root: Path) -> tuple[Any, Any]:
    """The only model-loading path; unreachable from setup/audit modes."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = MODELS[actor]
    kwargs = {"revision": spec["revision"], "cache_dir": str(cache_root), "token": os.environ.get("HF_TOKEN")}
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], **kwargs)
    # transformers==4.55.2 accepts the legacy torch_dtype alias here.  Passing
    # dtype is forwarded into the concrete model constructor and fails before
    # weights are loaded; the requested/runtime dtype remains bfloat16.
    model = AutoModelForCausalLM.from_pretrained(spec["id"], torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True, **kwargs)
    model.eval()
    model.config.use_cache = False
    return model, tokenizer


def release_accelerator_cache() -> int:
    """Release process-owned accelerator allocations and report residual bytes."""
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return int(torch.cuda.memory_allocated())
    except Exception:
        return 0


def measure_margin(model: Any, prompt_ids: list[int], positive_id: int, negative_id: int, layer: int, *, delta: np.ndarray | None = None, gradient: bool = False) -> dict[str, Any]:
    """Measure native/high-precision margins and the realized layer delta."""
    import torch

    layer_module = _layer_stack(model)[layer]
    head = _lm_head(model)
    captured: dict[str, Any] = {}
    delta_tensor = None if delta is None else torch.as_tensor(delta, dtype=torch.float32)

    def layer_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        position = len(prompt_ids) - 1
        captured["layer_hidden"] = hidden
        captured["before"] = hidden[:, position, :].detach().float()
        if delta_tensor is None:
            captured["after"] = captured["before"]
            return output
        changed = hidden.clone()
        changed[:, position, :] += delta_tensor.to(hidden.device, dtype=hidden.dtype)
        captured["after"] = changed[:, position, :].detach().float()
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    def head_pre_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
        captured["head_input"] = inputs[0]

    layer_handle = layer_module.register_forward_hook(layer_hook)
    head_handle = head.register_forward_pre_hook(head_pre_hook)
    try:
        ids = torch.as_tensor([prompt_ids], device=model.device, dtype=torch.long)
        context = torch.enable_grad() if gradient else torch.inference_mode()
        with context:
            outputs = model(input_ids=ids, use_cache=False)
            native_logits = outputs.logits[0, -1]
            native_margin = native_logits[positive_id].float() - native_logits[negative_id].float()
            final_hidden = captured["head_input"][0, -1].float()
            rows = head.weight[[positive_id, negative_id]].detach().float()
            high_margin = torch.sum(final_hidden * rows[0], dtype=torch.float32) - torch.sum(final_hidden * rows[1], dtype=torch.float32)
            result_gradient = None
            if gradient:
                full = torch.autograd.grad(high_margin, captured["layer_hidden"], retain_graph=False, create_graph=False, allow_unused=False)[0]
                result_gradient = full[0, len(prompt_ids) - 1].detach().float().cpu().numpy()
        before = captured["before"].cpu().numpy()[0]
        after = captured["after"].cpu().numpy()[0]
        return {"native_margin": float(native_margin.detach().cpu()), "high_precision_margin": float(high_margin.detach().cpu()), "gradient": result_gradient, "activation": before, "realized_delta": after - before, "native_verdict": int(float(native_margin.detach().cpu()) >= 0), "high_precision_verdict": int(float(high_margin.detach().cpu()) >= 0)}
    finally:
        layer_handle.remove()
        head_handle.remove()
        model.zero_grad(set_to_none=True)


def _save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _valid(path: Path) -> bool:
    sidecar = path.with_suffix(".json")
    if not path.is_file() or not sidecar.is_file():
        return False
    value = json.loads(sidecar.read_text(encoding="utf-8"))
    return value.get("complete") is True and value.get("artifact_sha256") == sha256_file(path)


def _directions(persistent_root: Path, actor: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = persistent_root / "checkpoints" / actor / "directions.npz"
    if sha256_file(path) != DIRECTION_ARTIFACT_SHA256[actor]:
        raise RuntimeError(f"V2_FROZEN_DIRECTION_HASH_MISMATCH:{actor}")
    with np.load(path, allow_pickle=False) as archive:
        vectors = np.concatenate([archive["directions"], archive["random_orthogonal"]], axis=0).astype(np.float32)
        ids = np.concatenate([archive["direction_ids"].astype(str), np.asarray([f"random_orthogonal_{index}" for index in range(len(archive["random_orthogonal"]))])])
        families = np.asarray(["relation"] * 8 + ["language"] * 4 + ["sentiment"] * 8 + ["truth"] * 8 + ["token"] * 4 + ["random"] * 8)
    if vectors.shape != (40, MODELS[actor]["hidden_size"]):
        raise RuntimeError("V2_DIRECTION_ARTIFACT_SHAPE_MISMATCH")
    return vectors, ids, families


def _render(record: dict[str, Any], template_id: str, mapping_id: str) -> dict[str, Any]:
    return render(_family(record), template_id, mapping_id, record["prompt_fields"])


def finalize_calibration(root: Path, grid: list[float]) -> dict[str, Any]:
    files = sorted((root / "calibration").glob("*.npz"))
    if not files:
        raise RuntimeError("V2_CALIBRATION_SHARDS_MISSING")
    loaded = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            loaded.append({key: data[key] for key in data.files})
    base_margins = np.asarray([abs(float(data["base_high_precision_margin"])) for data in loaded])
    repeat_errors = np.asarray([abs(float(data["base_high_precision_margin"]) - float(data["base_repeat_high_precision_margin"])) for data in loaded])
    resolution_floor = max(float(np.quantile(repeat_errors, 0.95)), float(np.finfo(np.float32).eps * max(np.median(base_margins), 1.0) * 8.0), 1e-7)
    verdict_disagreement = float(np.mean([int(data["base_native_verdict"]) != int(data["base_high_precision_verdict"]) for data in loaded]))
    if verdict_disagreement > 0.01:
        raise RuntimeError("HIGH_PRECISION_ENDPOINT_NOT_EQUIVALENT")
    by_key: dict[tuple[str, str], dict[str, dict[str, np.ndarray]]] = {}
    for data in loaded:
        key = (str(data["item_id"]), str(data["template"]))
        by_key.setdefault(key, {})[str(data["mapping"])] = data
    candidates = []
    for epsilon_index, epsilon in enumerate(grid):
        slopes, changes, ratios, cosines, saturations, adjacent, realized_valid = [], [], [], [], [], [], []
        for data in loaded:
            mask = np.isclose(data["epsilons"].astype(float), epsilon)
            local = data["finite_difference"][mask].astype(float)
            slopes.extend(local.tolist())
            changes.extend(np.abs(data["high_plus_margin"][mask].astype(float) - data["high_minus_margin"][mask].astype(float)).tolist())
            ratios.extend(data["realized_norm_ratio"][mask].astype(float).tolist())
            cosines.extend(data["realized_cosine"][mask].astype(float).tolist())
            saturations.extend((np.abs(data["high_plus_margin"][mask].astype(float)) >= 20.0).tolist())
            saturations.extend((np.abs(data["high_minus_margin"][mask].astype(float)) >= 20.0).tolist())
            realized_valid.extend(data["realized_valid"][mask].astype(bool).tolist())
            if epsilon_index + 1 < len(grid):
                next_mask = np.isclose(data["epsilons"].astype(float), grid[epsilon_index + 1])
                next_slopes = data["finite_difference"][next_mask].astype(float)
                adjacent.extend((np.abs(local - next_slopes) / np.maximum(np.abs(local), 1e-8)).tolist())
        mapping_agreements = []
        for values in by_key.values():
            required = {mapping.mapping_id for mapping in MAPPINGS}
            if not required.issubset(values):
                continue
            family_vectors = {}
            for symbol, standard, reversed_id in (("AB", "MAPPING_AB_STANDARD", "MAPPING_AB_REVERSED"), ("12", "MAPPING_12_STANDARD", "MAPPING_12_REVERSED")):
                left, right = values[standard], values[reversed_id]
                left_mask = np.isclose(left["epsilons"].astype(float), epsilon)
                right_mask = np.isclose(right["epsilons"].astype(float), epsilon)
                indices = left["fd_direction_indices"][left_mask].astype(int)
                semantic = indices < 28
                family_vectors[symbol] = (left["finite_difference"][left_mask].astype(float)[semantic] + right["finite_difference"][right_mask].astype(float)[semantic]) / 2.0
            if np.std(family_vectors["AB"]) > 0 and np.std(family_vectors["12"]) > 0:
                mapping_agreements.append(float(np.corrcoef(family_vectors["AB"], family_vectors["12"])[0, 1]))
        metrics = {
            "epsilon": float(epsilon),
            "median_cosine": float(np.median(cosines)),
            "median_norm_ratio": float(np.median(ratios)),
            "resolvable_fraction": float(np.mean(np.asarray(changes) > 4.0 * resolution_floor)),
            "median_adjacent_slope_relative_difference": float(np.median(adjacent)) if adjacent else None,
            "saturation_fraction": float(np.mean(saturations)),
            "invalid_realized_measurement_fraction": float(1.0 - np.mean(realized_valid)),
            "median_mapping_paired_AB_12_agreement": float(np.median(mapping_agreements)) if mapping_agreements else 0.0,
            "base_native_high_precision_verdict_disagreement_fraction": verdict_disagreement,
        }
        metrics["passed"] = bool(
            metrics["median_cosine"] >= CALIBRATION_THRESHOLDS.median_alignment_min
            and CALIBRATION_THRESHOLDS.median_norm_ratio_min <= metrics["median_norm_ratio"] <= CALIBRATION_THRESHOLDS.median_norm_ratio_max
            and metrics["resolvable_fraction"] >= CALIBRATION_THRESHOLDS.resolvable_fraction_min
            and metrics["median_adjacent_slope_relative_difference"] is not None
            and metrics["median_adjacent_slope_relative_difference"] <= CALIBRATION_THRESHOLDS.adjacent_slope_relative_difference_max
            and metrics["saturation_fraction"] < CALIBRATION_THRESHOLDS.saturation_fraction_max
            and metrics["invalid_realized_measurement_fraction"] <= CALIBRATION_THRESHOLDS.invalid_realized_measurement_fraction_max
            and metrics["median_mapping_paired_AB_12_agreement"] >= 0.60
            and verdict_disagreement <= 0.01
        )
        candidates.append(metrics)
    passing = [value for value in candidates if value["passed"]]
    if not passing:
        raise RuntimeError("V2_PERTURBATION_CALIBRATION_FAILED")
    selected = passing[0]
    result = {"schema_version": 1, "selected_absolute_l2": selected["epsilon"], "high_precision_resolution_floor": resolution_floor, "candidates": candidates, "complete": True, "status": "V2_PERTURBATION_CALIBRATION_PASS"}
    write_json(root / "CALIBRATION_SELECTION.json", result)
    return result


def run_actor(actor: str, persistent_root: Path, work_root: Path) -> Path:
    setup = persistent_root / "outputs" / "causal-fingerprint-v2-setup"
    freeze = json.loads((setup / "V2_DATASET_FREEZE.json").read_text(encoding="utf-8"))
    grid = json.loads((setup / "V2_PERTURBATION_GRID_CONTRACT.json").read_text(encoding="utf-8"))["actors"][actor]["absolute_l2_grid"]
    directions, direction_ids, direction_families = _directions(persistent_root, actor)
    validation_indices = []
    for family in ("relation", "truth", "sentiment", "language", "token", "random"):
        validation_indices.extend(np.flatnonzero(direction_families == family)[:2].tolist())
    model, tokenizer = load_actor(actor, work_root / "cache" / "huggingface")
    started = time.monotonic()
    peak_vram = 0
    root = persistent_root / "checkpoints" / actor / "causal_fingerprint_v2"
    manifest: Path | None = None
    try:
        # Audit all candidate mappings before the first intervention.
        for family in sorted({_family(record) for record in freeze["records"]}):
            for mapping in MAPPINGS:
                rendered = render(family, sorted(TEMPLATES[family])[0], mapping.mapping_id, next(record["prompt_fields"] for record in freeze["records"] if _family(record) == family))
                _candidate_id(tokenizer, rendered["positive_candidate"])
                _candidate_id(tokenizer, rendered["negative_candidate"])
        for purpose, stage in (("V2_NUMERICAL_CALIBRATION", "calibration"), ("V2_MEASUREMENT_VALIDATION", "validation")):
            selected_epsilon_path = root / "CALIBRATION_SELECTION.json"
            if stage == "validation":
                if not selected_epsilon_path.is_file():
                    raise RuntimeError("V2_CALIBRATION_SELECTION_REQUIRED")
                selection = json.loads(selected_epsilon_path.read_text(encoding="utf-8"))
                if selection.get("complete") is not True:
                    raise RuntimeError("V2_CALIBRATION_SELECTION_INCOMPLETE")
                selected_epsilon = float(selection["selected_absolute_l2"])
                stage_grid = [selected_epsilon]
            else:
                stage_grid = list(map(float, grid))
            for record in [value for value in freeze["records"] if value["purpose"] == purpose]:
                family = _family(record)
                for template_id in sorted(TEMPLATES[family]):
                    for mapping in MAPPINGS:
                        rendered = _render(record, template_id, mapping.mapping_id)
                        pids = _prompt_ids(tokenizer, rendered["prompt"])
                        positive_id = _candidate_id(tokenizer, rendered["positive_candidate"])
                        negative_id = _candidate_id(tokenizer, rendered["negative_candidate"])
                        target = root / stage / f"{record['item_id'].replace(':', '_')}__{template_id}__{mapping.mapping_id}.npz"
                        if _valid(target):
                            continue
                        base = measure_margin(model, pids, positive_id, negative_id, MODELS[actor]["layer"], gradient=True)
                        base_repeat = measure_margin(model, pids, positive_id, negative_id, MODELS[actor]["layer"], gradient=False) if stage == "calibration" else base
                        effects = np.asarray(base["gradient"], dtype=np.float32) @ directions.T
                        fd_rows = []
                        realized_plus_rows = []
                        realized_minus_rows = []
                        for direction_index in validation_indices:
                            direction = directions[direction_index]
                            for epsilon in stage_grid:
                                plus = measure_margin(model, pids, positive_id, negative_id, MODELS[actor]["layer"], delta=np.float32(epsilon) * direction)
                                minus = measure_margin(model, pids, positive_id, negative_id, MODELS[actor]["layer"], delta=-np.float32(epsilon) * direction)
                                plus_projection = float(np.asarray(plus["realized_delta"], dtype=np.float64) @ direction.astype(np.float64))
                                minus_projection = float(np.asarray(minus["realized_delta"], dtype=np.float64) @ direction.astype(np.float64))
                                denominator = plus_projection - minus_projection
                                requested_plus = np.float32(epsilon) * direction
                                requested_minus = -requested_plus
                                realized_plus = np.asarray(plus["realized_delta"], dtype=np.float32)
                                realized_minus = np.asarray(minus["realized_delta"], dtype=np.float32)
                                realized_norm = float(np.linalg.norm(realized_plus))
                                realized_minus_norm = float(np.linalg.norm(realized_minus))
                                requested_norm = float(np.linalg.norm(requested_plus))
                                cosine = float(realized_plus.astype(np.float64) @ requested_plus.astype(np.float64) / max(realized_norm * requested_norm, 1e-30))
                                minus_cosine = float(realized_minus.astype(np.float64) @ requested_minus.astype(np.float64) / max(realized_minus_norm * requested_norm, 1e-30))
                                plus_ratio = realized_norm / max(requested_norm, 1e-30)
                                minus_ratio = realized_minus_norm / max(requested_norm, 1e-30)
                                conservative_ratio = min(plus_ratio, minus_ratio)
                                conservative_cosine = min(cosine, minus_cosine)
                                realized_valid = bool(
                                    abs(denominator) > 1e-12
                                    and conservative_cosine >= CALIBRATION_THRESHOLDS.per_measurement_alignment_min
                                    and min(plus_ratio, minus_ratio) >= CALIBRATION_THRESHOLDS.per_measurement_norm_ratio_min
                                    and max(plus_ratio, minus_ratio) <= CALIBRATION_THRESHOLDS.per_measurement_norm_ratio_max
                                )
                                slope = float((plus["high_precision_margin"] - minus["high_precision_margin"]) / denominator) if abs(denominator) > 1e-12 else float("nan")
                                fd_rows.append((direction_index, epsilon, slope, denominator, conservative_ratio, conservative_cosine, plus["high_precision_margin"], minus["high_precision_margin"], plus["native_margin"], minus["native_margin"], plus_ratio, minus_ratio, cosine, minus_cosine, int(realized_valid)))
                                realized_plus_rows.append(realized_plus)
                                realized_minus_rows.append(realized_minus)
                        rows = np.asarray(fd_rows, dtype=np.float64)
                        _save_npz(target, gradient=np.asarray(base["gradient"], dtype=np.float32), effects=np.asarray(effects, dtype=np.float32), direction_ids=direction_ids, direction_families=direction_families, fd_direction_indices=rows[:, 0].astype(np.int32), epsilons=rows[:, 1].astype(np.float32), finite_difference=rows[:, 2].astype(np.float32), realized_denominator=rows[:, 3].astype(np.float32), realized_norm_ratio=rows[:, 4].astype(np.float32), realized_cosine=rows[:, 5].astype(np.float32), high_plus_margin=rows[:, 6].astype(np.float32), high_minus_margin=rows[:, 7].astype(np.float32), native_plus_margin=rows[:, 8].astype(np.float32), native_minus_margin=rows[:, 9].astype(np.float32), realized_plus_norm_ratio=rows[:, 10].astype(np.float32), realized_minus_norm_ratio=rows[:, 11].astype(np.float32), realized_plus_cosine=rows[:, 12].astype(np.float32), realized_minus_cosine=rows[:, 13].astype(np.float32), realized_valid=rows[:, 14].astype(np.bool_), realized_plus_delta=np.stack(realized_plus_rows).astype(np.float32), realized_minus_delta=np.stack(realized_minus_rows).astype(np.float32), base_native_margin=np.asarray(base["native_margin"], dtype=np.float32), base_high_precision_margin=np.asarray(base["high_precision_margin"], dtype=np.float32), base_repeat_high_precision_margin=np.asarray(base_repeat["high_precision_margin"], dtype=np.float32), base_native_verdict=np.asarray(base["native_verdict"], dtype=np.int8), base_high_precision_verdict=np.asarray(base["high_precision_verdict"], dtype=np.int8), activation=np.asarray(base["activation"], dtype=np.float32), item_id=np.asarray(record["item_id"]), group_id=np.asarray(record["group_id"]), dataset_id=np.asarray(record["dataset_id"]), family=np.asarray(family), template=np.asarray(template_id), mapping=np.asarray(mapping.mapping_id))
                        write_json(target.with_suffix(".json"), receipt(f"V2_{stage.upper()}_SHARD", target, actor=actor, source_commit="227c452bece0ac1d3a664444f69bf88b9199d4c0"))
                        try:
                            import torch

                            peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated()))
                        except Exception:
                            pass
            if stage == "calibration":
                existing_selection = None
                if selected_epsilon_path.is_file():
                    existing_selection = json.loads(selected_epsilon_path.read_text(encoding="utf-8"))
                if not existing_selection or existing_selection.get("complete") is not True:
                    selection = finalize_calibration(root, list(map(float, grid)))
                    selection["actor"] = actor
                    write_json(selected_epsilon_path, selection)
        unfinished = sorted(root.rglob("*.part.npz"))
        if unfinished:
            raise RuntimeError(f"V2_UNFINISHED_PART_FILES:{actor}:{len(unfinished)}")
        manifest = root / "manifest.json"
        artifacts = sorted(path for path in root.rglob("*.npz") if not path.name.endswith(".part.npz"))
        write_json(manifest, {"schema_version": 1, "actor": actor, "stage": "CAUSAL_FINGERPRINT_V2_PILOT", "files": [{"path": path.as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in artifacts], "scientific_summary_sealed": True})
    finally:
        # These assignments release the caller's owning references. Merely
        # deleting arguments inside a helper would leave these references live.
        model = None
        tokenizer = None
        final_vram = release_accelerator_cache()
    if manifest is None:
        raise RuntimeError("V2_PILOT_MANIFEST_NOT_CREATED")
    if final_vram != 0:
        raise RuntimeError(f"V2_ACTOR_UNLOAD_INCOMPLETE:{actor}:{final_vram}")
    write_json(root / "UNLOAD.json", {
        "schema_version": 1,
        "actor": actor,
        "complete": True,
        "final_compute_process_vram_bytes": final_vram,
        "status": "V2_ACTOR_FULLY_UNLOADED",
    })
    write_json(root / "PILOT.json", receipt(
        "CAUSAL_FINGERPRINT_V2_PILOT",
        manifest,
        actor=actor,
        runtime_seconds=time.monotonic() - started,
        peak_vram_bytes=peak_vram,
        final_compute_process_vram_bytes=final_vram,
        unload_receipt_sha256=sha256_file(root / "UNLOAD.json"),
    ))
    print(f"{actor.upper()}_V2_PILOT_COMPLETE_HASH_VERIFIED_AND_ACTOR_UNLOADED")
    return manifest


def _verify_actor_completion(root: Path, actor: str) -> None:
    manifest = root / "manifest.json"
    pilot = root / "PILOT.json"
    unload = root / "UNLOAD.json"
    if not all(path.is_file() for path in (manifest, pilot, unload)):
        raise RuntimeError(f"V2_ACTOR_COMPLETION_RECEIPT_MISSING:{actor}")
    pilot_value = json.loads(pilot.read_text(encoding="utf-8"))
    unload_value = json.loads(unload.read_text(encoding="utf-8"))
    if pilot_value.get("complete") is not True or pilot_value.get("artifact_sha256") != sha256_file(manifest):
        raise RuntimeError(f"V2_ACTOR_MANIFEST_HASH_MISMATCH:{actor}")
    if unload_value.get("complete") is not True or unload_value.get("final_compute_process_vram_bytes") != 0:
        raise RuntimeError(f"V2_ACTOR_UNLOAD_NOT_VERIFIED:{actor}")
    if pilot_value.get("unload_receipt_sha256") != sha256_file(unload):
        raise RuntimeError(f"V2_ACTOR_UNLOAD_RECEIPT_HASH_MISMATCH:{actor}")


def _load_stage_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(path for path in root.glob("*.npz") if not path.name.endswith(".part.npz")):
        if not _valid(path):
            raise RuntimeError(f"V2_SHARD_INVALID:{path.name}")
        with np.load(path, allow_pickle=False) as data:
            rows.append({key: data[key] for key in data.files})
    if not rows:
        raise RuntimeError("V2_VALIDATION_SHARDS_MISSING")
    return rows


def finalize_actor_validation(persistent_root: Path, actor: str) -> dict[str, Any]:
    root = persistent_root / "checkpoints" / actor / "causal_fingerprint_v2"
    _verify_actor_completion(root, actor)
    rows = _load_stage_rows(root / "validation")
    verdict_disagreement = float(np.mean([
        int(row["base_native_verdict"]) != int(row["base_high_precision_verdict"])
        for row in rows
    ]))
    if verdict_disagreement > 0.01:
        raise RuntimeError("HIGH_PRECISION_ENDPOINT_NOT_EQUIVALENT")
    paired: dict[tuple[str, str, str, str, str], dict[str, dict[str, np.ndarray]]] = {}
    for row in rows:
        key = (str(row["item_id"]), str(row["group_id"]), str(row["dataset_id"]), str(row["family"]), str(row["template"]))
        paired.setdefault(key, {})[str(row["mapping"])] = row
    measurements = []
    environment_rows = []
    for key, values in paired.items():
        required = {mapping.mapping_id for mapping in MAPPINGS}
        if not required.issubset(values):
            raise RuntimeError(f"V2_MAPPING_PAIR_INCOMPLETE:{key}")
        item, group, dataset, family, template = key
        for symbol, standard, reversed_id in (("AB", "MAPPING_AB_STANDARD", "MAPPING_AB_REVERSED"), ("12", "MAPPING_12_STANDARD", "MAPPING_12_REVERSED")):
            left, right = values[standard], values[reversed_id]
            gradient_pair = (left["effects"].astype(float) + right["effects"].astype(float)) / 2.0
            token_contrast = (left["effects"].astype(float) - right["effects"].astype(float)) / 2.0
            if not np.array_equal(left["fd_direction_indices"], right["fd_direction_indices"]):
                raise RuntimeError("V2_FD_DIRECTION_ALIGNMENT_FAILED")
            if not np.all(left["realized_valid"].astype(bool)) or not np.all(right["realized_valid"].astype(bool)):
                raise RuntimeError("V2_VALIDATION_REALIZED_PERTURBATION_INVALID")
            finite_pair = (left["finite_difference"].astype(float) + right["finite_difference"].astype(float)) / 2.0
            indices = left["fd_direction_indices"].astype(int)
            saturation = (np.abs(left["high_plus_margin"].astype(float)) >= 20.0) | (np.abs(left["high_minus_margin"].astype(float)) >= 20.0) | (np.abs(right["high_plus_margin"].astype(float)) >= 20.0) | (np.abs(right["high_minus_margin"].astype(float)) >= 20.0)
            measurements.append({"indices": indices, "gradient": gradient_pair[indices], "finite": finite_pair, "saturation": saturation})
            environment_rows.append({"group": f"{dataset}:{group}", "family": "language" if family.startswith("language_") else family, "environment": (family, template, symbol), "semantic": gradient_pair, "token": token_contrast})
    def numerical_for(direction_test: Any) -> Any:
        gradients, finite, saturation = [], [], []
        for measurement in measurements:
            mask = np.asarray([direction_test(int(index)) for index in measurement["indices"]], dtype=bool)
            gradients.append(measurement["gradient"][mask])
            finite.append(measurement["finite"][mask])
            saturation.append(measurement["saturation"][mask])
        return diagnostics(np.concatenate(gradients), np.concatenate(finite), np.concatenate(saturation), 0.0)

    numeric = numerical_for(lambda index: index < 28)
    random_numeric = numerical_for(lambda index: index >= 32)
    token_numeric = numerical_for(lambda index: 28 <= index < 32)
    groups_by_family: dict[str, list[str]] = {}
    for row in environment_rows:
        groups_by_family.setdefault(row["family"], []).append(row["group"])
    left_group_sets = {family: set(sorted(set(groups))[::2]) for family, groups in groups_by_family.items()}
    environments = sorted(set(row["environment"] for row in environment_rows))
    half_fingerprints = []
    token_fingerprints = []
    for half in (0, 1):
        blocks, token_blocks = [], []
        for environment in environments:
            selected = [row for row in environment_rows if row["environment"] == environment and ((row["group"] in left_group_sets[row["family"]]) == (half == 0))]
            if not selected:
                raise RuntimeError(f"V2_RELIABILITY_ENVIRONMENT_EMPTY:{environment}:{half}")
            blocks.append(np.mean(np.stack([row["semantic"] for row in selected]), axis=0))
            token_blocks.append(np.mean(np.stack([row["token"] for row in selected]), axis=0))
        half_fingerprints.append(np.stack(blocks, axis=1))
        token_fingerprints.append(np.stack(token_blocks, axis=1))
    similarities = per_direction_fingerprint_similarity(half_fingerprints[0], half_fingerprints[1])
    family_similarity = {family: float(np.median(similarities[part])) for family, part in FAMILY_SLICES.items() if family not in {"token", "random"}}
    gram = causal_gram_reliability(half_fingerprints[0], half_fingerprints[1])
    environment_index = {environment: index for index, environment in enumerate(environments)}
    paired_environment_indices = []
    for family, template, symbol in environments:
        if symbol != "AB":
            continue
        counterpart = (family, template, "12")
        if counterpart not in environment_index:
            raise RuntimeError(f"V2_AB_12_ENVIRONMENT_PAIR_MISSING:{family}:{template}")
        paired_environment_indices.append((environment_index[(family, template, "AB")], environment_index[counterpart]))
    mapping_agreements = []
    for half in (0, 1):
        ab = half_fingerprints[half][:, [pair[0] for pair in paired_environment_indices]].ravel()
        one_two = half_fingerprints[half][:, [pair[1] for pair in paired_environment_indices]].ravel()
        mapping_agreements.append(float(np.corrcoef(ab, one_two)[0, 1]))
    semantic_flat = np.concatenate([value.ravel() for value in half_fingerprints])
    token_flat = np.concatenate([value.ravel() for value in token_fingerprints])
    semantic_token_correlation = float(np.corrcoef(semantic_flat, token_flat)[0, 1])
    result = {
        "correlation": numeric.correlation,
        "sign_agreement": numeric.sign_agreement,
        "median_absolute_relative_error": numeric.median_absolute_relative_error,
        "saturation_fraction": numeric.saturation_fraction,
        "median_per_direction_fingerprint_similarity": float(np.median(similarities)),
        "family_fingerprint_similarity": family_similarity,
        "causal_gram_reliability": gram,
        "mapping_family_agreement": float(np.median(mapping_agreements)),
        "semantic_token_contrast_correlation": semantic_token_correlation,
        "random_direction_reliability": float(np.median(similarities[FAMILY_SLICES["random"]])),
        "random_direction_numerical": random_numeric.__dict__,
        "token_control_numerical": token_numeric.__dict__,
        "endpoint_environment_count": len(environments),
        "base_native_high_precision_verdict_disagreement_fraction": verdict_disagreement,
    }
    return result


def finalize_v2(persistent_root: Path) -> str:
    actors = {actor: finalize_actor_validation(persistent_root, actor) for actor in ("llama", "gemma")}
    status = decide(actors)
    target = persistent_root / "checkpoints" / "CAUSAL_FINGERPRINT_V2_GATE.json"
    write_json(target, {"schema_version": 1, "status": status, "actors": actors})
    print(status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["AUDIT_ONLY", "RUN_LLAMA_V2_PILOT", "RUN_GEMMA_V2_PILOT", "FINALIZE_V2_PILOT_CPU"], required=True)
    parser.add_argument("--persistent-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--allow-model-inference", action="store_true")
    args = parser.parse_args()
    setup = args.persistent_root / "outputs" / "causal-fingerprint-v2-setup"
    required = ["V2_MEASUREMENT_CONTRACT.json", "V2_DATASET_FREEZE.json", "V2_PERTURBATION_GRID_CONTRACT.json", "V2_SCORING_CONTRACT.json", "V2_MAPPING_CONTRACT.json", "V2_RELIABILITY_CONTRACT.json"]
    if not all((setup / name).is_file() for name in required):
        raise RuntimeError("V2_SETUP_CONTRACT_MISSING")
    setup_receipt_path = setup / "V2_SETUP_FREEZE_RECEIPT.json"
    if not setup_receipt_path.is_file():
        raise RuntimeError("V2_SETUP_FREEZE_RECEIPT_MISSING")
    setup_receipt = json.loads(setup_receipt_path.read_text(encoding="utf-8"))
    for name, expected in setup_receipt.get("contract_sha256", {}).items():
        if sha256_file(setup / name) != expected:
            raise RuntimeError(f"V2_SETUP_CONTRACT_HASH_MISMATCH:{name}")
    original_contract = args.persistent_root / "outputs" / "setup-v001" / "STUDY_CONTRACT.json"
    if sha256_file(original_contract) != setup_receipt.get("original_study_contract_sha256"):
        raise RuntimeError("V2_ORIGINAL_STUDY_CONTRACT_HASH_MISMATCH")
    for actor in ("llama", "gemma"):
        _directions(args.persistent_root, actor)
    if args.mode == "AUDIT_ONLY":
        print("CPU_ONLY_V2_RUNTIME_AUDIT_PASS")
        return 0
    if args.mode == "FINALIZE_V2_PILOT_CPU":
        finalize_v2(args.persistent_root)
        return 0
    if not args.allow_model_inference:
        raise RuntimeError("V2_MODEL_INFERENCE_EXPLICIT_FLAG_REQUIRED")
    actor = "llama" if "LLAMA" in args.mode else "gemma"
    run_actor(actor, args.persistent_root, args.work_root)
    print(f"{actor.upper()}_V2_PILOT_ARTIFACTS_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
