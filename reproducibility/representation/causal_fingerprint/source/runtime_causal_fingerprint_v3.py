from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from geometry_endorsement.causal_fingerprint_v2.archive_audit import sha256_file
from geometry_endorsement.causal_fingerprint_v2.reporting import receipt, write_json
from geometry_endorsement.causal_fingerprint_v3.calibration_selection import choose_smallest_passing
from geometry_endorsement.causal_fingerprint_v3.checkpoint_archive import publish_archive
from geometry_endorsement.causal_fingerprint_v3.runtime_stages import assert_stage_allowed
from geometry_endorsement.causal_fingerprint_v3.two_sided_linearity import two_sided_linearity
from geometry_endorsement.causal_fingerprint_v3.validation_gate import actor_pass, decide as decide_validation
from geometry_endorsement.semantic_control_geometry.prompt_contracts import MAPPINGS, TEMPLATES, render


MODES = (
    "AUDIT_AND_HASH_VERIFY",
    "RUN_LLAMA_V3_CALIBRATION",
    "RUN_GEMMA_V3_CALIBRATION",
    "RUN_LLAMA_V3_VALIDATION",
    "RUN_GEMMA_V3_VALIDATION",
    "FINALIZE_V3",
    "VERIFY_V3_ARTIFACTS",
)


class _CheckpointPublisher:
    def __init__(
        self,
        stage_root: Path,
        target: Path,
        logical_stage: str,
        interval_seconds: float = 300.0,
    ) -> None:
        self.stage_root = stage_root
        self.target = target
        self.logical_stage = logical_stage
        self.interval_seconds = interval_seconds
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._failure: BaseException | None = None
        self._thread = threading.Thread(
            target=self._periodic,
            name=f"checkpoint-{logical_stage}",
            daemon=True,
        )

    def _publish(self) -> None:
        with self._lock:
            publish_archive(self.stage_root, self.target, logical_stage=self.logical_stage)

    def _periodic(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self._publish()
            except BaseException as error:
                self._failure = error
                self._stop.set()
                return

    def start(self) -> None:
        self._publish()
        self._thread.start()

    def completed_group(self) -> None:
        self.check()
        self._publish()

    def check(self) -> None:
        if self._failure is not None:
            raise RuntimeError("V3_PERIODIC_CHECKPOINT_FAILED") from self._failure

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        self.check()


def _load_v2_runtime() -> Any:
    path = REPO_ROOT / "scripts" / "runtime_causal_fingerprint_v2.py"
    spec = importlib.util.spec_from_file_location("frozen_v2_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("V2_RUNTIME_IMPORT_FAILED")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prompt_ids(tokenizer: Any, text: str) -> list[int]:
    """Normalize both list and BatchEncoding returns without changing tokenization."""
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True
    )
    if isinstance(encoded, dict) or hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    values = [int(value) for value in encoded]
    if not values:
        raise RuntimeError("EMPTY_PROMPT")
    return values


def _setup(root: Path) -> Path:
    candidates = (root / "frozen_setup", root / "outputs" / "causal-fingerprint-v3-setup", root)
    for candidate in candidates:
        if (candidate / "V3_STUDY_CONTRACT.json").is_file():
            return candidate
    raise RuntimeError("V3_FROZEN_SETUP_NOT_FOUND")


def _receipts(results_root: Path) -> dict[str, dict[str, Any]]:
    values = {}
    receipt_root = results_root / "receipts"
    if receipt_root.is_dir():
        for path in receipt_root.glob("*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            values[str(value.get("stage", path.stem))] = value
    return values


def _write_stage_receipt(results_root: Path, stage: str, **values: Any) -> Path:
    path = results_root / "receipts" / f"{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {"schema_version": 1, "stage": stage, "complete": True, "hash_verified": True, **values})
    return path


def audit(setup_root: Path, results_root: Path) -> None:
    verifier_path = REPO_ROOT / "scripts" / "verify_causal_fingerprint_v3.py"
    spec = importlib.util.spec_from_file_location("v3_verify", verifier_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("V3_VERIFIER_IMPORT_FAILED")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verifier.verify(setup_root)
    _write_stage_receipt(results_root, "AUDIT_AND_HASH_VERIFY", status="V3_RUNTIME_AUDIT_PASS")
    print("V3_RUNTIME_AUDIT_PASS")


def _record_family(record: dict[str, Any]) -> str:
    return str(record["family"])


def _run_measurements(actor: str, purpose: str, stage: str, setup_root: Path,
                      study_root: Path, results_root: Path, work_root: Path) -> tuple[Path, float, int]:
    v2 = _load_v2_runtime()
    freeze_name = "V3_CALIBRATION_DATASET_FREEZE.json" if purpose == "V3_NUMERICAL_CALIBRATION" else "V3_VALIDATION_DATASET_FREEZE.json"
    records = json.loads((setup_root / freeze_name).read_text(encoding="utf-8"))["records"]
    grid_contract = json.loads((setup_root / "V3_PERTURBATION_GRID_CONTRACT.json").read_text(encoding="utf-8"))
    if stage == "calibration":
        grid = list(map(float, grid_contract["actors"][actor]["absolute_l2_grid"]))
    else:
        selection_path = results_root / "checkpoints" / actor / "causal_fingerprint_v3" / "CALIBRATION_SELECTION.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("status") != "V3_PERTURBATION_CALIBRATION_PASS":
            raise RuntimeError("V3_CALIBRATION_SELECTION_REQUIRED")
        grid = [float(selection["selected_absolute_l2"])]
    directions, direction_ids, direction_families = v2._directions(study_root, actor)
    direction_indices: list[int] = []
    for family in ("relation", "truth", "sentiment", "language", "token", "random"):
        direction_indices.extend(np.flatnonzero(direction_families == family)[:2].tolist())
    root = results_root / "checkpoints" / actor / "causal_fingerprint_v3"
    logical_stage = f"run_{actor}_v3_{stage}"
    checkpoint = _CheckpointPublisher(
        root / stage,
        root / f"{stage}_checkpoint.zip",
        logical_stage,
    )
    checkpoint.start()
    started = time.monotonic()
    peak_vram = 0
    checkpoint_failure: BaseException | None = None
    model = None
    tokenizer = None
    try:
        model, tokenizer = v2.load_actor(actor, work_root / "cache" / "huggingface")
        for record in records:
            family = _record_family(record)
            for template_id in sorted(TEMPLATES[family]):
                for mapping in MAPPINGS:
                    rendered = render(family, template_id, mapping.mapping_id, record["prompt_fields"])
                    pids = _prompt_ids(tokenizer, rendered["prompt"])
                    positive_id = v2._candidate_id(tokenizer, rendered["positive_candidate"])
                    negative_id = v2._candidate_id(tokenizer, rendered["negative_candidate"])
                    safe_item = str(record["item_id"]).replace(":", "_").replace("/", "_")
                    target = root / stage / f"{safe_item}__{template_id}__{mapping.mapping_id}.npz"
                    if v2._valid(target):
                        continue
                    base = v2.measure_margin(model, pids, positive_id, negative_id, v2.MODELS[actor]["layer"], gradient=True)
                    repeat = v2.measure_margin(model, pids, positive_id, negative_id, v2.MODELS[actor]["layer"], gradient=False) if stage == "calibration" else base
                    effects = np.asarray(base["gradient"], dtype=np.float32) @ directions.T
                    fd_rows, realized_plus_rows, realized_minus_rows = [], [], []
                    for direction_index in direction_indices:
                        direction = directions[direction_index]
                        for epsilon in grid:
                            requested_plus = np.float32(epsilon) * direction
                            plus = v2.measure_margin(model, pids, positive_id, negative_id, v2.MODELS[actor]["layer"], delta=requested_plus)
                            minus = v2.measure_margin(model, pids, positive_id, negative_id, v2.MODELS[actor]["layer"], delta=-requested_plus)
                            realized_plus = np.asarray(plus["realized_delta"], dtype=np.float32)
                            realized_minus = np.asarray(minus["realized_delta"], dtype=np.float32)
                            requested_norm = float(np.linalg.norm(requested_plus))
                            plus_norm, minus_norm = float(np.linalg.norm(realized_plus)), float(np.linalg.norm(realized_minus))
                            plus_projection = float(realized_plus.astype(float) @ direction.astype(float))
                            minus_projection = float(realized_minus.astype(float) @ direction.astype(float))
                            denominator = plus_projection - minus_projection
                            plus_cosine = float(realized_plus.astype(float) @ requested_plus.astype(float) / max(plus_norm * requested_norm, 1e-30))
                            minus_cosine = float(realized_minus.astype(float) @ (-requested_plus).astype(float) / max(minus_norm * requested_norm, 1e-30))
                            plus_ratio, minus_ratio = plus_norm / max(requested_norm, 1e-30), minus_norm / max(requested_norm, 1e-30)
                            valid = bool(abs(denominator) > 1e-12 and min(plus_cosine, minus_cosine) >= 0.95 and min(plus_ratio, minus_ratio) >= 0.50 and max(plus_ratio, minus_ratio) <= 1.50)
                            slope = float((plus["high_precision_margin"] - minus["high_precision_margin"]) / denominator) if abs(denominator) > 1e-12 else float("nan")
                            fd_rows.append((direction_index, epsilon, slope, denominator, min(plus_ratio, minus_ratio), min(plus_cosine, minus_cosine), plus["high_precision_margin"], minus["high_precision_margin"], plus["native_margin"], minus["native_margin"], plus_ratio, minus_ratio, plus_cosine, minus_cosine, int(valid)))
                            realized_plus_rows.append(realized_plus)
                            realized_minus_rows.append(realized_minus)
                    rows = np.asarray(fd_rows, dtype=np.float64)
                    v2._save_npz(target, gradient=np.asarray(base["gradient"], dtype=np.float32), effects=np.asarray(effects, dtype=np.float32), direction_ids=direction_ids, direction_families=direction_families, fd_direction_indices=rows[:, 0].astype(np.int32), epsilons=rows[:, 1].astype(np.float32), finite_difference=rows[:, 2].astype(np.float32), realized_denominator=rows[:, 3].astype(np.float32), realized_norm_ratio=rows[:, 4].astype(np.float32), realized_cosine=rows[:, 5].astype(np.float32), high_plus_margin=rows[:, 6].astype(np.float32), high_minus_margin=rows[:, 7].astype(np.float32), native_plus_margin=rows[:, 8].astype(np.float32), native_minus_margin=rows[:, 9].astype(np.float32), realized_plus_norm_ratio=rows[:, 10].astype(np.float32), realized_minus_norm_ratio=rows[:, 11].astype(np.float32), realized_plus_cosine=rows[:, 12].astype(np.float32), realized_minus_cosine=rows[:, 13].astype(np.float32), realized_valid=rows[:, 14].astype(np.bool_), realized_plus_delta=np.stack(realized_plus_rows), realized_minus_delta=np.stack(realized_minus_rows), base_native_margin=np.asarray(base["native_margin"], dtype=np.float32), base_high_precision_margin=np.asarray(base["high_precision_margin"], dtype=np.float32), base_repeat_high_precision_margin=np.asarray(repeat["high_precision_margin"], dtype=np.float32), base_native_verdict=np.asarray(base["native_verdict"], dtype=np.int8), base_high_precision_verdict=np.asarray(base["high_precision_verdict"], dtype=np.int8), item_id=np.asarray(record["item_id"]), group_id=np.asarray(record["group_id"]), dataset_id=np.asarray(record["dataset_id"]), family=np.asarray(family), template=np.asarray(template_id), mapping=np.asarray(mapping.mapping_id))
                    write_json(target.with_suffix(".json"), receipt(f"V3_{stage.upper()}_SHARD", target, actor=actor))
                    checkpoint.check()
                    try:
                        import torch
                        peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated()))
                    except Exception:
                        pass
            checkpoint.completed_group()
    finally:
        try:
            checkpoint.close()
        except BaseException as error:
            checkpoint_failure = error
        unload_failure: BaseException | None = None
        if model is not None:
            try:
                model.to("cpu")
            except BaseException as error:
                unload_failure = error
        model = None
        tokenizer = None
        final_vram = v2.release_accelerator_cache()
    if checkpoint_failure is not None:
        raise RuntimeError("V3_CHECKPOINT_PUBLISHER_CLOSE_FAILED") from checkpoint_failure
    if unload_failure is not None:
        raise RuntimeError(f"V3_ACTOR_CPU_UNLOAD_FAILED:{actor}") from unload_failure
    if final_vram != 0:
        raise RuntimeError(f"V3_ACTOR_UNLOAD_INCOMPLETE:{actor}:{final_vram}")
    write_json(root / f"UNLOAD_AFTER_{stage.upper()}.json", {"actor": actor, "stage": stage, "complete": True, "final_compute_process_vram_bytes": 0})
    return root, time.monotonic() - started, peak_vram


def _finalize_calibration(root: Path, actor: str, setup_root: Path) -> dict[str, Any]:
    v2 = _load_v2_runtime()
    loaded = []
    for path in sorted((root / "calibration").glob("*.npz")):
        if not v2._valid(path):
            raise RuntimeError(f"V3_CALIBRATION_SHARD_INVALID:{path.name}")
        with np.load(path, allow_pickle=False) as data:
            loaded.append({key: data[key] for key in data.files})
    if not loaded:
        raise RuntimeError("V3_CALIBRATION_SHARDS_MISSING")
    contract = json.loads((setup_root / "V3_PERTURBATION_GRID_CONTRACT.json").read_text(encoding="utf-8"))
    points = contract["actors"][actor]["points"]
    grid = np.asarray(contract["actors"][actor]["absolute_l2_grid"], dtype=float)
    base = np.asarray([abs(float(row["base_high_precision_margin"])) for row in loaded])
    repeat = np.asarray([abs(float(row["base_high_precision_margin"]) - float(row["base_repeat_high_precision_margin"])) for row in loaded])
    resolution = max(float(np.quantile(repeat, 0.95)), float(np.finfo(np.float32).eps * max(np.median(base), 1.0) * 8.0), 1e-7)
    verdict = float(np.mean([int(row["base_native_verdict"]) != int(row["base_high_precision_verdict"]) for row in loaded]))
    by_key: dict[tuple[str, str], dict[str, dict[str, np.ndarray]]] = {}
    for row in loaded:
        by_key.setdefault((str(row["item_id"]), str(row["template"])), {})[str(row["mapping"])] = row
    candidates = []
    for index, point in enumerate(points):
        epsilon = grid[index]
        slopes_by_index, changes, ratios, cosines, saturation, valid = [], [], [], [], [], []
        for row in loaded:
            arrays = []
            for grid_value in grid:
                mask = np.isclose(row["epsilons"].astype(float), grid_value)
                arrays.append(row["finite_difference"][mask].astype(float))
            slopes_by_index.append(arrays)
            mask = np.isclose(row["epsilons"].astype(float), epsilon)
            changes.extend(np.abs(row["high_plus_margin"][mask].astype(float) - row["high_minus_margin"][mask].astype(float)))
            ratios.extend(row["realized_norm_ratio"][mask].astype(float))
            cosines.extend(row["realized_cosine"][mask].astype(float))
            saturation.extend(np.abs(row["high_plus_margin"][mask].astype(float)) >= 20.0)
            saturation.extend(np.abs(row["high_minus_margin"][mask].astype(float)) >= 20.0)
            valid.extend(row["realized_valid"][mask].astype(bool))
        linearity = {"lower_adjacent_relative_difference": None, "upper_adjacent_relative_difference": None}
        if 0 < index < len(grid) - 1:
            linearity = two_sided_linearity(
                np.concatenate([values[index - 1] for values in slopes_by_index]),
                np.concatenate([values[index] for values in slopes_by_index]),
                np.concatenate([values[index + 1] for values in slopes_by_index]),
            )
        agreements = []
        for mappings in by_key.values():
            required = {mapping.mapping_id for mapping in MAPPINGS}
            if not required.issubset(mappings):
                continue
            vectors = {}
            for symbol, standard, reversed_id in (("AB", "MAPPING_AB_STANDARD", "MAPPING_AB_REVERSED"), ("12", "MAPPING_12_STANDARD", "MAPPING_12_REVERSED")):
                left, right = mappings[standard], mappings[reversed_id]
                left_mask, right_mask = np.isclose(left["epsilons"].astype(float), epsilon), np.isclose(right["epsilons"].astype(float), epsilon)
                indices = left["fd_direction_indices"][left_mask].astype(int)
                vectors[symbol] = (left["finite_difference"][left_mask].astype(float)[indices < 28] + right["finite_difference"][right_mask].astype(float)[indices < 28]) / 2.0
            if np.std(vectors["AB"]) > 0 and np.std(vectors["12"]) > 0:
                agreements.append(float(np.corrcoef(vectors["AB"], vectors["12"])[0, 1]))
        candidates.append({**point, "median_cosine": float(np.median(cosines)), "median_norm_ratio": float(np.median(ratios)), "resolvable_fraction": float(np.mean(np.asarray(changes) > 4.0 * resolution)), "saturation_fraction": float(np.mean(saturation)), "invalid_realized_measurement_fraction": float(1.0 - np.mean(valid)), "median_mapping_paired_AB_12_agreement": float(np.median(agreements)) if agreements else 0.0, "base_native_high_precision_verdict_disagreement_fraction": verdict, **linearity})
    selection = choose_smallest_passing(candidates)
    selection.update({"actor": actor, "high_precision_resolution_floor": resolution})
    write_json(root / "CALIBRATION_SELECTION.json", selection)
    return selection


def _finalize_validation_with_v2_adapter(root: Path, actor: str, study_root: Path, work_root: Path) -> dict[str, Any]:
    v2 = _load_v2_runtime()
    shadow = work_root / "v3_validation_adapter"
    target = shadow / "checkpoints" / actor / "causal_fingerprint_v2"
    target.mkdir(parents=True, exist_ok=True)
    validation = target / "validation"
    validation.mkdir(exist_ok=True)
    for source in sorted((root / "validation").glob("*")):
        if source.is_file():
            shutil.copy2(source, validation / source.name)
    write_json(target / "manifest.json", {"actor": actor, "files": [], "scientific_summary_sealed": True})
    write_json(target / "UNLOAD.json", {"complete": True, "final_compute_process_vram_bytes": 0})
    write_json(target / "PILOT.json", {"complete": True, "artifact_sha256": sha256_file(target / "manifest.json"), "unload_receipt_sha256": sha256_file(target / "UNLOAD.json")})
    result = v2.finalize_actor_validation(shadow, actor)
    result["actor"] = actor
    result["v2_validation_implementation_adapter"] = True
    result["invalid_realized_measurement_fraction"] = 0.0
    write_json(root / "VALIDATION_RESULT.json", result)
    return result


def run_actor_mode(mode: str, setup_root: Path, study_root: Path, results_root: Path, work_root: Path) -> None:
    actor = "llama" if "LLAMA" in mode else "gemma"
    stage = "calibration" if "CALIBRATION" in mode else "validation"
    assert_stage_allowed(mode, _receipts(results_root))
    root, runtime_seconds, peak_vram = _run_measurements(actor, f"V3_{'NUMERICAL_CALIBRATION' if stage == 'calibration' else 'MEASUREMENT_VALIDATION'}", stage, setup_root, study_root, results_root, work_root)
    if stage == "calibration":
        try:
            result = _finalize_calibration(root, actor, setup_root)
            status = result["status"]
        except RuntimeError as error:
            if str(error) != "V3_PERTURBATION_CALIBRATION_FAILED":
                raise
            _write_stage_receipt(results_root, mode, status=str(error), actor=actor, scientific_failure=True)
            print(str(error))
            raise SystemExit(40) from error
    else:
        result = _finalize_validation_with_v2_adapter(root, actor, study_root, work_root)
        status = "V3_ACTOR_VALIDATION_PASS" if actor_pass(result) else "V3_ACTOR_VALIDATION_FAILED"
    _write_stage_receipt(results_root, mode, status=status, actor=actor, runtime_seconds=runtime_seconds, peak_vram_bytes=peak_vram, final_compute_process_vram_bytes=0, result_sha256=sha256_file(root / ("CALIBRATION_SELECTION.json" if stage == "calibration" else "VALIDATION_RESULT.json")))
    unload_stage = f"UNLOAD_{actor.upper()}_AFTER_{stage.upper()}"
    _write_stage_receipt(results_root, unload_stage, status="V3_ACTOR_FULLY_UNLOADED", actor=actor, final_compute_process_vram_bytes=0)
    print(f"{mode}_COMPLETE_HASH_VERIFIED_AND_ACTOR_UNLOADED")


def finalize(results_root: Path) -> str:
    receipts = _receipts(results_root)
    assert_stage_allowed("FINALIZE_V3", receipts)
    actors = {}
    for actor in ("llama", "gemma"):
        path = results_root / "checkpoints" / actor / "causal_fingerprint_v3" / "VALIDATION_RESULT.json"
        actors[actor] = json.loads(path.read_text(encoding="utf-8"))
    disposition = decide_validation(actors)
    write_json(results_root / "CAUSAL_FINGERPRINT_MEASUREMENT_V3_REPORT.json", {"study_id": "CAUSAL_FINGERPRINT_MEASUREMENT_V3", "terminal_disposition": disposition, "actors": actors, "complete": True})
    _write_stage_receipt(results_root, "FINALIZE_V3", status=disposition)
    print(disposition)
    return disposition


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--setup-root", type=Path, required=True)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--allow-model-inference", action="store_true")
    args = parser.parse_args()
    setup_root = _setup(args.setup_root)
    if args.mode == "AUDIT_AND_HASH_VERIFY":
        audit(setup_root, args.results_root)
    elif args.mode.startswith("RUN_"):
        if not args.allow_model_inference:
            raise RuntimeError("V3_MODEL_INFERENCE_NOT_AUTHORIZED")
        run_actor_mode(args.mode, setup_root, args.study_root, args.results_root, args.work_root)
    elif args.mode == "FINALIZE_V3":
        disposition = finalize(args.results_root)
        if disposition == "CAUSAL_FINGERPRINT_MEASUREMENT_V3_INVALID":
            return 40
    else:
        report = args.results_root / "CAUSAL_FINGERPRINT_MEASUREMENT_V3_REPORT.json"
        if not report.is_file():
            raise RuntimeError("V3_FINAL_REPORT_MISSING")
        _write_stage_receipt(args.results_root, "VERIFY_V3_ARTIFACTS", status="V3_RESULTS_HASH_VERIFIED", report_sha256=sha256_file(report))
        print("V3_RESULTS_HASH_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
