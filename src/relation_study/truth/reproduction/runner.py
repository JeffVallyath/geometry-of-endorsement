from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import analyze
from .cache import ChunkStore, concatenate_chunks
from .config import RunConfig, load_config
from .data import assign_grouped_splits, fetch_dataset, load_examples, write_split_manifest
from .model import extract_batch, load_model
from .prompts import prompt_contract_hash
from .provenance import collect_environment, write_json_atomic


FINAL_ARTIFACTS = (
    "TRUTH_CONTROL_V2_REPORT.md",
    "truth_control_v2_results.json",
    "layerwise_signed_separation.png",
    "environment.json",
    "run.log",
    "split_manifest.csv",
    "cache_manifest.json",
    "TRUTH_CONTROL_V2_SMOKE_REPORT.json",
)
SMOKE_ATOL = 2e-3


def _logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("truth_control_v2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)sZ %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _signature(
    config: RunConfig,
    model_revision: str,
    artifacts: list[Any],
    record_ids: list[str],
) -> str:
    payload = {
        "schema": "truth_control_v2_cache_v1",
        "config": config.digest,
        "model_revision": model_revision,
        "datasets": {artifact.name: artifact.sha256 for artifact in artifacts},
        "record_ids_sha256": hashlib.sha256("\n".join(record_ids).encode()).hexdigest(),
        "prompt_contract_sha256": prompt_contract_hash(config.prompt),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _chunk_ranges(examples: Any, chunk_size: int) -> list[np.ndarray]:
    ranges: list[np.ndarray] = []
    schemes = examples["scheme"].to_numpy()
    for scheme in ("primary", "transfer"):
        rows = np.flatnonzero(schemes == scheme)
        ranges.extend(
            rows[start : start + chunk_size]
            for start in range(0, len(rows), chunk_size)
        )
    return ranges


def _smoke_validation_rows(
    chunk_ranges: list[np.ndarray],
    batch_size: int,
) -> np.ndarray:
    if not chunk_ranges:
        raise RuntimeError("Smoke validation requires at least one cache chunk.")
    first = np.asarray(chunk_ranges[0], dtype=np.int64)
    if len(first) < batch_size:
        raise RuntimeError("The first smoke chunk is smaller than cache.batch_size.")
    return first[:batch_size]


def _allclose_with_max(
    left: Any,
    right: Any,
    atol: float = SMOKE_ATOL,
) -> tuple[bool, float]:
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    if left_array.shape != right_array.shape:
        return False, float("inf")
    maximum = (
        float(np.max(np.abs(left_array - right_array)))
        if left_array.size
        else 0.0
    )
    return (
        bool(np.allclose(left_array, right_array, rtol=0.0, atol=atol)),
        maximum,
    )


def _persist_final(
    output_dir: Path,
    persistent_dir: Path | None,
) -> dict[str, str]:
    if persistent_dir is None:
        return {}
    persistent_dir.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for name in FINAL_ARTIFACTS:
        source = output_dir / name
        if not source.exists():
            continue
        target = persistent_dir / name
        partial = target.with_suffix(target.suffix + ".part")
        shutil.copy2(source, partial)
        digest = hashlib.sha256(partial.read_bytes()).hexdigest()
        partial.replace(target)
        digests[name] = digest
    write_json_atomic(
        {"protocol": "TRUTH_CONTROL_V2_NEUTRAL_MAPPING", "files": digests},
        persistent_dir / "final_artifact_manifest.json",
    )
    return digests


def dry_run_plan(
    config: RunConfig,
    output_dir: str | Path,
    persistent_dir: str | Path | None,
) -> dict[str, Any]:
    return {
        "status": "DRY_RUN_ONLY",
        "protocol": "TRUTH_CONTROL_V2_NEUTRAL_MAPPING",
        "config": str(config.path),
        "config_sha256": config.digest,
        "mode": config.mode,
        "model": config.model,
        "datasets": [spec.__dict__ for spec in config.datasets],
        "output_dir": str(Path(output_dir).resolve()),
        "persistent_dir": str(Path(persistent_dir).resolve()) if persistent_dir else None,
    }


def _record_dicts(frame: Any) -> list[dict[str, str]]:
    return [
        {
            "item_id": str(row.item_id),
            "statement": str(row.statement),
            "scheme": str(row.scheme),
        }
        for row in frame.itertuples(index=False)
    ]


def _mapping_counts(examples: Any) -> dict[str, int]:
    counts = examples.groupby(["split", "scheme", "mapping"]).size()
    return {
        ":".join(map(str, key)): int(value)
        for key, value in counts.items()
    }


def run(
    config_path: str | Path,
    output_dir: str | Path,
    persistent_dir: str | Path | None = None,
    dry_run: bool = False,
    require_smoke_mode: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    if require_smoke_mode and config.mode != "smoke":
        raise RuntimeError("--smoke-test requires run.mode=smoke.")
    if dry_run:
        return dry_run_plan(config, output_dir, persistent_dir)

    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    persistent = Path(persistent_dir).resolve() if persistent_dir else None
    logger = _logger(target)
    logger.info(
        "Starting %s (%s); config=%s",
        config.raw["run"]["name"],
        config.mode,
        config.digest,
    )

    artifacts = [fetch_dataset(spec, target / "data") for spec in config.datasets]
    examples = load_examples(
        artifacts,
        config.data["statement_column"],
        config.data["label_column"],
        list(config.data["group_columns"]),
        config.prompt,
        config.data.get("max_examples_per_dataset"),
    )
    examples = assign_grouped_splits(
        examples,
        int(config.raw["run"]["seed"]),
        float(config.analysis["train_fraction"]),
        float(config.analysis["dev_fraction"]),
        float(config.analysis["test_fraction"]),
    )
    write_split_manifest(examples, target / "split_manifest.csv")
    logger.info(
        "Prepared %d prompt records across %d grouped units",
        len(examples),
        examples["group_id"].nunique(),
    )

    loaded = load_model(config)
    logger.info(
        "Loaded %s at revision %s on %s (%d MiB)",
        config.model["id"],
        loaded.model_revision,
        loaded.device_name,
        loaded.gpu_memory_mib,
    )
    signature = _signature(
        config,
        loaded.model_revision,
        artifacts,
        examples["record_id"].tolist(),
    )
    store = ChunkStore(target, signature, persistent)
    chunks: list[dict[str, np.ndarray]] = []
    chunk_ranges = _chunk_ranges(examples, int(config.cache["chunk_size"]))
    batch_size = int(config.cache["batch_size"])

    for chunk_index, rows in enumerate(chunk_ranges):
        expected_ids = examples.iloc[rows]["record_id"].tolist()
        cached = store.load(chunk_index, expected_ids)
        if cached is not None:
            chunks.append(cached)
            logger.info("Restored chunk %d/%d", chunk_index + 1, len(chunk_ranges))
            continue
        chunk_frame = examples.iloc[rows]
        if chunk_frame["scheme"].nunique() != 1:
            raise RuntimeError("A cache chunk crossed verbalizer schemes.")
        batch_results = []
        prompt_hashes: list[str] = []
        for start in range(0, len(chunk_frame), batch_size):
            batch = chunk_frame.iloc[start : start + batch_size]
            result = extract_batch(
                loaded,
                _record_dicts(batch),
                config.prompt,
                config.cache["activation_dtype"],
            )
            batch_results.append(result)
            prompt_hashes.extend(
                hashlib.sha256(text.encode()).hexdigest()
                for text in result.rendered_prompts
            )
        store.save(
            chunk_index,
            expected_ids,
            activations=np.concatenate([item.activations for item in batch_results]),
            true_logp=np.concatenate([item.true_logp for item in batch_results]),
            false_logp=np.concatenate([item.false_logp for item in batch_results]),
            truth_scores=np.concatenate([item.truth_scores for item in batch_results]),
            predicted_labels=np.concatenate([item.predicted_labels for item in batch_results]),
            prompt_token_indices=np.concatenate(
                [item.prompt_token_indices for item in batch_results]
            ),
            prompt_sha256=np.asarray(prompt_hashes),
            mapping_names=np.asarray(
                [name for item in batch_results for name in item.mapping_names]
            ),
            candidate_token_lengths=np.concatenate(
                [item.candidate_token_lengths for item in batch_results]
            ),
        )
        loaded_chunk = store.load(chunk_index, expected_ids)
        if loaded_chunk is None:
            raise RuntimeError("Saved cache chunk could not be reloaded.")
        chunks.append(loaded_chunk)
        logger.info("Wrote and verified chunk %d/%d", chunk_index + 1, len(chunk_ranges))

    combined = concatenate_chunks(chunks)
    if [str(value) for value in combined["record_ids"]] != examples["record_id"].tolist():
        raise RuntimeError("Concatenated cache order differs from split_manifest.csv.")

    if config.mode == "smoke":
        validation_rows = _smoke_validation_rows(chunk_ranges, batch_size)
        validation_frame = examples.iloc[validation_rows]
        if validation_frame["scheme"].nunique() != 1:
            raise RuntimeError("Smoke validation crossed verbalizer schemes.")
        records = _record_dicts(validation_frame)
        repeat_a = extract_batch(
            loaded, records, config.prompt, config.cache["activation_dtype"]
        )
        repeat_b = extract_batch(
            loaded, records, config.prompt, config.cache["activation_dtype"]
        )
        deterministic, deterministic_max = _allclose_with_max(
            repeat_a.activations, repeat_b.activations
        )
        cache_matches, cache_max = _allclose_with_max(
            combined["activations"][validation_rows], repeat_a.activations
        )
        score_matches, score_max = _allclose_with_max(
            combined["truth_scores"][validation_rows], repeat_a.truth_scores
        )
        repeat_hashes = np.asarray(
            [hashlib.sha256(text.encode()).hexdigest() for text in repeat_a.rendered_prompts]
        )
        checks = {
            "deterministic_repeat_within_atol_0.002": deterministic,
            "cache_matches_repeat_within_atol_0.002": cache_matches,
            "cache_truth_scores_match_within_atol_0.002": score_matches,
            "cache_prompt_hashes_match": bool(
                np.array_equal(combined["prompt_sha256"][validation_rows], repeat_hashes)
            ),
            "cache_prompt_token_indices_match": bool(
                np.array_equal(
                    combined["prompt_token_indices"][validation_rows],
                    repeat_a.prompt_token_indices,
                )
            ),
            "cache_predictions_match": bool(
                np.array_equal(
                    combined["predicted_labels"][validation_rows],
                    repeat_a.predicted_labels,
                )
            ),
            "cache_mapping_names_match": bool(
                np.array_equal(
                    combined["mapping_names"][validation_rows],
                    np.asarray(repeat_a.mapping_names),
                )
            ),
            "cache_candidate_lengths_match": bool(
                np.array_equal(
                    combined["candidate_token_lengths"][validation_rows],
                    repeat_a.candidate_token_lengths,
                )
            ),
            "hidden_state_layer_count_matches": bool(
                repeat_a.activations.shape[1]
                == int(loaded.model.config.num_hidden_layers)
            ),
            "candidate_answer_scoring_finite": bool(
                np.isfinite(repeat_a.true_logp).all()
                and np.isfinite(repeat_a.false_logp).all()
                and np.isfinite(repeat_a.truth_scores).all()
            ),
        }
        smoke = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "protocol": "TRUTH_CONTROL_V2_NEUTRAL_MAPPING",
            "model_loaded": True,
            "model_id": config.model["id"],
            "model_revision": loaded.model_revision,
            "rendered_prompt": repeat_a.rendered_prompts[0],
            "rendered_prompt_sha256": hashlib.sha256(
                repeat_a.rendered_prompts[0].encode()
            ).hexdigest(),
            "final_prompt_token_index": int(repeat_a.prompt_token_indices[0]),
            "hidden_state_layers": int(repeat_a.activations.shape[1]),
            "expected_hidden_state_layers": int(loaded.model.config.num_hidden_layers),
            "validation_batch_size": int(len(validation_rows)),
            "deterministic_repeat_max_abs": deterministic_max,
            "cache_repeat_max_abs": cache_max,
            "cache_truth_scores_max_abs": score_max,
            "cache_signature": signature,
            **checks,
        }
        write_json_atomic(smoke, target / "TRUTH_CONTROL_V2_SMOKE_REPORT.json")
        if smoke["status"] != "PASS":
            raise RuntimeError("V2 smoke determinism/cache contract failed.")
        logger.info("V2 smoke contracts passed")

    results = analyze(
        combined["activations"],
        combined["truth_scores"],
        combined["predicted_labels"],
        examples,
        config.analysis,
        int(config.raw["run"]["seed"]),
        config.mode,
        checkpoint_dir=target / "analysis_checkpoints",
        persistent_checkpoint_dir=persistent / "analysis_checkpoints" if persistent else None,
        checkpoint_signature=signature,
    )
    results.update(
        {
            "cache_signature": signature,
            "model_revision": loaded.model_revision,
            "dataset_sha256": {
                artifact.name: artifact.sha256 for artifact in artifacts
            },
            "config_sha256": config.digest,
            "prompt_contract_sha256": prompt_contract_hash(config.prompt),
            "mapping_counts": _mapping_counts(examples),
            "split_manifest_sha256": hashlib.sha256(
                (target / "split_manifest.csv").read_bytes()
            ).hexdigest(),
        }
    )
    from .reporting import write_results

    write_results(results, target)
    repo_root = Path(__file__).resolve().parents[4]
    environment = collect_environment(
        repo_root,
        config,
        loaded.model_revision,
        artifacts,
        {"name": loaded.device_name, "memory_total_mib": loaded.gpu_memory_mib},
    )
    write_json_atomic(environment, target / "environment.json")
    logger.info("Terminal disposition: %s", results["terminal_disposition"])
    for handler in logger.handlers:
        handler.flush()
    results["persisted_final_sha256"] = _persist_final(target, persistent)
    return results
