from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geometry_of_truth.common.artifacts import repository_root, sha256_file

from .contracts import (
    CACHE_MANIFEST_SHA256,
    CACHE_SIGNATURE,
    CONFIG_SHA256,
    EXPECTED,
    MODEL_ID,
    MODEL_REVISION,
    RANDOM_SEED,
    load_bundle,
)


def verify_cache(cache_root: str | Path, split: pd.DataFrame) -> list[Path]:
    root = Path(cache_root).expanduser().resolve()
    manifest_path = root / "cache_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Analysis reproduction requires the retained activation cache")
    if sha256_file(manifest_path) != CACHE_MANIFEST_SHA256:
        raise RuntimeError("Activation cache manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("signature") != CACHE_SIGNATURE:
        raise RuntimeError("Activation cache signature mismatch")
    paths = []
    records = 0
    for name, entry in sorted(manifest.get("chunks", {}).items()):
        path = root / "activation_chunks" / name
        if not path.is_file() or sha256_file(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"Activation cache chunk mismatch  {name}")
        paths.append(path)
        records += int(entry["records"])
    if records != len(split):
        raise RuntimeError(f"Activation cache covers {records} rows instead of {len(split)}")
    return paths


def _load_cache(paths: list[Path], split: pd.DataFrame) -> dict[str, np.ndarray]:
    from .reproduction.cache import concatenate_chunks

    chunks = []
    cursor = 0
    required = {
        "signature", "record_ids", "activations", "true_logp", "false_logp",
        "truth_scores", "predicted_labels", "prompt_token_indices",
        "prompt_sha256", "mapping_names", "candidate_token_lengths",
    }
    for path in paths:
        with np.load(path, allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        if set(data) != required or str(data["signature"].item()) != CACHE_SIGNATURE:
            raise RuntimeError(f"Activation cache schema mismatch  {path.name}")
        ids = [str(value) for value in data["record_ids"]]
        expected = split.iloc[cursor:cursor + len(ids)]["record_id"].astype(str).tolist()
        if ids != expected or data["activations"].shape != (len(ids), 32, 4096):
            raise RuntimeError(f"Activation cache linkage mismatch  {path.name}")
        cursor += len(ids)
        chunks.append(data)
    combined = concatenate_chunks(chunks)
    if combined["activations"].shape != (len(split), 32, 4096):
        raise RuntimeError("Combined activation cache shape mismatch")
    return combined


def _core(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "protocol", "terminal_disposition", "mode",
        "selected_layer", "selection_rule", "training_partitions",
        "development_layer_sweep", "confirmatory_layer_candidates",
        "confirmatory_layer_results", "primary_test_T",
        "directional_consensus_C", "permutation_null", "native_answer_controls",
        "native_gate_values", "checks", "auroc_role", "v1_disposition_changed",
    )
    return {key: result[key] for key in keys}


def compare_analysis(actual: dict[str, Any], frozen: dict[str, Any]) -> pd.DataFrame:
    if _core(actual) != _core(frozen):
        raise RuntimeError("Recomputed analysis differs from the frozen result")
    selected = str(actual["selected_layer"])
    rows = [
        ("selected layer", actual["selected_layer"], EXPECTED["selected_layer"]),
        ("primary T", actual["primary_test_T"], EXPECTED["primary_T"]),
        ("consensus C", actual["directional_consensus_C"], EXPECTED["consensus_C"]),
        ("permutation p", actual["permutation_null"]["p_greater_equal"], EXPECTED["permutation_p"]),
        ("held out 1/2 T", actual["confirmatory_layer_results"][selected]["test"]["transfer"]["overall"]["T"], EXPECTED["transfer_T"]),
    ]
    return pd.DataFrame([{"quantity": name, "reproduced": value, "frozen": expected, "pass": value == expected} for name, value, expected in rows])


def reproduce_analysis(cache_root: str | Path, output_root: str | Path, seed: int = RANDOM_SEED) -> dict[str, Any]:
    if seed != RANDOM_SEED:
        raise RuntimeError("Analysis reproduction requires seed 314159")
    bundle = load_bundle()
    chunks = verify_cache(cache_root, bundle["split"])
    combined = _load_cache(chunks, bundle["split"])
    root = repository_root()
    from .reproduction.analysis import analyze
    from .reproduction.config import load_config

    config = load_config(root / "configs" / "truth_control_v2.yaml")
    if config.digest != CONFIG_SHA256:
        raise RuntimeError("Frozen Truth configuration hash mismatch")
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = analyze(
        combined["activations"], combined["truth_scores"], combined["predicted_labels"],
        bundle["split"], config.analysis, seed, "full",
        checkpoint_dir=output / "analysis_checkpoints",
        checkpoint_signature=CACHE_SIGNATURE,
    )
    comparison = compare_analysis(result, bundle["v2"])
    artifact = output / "analysis_reproduction.json"
    artifact.write_text(json.dumps(_core(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "result": result,
        "comparison": comparison,
        "artifact": artifact,
        "artifact_sha256": sha256_file(artifact),
        "elapsed_seconds": time.perf_counter() - started,
    }


def gpu_preflight(minimum_mib: int = 23000) -> pd.DataFrame:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("Full reproduction requires PyTorch with CUDA") from exc
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Full reproduction requires a CUDA GPU with native BF16")
    props = torch.cuda.get_device_properties(0)
    total = int(props.total_memory // (1024 * 1024))
    free = int(torch.cuda.mem_get_info()[0] // (1024 * 1024))
    if total < minimum_mib or free < minimum_mib:
        raise RuntimeError(f"Full reproduction requires {minimum_mib} MiB total and free VRAM")
    return pd.DataFrame([("GPU", props.name), ("total VRAM MiB", total), ("free VRAM MiB", free), ("native BF16", True)], columns=["field", "value"])


def reproduce_full(output_root: str | Path, persistent_root: str | Path | None = None) -> dict[str, Any]:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("Full reproduction requires HF_TOKEN")
    gpu_preflight()
    from huggingface_hub import HfApi
    resolved = HfApi(token=os.environ["HF_TOKEN"]).model_info(MODEL_ID, revision="main").sha
    if resolved != MODEL_REVISION:
        raise RuntimeError(f"Model main resolves to {resolved} instead of the frozen revision")
    from .reproduction.runner import run

    root = repository_root()
    output = Path(output_root).expanduser().resolve()
    persistent = Path(persistent_root).expanduser().resolve() if persistent_root else None
    result = run(root / "configs" / "truth_control_v2.yaml", output, persistent_dir=persistent)
    bundle = load_bundle(root)
    selected = str(result["selected_layer"])
    checks = {
        "selected layer": result["selected_layer"] == EXPECTED["selected_layer"],
        "primary T tolerance": abs(result["primary_test_T"] - EXPECTED["primary_T"]) <= 0.02,
        "consensus C tolerance": abs(result["directional_consensus_C"] - EXPECTED["consensus_C"]) <= 0.002,
        "transfer T tolerance": abs(result["confirmatory_layer_results"][selected]["test"]["transfer"]["overall"]["T"] - EXPECTED["transfer_T"]) <= 0.02,
        "permutation exceedances": result["permutation_null"]["count_greater_equal_observed"] == 0,
        "model revision": result["model_revision"] == MODEL_REVISION,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Full Truth reproduction tolerance failure  {checks}")
    return {"result": result, "checks": pd.DataFrame([{"check": key, "pass": value} for key, value in checks.items()])}
