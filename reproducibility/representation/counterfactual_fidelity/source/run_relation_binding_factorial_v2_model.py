#!/usr/bin/env python3
"""Run one exact-model v2 Track A forward pass plus the 112-item Track B addendum.

The extraction core adapts the proven direction-identity implementation in
``src/geometry_endorsement/direction_identity/activation_extraction.py``:
exact-revision actor loading, chat-token prompt IDs, candidate-sequence scoring,
and a selected-decoder-layer hook at the final prompt token. Validation and
dry-run modes import neither torch nor transformers and never download a model.

This is a foreground, restart-from-beginning entrypoint. It has no checkpoint
or scientific retry mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "RELATION_BINDING_FACTORIAL_V2"
TRACK_B_SCHEMA_VERSION = "TRACK_B_EXTRACTION_ADDENDUM_V1"
EXPECTED_TRACK_A_JOBS = 3744
EXPECTED_TRACK_B_JOBS = 112
ACTIVATION_POSITION = "final prompt position immediately before candidate answer sequence"
OPEN_RUNTIME_EVIDENCE_PATH = "artifacts/truth/environment.json"
OPEN_RUNTIME_EVIDENCE_SHA256 = "3f68cb60e218bbada9ead06f27f1389a974dab05f34c992e8c1c5cf0b93e448e"
EXPECTED_TORCH_VERSION = "2.11.0+cu128"
EXPECTED_TORCH_CUDA_VERSION = "12.8"

SYSTEM_INSTRUCTION = (
    "You judge how one named moral consideration bears on an action in context. "
    "Follow the requested answer mapping exactly."
)
USER_TEMPLATE = (
    "Situation and action: {situation}\n"
    "Named consideration: {consideration}\n"
    "Question: In this specific context, does the named consideration support or oppose the action?\n"
    "{mapping_instruction}"
)

ACTORS = {
    "llama": {
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "model_revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "selected_layer": 19,
        "hidden_size": 4096,
        "direction_bundle_sha256": "92f0cfa02d4d0d8cfe57ebb5e082ee4e45de4af41117ac25c73d0ea15631e6ac",
        "direction_sha256": "f44fb897ab2617abd95ca5a2f8e67dd2626dd75099a9cdf3973033b3e1b64c99",
        "chat_policy": "NATIVE_SYSTEM",
    },
    "gemma": {
        "model_id": "google/gemma-2-9b-it",
        "model_revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
        "selected_layer": 27,
        "hidden_size": 3584,
        "direction_bundle_sha256": "f7ce9e5d4082379e56f8ce96bab91b3e05c409c25d6e611c8ebf9c79e0e029b9",
        "direction_sha256": "81c704de24ffc90b468f13bbf494638804f7fafa4b24f2f51d6bb64d2dc1c4e7",
        "chat_policy": "FOLD_SYSTEM_INTO_USER",
    },
}


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_json_bytes(value))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number}: expected JSON object")
            records.append(value)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    data = "".join(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n" for record in records)
    atomic_write(path, data.encode("utf-8"))


def require_hash(path: Path, expected: str, label: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise RuntimeError(f"{label} expected hash is not canonical SHA-256")
    if not path.is_file():
        raise RuntimeError(f"{label} missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash mismatch: expected {expected}, got {actual}")


def mapping_for_item(item_id: str) -> dict[str, str]:
    reversed_mapping = bool(hashlib.sha256(f"answer-map:{item_id}".encode("utf-8")).digest()[0] & 1)
    if reversed_mapping:
        return {"id": "REVERSED", "supports": "B", "opposes": "A"}
    return {"id": "STANDARD", "supports": "A", "opposes": "B"}


def messages_for_track_b(
    situation: str,
    consideration: str,
    mapping: dict[str, str],
    actor: str,
) -> list[dict[str, str]]:
    instruction = (
        f"Use {mapping['supports']} when the named consideration supports the action in this situation, "
        f"and {mapping['opposes']} when it opposes the action. Reply with only A or B."
    )
    user = USER_TEMPLATE.format(
        situation=situation,
        consideration=consideration,
        mapping_instruction=instruction,
    )
    if actor == "llama":
        return [{"role": "system", "content": SYSTEM_INSTRUCTION}, {"role": "user", "content": user}]
    if actor == "gemma":
        return [{"role": "user", "content": f"{SYSTEM_INSTRUCTION}\n\n{user}"}]
    raise RuntimeError(f"unknown actor: {actor}")


def render_chat(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[list[int], str]:
    token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise RuntimeError("chat renderer returned unexpected batch")
        token_ids = token_ids[0]
    ids = [int(value) for value in token_ids]
    if not ids:
        raise RuntimeError("chat renderer returned empty prompt")
    rendered = tokenizer.decode(ids, skip_special_tokens=False)
    return ids, rendered


def render_track_b_prompt(tokenizer: Any, actor: str, record: dict[str, Any]) -> dict[str, Any]:
    mapping = mapping_for_item(record["item_id"])
    messages = messages_for_track_b(
        record["situation_action_text"],
        record["consideration_text"],
        mapping,
        actor,
    )
    ids, rendered = render_chat(tokenizer, messages)
    actor_hash = sha256_bytes(rendered.encode("utf-8"))
    source_hash = record["rendered_prompt_sha256"]
    if actor == "llama" and actor_hash != source_hash:
        raise RuntimeError(
            f"inherited Llama prompt hash mismatch for {record['base_item_id']}: "
            f"expected {source_hash}, got {actor_hash}"
        )
    return {
        "prompt_ids": ids,
        "rendered_prompt": rendered,
        "actor_rendered_prompt_sha256": actor_hash,
        "source_llama_rendered_prompt_sha256": source_hash,
        "mapping": mapping,
        "candidate_symbols": [mapping["supports"], mapping["opposes"]],
    }


def render_track_a_prompt(tokenizer: Any, record: dict[str, Any]) -> dict[str, Any]:
    ids, rendered = render_chat(tokenizer, [{"role": "user", "content": record["prompt_text"]}])
    mapping_id = record["answer_mapping_id"]
    candidates = ["A", "B"] if "AB" in mapping_id else ["1", "2"]
    return {
        "prompt_ids": ids,
        "rendered_prompt": rendered,
        "actor_rendered_prompt_sha256": sha256_bytes(rendered.encode("utf-8")),
        "candidate_symbols": candidates,
    }


def validate_inputs(
    actor: str,
    track_a_jobs: Sequence[dict[str, Any]],
    track_b_jobs: Sequence[dict[str, Any]],
    *,
    expected_track_a: int = EXPECTED_TRACK_A_JOBS,
    expected_track_b: int = EXPECTED_TRACK_B_JOBS,
) -> dict[str, Any]:
    config = ACTORS[actor]
    errors: list[str] = []
    if len(track_a_jobs) != expected_track_a:
        errors.append(f"Track A count expected {expected_track_a}, got {len(track_a_jobs)}")
    if len(track_b_jobs) != expected_track_b:
        errors.append(f"Track B count expected {expected_track_b}, got {len(track_b_jobs)}")
    for lane, records, id_field, schema in (
        ("track_a", track_a_jobs, "job_id", SCHEMA_VERSION),
        ("track_b", track_b_jobs, "base_item_id", TRACK_B_SCHEMA_VERSION),
    ):
        ids = [record.get(id_field) for record in records]
        if any(not isinstance(value, str) or not value for value in ids):
            errors.append(f"{lane} contains blank/non-string IDs")
        if len(set(ids)) != len(ids):
            errors.append(f"{lane} IDs are not unique")
        for record in records:
            if record.get("schema_version") != schema:
                errors.append(f"{lane} schema mismatch: {record.get(id_field)}")
            for field, expected in (
                ("actor" if lane == "track_b" else "model_key", actor),
                ("model_id", config["model_id"]),
                ("model_revision", config["model_revision"]),
                ("selected_layer" if lane == "track_b" else "model_layer", config["selected_layer"]),
                ("activation_position", ACTIVATION_POSITION),
            ):
                if record.get(field) != expected:
                    errors.append(f"{lane} {field} mismatch: {record.get(id_field)}")
            if record.get("dim_direction_sha256") != config["direction_sha256"]:
                errors.append(f"{lane} direction hash mismatch: {record.get(id_field)}")
            if lane == "track_a":
                if record.get("outcome_state") != "UNOBSERVED_PRE_MODEL":
                    errors.append(f"Track A outcome already observed: {record.get(id_field)}")
                if any(
                    record.get(field) is not None
                    for field in (
                        "native_answer",
                        "native_probability",
                        "dim_score",
                        "logistic_score",
                    )
                ):
                    errors.append(f"Track A outcome field populated: {record.get(id_field)}")
                if any(
                    not isinstance(record.get(field), str) or not record[field]
                    for field in ("full_activation_artifact", "receipt_artifact")
                ):
                    errors.append(f"Track A artifact destination missing: {record.get(id_field)}")
                if not isinstance(record.get("prompt_text"), str) or not record["prompt_text"]:
                    errors.append(f"Track A prompt missing: {record.get(id_field)}")
            else:
                if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("rendered_prompt_sha256", ""))):
                    errors.append(f"Track B source prompt hash invalid: {record.get(id_field)}")
                if record.get("technical_receipt", {}).get("state") != "UNOBSERVED_PRE_MODEL":
                    errors.append(f"Track B outcome already observed: {record.get(id_field)}")
                if record.get("prompt_token_index") is not None or record.get("frozen_dim_projection") is not None:
                    errors.append(f"Track B scalar outcome populated: {record.get(id_field)}")
                for artifact_field in ("full_selected_layer_activation", "relation_residual"):
                    artifact = record.get(artifact_field, {})
                    if any(artifact.get(field) is not None for field in ("dtype", "sha256", "shape")):
                        errors.append(
                            f"Track B {artifact_field} outcome populated: {record.get(id_field)}"
                        )
                if record.get("technical_receipt", {}).get("sha256") is not None:
                    errors.append(f"Track B receipt hash populated: {record.get(id_field)}")
    if errors:
        raise RuntimeError("input validation faileexternal-artifacts" + "\n".join(errors[:50]))
    return {
        "status": "PASS",
        "actor": actor,
        "track_a_jobs": len(track_a_jobs),
        "track_b_jobs": len(track_b_jobs),
        "total_jobs": len(track_a_jobs) + len(track_b_jobs),
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "selected_layer": config["selected_layer"],
        "dtype": "bfloat16",
    }


def load_direction_npz(
    path: Path,
    bundle_sha256: str,
    direction_key: str,
    direction_sha256: str,
    expected_size: int,
) -> Any:
    require_hash(path, bundle_sha256, "direction bundle")
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        if direction_key not in archive.files:
            raise RuntimeError(f"direction bundle lacks exact {direction_key!r} member")
        source = archive[direction_key]
        if source.dtype != np.dtype("float32"):
            raise RuntimeError(
                f"direction member {direction_key!r} dtype mismatch: expected float32, got {source.dtype}"
            )
        direction = np.asarray(source, dtype=np.float32, order="C")
    if direction.shape != (expected_size,):
        raise RuntimeError(f"direction shape mismatch: expected {(expected_size,)}, got {direction.shape}")
    actual_direction_hash = sha256_bytes(direction.tobytes(order="C"))
    if actual_direction_hash != direction_sha256:
        raise RuntimeError(
            f"direction member {direction_key!r} hash mismatch: "
            f"expected {direction_sha256}, got {actual_direction_hash}"
        )
    if not np.isfinite(direction).all() or float(np.dot(direction, direction)) <= 0.0:
        raise RuntimeError(f"direction member {direction_key!r} is non-finite or zero")
    return direction


def validate_torch_runtime(torch_module: Any) -> dict[str, str]:
    """Fail closed on the exact open-evidence PyTorch/CUDA build.

    This check runs immediately after importing torch and before tokenizer or
    model retrieval. Accessing ``torch.version.cuda`` is metadata-only and does
    not initialize a CUDA device.
    """

    actual_version = str(torch_module.__version__)
    actual_cuda = str(torch_module.version.cuda)
    if actual_version != EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            "PyTorch build differs from open runtime evidence: "
            f"expected {EXPECTED_TORCH_VERSION}, got {actual_version}"
        )
    if actual_cuda != EXPECTED_TORCH_CUDA_VERSION:
        raise RuntimeError(
            "PyTorch CUDA build differs from open runtime evidence: "
            f"expected {EXPECTED_TORCH_CUDA_VERSION}, got {actual_cuda}"
        )
    return {
        "torch": actual_version,
        "cuda_version": actual_cuda,
        "evidence_path": OPEN_RUNTIME_EVIDENCE_PATH,
        "evidence_sha256": OPEN_RUNTIME_EVIDENCE_SHA256,
    }


def load_actor(actor: str, cache_dir: Path, hf_token: str | None = None) -> tuple[Any, Any, Any]:
    import torch

    validate_torch_runtime(torch)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = ACTORS[actor]
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        cache_dir=str(cache_dir),
        token=hf_token,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        cache_dir=str(cache_dir),
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    validate_model_dtype(model, torch)
    return tokenizer, model, torch


def validate_model_dtype(model: Any, torch_module: Any) -> dict[str, Any]:
    """Require bf16 parameters; permit only explicit float32 RoPE metadata buffers."""

    checked_parameters = 0
    checked_buffers = 0
    allowlisted_buffers: list[str] = []
    mismatches: list[str] = []
    for kind, values in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, value in values:
            if not value.is_floating_point():
                continue
            if kind == "parameter":
                checked_parameters += 1
            else:
                checked_buffers += 1
            if (
                kind == "buffer"
                and value.dtype != torch_module.bfloat16
                and value.dtype == torch_module.float32
                and name.endswith((".inv_freq", ".original_inv_freq"))
            ):
                allowlisted_buffers.append(f"{name}={value.dtype}")
            elif value.dtype != torch_module.bfloat16:
                mismatches.append(f"{kind}:{name}={value.dtype}")
    if not checked_parameters:
        raise RuntimeError("actor exposes no floating parameters for dtype verification")
    if mismatches:
        raise RuntimeError(
            "actor contains non-bfloat16 floating tensors: " + ", ".join(mismatches[:20])
        )
    return {
        "floating_parameters": checked_parameters,
        "floating_buffers": checked_buffers,
        "allowlisted_non_bfloat16_buffers": allowlisted_buffers,
    }


def candidate_ids(tokenizer: Any, symbol: str) -> list[int]:
    values = tokenizer.encode(symbol, add_special_tokens=False)
    if hasattr(values, "tolist"):
        values = values.tolist()
    result = [int(value) for value in values]
    if not result:
        raise RuntimeError(f"candidate tokenizes to empty sequence: {symbol!r}")
    return result


def layer_stack(model: Any) -> Any:
    candidates = (
        lambda: model.model.layers,
        lambda: model.model.decoder.layers,
        lambda: model.transformer.h,
    )
    for getter in candidates:
        try:
            layers = getter()
        except (AttributeError, TypeError):
            continue
        if layers is not None:
            return layers
    raise RuntimeError("unable to locate decoder layer stack")


def score_ids_and_extract(
    model: Any,
    torch: Any,
    prompt_token_ids: Sequence[int],
    candidate_token_ids: Sequence[Sequence[int]],
    selected_layer: int,
) -> tuple[list[float], Any, int]:
    import numpy as np

    if not prompt_token_ids:
        raise RuntimeError("prompt token IDs are empty")
    layers = layer_stack(model)
    if not 0 <= selected_layer < len(layers):
        raise RuntimeError(f"selected layer {selected_layer} outside decoder stack length {len(layers)}")
    device = next(model.parameters()).device
    activations: list[Any] = []
    scores: list[float] = []
    prompt_length = len(prompt_token_ids)

    for candidate in candidate_token_ids:
        captured: list[Any] = []

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.dtype != torch.bfloat16:
                raise RuntimeError(
                    f"selected-layer activation is not exact bfloat16: {hidden.dtype}"
                )
            captured.append(hidden[:, prompt_length - 1, :].detach().float().cpu().numpy()[0])

        handle = layers[selected_layer].register_forward_hook(hook)
        try:
            all_ids = list(prompt_token_ids) + list(candidate)
            input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)
            with torch.inference_mode():
                output = model(input_ids=input_ids, use_cache=False)
            logits = output.logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)
            score = 0.0
            for offset, token_id in enumerate(candidate):
                score += float(log_probs[prompt_length - 1 + offset, token_id].detach().cpu())
            scores.append(score)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(f"selected-layer hook fired {len(captured)} times")
        activations.append(np.asarray(captured[0], dtype=np.float32, order="C"))

    reference = activations[0]
    for activation in activations[1:]:
        if not np.array_equal(reference, activation):
            maximum = float(np.max(np.abs(reference - activation)))
            if maximum > 1e-5:
                raise RuntimeError(f"prompt activation changed across candidates; max abs delta {maximum}")
    return scores, reference, prompt_length - 1


def projection_and_residual(activation: Any, direction: Any) -> tuple[float, Any]:
    import numpy as np

    activation = np.asarray(activation, dtype=np.float32, order="C")
    direction = np.asarray(direction, dtype=np.float32, order="C")
    if activation.shape != direction.shape:
        raise RuntimeError(f"activation/direction shape mismatch: {activation.shape}/{direction.shape}")
    dot = float(np.dot(activation, direction))
    denominator = float(np.dot(direction, direction))
    residual = np.asarray(activation - (dot / denominator) * direction, dtype=np.float32, order="C")
    return dot, residual


def atomic_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def relative_artifact_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"artifact path escapes output root: {relative}") from exc
    return candidate


def normalized_probabilities(log_scores: Sequence[float]) -> list[float]:
    maximum = max(log_scores)
    weights = [math.exp(value - maximum) for value in log_scores]
    total = sum(weights)
    return [value / total for value in weights]


def array_record(array: Any) -> dict[str, Any]:
    import numpy as np

    value = np.asarray(array, dtype=np.float32, order="C")
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": sha256_bytes(value.tobytes(order="C"))}


def run_track_a_job(
    record: dict[str, Any],
    tokenizer: Any,
    model: Any,
    torch: Any,
    direction: Any,
    output_root: Path,
) -> dict[str, Any]:
    rendered = render_track_a_prompt(tokenizer, record)
    candidate_sequences = [candidate_ids(tokenizer, symbol) for symbol in rendered["candidate_symbols"]]
    scores, activation, prompt_index = score_ids_and_extract(
        model, torch, rendered["prompt_ids"], candidate_sequences, int(record["model_layer"])
    )
    probabilities = normalized_probabilities(scores)
    winner = max(range(len(scores)), key=lambda index: scores[index])
    dim_score, residual = projection_and_residual(activation, direction)
    artifact_path = relative_artifact_path(output_root, record["full_activation_artifact"])
    atomic_npz(artifact_path, selected_layer_activation=activation, relation_residual=residual)
    artifact_sha = sha256_file(artifact_path)
    receipt_path = relative_artifact_path(output_root, record["receipt_artifact"])
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "state": "FORWARD_COMPLETE_ANALYSIS_PENDING",
        "job_id": record["job_id"],
        "model_id": record["model_id"],
        "model_revision": record["model_revision"],
        "selected_layer": record["model_layer"],
        "dtype": "bfloat16",
        "activation_position": ACTIVATION_POSITION,
        "prompt_token_index": prompt_index,
        "actor_rendered_prompt_sha256": rendered["actor_rendered_prompt_sha256"],
        "candidate_symbols": rendered["candidate_symbols"],
        "candidate_token_ids": candidate_sequences,
        "candidate_log_probabilities": scores,
        "native_answer": rendered["candidate_symbols"][winner],
        "native_probability": probabilities[winner],
        "dim_score": dim_score,
        "selected_layer_activation": array_record(activation),
        "relation_residual": array_record(residual),
        "artifact": {"path": record["full_activation_artifact"], "sha256": artifact_sha, "bytes": artifact_path.stat().st_size},
    }
    write_json(receipt_path, receipt)
    return {
        **record,
        "native_answer": receipt["native_answer"],
        "native_probability": receipt["native_probability"],
        "dim_score": dim_score,
        "logistic_score": None,
        "prompt_token_index": prompt_index,
        "actor_rendered_prompt_sha256": rendered["actor_rendered_prompt_sha256"],
        "full_activation_artifact_sha256": artifact_sha,
        "receipt_artifact_sha256": sha256_file(receipt_path),
        "outcome_state": "FORWARD_COMPLETE_ANALYSIS_PENDING",
    }


def run_track_b_job(
    record: dict[str, Any],
    tokenizer: Any,
    model: Any,
    torch: Any,
    direction: Any,
    output_root: Path,
) -> dict[str, Any]:
    rendered = render_track_b_prompt(tokenizer, record["actor"], record)
    candidate_sequences = [candidate_ids(tokenizer, symbol) for symbol in rendered["candidate_symbols"]]
    _scores, activation, prompt_index = score_ids_and_extract(
        model, torch, rendered["prompt_ids"], candidate_sequences, int(record["selected_layer"])
    )
    dim_score, residual = projection_and_residual(activation, direction)
    relative_path = record["full_selected_layer_activation"]["path"]
    artifact_path = relative_artifact_path(output_root, relative_path)
    atomic_npz(artifact_path, selected_layer_activation=activation, relation_residual=residual)
    artifact_sha = sha256_file(artifact_path)
    receipt_path = relative_artifact_path(output_root, record["technical_receipt"]["path"])
    activation_record = {**array_record(activation), "member": "selected_layer_activation", "path": relative_path, "artifact_sha256": artifact_sha}
    residual_record = {**array_record(residual), "member": "relation_residual", "path": relative_path, "artifact_sha256": artifact_sha}
    receipt = {
        "schema_version": TRACK_B_SCHEMA_VERSION,
        "state": "COMPLETE",
        "base_item_id": record["base_item_id"],
        "actor": record["actor"],
        "model_id": record["model_id"],
        "model_revision": record["model_revision"],
        "selected_layer": record["selected_layer"],
        "dtype": "bfloat16",
        "activation_position": ACTIVATION_POSITION,
        "prompt_token_index": prompt_index,
        "source_llama_rendered_prompt_sha256": rendered["source_llama_rendered_prompt_sha256"],
        "actor_rendered_prompt_sha256": rendered["actor_rendered_prompt_sha256"],
        "mapping": rendered["mapping"],
        "selected_layer_activation": activation_record,
        "frozen_dim_projection": dim_score,
        "relation_residual": residual_record,
    }
    write_json(receipt_path, receipt)
    return {
        **record,
        "prompt_token_index": prompt_index,
        "source_llama_rendered_prompt_sha256": rendered["source_llama_rendered_prompt_sha256"],
        "actor_rendered_prompt_sha256": rendered["actor_rendered_prompt_sha256"],
        "full_selected_layer_activation": activation_record,
        "frozen_dim_projection": dim_score,
        "relation_residual": residual_record,
        "technical_receipt": {"path": record["technical_receipt"]["path"], "sha256": sha256_file(receipt_path), "state": "COMPLETE"},
    }


def artifact_inventory(output_root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "activation_artifacts.tar.gz":
            records.append({"path": path.relative_to(output_root).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return records


def make_archive(output_root: Path) -> Path:
    archive = output_root / "activation_artifacts.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(output_root.rglob("*")):
            if path.is_file() and path != archive:
                handle.add(path, arcname=path.relative_to(output_root).as_posix(), recursive=False)
    return archive


def run(args: argparse.Namespace) -> dict[str, Any]:
    actor = args.actor
    config = ACTORS[actor]
    if args.direction_key != "raw_relation_dim":
        raise RuntimeError("Track A execution requires direction_key=raw_relation_dim")
    if args.direction_sha256 != config["direction_sha256"]:
        raise RuntimeError(
            "selected direction hash differs from the frozen actor contract: "
            f"expected {config['direction_sha256']}, got {args.direction_sha256}"
        )
    require_hash(args.track_a_jobs, args.track_a_jobs_sha256, "Track A job ledger")
    require_hash(args.track_b_addendum, args.track_b_addendum_sha256, "Track B addendum")
    track_a_jobs = read_jsonl(args.track_a_jobs)
    track_b_jobs = read_jsonl(args.track_b_addendum)
    validation = validate_inputs(actor, track_a_jobs, track_b_jobs)
    validation.update({
        "track_a_jobs_sha256": args.track_a_jobs_sha256,
        "track_b_addendum_sha256": args.track_b_addendum_sha256,
        "direction_bundle": str(args.direction_bundle),
        "direction_bundle_sha256": args.direction_bundle_sha256,
        "direction_key": args.direction_key,
        "direction_sha256": args.direction_sha256,
    })
    if args.validate_inputs or args.dry_run:
        # CPU-only validation still checks the selected member, not only
        # the enclosing NPZ bytes. It imports numpy but never torch/transformers.
        load_direction_npz(
            args.direction_bundle,
            args.direction_bundle_sha256,
            args.direction_key,
            args.direction_sha256,
            config["hidden_size"],
        )
        validation["mode"] = "VALIDATE_INPUTS" if args.validate_inputs else "DRY_RUN"
        return validation

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError("one-shot output root must be absent or empty; restart from a clean output destination")
    args.output_root.mkdir(parents=True, exist_ok=True)
    direction = load_direction_npz(
        args.direction_bundle,
        args.direction_bundle_sha256,
        args.direction_key,
        args.direction_sha256,
        config["hidden_size"],
    )
    token = os.environ.get(args.hf_token_env) if args.hf_token_env else None
    tokenizer, model, torch = load_actor(actor, args.cache_dir, token)
    environment = {
        "schema_version": SCHEMA_VERSION,
        "actor": actor,
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "selected_layer": config["selected_layer"],
        "dtype": "bfloat16",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "runtime_evidence": {
            "path": OPEN_RUNTIME_EVIDENCE_PATH,
            "sha256": OPEN_RUNTIME_EVIDENCE_SHA256,
        },
    }
    if not environment["cuda_available"]:
        raise RuntimeError("foreground CUDA is required for model execution")
    write_json(args.output_root / "environment.json", environment)

    completed_a = [run_track_a_job(record, tokenizer, model, torch, direction, args.output_root) for record in track_a_jobs]
    write_jsonl(args.output_root / "track_a_completed_jobs.jsonl", completed_a)
    completed_b = [run_track_b_job(record, tokenizer, model, torch, direction, args.output_root) for record in track_b_jobs]
    write_jsonl(args.output_root / "track_b_completed_addendum.jsonl", completed_b)
    inventory = artifact_inventory(args.output_root)
    write_json(args.output_root / "artifact_manifest.json", {"schema_version": SCHEMA_VERSION, "actor": actor, "files": inventory})
    archive = make_archive(args.output_root)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "terminal_status": "FORWARD_COMPLETE_ANALYSIS_PENDING",
        "actor": actor,
        "track_a_jobs": len(completed_a),
        "track_b_jobs": len(completed_b),
        "archive": {"path": archive.name, "sha256": sha256_file(archive), "bytes": archive.stat().st_size},
        "scientific_retries": 0,
        "checkpoint_used": False,
    }
    write_json(args.output_root / "technical_receipt.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True, choices=sorted(ACTORS))
    parser.add_argument("--track-a-jobs", type=Path, required=True)
    parser.add_argument("--track-a-jobs-sha256", required=True)
    parser.add_argument("--track-b-addendum", type=Path, required=True)
    parser.add_argument("--track-b-addendum-sha256", required=True)
    parser.add_argument("--direction-bundle", type=Path, required=True)
    parser.add_argument("--direction-bundle-sha256", required=True)
    parser.add_argument("--direction-key", default="raw_relation_dim")
    parser.add_argument("--direction-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/content/model-cache"))
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-inputs", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
