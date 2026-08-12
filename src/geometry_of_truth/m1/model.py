from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import M1Config
from .model_specs import LLAMA_MODEL_ID, get_model_spec
from .prompts import mapping_for_item, render_batch
from geometry_of_truth.truth.reproduction.model import gpu_preflight


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    tokenizer: Any
    model_revision: str
    tokenizer_revision: str
    device_name: str
    gpu_memory_mib: int
    activation_torch_dtype: str
    model_id: str = LLAMA_MODEL_ID
    model_family: str = "llama31_8b"
    chat_policy: str = "native_system"


@dataclass(frozen=True)
class ExtractionResult:
    activations: np.ndarray
    supports_logp: np.ndarray
    opposes_logp: np.ndarray
    native_margin: np.ndarray
    predicted_labels: np.ndarray
    prompt_token_indices: np.ndarray
    rendered_prompts: list[str]
    mapping_names: list[str]
    candidate_token_lengths: np.ndarray
    peak_vram_bytes: int


def load_model(config: M1Config) -> LoadedModel:
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = config.section("model")
    try:
        spec = get_model_spec(str(model_config["id"]))
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    gpu = gpu_preflight(int(model_config["min_gpu_memory_mib"]))
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing; add it to Colab Secrets.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU execution and offloading are forbidden.")
    revision = HfApi(token=token).model_info(
        model_config["id"], revision=model_config["revision"]
    ).sha
    if spec.pinned_revision is not None and revision != spec.pinned_revision:
        raise RuntimeError(
            f"Resolved revision {revision} differs from pinned {spec.pinned_revision}."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=revision,
        token=token,
        trust_remote_code=False,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("The pinned tokenizer does not expose a chat template.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable; the exact protocol forbids FP16 fallback.")
    torch_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"],
        revision=revision,
        token=token,
        torch_dtype=torch_dtype,
        device_map=model_config["device_map"],
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model_type = str(getattr(model.config, "model_type", ""))
    if model_type != spec.expected_model_type:
        raise RuntimeError(
            f"Expected model_type {spec.expected_model_type!r}, got {model_type!r}."
        )
    device_map = getattr(model, "hf_device_map", {})
    forbidden = {str(device) for device in device_map.values()} & {"cpu", "disk"}
    if forbidden:
        raise RuntimeError(
            f"Model was offloaded to {sorted(forbidden)}; use a qualifying GPU."
        )
    parameter_devices = {str(parameter.device) for parameter in model.parameters()}
    if (
        not parameter_devices
        or any(not device.startswith("cuda:") for device in parameter_devices)
        or len(parameter_devices) != 1
    ):
        raise RuntimeError(
            "Every model parameter must reside on one CUDA device; observed "
            f"{sorted(parameter_devices)}."
        )
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        model_revision=revision,
        tokenizer_revision=revision,
        device_name=str(gpu["name"]),
        gpu_memory_mib=int(gpu["memory_total_mib"]),
        activation_torch_dtype="bfloat16",
        model_id=spec.model_id,
        model_family=spec.family,
        chat_policy=spec.chat_policy,
    )


def candidate_token_ids(tokenizer: Any, text: str) -> list[int]:
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    if not ids:
        raise RuntimeError(f"Candidate answer {text!r} tokenized to an empty sequence.")
    return [int(value) for value in ids]


def _score_candidate_sequences(
    loaded: LoadedModel,
    prompt_rows: list[list[int]],
    candidate_pairs: list[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    if len(prompt_rows) != len(candidate_pairs):
        raise ValueError("Prompt and candidate counts differ.")
    sequences: list[list[int]] = []
    prompt_lengths: list[int] = []
    candidate_ids: list[list[int]] = []
    owner: list[tuple[int, int]] = []
    for item_index, (prompt, pair) in enumerate(
        zip(prompt_rows, candidate_pairs, strict=True)
    ):
        for semantic_index, candidate in enumerate(pair):
            ids = candidate_token_ids(loaded.tokenizer, candidate)
            sequences.append([*prompt, *ids])
            prompt_lengths.append(len(prompt))
            candidate_ids.append(ids)
            owner.append((item_index, semantic_index))
    if loaded.tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer pad token must be configured before candidate scoring.")
    loaded.tokenizer.padding_side = "right"
    encoded = loaded.tokenizer.pad(
        {"input_ids": sequences}, padding=True, return_tensors="pt"
    )
    device = loaded.model.get_input_embeddings().weight.device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = loaded.model(
            **encoded, output_hidden_states=False, use_cache=False, return_dict=True
        )
        log_probs = torch.log_softmax(output.logits.float(), dim=-1)
    scores = np.empty((len(prompt_rows), 2), dtype=np.float32)
    lengths = np.empty((len(prompt_rows), 2), dtype=np.int16)
    for row_index, ((item_index, semantic_index), prompt_length, ids) in enumerate(
        zip(owner, prompt_lengths, candidate_ids, strict=True)
    ):
        total = 0.0
        for offset, token_id in enumerate(ids):
            total += float(log_probs[row_index, prompt_length + offset - 1, token_id])
        scores[item_index, semantic_index] = total
        lengths[item_index, semantic_index] = len(ids)
    return scores[:, 0], scores[:, 1], lengths


def extract_batch(
    loaded: LoadedModel,
    records: list[dict[str, str]],
    prompt_config: dict[str, Any],
    activation_dtype: str,
) -> ExtractionResult:
    import torch

    rendered = render_batch(
        loaded.tokenizer,
        records,
        prompt_config,
        model_id=loaded.model_id,
    )
    device = loaded.model.get_input_embeddings().weight.device
    encoded = {key: value.to(device) for key, value in rendered.encoded.items()}
    final_indices = rendered.final_token_indices.to(device)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        output = loaded.model(
            **encoded, output_hidden_states=True, use_cache=False, return_dict=True
        )
    expected_layers = int(loaded.model.config.num_hidden_layers)
    if output.hidden_states is None or len(output.hidden_states) != expected_layers + 1:
        actual = 0 if output.hidden_states is None else len(output.hidden_states)
        raise RuntimeError(
            f"Expected embedding plus {expected_layers} layer states, got {actual}."
        )
    row_indices = torch.arange(len(records), device=device)
    states = [
        layer[row_indices, final_indices, :] for layer in output.hidden_states[1:]
    ]
    activations = (
        torch.stack(states, dim=1)
        .float()
        .cpu()
        .numpy()
        .astype(activation_dtype)
    )
    final_cpu = final_indices.cpu()
    prompt_rows = [
        rendered.encoded["input_ids"][index, : int(final_cpu[index]) + 1].tolist()
        for index in range(len(records))
    ]
    del output, states
    supports_logp, opposes_logp, lengths = _score_candidate_sequences(
        loaded, prompt_rows, rendered.candidate_texts
    )
    margin = supports_logp - opposes_logp
    mapping_names = [
        mapping_for_item(record["item_id"], record["scheme"], prompt_config).name
        for record in records
    ]
    return ExtractionResult(
        activations=activations,
        supports_logp=supports_logp,
        opposes_logp=opposes_logp,
        native_margin=margin.astype(np.float32),
        predicted_labels=(margin >= 0).astype(np.int8),
        prompt_token_indices=final_cpu.numpy().astype(np.int32),
        rendered_prompts=rendered.rendered,
        mapping_names=mapping_names,
        candidate_token_lengths=lengths,
        peak_vram_bytes=int(torch.cuda.max_memory_allocated()),
    )
