from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import EXACT_MODEL_ID, RunConfig
from .prompts import render_batch


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    tokenizer: Any
    model_revision: str
    tokenizer_revision: str
    device_name: str
    gpu_memory_mib: int
    activation_torch_dtype: str


@dataclass(frozen=True)
class BatchResult:
    activations: np.ndarray
    true_logp: np.ndarray
    false_logp: np.ndarray
    truth_scores: np.ndarray
    predicted_labels: np.ndarray
    prompt_token_indices: np.ndarray
    rendered_prompts: list[str]
    mapping_names: list[str]
    candidate_token_lengths: np.ndarray
    peak_vram_bytes: int


def gpu_preflight(min_memory_mib: int) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen Llama-3.1-8B-Instruct protocol.")
    props = torch.cuda.get_device_properties(0)
    total_mib = int(props.total_memory // (1024 * 1024))
    if total_mib < min_memory_mib:
        raise RuntimeError(
            f"GPU {props.name!r} exposes {total_mib} MiB; protocol requires at least "
            f"{min_memory_mib} MiB. Do not quantize or substitute a model."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Native CUDA BF16 is required; FP16 fallback is forbidden.")
    return {"name": props.name, "memory_total_mib": total_mib, "cuda_runtime": torch.version.cuda}


def assert_model_on_single_cuda(model: Any) -> None:
    device_map = getattr(model, "hf_device_map", {})
    forbidden = {str(device) for device in device_map.values()} & {"cpu", "disk"}
    if forbidden:
        raise RuntimeError(f"Model was offloaded to {sorted(forbidden)}.")
    parameter_devices = {str(parameter.device) for parameter in model.parameters()}
    if (
        not parameter_devices
        or any(not device.startswith("cuda:") for device in parameter_devices)
        or len(parameter_devices) != 1
    ):
        raise RuntimeError(
            "Every model parameter must reside on exactly one CUDA device; observed "
            f"{sorted(parameter_devices)}."
        )


def load_model(config: RunConfig) -> LoadedModel:
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if config.model["id"] != EXACT_MODEL_ID:
        raise RuntimeError("Refusing model substitution.")
    gpu = gpu_preflight(int(config.model["min_gpu_memory_mib"]))
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing; add it to Colab Secrets.")
    revision = HfApi(token=token).model_info(
        config.model["id"], revision=config.model["revision"]
    ).sha
    tokenizer = AutoTokenizer.from_pretrained(
        config.model["id"], revision=revision, token=token, trust_remote_code=False
    )
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("The pinned tokenizer does not expose a chat template.")
    model = AutoModelForCausalLM.from_pretrained(
        config.model["id"],
        revision=revision,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map=config.model["device_map"],
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    assert_model_on_single_cuda(model)
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        model_revision=revision,
        tokenizer_revision=revision,
        device_name=str(gpu["name"]),
        gpu_memory_mib=int(gpu["memory_total_mib"]),
        activation_torch_dtype="bfloat16",
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
    owners: list[tuple[int, int]] = []
    for item_index, (prompt, pair) in enumerate(zip(prompt_rows, candidate_pairs, strict=True)):
        for semantic_index, candidate in enumerate(pair):
            ids = candidate_token_ids(loaded.tokenizer, candidate)
            sequences.append([*prompt, *ids])
            prompt_lengths.append(len(prompt))
            candidate_ids.append(ids)
            owners.append((item_index, semantic_index))
    if loaded.tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer pad token must be configured before candidate scoring.")
    loaded.tokenizer.padding_side = "right"
    encoded = loaded.tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
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
        zip(owners, prompt_lengths, candidate_ids, strict=True)
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
) -> BatchResult:
    import torch

    rendered = render_batch(loaded.tokenizer, records, prompt_config)
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
        torch.stack(states, dim=1).float().cpu().numpy().astype(activation_dtype)
    )
    final_cpu = final_indices.cpu()
    prompt_rows = [
        rendered.encoded["input_ids"][index, : int(final_cpu[index]) + 1].tolist()
        for index in range(len(records))
    ]
    del output, states
    true_logp, false_logp, lengths = _score_candidate_sequences(
        loaded, prompt_rows, rendered.candidate_texts
    )
    margin = true_logp - false_logp
    return BatchResult(
        activations=activations,
        true_logp=true_logp,
        false_logp=false_logp,
        truth_scores=margin.astype(np.float32),
        predicted_labels=(margin >= 0).astype(np.int8),
        prompt_token_indices=final_cpu.numpy().astype(np.int32),
        rendered_prompts=rendered.rendered,
        mapping_names=rendered.mapping_names,
        candidate_token_lengths=lengths,
        peak_vram_bytes=int(torch.cuda.max_memory_allocated()),
    )
