from __future__ import annotations

import gc
import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class CandidateScore:
    first_logp: float
    second_logp: float
    margin: float
    first_tokens: tuple[int, ...]
    second_tokens: tuple[int, ...]
    prompt_token_sha256: str
    length_normalized_margin: float


@dataclass(frozen=True)
class CandidateActivationScore:
    score: CandidateScore
    activation: np.ndarray
    prompt_token_index: int


def token_ids_sha256(ids: Sequence[int]) -> str:
    payload = b"".join(int(value).to_bytes(8, "little", signed=True) for value in ids)
    return hashlib.sha256(payload).hexdigest()


def candidate_token_ids(tokenizer: Any, text: str) -> list[int]:
    ids = [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
    if not ids:
        raise RuntimeError(f"Candidate answer {text!r} tokenized to an empty sequence.")
    return ids


def configure_frozen_padding(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither pad nor EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


def score_one_prompt(
    loaded: Any,
    prompt_ids: Sequence[int],
    candidates: tuple[str, str],
    *,
    batch_size: int = 1,
) -> CandidateScore:
    """Score exactly one prompt with the repaired Gate 2 convention.

    The two candidate continuations share a model call. `batch_size` refers to
    prompts, not candidate sequences, and any value other than one fails.
    """
    if batch_size != 1:
        raise RuntimeError("Dynamic multi-prompt candidate scoring is forbidden; batch_size must equal 1.")
    import torch

    prompt = [int(value) for value in prompt_ids]
    if not prompt:
        raise RuntimeError("Prompt token sequence is empty.")
    configure_frozen_padding(loaded.tokenizer)
    answer_ids = [candidate_token_ids(loaded.tokenizer, text) for text in candidates]
    sequences = [[*prompt, *ids] for ids in answer_ids]
    encoded = loaded.tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
    device = loaded.model.get_input_embeddings().weight.device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = loaded.model(**encoded, output_hidden_states=False, use_cache=False, return_dict=True)
        log_probs = torch.log_softmax(output.logits.float(), dim=-1)
    sums: list[np.float32] = []
    means: list[np.float32] = []
    for row, ids in enumerate(answer_ids):
        values = [log_probs[row, len(prompt) + offset - 1, token_id].float() for offset, token_id in enumerate(ids)]
        total = torch.stack(values).sum(dtype=torch.float32)
        sums.append(np.float32(total.item()))
        means.append(np.float32((total / len(ids)).item()))
    margin = np.float32(sums[0] - sums[1])
    normalized = np.float32(means[0] - means[1])
    return CandidateScore(
        first_logp=float(sums[0]),
        second_logp=float(sums[1]),
        margin=float(margin),
        first_tokens=tuple(answer_ids[0]),
        second_tokens=tuple(answer_ids[1]),
        prompt_token_sha256=token_ids_sha256(prompt),
        length_normalized_margin=float(normalized),
    )


def score_one_prompt_with_activation(
    loaded: Any,
    prompt_ids: Sequence[int],
    candidates: tuple[str, str],
    *,
    selected_layer: int,
    batch_size: int = 1,
) -> CandidateActivationScore:
    if batch_size != 1:
        raise RuntimeError("Dynamic multi-prompt candidate scoring is forbidden; batch_size must equal 1.")
    import torch

    prompt = [int(value) for value in prompt_ids]
    configure_frozen_padding(loaded.tokenizer)
    answer_ids = [candidate_token_ids(loaded.tokenizer, text) for text in candidates]
    encoded = loaded.tokenizer.pad({"input_ids": [[*prompt, *ids] for ids in answer_ids]}, padding=True, return_tensors="pt")
    device = loaded.model.get_input_embeddings().weight.device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = loaded.model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
        log_probs = torch.log_softmax(output.logits.float(), dim=-1)
    sums = []
    means = []
    for row, ids in enumerate(answer_ids):
        values = torch.stack([log_probs[row, len(prompt) + offset - 1, token_id].float() for offset, token_id in enumerate(ids)])
        total = values.sum(dtype=torch.float32)
        sums.append(np.float32(total.item()))
        means.append(np.float32((total / len(ids)).item()))
    candidate = CandidateScore(float(sums[0]), float(sums[1]), float(np.float32(sums[0] - sums[1])), tuple(answer_ids[0]), tuple(answer_ids[1]), token_ids_sha256(prompt), float(np.float32(means[0] - means[1])))
    activation = output.hidden_states[int(selected_layer) + 1][0, len(prompt) - 1].float().cpu().numpy().astype(np.float32)
    return CandidateActivationScore(candidate, activation, len(prompt) - 1)


def unload_model(loaded: Any) -> None:
    """Release the sole sequential model before the next jury member."""
    import torch

    model = getattr(loaded, "model", None)
    tokenizer = getattr(loaded, "tokenizer", None)
    try:
        object.__setattr__(loaded, "model", None)
        object.__setattr__(loaded, "tokenizer", None)
    except (AttributeError, TypeError):
        pass
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
