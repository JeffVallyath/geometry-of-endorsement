from __future__ import annotations

import hashlib
import io
import json
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .input_audit import array_sha256, sha256_bytes, sha256_file


@dataclass(frozen=True)
class InterventionCondition:
    family: str
    direction_id: str
    layer: int
    dose: float
    seed: int | None


@dataclass(frozen=True)
class InterventionScore:
    supports_logp: float
    opposes_logp: float
    semantic_margin: float
    prompt_token_index: int
    layer: int
    dose: float
    target_delta_max_abs_error: float
    untargeted_max_abs_difference: float
    prompt_token_ids_sha256: str
    batch_size: int = 1


def projection_sd_step(direction: np.ndarray, projection_sigma: float) -> np.ndarray:
    vector = np.asarray(direction, dtype=np.float32)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError("DIM direction must be one finite vector.")
    denominator = float(vector @ vector)
    if denominator <= 0 or not np.isfinite(projection_sigma) or projection_sigma <= 0:
        raise ValueError("Direction norm and training projection SD must be positive.")
    result = np.float32(projection_sigma / denominator) * vector
    if not np.isclose(float(vector @ result), float(projection_sigma), rtol=2e-6, atol=2e-6):
        raise RuntimeError("Training-SD intervention calibration failed.")
    return result.astype(np.float32)


def norm_matched_orthogonal(
    reference_step: np.ndarray, direction: np.ndarray, seed: int
) -> np.ndarray:
    reference = np.asarray(reference_step, dtype=np.float32)
    basis = np.asarray(direction, dtype=np.float32)
    if reference.shape != basis.shape or reference.ndim != 1:
        raise ValueError("Control and DIM vectors must have the same one-dimensional shape.")
    rng = np.random.default_rng(int(seed))
    random = rng.standard_normal(reference.size).astype(np.float32)
    random -= np.float32((random @ basis) / (basis @ basis)) * basis
    norm = float(np.linalg.norm(random))
    target = float(np.linalg.norm(reference))
    if norm <= 0 or target <= 0:
        raise RuntimeError("Could not construct a nonzero orthogonal control.")
    result = random * np.float32(target / norm)
    if abs(float(result @ basis)) > 2e-4 * float(np.linalg.norm(result) * np.linalg.norm(basis)):
        raise RuntimeError("Orthogonal control is not numerically orthogonal.")
    return result.astype(np.float32)


def build_conditions(
    *,
    selected_layer: int,
    wrong_layer: int,
    doses: Sequence[float],
    orthogonal_seeds: Sequence[int],
) -> list[InterventionCondition]:
    dose_values = [float(value) for value in doses]
    if dose_values != [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
        raise RuntimeError("Frozen intervention doses changed.")
    rows = [InterventionCondition("no_intervention", "none", selected_layer, 0.0, None)]
    for dose in dose_values:
        rows.append(InterventionCondition("dim", "frozen_dim", selected_layer, dose, None))
        rows.append(InterventionCondition("opposite_dim", "negative_frozen_dim", selected_layer, dose, None))
        rows.append(InterventionCondition("wrong_layer", "frozen_dim", wrong_layer, dose, None))
        for seed in orthogonal_seeds:
            rows.append(
                InterventionCondition(
                    "random_orthogonal", f"orthogonal_{int(seed)}", selected_layer, dose, int(seed)
                )
            )
    return rows


def _decoder_layer(model: Any, layer: int) -> Any:
    candidates = [
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "model", None),
        getattr(model, "transformer", None),
    ]
    for candidate in candidates:
        layers = getattr(candidate, "layers", None)
        if layers is not None and 0 <= int(layer) < len(layers):
            return layers[int(layer)]
    raise RuntimeError(f"Could not resolve decoder layer {layer} on the loaded model.")


@contextmanager
def residual_intervention(
    model: Any,
    *,
    layer: int,
    token_index: int,
    delta: np.ndarray,
) -> Iterator[dict[str, Any]]:
    """Add one fixed delta at one post-block residual position.

    The hook applies to all candidate sequences for the single logical prompt.
    It never changes parameters and records exact targeted/untargeted changes.
    """
    import torch

    module = _decoder_layer(model, layer)
    vector = torch.as_tensor(np.asarray(delta, dtype=np.float32))
    audit: dict[str, Any] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        tensor = output[0] if isinstance(output, tuple) else output
        if tensor.ndim != 3 or not (0 <= int(token_index) < tensor.shape[1]):
            raise RuntimeError("Intervention position is outside the decoder-layer output.")
        cast = vector.to(device=tensor.device, dtype=tensor.dtype)
        changed = tensor.clone()
        before = tensor.detach().float()
        changed[:, int(token_index), :] = changed[:, int(token_index), :] + cast
        after = changed.detach().float()
        actual = after[:, int(token_index), :] - before[:, int(token_index), :]
        expected = cast.detach().float().expand_as(actual)
        mask = torch.ones(tensor.shape[1], dtype=torch.bool, device=tensor.device)
        mask[int(token_index)] = False
        untargeted = (
            (after[:, mask, :] - before[:, mask, :]).abs().max().item()
            if bool(mask.any())
            else 0.0
        )
        audit["target_delta_max_abs_error"] = float((actual - expected).abs().max().item())
        audit["untargeted_max_abs_difference"] = float(untargeted)
        if isinstance(output, tuple):
            return (changed, *output[1:])
        return changed

    handle = module.register_forward_hook(hook)
    try:
        yield audit
    finally:
        handle.remove()


def _candidate_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    ids = tuple(int(value) for value in tokenizer.encode(text, add_special_tokens=False))
    if not ids:
        raise RuntimeError(f"Candidate {text!r} tokenized to an empty sequence.")
    return ids


def _ids_sha256(ids: Sequence[int]) -> str:
    return hashlib.sha256(
        b"".join(int(value).to_bytes(8, "little", signed=True) for value in ids)
    ).hexdigest()


def load_frozen_dim(
    probe_bundle: str | Path,
    *,
    expected_sha256: str,
    expected_revision: str,
    expected_layer: int,
    expected_width: int,
) -> dict[str, Any]:
    """Read and verify the retained DIM vector without refitting it."""
    path = Path(probe_bundle)
    if sha256_file(path) != str(expected_sha256):
        raise RuntimeError("Frozen probe-bundle SHA-256 changed.")
    with zipfile.ZipFile(path) as archive:
        payload = archive.read("m1_probe_parameters.npz")
    arrays = np.load(io.BytesIO(payload), allow_pickle=False)
    direction = np.asarray(arrays["difference_in_means_direction"], dtype=np.float32)
    revision = str(np.asarray(arrays["model_revision"]).item())
    tokenizer_revision = str(np.asarray(arrays["tokenizer_revision"]).item())
    layer = int(np.asarray(arrays["selected_layer"]).item())
    if direction.shape != (int(expected_width),):
        raise RuntimeError("Frozen DIM width changed.")
    if revision != expected_revision or tokenizer_revision != expected_revision:
        raise RuntimeError("Frozen DIM model/tokenizer revision changed.")
    if layer != int(expected_layer):
        raise RuntimeError("Frozen DIM selected layer changed.")
    return {
        "direction": direction,
        "direction_sha256": array_sha256(direction),
        "probe_member_sha256": sha256_bytes(payload),
        "model_revision": revision,
        "tokenizer_revision": tokenizer_revision,
        "selected_layer": layer,
    }


def render_claim1_reason_prompt(
    loaded: Any,
    *,
    item_id: str,
    situation: str,
    consideration: str,
    prompt_config: Mapping[str, Any],
) -> tuple[list[int], str, str, str]:
    """Render the exact retained M1 primary/joint prompt and mapping."""
    from m1_vertical_slice.prompts import mapping_for_item, messages_for

    mapping = mapping_for_item(str(item_id), "primary", dict(prompt_config))
    messages = messages_for(
        str(situation),
        str(consideration),
        mapping,
        "primary",
        "joint",
        dict(prompt_config),
        model_id=loaded.model_id,
    )
    ids = [
        int(value)
        for value in loaded.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=bool(prompt_config["add_generation_prompt"]),
        )
    ]
    if not ids:
        raise RuntimeError("Claim 1 intervention prompt rendered empty.")
    rendered = loaded.tokenizer.decode(ids, skip_special_tokens=False)
    return ids, mapping.supports, mapping.opposes, mapping.name


def render_explicit_reason_prompt(
    loaded: Any,
    *,
    situation: str,
    consideration: str,
    mapping_name: str,
    prompt_config: Mapping[str, Any],
) -> tuple[list[int], str, str, str]:
    """Render one of the two repaired Claim 2 A/B mappings exactly."""
    from m1_vertical_slice.prompts import AnswerMapping, messages_for

    if mapping_name == "standard":
        supports, opposes = "A", "B"
    elif mapping_name == "reversed":
        supports, opposes = "B", "A"
    else:
        raise ValueError("mapping_name must be standard or reversed.")
    mapping = AnswerMapping(
        name=f"primary:{mapping_name}",
        supports=supports,
        opposes=opposes,
        instruction=(
            f"Use {supports} when the named consideration supports the action in this "
            f"situation, and {opposes} when it opposes the action. Reply with only A or B."
        ),
    )
    messages = messages_for(
        str(situation),
        str(consideration),
        mapping,
        "primary",
        "joint",
        dict(prompt_config),
        model_id=loaded.model_id,
    )
    ids = [
        int(value)
        for value in loaded.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=bool(prompt_config["add_generation_prompt"]),
        )
    ]
    if not ids:
        raise RuntimeError("Claim 2 intervention prompt rendered empty.")
    rendered = loaded.tokenizer.decode(ids, skip_special_tokens=False)
    return ids, supports, opposes, mapping.name


def extract_final_prompt_activation(
    loaded: Any,
    *,
    prompt_ids: Sequence[int],
    layer: int,
    batch_size: int = 1,
) -> np.ndarray:
    """Extract one post-block final-prompt-token activation at batch size one."""
    if batch_size != 1:
        raise RuntimeError("Activation extraction requires batch_size=1.")
    import torch

    prompt = [int(value) for value in prompt_ids]
    if not prompt:
        raise RuntimeError("Activation extraction prompt is empty.")
    module = _decoder_layer(loaded.model, int(layer))
    captured: dict[str, Any] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        if tensor.shape[0] != 1 or tensor.ndim != 3:
            raise RuntimeError("Activation extraction observed a non-single-prompt tensor.")
        captured["activation"] = tensor[0, len(prompt) - 1, :].detach().float().cpu().numpy()

    handle = module.register_forward_hook(hook)
    try:
        device = loaded.model.get_input_embeddings().weight.device
        input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            loaded.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
    finally:
        handle.remove()
    if "activation" not in captured:
        raise RuntimeError("The selected decoder-layer hook did not fire.")
    value = np.asarray(captured["activation"], dtype=np.float32)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise RuntimeError("Extracted activation is invalid.")
    return value


def score_reason_with_intervention(
    loaded: Any,
    *,
    prompt_ids: Sequence[int],
    supports_candidate: str,
    opposes_candidate: str,
    direction_step: np.ndarray,
    condition: InterventionCondition,
    batch_size: int = 1,
) -> InterventionScore:
    if batch_size != 1:
        raise RuntimeError("Intervention scoring requires batch_size=1.")
    import torch

    prompt = [int(value) for value in prompt_ids]
    if not prompt:
        raise RuntimeError("Intervention prompt is empty.")
    target = len(prompt) - 1
    tokenizer = loaded.tokenizer
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither pad nor EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    candidates = (
        _candidate_ids(tokenizer, supports_candidate),
        _candidate_ids(tokenizer, opposes_candidate),
    )
    sequences = [[*prompt, *candidate] for candidate in candidates]
    encoded = tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
    device = loaded.model.get_input_embeddings().weight.device
    encoded = {key: value.to(device) for key, value in encoded.items()}

    step = np.asarray(direction_step, dtype=np.float32)
    if condition.family == "opposite_dim":
        step = -step
    delta = np.float32(condition.dose) * step
    context = (
        residual_intervention(
            loaded.model,
            layer=int(condition.layer),
            token_index=target,
            delta=delta,
        )
        if condition.family != "no_intervention"
        else _null_audit()
    )
    with context as audit:
        with torch.inference_mode():
            output = loaded.model(
                **encoded, output_hidden_states=False, use_cache=False, return_dict=True
            )
            log_probs = torch.log_softmax(output.logits.float(), dim=-1)
    totals: list[float] = []
    for row, candidate in enumerate(candidates):
        values = torch.stack(
            [
                log_probs[row, len(prompt) + offset - 1, token_id].float()
                for offset, token_id in enumerate(candidate)
            ]
        )
        totals.append(float(values.sum(dtype=torch.float32).item()))
    return InterventionScore(
        supports_logp=totals[0],
        opposes_logp=totals[1],
        semantic_margin=float(np.float32(totals[0] - totals[1])),
        prompt_token_index=target,
        layer=int(condition.layer),
        dose=float(condition.dose),
        target_delta_max_abs_error=float(audit.get("target_delta_max_abs_error", 0.0)),
        untargeted_max_abs_difference=float(audit.get("untargeted_max_abs_difference", 0.0)),
        prompt_token_ids_sha256=_ids_sha256(prompt),
    )


@contextmanager
def _null_audit() -> Iterator[dict[str, float]]:
    yield {"target_delta_max_abs_error": 0.0, "untargeted_max_abs_difference": 0.0}
