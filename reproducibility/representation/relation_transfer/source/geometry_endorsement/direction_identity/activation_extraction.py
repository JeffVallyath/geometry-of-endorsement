from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hashlib
import numpy as np


SETUP_ALLOWED_MODES = frozenset({"AUDIT_ONLY", "SYNTHETIC_SMOKE_TEST"})
ALL_MODES = frozenset({"AUDIT_ONLY", "RUN_LLAMA_EXTRACTION", "RUN_LLAMA_INTERVENTION",
    "RUN_GEMMA_EXTRACTION", "RUN_GEMMA_INTERVENTION", "ANALYZE_CPU", "BUILD_FINAL_REPORT"})


@dataclass(frozen=True)
class ExtractionContract:
    layer: int
    batch_size: int = 1
    activation_position: str = "final_prompt_position_immediately_before_candidate_answer_sequence"
    model_dtype: str = "bfloat16"
    score_dtype: str = "float32"

    def validate(self) -> None:
        if self.batch_size != 1: raise RuntimeError("Frozen extraction requires batch_size=1")
        if self.layer < 0: raise RuntimeError("Layer selection is forbidden")
        if self.activation_position != "final_prompt_position_immediately_before_candidate_answer_sequence":
            raise RuntimeError("Activation-position drift")


def enforce_setup_mode(mode: str) -> None:
    if mode not in SETUP_ALLOWED_MODES:
        raise RuntimeError(f"SETUP_ONLY_MODEL_INFERENCE_BLOCKED:{mode}")


def checkpoint_key(model: str, dataset: str, template: str, mapping: str) -> str:
    return f"{model}/{dataset}/{template}/{mapping}"


def load_actor(model_id: str, revision: str, cache_dir: str | Path, hf_token: str | None = None) -> tuple[Any, Any]:
    """Lazy real-runtime loader; never called by setup or synthetic validation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kwargs = {"revision": revision, "cache_dir": str(cache_dir), "token": hf_token}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
        device_map="auto", low_cpu_mem_usage=True, **kwargs)
    model.eval()
    return model, tokenizer


def prompt_ids(tokenizer: Any, rendered_text: str) -> list[int]:
    messages = [{"role": "user", "content": rendered_text}]
    ids = list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    if not ids: raise RuntimeError("Empty chat prompt")
    return ids


def candidate_ids(tokenizer: Any, candidate: str) -> list[int]:
    ids = list(tokenizer.encode(candidate, add_special_tokens=False))
    if not ids: raise RuntimeError(f"Empty candidate sequence: {candidate!r}")
    return ids


def _layer_stack(model: Any) -> Any:
    for path in (("model", "layers"), ("model", "model", "layers")):
        value=model
        try:
            for part in path: value=getattr(value, part)
            return value
        except AttributeError: continue
    raise RuntimeError("Unsupported decoder layer topology")


def score_and_extract(model: Any, tokenizer: Any, rendered_text: str,
                      candidates: tuple[str,str], layer: int,
                      intervention_delta: np.ndarray | None = None) -> dict[str, Any]:
    return score_ids_and_extract(model, tokenizer, prompt_ids(tokenizer, rendered_text), candidates, layer, intervention_delta)


def score_ids_and_extract(model: Any, tokenizer: Any, pids: list[int],
                          candidates: tuple[str,str], layer: int,
                          intervention_delta: np.ndarray | None = None) -> dict[str, Any]:
    import torch
    activation=None
    layers=_layer_stack(model)
    if layer >= len(layers): raise RuntimeError(f"Frozen layer {layer} absent")
    delta_tensor=None if intervention_delta is None else torch.as_tensor(intervention_delta, dtype=torch.float32)
    def hook(_module, _inputs, output):
        nonlocal activation
        hidden=output[0] if isinstance(output,tuple) else output
        activation=hidden[:,len(pids)-1,:].detach().float().cpu().numpy()[0]
        if delta_tensor is None: return output
        changed=hidden.clone(); changed[:,len(pids)-1,:] += delta_tensor.to(hidden.device,dtype=hidden.dtype)
        return (changed,*output[1:]) if isinstance(output,tuple) else changed
    handle=layers[layer].register_forward_hook(hook)
    try:
        logps={}; sequences={}
        with torch.inference_mode():
            for candidate in candidates:
                cids=candidate_ids(tokenizer,candidate); sequences[candidate]=cids
                ids=torch.tensor([pids+cids],device=model.device,dtype=torch.long)
                logits=model(input_ids=ids,use_cache=False).logits.float()
                total=torch.zeros((),device=logits.device,dtype=torch.float32)
                for offset, token_id in enumerate(cids):
                    total += torch.log_softmax(logits[0,len(pids)+offset-1],dim=-1)[token_id]
                logps[candidate]=float(total.cpu())
    finally: handle.remove()
    if activation is None: raise RuntimeError("Frozen activation hook did not fire")
    return {"activation":activation,"candidate_log_probabilities":logps,"candidate_token_ids":sequences}


def load_direction_npz(path: str | Path, bundle_sha256: str, direction_sha256: str,
                       key: str = "difference_in_means_direction") -> np.ndarray:
    source=Path(path); actual=hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != bundle_sha256: raise RuntimeError(f"FROZEN_INPUT_MISMATCH: bundle {actual}")
    with np.load(source,allow_pickle=False) as archive: direction=np.asarray(archive[key],dtype=np.float32)
    raw_hash=hashlib.sha256(direction.tobytes(order="C")).hexdigest()
    if raw_hash != direction_sha256: raise RuntimeError(f"FROZEN_INPUT_MISMATCH: direction {raw_hash}")
    return direction
