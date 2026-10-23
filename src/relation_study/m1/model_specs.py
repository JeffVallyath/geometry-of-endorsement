from __future__ import annotations

from dataclasses import dataclass


LLAMA_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LLAMA_MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
GEMMA_MODEL_ID = "google/gemma-2-9b-it"
GEMMA_MODEL_REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    expected_model_type: str
    chat_policy: str
    pinned_revision: str | None
    minimum_gpu_memory_mib: int = 23000


MODEL_SPECS = {
    LLAMA_MODEL_ID: ModelSpec(
        model_id=LLAMA_MODEL_ID,
        family="llama31_8b",
        expected_model_type="llama",
        chat_policy="native_system",
        pinned_revision=LLAMA_MODEL_REVISION,
    ),
    GEMMA_MODEL_ID: ModelSpec(
        model_id=GEMMA_MODEL_ID,
        family="gemma2_9b",
        expected_model_type="gemma2",
        chat_policy="fold_system_into_user",
        pinned_revision=GEMMA_MODEL_REVISION,
    ),
}


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_id]
    except KeyError as exc:
        allowed = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(
            f"Model substitution is forbidden; allowed exact models are {allowed}."
        ) from exc


def validate_model_revision(model_id: str, revision: str) -> None:
    spec = get_model_spec(model_id)
    if spec.pinned_revision is not None and revision != spec.pinned_revision:
        raise ValueError(
            f"{model_id} must use immutable revision {spec.pinned_revision}; "
            f"observed {revision}."
        )
