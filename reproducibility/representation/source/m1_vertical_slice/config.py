from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from .model_specs import LLAMA_MODEL_ID, get_model_spec, validate_model_revision

# Backward-compatible alias used by the frozen Llama importer and tests.
EXACT_MODEL_ID = LLAMA_MODEL_ID


class ConfigError(ValueError):
    """Raised when a run departs from the frozen M1 protocol."""


@dataclass(frozen=True)
class M1Config:
    raw: dict[str, Any]
    path: Path
    digest: str

    @property
    def mode(self) -> str:
        return str(self.raw["run"]["mode"])

    def section(self, name: str) -> dict[str, Any]:
        return self.raw[name]

    @property
    def extraction_digest(self) -> str:
        """Digest only fields capable of changing cached model outputs."""
        return canonical_digest(
            {
                "schema": "m1_activation_cache_v2",
                "model": self.raw["model"],
                "prompt": self.raw["prompt"],
                "cache": self.raw["cache"],
            }
        )


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> M1Config:
    resolved = Path(path).resolve()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a mapping.")
    _validate(raw)
    return M1Config(raw, resolved, canonical_digest(raw))


def _validate(raw: dict[str, Any]) -> None:
    required = {"run", "model", "data", "pilot", "prompt", "cache", "analysis"}
    missing = sorted(required - set(raw))
    if missing:
        raise ConfigError(f"Missing config sections: {missing}")
    mode = raw["run"].get("mode")
    if mode not in {"smoke", "full"}:
        raise ConfigError("run.mode must be smoke or full.")
    model = raw["model"]
    try:
        spec = get_model_spec(str(model.get("id")))
        validate_model_revision(str(model.get("id")), str(model.get("revision")))
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    if model.get("torch_dtype") != "bfloat16":
        raise ConfigError("The frozen primary dtype is bfloat16.")
    if model.get("device_map") != "auto":
        raise ConfigError("device_map must be auto; CPU or disk placement is forbidden.")
    if model.get("trust_remote_code") is not False:
        raise ConfigError("trust_remote_code must remain false.")
    if model.get("allow_fp16_fallback") is not False:
        raise ConfigError("The exact Colab protocol forbids FP16 fallback.")
    if model.get("quantization") not in {None, "none"}:
        raise ConfigError("Quantization is forbidden for this protocol.")
    if int(model.get("min_gpu_memory_mib", 0)) < 23000:
        raise ConfigError("The GPU gate may not be below 23,000 MiB.")
    if int(model.get("min_gpu_memory_mib", 0)) < spec.minimum_gpu_memory_mib:
        raise ConfigError(
            f"{spec.family} requires at least {spec.minimum_gpu_memory_mib:,} MiB."
        )
    pilot = raw["pilot"]
    for key in ("train_rows", "select_boards", "eval_boards"):
        if int(pilot.get(key, 0)) < 1:
            raise ConfigError(f"pilot.{key} must be positive.")
    if mode == "full":
        if not 1000 <= int(pilot["train_rows"]) <= 1500:
            raise ConfigError("Full pilot_train must contain 1,000-1,500 rows.")
        if not 300 <= 4 * int(pilot["select_boards"]) <= 500:
            raise ConfigError("Full pilot_select must contain 300-500 board-cell rows.")
        if not 500 <= 4 * int(pilot["eval_boards"]) <= 1000:
            raise ConfigError("Full pilot_eval must contain 500-1,000 board-cell rows.")
        if int(raw["analysis"].get("permutations", 0)) < 100:
            raise ConfigError("Full runs require at least 100 grouped label permutations.")
    schemes = raw["prompt"].get("schemes", {})
    if raw["prompt"].get("chat_template_source") != "pinned_tokenizer_revision":
        raise ConfigError("The chat template must come from the pinned tokenizer revision.")
    if raw["prompt"].get("add_generation_prompt") is not True:
        raise ConfigError("The frozen prompt ends at the assistant generation boundary.")
    if (
        raw["prompt"].get("stop_conditions")
        != "not_applicable_sequence_scoring_without_generation"
    ):
        raise ConfigError("Primary measurements use scoring, not generation stops.")
    if set(schemes) != {"primary", "transfer"}:
        raise ConfigError("Exactly primary and transfer prompt schemes are required.")
    for name, scheme in schemes.items():
        symbols = scheme.get("symbols", [])
        if len(symbols) != 2 or symbols[0] == symbols[1]:
            raise ConfigError(f"Prompt scheme {name} needs two distinct answer symbols.")
    if raw["analysis"].get("selection_metric") != "mean_mirrored_pairwise":
        raise ConfigError("Layer selection must use the frozen mirrored-pair mean.")
    if int(raw["analysis"].get("bootstrap_replicates", 0)) < 1:
        raise ConfigError("At least one bootstrap replicate is required.")
    analysis = raw["analysis"]
    if analysis.get("primary_checkerboard_metric") != "standardized_interaction_contrast":
        raise ConfigError("The primary checkerboard metric must be standardized I_b.")
    if analysis.get("text_baseline") != "sbert_interaction":
        raise ConfigError("The frozen text baseline must be sbert_interaction.")
    if float(analysis.get("practical_effect_sd", -1)) != 0.30:
        raise ConfigError("The frozen practical effect is 0.30 standardized I_b units.")
    if analysis.get("checkerboard_inference") != "dyadic_robust":
        raise ConfigError("Checkerboard inference must use the dyadic-robust estimator.")
    if "practical_lift" in analysis:
        raise ConfigError("The retired both-ways practical_lift field is forbidden.")
