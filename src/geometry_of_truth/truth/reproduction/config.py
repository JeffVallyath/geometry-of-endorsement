from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


EXACT_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"


class ConfigError(ValueError):
    """Raised when a run would depart from the frozen v2 protocol."""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    sha256: str


@dataclass(frozen=True)
class RunConfig:
    raw: dict[str, Any]
    path: Path
    digest: str

    @property
    def mode(self) -> str:
        return str(self.raw["run"]["mode"])

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def prompt(self) -> dict[str, Any]:
        return self.raw["prompt"]

    @property
    def cache(self) -> dict[str, Any]:
        return self.raw["cache"]

    @property
    def analysis(self) -> dict[str, Any]:
        return self.raw["analysis"]

    @property
    def datasets(self) -> list[DatasetSpec]:
        return [DatasetSpec(**item) for item in self.data["datasets"]]


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> RunConfig:
    resolved = Path(path).resolve()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a mapping.")
    _validate(raw)
    return RunConfig(raw=raw, path=resolved, digest=canonical_digest(raw))


def _validate(raw: dict[str, Any]) -> None:
    required = {"run", "model", "data", "prompt", "cache", "analysis"}
    missing = sorted(required - set(raw))
    if missing:
        raise ConfigError(f"Missing config sections: {missing}")

    mode = raw["run"].get("mode")
    if mode not in {"smoke", "full"}:
        raise ConfigError("run.mode must be smoke or full.")
    if raw["run"].get("protocol") != "TRUTH_CONTROL_V2_NEUTRAL_MAPPING":
        raise ConfigError("run.protocol must identify the frozen v2 control.")

    model = raw["model"]
    if model.get("id") != EXACT_MODEL_ID:
        raise ConfigError(f"Model substitution is forbidden; expected {EXACT_MODEL_ID}.")
    if model.get("torch_dtype") != "bfloat16":
        raise ConfigError("The frozen model precision is bfloat16.")
    if model.get("allow_fp16_fallback") is not False:
        raise ConfigError("FP16 fallback is forbidden.")
    if model.get("quantization") not in {None, "none"}:
        raise ConfigError("Quantization is forbidden.")
    if int(model.get("min_gpu_memory_mib", 0)) < 23000:
        raise ConfigError("The GPU gate may not be below 23,000 MiB.")
    if model.get("device_map") != "auto":
        raise ConfigError("The loader contract requires device_map=auto with an independent one-CUDA assertion.")

    datasets = raw["data"].get("datasets", [])
    if [item.get("name") for item in datasets] != ["cities", "neg_cities"]:
        raise ConfigError("Dataset order must be cities, neg_cities.")
    for item in datasets:
        digest = str(item.get("sha256", ""))
        if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
            raise ConfigError(f"Dataset {item.get('name')} needs a SHA-256 digest.")

    prompt = raw["prompt"]
    if prompt.get("chat_template_source") != "pinned_tokenizer_revision":
        raise ConfigError("The chat template must come from the pinned tokenizer revision.")
    if prompt.get("add_generation_prompt") is not True:
        raise ConfigError("The prompt must end at the assistant generation boundary.")
    if prompt.get("stop_conditions") != "not_applicable_sequence_scoring_without_generation":
        raise ConfigError("Candidate answers must be scored rather than generated.")
    schemes = prompt.get("schemes", {})
    if set(schemes) != {"primary", "transfer"}:
        raise ConfigError("Exactly primary and transfer schemes are required.")
    if [str(x) for x in schemes["primary"].get("symbols", [])] != ["A", "B"]:
        raise ConfigError("The primary neutral symbols must be A/B.")
    if [str(x) for x in schemes["transfer"].get("symbols", [])] != ["1", "2"]:
        raise ConfigError("The held-out neutral symbols must be 1/2.")
    prompt_text = json.dumps(prompt, sort_keys=True)
    if "Answer True" in prompt_text or "Answer False" in prompt_text:
        raise ConfigError("Countersemantic True/False answer instructions are forbidden.")

    cache = raw["cache"]
    if cache.get("activation_dtype") != "float16":
        raise ConfigError("Activation caches must use float16.")
    if int(cache.get("batch_size", 0)) < 1 or int(cache.get("chunk_size", 0)) < 1:
        raise ConfigError("Cache batch and chunk sizes must be positive.")

    analysis = raw["analysis"]
    fractions = [float(analysis[key]) for key in ("train_fraction", "dev_fraction", "test_fraction")]
    if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ConfigError("Train/dev/test fractions must be positive and sum to one.")
    expected = {
        "primary_scheme": "primary",
        "transfer_scheme": "transfer",
        "selection_metric": "min_primary_mapping_signed_standardized_separation",
        "primary_statistic": "mean_cross_partition_signed_standardized_separation",
        "permutation_unit": "training_group_sign_flip",
        "bootstrap_unit": "test_group_id",
    }
    for key, value in expected.items():
        if analysis.get(key) != value:
            raise ConfigError(f"analysis.{key} differs from the frozen v2 protocol.")
    partitions = int(analysis.get("training_partitions", 0))
    permutations = int(analysis.get("permutations", 0))
    bootstraps = int(analysis.get("bootstrap_replicates", 0))
    checkpoint_every = int(analysis.get("checkpoint_every_permutations", 0))
    if checkpoint_every < 1:
        raise ConfigError("analysis.checkpoint_every_permutations must be positive.")
    if mode == "full":
        if partitions != 8:
            raise ConfigError("Full v2 runs require exactly eight training partitions.")
        if permutations < 1000:
            raise ConfigError("Full v2 runs require at least 1,000 complete permutations.")
        if bootstraps < 2000:
            raise ConfigError("Full v2 runs require at least 2,000 group bootstraps.")
    elif partitions < 2 or permutations < 20 or bootstraps < 20:
        raise ConfigError("Smoke analysis requires >=2 partitions and >=20 null/bootstrap replicates.")
    if float(analysis.get("native_auroc_gate", 0.0)) != 0.70:
        raise ConfigError("The frozen native semantic AUROC gate is 0.70.")
    if float(analysis.get("permutation_p_gate", 0.0)) != 0.05:
        raise ConfigError("The frozen exact permutation p gate is 0.05.")
    if float(analysis.get("confidence_level", 0.0)) != 0.95:
        raise ConfigError("The frozen confidence level is 0.95.")
