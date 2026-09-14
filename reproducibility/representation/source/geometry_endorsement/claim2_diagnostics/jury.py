from __future__ import annotations

import itertools
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LoadedJuryModel:
    model: Any
    tokenizer: Any
    model_id: str
    model_revision: str
    tokenizer_revision: str
    family: str
    device_name: str
    gpu_memory_mib: int
    dtype: str
    attention_implementation: str
    loaded_at: float


def gpu_preflight(minimum_mib: int = 39000) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; jury execution requires an A100.")
    properties = torch.cuda.get_device_properties(0)
    name = str(properties.name)
    memory_mib = int(properties.total_memory // (1024 * 1024))
    if "A100" not in name or memory_mib < minimum_mib:
        raise RuntimeError(f"A100 with at least {minimum_mib} MiB required; observed {name} ({memory_mib} MiB).")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable.")
    return {"name": name, "memory_mib": memory_mib, "bf16": True}


def model_access_preflight(panel: list[dict[str, str]], token: str | None = None) -> pd.DataFrame:
    from huggingface_hub import HfApi

    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    rows = []
    for entry in panel:
        try:
            observed = api.model_info(entry["id"], revision=entry["revision"]).sha
            usable = observed == entry["revision"]
            error = "" if usable else f"resolved revision {observed}"
        except Exception as exc:  # access failures are recorded, never substituted
            observed = ""
            usable = False
            error = f"{type(exc).__name__}: {exc}"
        rows.append({**entry, "resolved_revision": observed, "usable": usable, "error": error})
    return pd.DataFrame(rows)


def load_jury_model(entry: dict[str, str], *, token: str | None = None) -> LoadedJuryModel:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gpu = gpu_preflight()
    credential = token or os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(entry["id"], revision=entry["revision"], token=credential, trust_remote_code=False)
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(f"{entry['id']} has no canonical chat template.")
    model = AutoModelForCausalLM.from_pretrained(
        entry["id"],
        revision=entry["revision"],
        token=credential,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.eval()
    devices = {str(parameter.device) for parameter in model.parameters()}
    if not devices or any(not device.startswith("cuda:") for device in devices):
        raise RuntimeError(f"Model offload is forbidden; observed devices {sorted(devices)}")
    attention = str(getattr(model.config, "_attn_implementation", "default"))
    return LoadedJuryModel(model, tokenizer, entry["id"], entry["revision"], entry["revision"], entry["family"], gpu["name"], gpu["memory_mib"], "bfloat16", attention, time.monotonic())


def evaluate_model_competence(scores: pd.DataFrame) -> pd.DataFrame:
    required = {"model_id", "item_id", "mapping", "semantic_margin", "gold_label"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Missing competence columns: {sorted(missing)}")
    rows = []
    for model_id, frame in scores.groupby("model_id", sort=False):
        pivot = frame.pivot(index="item_id", columns="mapping", values="semantic_margin")
        gold = frame.drop_duplicates("item_id").set_index("item_id")["gold_label"].astype(int)
        complete = pivot.dropna()
        mapping_consistency = float((np.sign(complete.iloc[:, 0]) == np.sign(complete.iloc[:, 1])).mean()) if len(complete) else 0.0
        mean_margin = complete.mean(axis=1)
        accuracy = float((np.sign(mean_margin) == gold.loc[mean_margin.index]).mean()) if len(mean_margin) else 0.0
        literal = frame["literal_first_minus_second_margin"] if "literal_first_minus_second_margin" in frame else frame["semantic_margin"]
        first_symbol_fraction = float((literal >= 0).mean()) if len(frame) else 1.0
        collapse = max(first_symbol_fraction, 1.0 - first_symbol_fraction)
        rows.append({"model_id": model_id, "mapping_consistency": mapping_consistency, "semantic_accuracy": accuracy, "maximum_symbol_fraction": collapse, "items": len(complete)})
    return pd.DataFrame(rows)


def apply_eligibility(competence: pd.DataFrame, rule: dict[str, float]) -> pd.DataFrame:
    result = competence.copy()
    result["eligible"] = (
        result["mapping_consistency"].ge(float(rule["minimum_mapping_consistency"]))
        & result["semantic_accuracy"].ge(float(rule["minimum_semantic_accuracy"]))
        & result["maximum_symbol_fraction"].le(float(rule["maximum_single_symbol_fraction"]))
    )
    return result


def hard_votes(scores: pd.DataFrame) -> pd.DataFrame:
    pivot = scores.pivot(index=["model_id", "family", "base_item_id"], columns="mapping", values="semantic_margin").reset_index()
    if not {"standard", "reversed"} <= set(pivot.columns):
        raise RuntimeError("Both jury mappings are required.")
    pivot["mapping_consistent"] = np.sign(pivot["standard"]) == np.sign(pivot["reversed"])
    pivot["vote"] = np.where(pivot["mapping_consistent"], np.sign((pivot["standard"] + pivot["reversed"]) / 2.0), np.nan)
    return pivot


def _binary_entropy(fraction: float) -> float:
    if not 0.0 < fraction < 1.0:
        return 0.0
    return float(-(fraction * np.log2(fraction) + (1.0 - fraction) * np.log2(1.0 - fraction)))


def aggregate_item_features(votes: pd.DataFrame, *, excluded_target_model: str) -> pd.DataFrame:
    pool = votes[votes["model_id"].ne(excluded_target_model)].copy()
    rows = []
    for base_item_id, frame in pool.groupby("base_item_id", sort=False):
        valid = frame.dropna(subset=["vote"])
        count = len(valid)
        fraction = float((valid["vote"] > 0).mean()) if count else np.nan
        majority = max(fraction, 1.0 - fraction) if count else np.nan
        family_votes = valid.groupby("family")["vote"].mean()
        family_fraction = float((family_votes > 0).mean()) if len(family_votes) else np.nan
        rows.append({
            "base_item_id": base_item_id,
            "leave_target_out_majority_fraction": majority,
            "leave_target_out_vote_margin": abs(2.0 * fraction - 1.0) if count else np.nan,
            "leave_target_out_vote_entropy": _binary_entropy(fraction) if count else np.nan,
            "valid_votes": count,
            "mapping_inconsistent_fraction": float((~frame["mapping_consistent"]).mean()),
            "family_balanced_majority_strength": max(family_fraction, 1.0 - family_fraction) if len(family_votes) else np.nan,
        })
    return pd.DataFrame(rows)


def five_of_seven_subsets(model_ids: list[str]) -> list[tuple[str, ...]]:
    if len(model_ids) != 7:
        raise ValueError("Five-of-seven robustness requires exactly seven frozen models.")
    return list(itertools.combinations(model_ids, 5))
