from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .util import canonical_digest, require_exact_keys


@dataclass(frozen=True)
class Claim2Config:
    path: Path
    raw: dict[str, Any]
    digest: str

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise RuntimeError(f"Missing configuration section: {name}")
        return value


def load_config(path: str | Path) -> Claim2Config:
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("Claim 2 configuration must be a mapping.")
    require_exact_keys(
        raw,
        {
            "schema_version", "protocol", "status_scope", "run", "model",
            "inputs", "frozen_geometry", "sampling", "generation", "controls",
            "review", "inference", "prediction", "specificity", "decision",
            "protected_path_markers",
        },
        "configuration",
    )
    if raw["schema_version"] != 1 or raw["status_scope"] != "development_only":
        raise RuntimeError("Only development-only Claim 2 schema v1 is supported.")
    if raw["run"].get("allow_remote_push") or raw["run"].get("allow_llm_judge"):
        raise RuntimeError("Remote push and LLM judging must remain disabled.")
    if int(raw["model"]["selected_layer"]) != 19:
        raise RuntimeError("The frozen Llama selected layer must remain 19.")
    sampling = raw["sampling"]
    if int(sampling["representative_items"]) != 80 or int(
        sampling["disagreement_items"]
    ) != 32:
        raise RuntimeError("The frozen 80/32 sampling contract changed.")
    if not raw["generation"].get("consideration_must_match_bytes"):
        raise RuntimeError("Consideration byte preservation must be enabled.")
    if raw["generation"]["families"] != [
        "syntactic_restructuring",
        "information_order",
        "lexical_substitution",
        "sentence_structure_or_voice",
    ]:
        raise RuntimeError("The four frozen rephrasing families changed.")
    if int(raw["controls"]["count_per_type"]) != 11:
        raise RuntimeError("The frozen hidden-control allocation changed.")
    if raw["inference"]["mappings"] != ["standard", "reversed"]:
        raise RuntimeError("Both frozen neutral answer mappings are required.")
    prediction = raw["prediction"]
    if len(prediction["seeds"]) != int(prediction["outer_repeats"]):
        raise RuntimeError("One fixed seed is required per outer repeat.")
    if int(prediction["bootstrap_replicates"]) < 2000:
        raise RuntimeError("At least 2,000 base-item bootstrap replicates are required.")
    if prediction["primary_stratum"] != "representative":
        raise RuntimeError("The representative stratum must remain primary.")
    if int(raw["specificity"]["random_direction_count"]) != 20:
        raise RuntimeError("The frozen random-direction count changed.")
    if int(raw["specificity"]["permuted_dim_count"]) != 20:
        raise RuntimeError("The frozen permuted-direction count changed.")
    decision = raw["decision"]
    probability_keys = (
        "minimum_joint_candidate_acceptance",
        "minimum_representative_items_with_multiple_accepted_fraction",
        "maximum_overall_reviewer_uncertainty_rate",
        "maximum_overall_reviewer_disagreement_rate",
        "minimum_identity_control_acceptance",
        "minimum_trivial_control_acceptance",
        "maximum_semantic_change_control_acceptance",
    )
    if any(
        not 0 <= float(decision[key]) <= 1
        for key in probability_keys
    ):
        raise RuntimeError("Every decision-rate threshold must be in [0, 1].")
    if int(decision["minimum_distinct_flipping_items"]) < 1:
        raise RuntimeError("The event-count decision threshold must be positive.")
    return Claim2Config(source, raw, canonical_digest(raw))
