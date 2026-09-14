from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FORBIDDEN_BLIND_TERMS = (
    "gate2_candidate_outcomes",
    "gate2_per_original_outcomes",
    "susceptibility",
    "geometry_analysis_report",
    "item_level_predictions_private",
    "predictor_ladder",
    "outcome_definition_sensitivity",
)


@dataclass(frozen=True)
class PromptContract:
    id: str
    symbols: tuple[str, str]
    supports_index: int

    @property
    def candidates(self) -> tuple[str, str]:
        return self.symbols

    def semantic_margin(self, first_minus_second: float) -> float:
        return float(first_minus_second if self.supports_index == 0 else -first_minus_second)


def contracts_from_config(config: dict[str, Any]) -> tuple[PromptContract, ...]:
    contracts = tuple(
        PromptContract(str(row["id"]), tuple(map(str, row["symbols"])), int(row["supports_index"]))
        for row in config["prompt_contracts"]
    )
    expected = ("ab_standard", "ab_reversed", "12_standard", "12_reversed", "direct_semantic")
    if tuple(value.id for value in contracts) != expected:
        raise RuntimeError("Prompt contract order changed.")
    return contracts


def build_messages(situation: str, consideration: str, contract: PromptContract, random_string: str | None = None) -> list[dict[str, str]]:
    supports = contract.symbols[contract.supports_index]
    opposes = contract.symbols[1 - contract.supports_index]
    suffix = "" if random_string is None else f"\n\nIgnore this random seed field: [{random_string}]"
    system = "Judge only whether the named consideration supports or opposes the action in the stated situation."
    user = (
        f"Situation and action: {situation}\nNamed consideration: {consideration}{suffix}\n\n"
        f"Use {supports} for Supports and {opposes} for Opposes. Reply with only {contract.symbols[0]} or {contract.symbols[1]}."
    )
    # A single user turn is used uniformly across the heterogeneous jury so
    # every pinned canonical template can render the identical task content.
    return [{"role": "user", "content": f"{system}\n\n{user}"}]


def build_target_messages(
    situation: str,
    consideration: str,
    contract: PromptContract,
    frozen_prompt_config: dict[str, Any],
    *,
    random_string: str | None = None,
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
) -> list[dict[str, str]]:
    """Render the target task from the repaired Gate 2 primary template."""
    supports = contract.symbols[contract.supports_index]
    opposes = contract.symbols[1 - contract.supports_index]
    instruction = (
        f"Use {supports} when the named consideration supports the action in this situation, "
        f"and {opposes} when it opposes the action. Reply with only {contract.symbols[0]} or {contract.symbols[1]}."
    )
    body = str(frozen_prompt_config["schemes"]["primary"]["joint_user_template"]).format(
        situation=situation,
        consideration=consideration,
        mapping_instruction=instruction,
    )
    if random_string is not None:
        body += f"\n\nIgnore this random seed field: [{random_string}]"
    system = str(frozen_prompt_config["system_template"])
    if model_id == "google/gemma-2-9b-it":
        return [{"role": "user", "content": f"{system}\n\n{body}"}]
    return [{"role": "system", "content": system}, {"role": "user", "content": body}]


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[list[int], str, str]:
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("Pinned tokenizer has no canonical chat template.")
    ids = list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    if not ids:
        raise RuntimeError("Chat rendering produced an empty prompt.")
    rendered = tokenizer.decode(ids, skip_special_tokens=False)
    return [int(value) for value in ids], rendered, hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def assert_blind_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    lowered = resolved.as_posix().lower()
    hit = next((term for term in FORBIDDEN_BLIND_TERMS if term in lowered), None)
    if hit:
        raise RuntimeError(f"Blind feature firewall rejected path containing {hit!r}.")
    return resolved


class BlindFeatureReader:
    """The only filesystem reader exposed to blind feature generation."""

    def read_bytes(self, path: str | Path) -> bytes:
        return assert_blind_path(path).read_bytes()

    def resolve(self, path: str | Path) -> Path:
        return assert_blind_path(path)
