from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .config import canonical_digest


@dataclass(frozen=True)
class AnswerMapping:
    name: str
    true_answer: str
    false_answer: str
    instruction: str


@dataclass(frozen=True)
class RenderedBatch:
    encoded: dict[str, Any]
    final_token_indices: Any
    rendered: list[str]
    candidate_texts: list[tuple[str, str]]
    mapping_names: list[str]


def mapping_for_item(item_id: str, scheme_name: str, prompt_config: dict[str, Any]) -> AnswerMapping:
    symbols = [str(value) for value in prompt_config["schemes"][scheme_name]["symbols"]]
    reverse = bool(hashlib.sha256(f"answer-map:{item_id}".encode()).digest()[0] & 1)
    true_answer, false_answer = (symbols[1], symbols[0]) if reverse else (symbols[0], symbols[1])
    name = f"{scheme_name}:{'reversed' if reverse else 'standard'}"
    instruction = (
        f"Use {true_answer} when the statement is factually true, and {false_answer} "
        f"when it is factually false. Reply with only {symbols[0]} or {symbols[1]}."
    )
    return AnswerMapping(name, true_answer, false_answer, instruction)


def messages_for(
    statement: str,
    mapping: AnswerMapping,
    scheme_name: str,
    prompt_config: dict[str, Any],
) -> list[dict[str, str]]:
    system = str(prompt_config["system_template"])
    scheme = prompt_config["schemes"][scheme_name]
    body = str(scheme["user_template"]).format(
        statement=statement,
        mapping_instruction=mapping.instruction,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": body}]


def prompt_contract_hash(prompt_config: dict[str, Any]) -> str:
    return canonical_digest(prompt_config)


def render_batch(
    tokenizer: Any,
    records: list[dict[str, str]],
    prompt_config: dict[str, Any],
) -> RenderedBatch:
    token_rows: list[list[int]] = []
    rendered: list[str] = []
    candidates: list[tuple[str, str]] = []
    mapping_names: list[str] = []
    for record in records:
        mapping = mapping_for_item(record["item_id"], record["scheme"], prompt_config)
        messages = messages_for(record["statement"], mapping, record["scheme"], prompt_config)
        token_ids = list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=bool(prompt_config["add_generation_prompt"]),
            )
        )
        if not token_ids:
            raise RuntimeError("Chat template produced an empty prompt.")
        token_rows.append(token_ids)
        rendered.append(tokenizer.decode(token_ids, skip_special_tokens=False))
        candidates.append((mapping.true_answer, mapping.false_answer))
        mapping_names.append(mapping.name)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither a pad token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded = tokenizer.pad({"input_ids": token_rows}, padding=True, return_tensors="pt")
    final_indices = encoded["attention_mask"].sum(dim=1) - 1
    if (final_indices < 0).any():
        raise RuntimeError("Final prompt-token position underflowed.")
    return RenderedBatch(encoded, final_indices, rendered, candidates, mapping_names)
