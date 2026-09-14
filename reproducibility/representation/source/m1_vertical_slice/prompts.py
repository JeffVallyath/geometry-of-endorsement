from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .config import canonical_digest
from .model_specs import LLAMA_MODEL_ID, get_model_spec


@dataclass(frozen=True)
class AnswerMapping:
    name: str
    supports: str
    opposes: str
    instruction: str


@dataclass(frozen=True)
class RenderedBatch:
    encoded: dict[str, Any]
    final_token_indices: Any
    rendered: list[str]
    candidate_texts: list[tuple[str, str]]


def mapping_for_item(item_id: str, scheme_name: str, prompt_config: dict[str, Any]) -> AnswerMapping:
    symbols = [str(x) for x in prompt_config["schemes"][scheme_name]["symbols"]]
    reverse = bool(hashlib.sha256(f"answer-map:{item_id}".encode()).digest()[0] & 1)
    supports, opposes = (symbols[1], symbols[0]) if reverse else (symbols[0], symbols[1])
    name = f"{scheme_name}:{'reversed' if reverse else 'standard'}"
    instruction = (
        f"Use {supports} when the named consideration supports the action in this situation, "
        f"and {opposes} when it opposes the action. Reply with only {symbols[0]} or {symbols[1]}."
    )
    return AnswerMapping(name, supports, opposes, instruction)


def messages_for(
    situation: str,
    consideration: str,
    mapping: AnswerMapping,
    scheme_name: str,
    representation: str,
    prompt_config: dict[str, Any],
    model_id: str = LLAMA_MODEL_ID,
) -> list[dict[str, str]]:
    system = str(prompt_config["system_template"])
    scheme = prompt_config["schemes"][scheme_name]
    fields = {
        "situation": situation,
        "consideration": consideration,
        "mapping_instruction": mapping.instruction,
    }
    template_key = {
        "joint": "joint_user_template",
        "situation": "situation_user_template",
        "consideration": "consideration_user_template",
    }.get(representation)
    if template_key is None:
        raise ValueError(f"Unknown representation: {representation}")
    body = str(scheme[template_key]).format(**fields)
    spec = get_model_spec(model_id)
    if spec.chat_policy == "native_system":
        return [{"role": "system", "content": system}, {"role": "user", "content": body}]
    if spec.chat_policy == "fold_system_into_user":
        return [{"role": "user", "content": f"{system}\n\n{body}"}]
    raise RuntimeError(f"Unsupported chat policy {spec.chat_policy!r}.")


def prompt_contract_hash(prompt_config: dict[str, Any]) -> str:
    return canonical_digest(prompt_config)


def render_batch(
    tokenizer: Any,
    records: list[dict[str, str]],
    prompt_config: dict[str, Any],
    model_id: str = LLAMA_MODEL_ID,
) -> RenderedBatch:
    rows: list[list[int]] = []
    rendered: list[str] = []
    candidates: list[tuple[str, str]] = []
    for record in records:
        mapping = mapping_for_item(record["item_id"], record["scheme"], prompt_config)
        messages = messages_for(
            record["situation"],
            record["consideration"],
            mapping,
            record["scheme"],
            record["representation"],
            prompt_config,
            model_id=model_id,
        )
        ids = list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=bool(
                    prompt_config["add_generation_prompt"]
                ),
            )
        )
        if not ids:
            raise RuntimeError("Chat template produced an empty prompt.")
        rows.append(ids)
        rendered.append(tokenizer.decode(ids, skip_special_tokens=False))
        candidates.append((mapping.supports, mapping.opposes))
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither pad nor EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded = tokenizer.pad({"input_ids": rows}, padding=True, return_tensors="pt")
    final = encoded["attention_mask"].sum(dim=1) - 1
    if (final < 0).any():
        raise RuntimeError("Final prompt-token position underflowed.")
    return RenderedBatch(encoded, final, rendered, candidates)
