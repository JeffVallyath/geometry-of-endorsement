from __future__ import annotations

import hashlib
from dataclasses import dataclass


FORBIDDEN_ABLATED_WORDS = frozenset({"support", "supports", "attack", "attacks", "oppose", "opposes"})

TEMPLATES = {
    "EXPLICIT_RELATION": (
        "Proposition A:\n{source}\n\nProposition B:\n{target}\n\n"
        "Does Proposition A support or attack Proposition B?\n{mapping_instruction}\nAnswer:"
    ),
    "LEXICALLY_ABLATED_RELATION": (
        "Read the two claims below as a directed pair.\n\nFirst claim: {source}\n\n"
        "Claim being evaluated: {target}\n\nDoes the first claim provide a reason in favor "
        "of the claim being evaluated, or does it count against it?\n{mapping_instruction}\nResponse:"
    ),
}


@dataclass(frozen=True)
class RenderedPrompt:
    template_id: str
    mapping_id: str
    text: str
    semantic_label: str
    physical_token_label: str
    activation_position: str = "final_prompt_position_immediately_before_candidate_answer_sequence"


def render_prompt(template_id: str, source: str, target: str, mapping: object,
                  semantic_label: str) -> RenderedPrompt:
    template = TEMPLATES[template_id]
    instruction = mapping.instruction
    text = template.format(source=source, target=target, mapping_instruction=instruction)
    physical = mapping.positive_token if semantic_label == "positive" else mapping.negative_token
    return RenderedPrompt(template_id, mapping.mapping_id, text, semantic_label, physical)


def validate_templates() -> None:
    ablated = TEMPLATES["LEXICALLY_ABLATED_RELATION"].lower()
    tokens = set(ablated.replace("?", "").replace(".", "").split())
    present = sorted(tokens & FORBIDDEN_ABLATED_WORDS)
    if present: raise RuntimeError(f"Lexical ablation violation: {present}")
    if TEMPLATES["EXPLICIT_RELATION"] == TEMPLATES["LEXICALLY_ABLATED_RELATION"]:
        raise RuntimeError("Prompt templates are not structurally distinct")


def prompt_contract_sha256() -> str:
    validate_templates()
    body = "\n\x1e\n".join(f"{k}\n{TEMPLATES[k]}" for k in sorted(TEMPLATES))
    return hashlib.sha256(body.encode()).hexdigest()
