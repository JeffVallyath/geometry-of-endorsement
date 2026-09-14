from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class AnswerMapping:
    mapping_id: str
    positive_token: str
    negative_token: str

    @property
    def instruction(self) -> str:
        return (f"Use {self.positive_token} for the positive relation and "
                f"{self.negative_token} for the negative relation. Return only one candidate.")


MAPPINGS = (
    AnswerMapping("MAPPING_AB_STANDARD", "A", "B"),
    AnswerMapping("MAPPING_AB_REVERSED", "B", "A"),
    AnswerMapping("MAPPING_12_STANDARD", "1", "2"),
    AnswerMapping("MAPPING_12_REVERSED", "2", "1"),
)


def factorial_rows() -> list[dict[str, str]]:
    return [{"mapping_id": m.mapping_id, "semantic_label": semantic,
             "physical_token_label": m.positive_token if semantic == "positive" else m.negative_token}
            for m in MAPPINGS for semantic in ("positive", "negative")]


def assert_semantic_token_orthogonality() -> None:
    rows = factorial_rows()
    for family in ({"A", "B"}, {"1", "2"}):
        subset = [r for r in rows if r["physical_token_label"] in family]
        table = {(r["semantic_label"], r["physical_token_label"]) for r in subset}
        expected = {(s, t) for s in ("positive", "negative") for t in family}
        if table != expected: raise RuntimeError(f"Factorial not crossed for {family}: {table}")


def candidate_token_sequences(tokenize: Callable[[str], list[int]]) -> dict[str, list[int]]:
    result = {token: list(tokenize(token)) for token in ("A", "B", "1", "2")}
    if any(not sequence for sequence in result.values()):
        raise RuntimeError("Candidate tokenization produced an empty sequence")
    return result


def semantic_margin(mapping: AnswerMapping, sequence_log_probabilities: dict[str, float]) -> float:
    return float(sequence_log_probabilities[mapping.positive_token] - sequence_log_probabilities[mapping.negative_token])
