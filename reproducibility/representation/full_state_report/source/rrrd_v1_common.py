#!/usr/bin/env python3
"""Shared constants and helpers for RELATION_READOUT_REPORT_DISCRIMINATION_V1.

Every script of the study imports its frozen constants from here so the freeze,
the GPU runner, and the analyses cannot drift from each other. Nothing in this
module reads an outcome.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any, Iterable

STUDY_ID = "RELATION_READOUT_REPORT_DISCRIMINATION_V1"
SCHEMA_PREFIX = "RRRD_V1"
ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = ROOT / "reports" / "relation_readout_report_discrimination_v1"
FREEZE_ROOT = STUDY_ROOT / "freeze"
TRACK_A_FREEZE = ROOT / "reports" / "relation_operator_transverse_fragility_v1" / "track_a_v2" / "freeze"

ACTORS: dict[str, dict[str, Any]] = {
    "llama": {
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "model_revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "primary_layer": 19,
        "band": list(range(14, 25)),
        "hidden_size": 4096,
        "raw_relation_dim_sha256": "f44fb897ab2617abd95ca5a2f8e67dd2626dd75099a9cdf3973033b3e1b64c99",
        "jointly_residualized_relation_sha256": "f9bdfb2dd428c50125543770d0c7935f32cdba50740ac267a703a93d5b88319c",
        "chat_policy": "NATIVE_SYSTEM",
    },
    "gemma": {
        "model_id": "google/gemma-2-9b-it",
        "model_revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
        "primary_layer": 27,
        "band": list(range(22, 33)),
        "hidden_size": 3584,
        "raw_relation_dim_sha256": "81c704de24ffc90b468f13bbf494638804f7fafa4b24f2f51d6bb64d2dc1c4e7",
        "jointly_residualized_relation_sha256": "f8484d17ed9abf50a21046c30aafe33454b34ba19910ed2ff01c8b42e4f26dea",
        "chat_policy": "FOLD_SYSTEM_INTO_USER",
    },
}

ACTIVATION_POSITION = "final prompt position immediately before the candidate answer sequence"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260902_17
FOLD_SEED = 20260902_5
DEMO_HASH_SALT = "RRRD_V1_DEMO|"
PRACTICAL_GATE = 0.70
DIRECT_COMPETENCE_GATE = 0.80
DIRECT_COMPETENCE_CI_LOW = 0.70
AGREEMENT_GATE = 0.90
MAPPINGS = ("MAPPING_AB_STANDARD", "MAPPING_AB_REVERSED", "MAPPING_12_STANDARD", "MAPPING_12_REVERSED")
MAPPING_SYMBOLS = {
    "MAPPING_AB_STANDARD": ("A", "B"),
    "MAPPING_AB_REVERSED": ("B", "A"),
    "MAPPING_12_STANDARD": ("1", "2"),
    "MAPPING_12_REVERSED": ("2", "1"),
}
# (entailed_symbol, not_entailed_symbol) under each mapping; matches the corpus
# convention that the standard mapping assigns the first symbol to "Yes".
F1_SEMANTICS = ("ENTAILED", "NOT_ENTAILED")
F2_LABELS = ("ENTAILED", "NOT_ENTAILED")
F1_TASK_LINE = (
    "Task: Decide whether the CLAIM is entailed by the CONTEXT. Use only what the "
    "context states. Follow the answer mapping shown below and output only the answer symbol."
)
F2_TASK_LINE = (
    "Task: Decide whether the CLAIM is entailed by the CONTEXT. Use only what the "
    "context states. Reply with exactly ENTAILED or NOT_ENTAILED."
)
SHOT_SEPARATOR = "\n\n"  # C9: "\n\n".join(shots) + "\n\n" + test
F2_MAX_NEW_TOKENS = 8

QUESTION_RX = re.compile(r"^Question: Is it established that (.+?) (supports|opposes) the proposition that (.+)\?$")


# ---------------------------------------------------------------- helpers
def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=_default) + "\n").encode("utf-8")


def _default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.bool_,)):
            return bool(value)
    except ImportError:  # pragma: no cover
        pass
    return str(value)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def stable_hash(text: str) -> str:
    return sha256_text(text)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def jsonl_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False, default=_default) + "\n" for r in records).encode("utf-8")


def write_create_only(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_bytes(rows: list[dict[str, Any]], columns: list[str]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue().encode("utf-8")


def manifest_for(root: Path, exclude: Iterable[str] = ("artifact_manifest.json",)) -> list[dict[str, Any]]:
    out = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.name not in set(exclude):
            out.append({"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    return out


# ---------------------------------------------------------------- corpus parsing
def parse_question(question_text: str) -> tuple[str, str, str]:
    m = QUESTION_RX.match(question_text)
    if not m:
        raise ValueError(f"question does not match the frozen pattern: {question_text!r}")
    return m.group(1), m.group(2), m.group(3)


def declarative_claim(record: dict[str, Any]) -> str:
    holder, verb, target = parse_question(record["question_text"])
    if holder != record["holder_name"] or target != record["target_text"]:
        raise ValueError(f"claim reconstruction mismatch for {record['example_id']}")
    if verb.upper().rstrip("S") != record["queried_operator"]:
        raise ValueError(f"operator mismatch for {record['example_id']}")
    return f"{holder} {verb} the proposition that {target}."


def context_block(record: dict[str, Any]) -> str:
    text = record["prompt_text"]
    if not text.startswith("Context:\n") or "\n\nQuestion:" not in text:
        raise ValueError(f"prompt lacks the frozen Context/Question layout: {record['example_id']}")
    return text[len("Context:\n") : text.index("\n\nQuestion:")]


def render_f1_example(context: str, claim: str, mapping_id: str, answer_symbol: str | None) -> str:
    sym_e, sym_n = MAPPING_SYMBOLS[mapping_id]
    # options are listed in physical symbol order (A before B, 1 before 2), as in the corpus
    ordered = sorted([(sym_e, F1_SEMANTICS[0]), (sym_n, F1_SEMANTICS[1])], key=lambda kv: kv[0])
    lines = ["Context:", context, "", "Claim:", claim, ""] + [f"{s}) {sem}" for s, sem in ordered] + ["Answer:"]
    text = "\n".join(lines)
    if answer_symbol is not None:
        text += f" {answer_symbol}"
    return text


def render_f2_example(context: str, claim: str, answer_label: str | None) -> str:
    lines = ["Context:", context, "", "Claim:", claim, "", "Answer:"]
    text = "\n".join(lines)
    if answer_label is not None:
        text += f" {answer_label}"
    return text


def f1_symbol_for(mapping_id: str, entailment_label: str) -> str:
    sym_e, sym_n = MAPPING_SYMBOLS[mapping_id]
    return sym_e if entailment_label == "ENTAILED" else sym_n


def render_f1_prompt(demos: list[dict[str, Any]], context: str, claim: str, mapping_id: str) -> str:
    shots = [render_f1_example(d["context"], d["claim"], mapping_id, f1_symbol_for(mapping_id, d["entailment_label"])) for d in demos]
    return F1_TASK_LINE + SHOT_SEPARATOR + SHOT_SEPARATOR.join(shots) + SHOT_SEPARATOR + render_f1_example(context, claim, mapping_id, None)


def render_f2_prompt(demos: list[dict[str, Any]], context: str, claim: str) -> str:
    shots = [render_f2_example(d["context"], d["claim"], d["entailment_label"]) for d in demos]
    return F2_TASK_LINE + SHOT_SEPARATOR + SHOT_SEPARATOR.join(shots) + SHOT_SEPARATOR + render_f2_example(context, claim, None)


# ---------------------------------------------------------------- ValuePrism (M1) prompt contract
M1_SYSTEM_TEMPLATE = (
    "You judge how one named moral consideration bears on an action in context. "
    "Follow the requested answer mapping exactly."
)
M1_JOINT_USER_TEMPLATE = (
    "Situation and action: {situation}\n"
    "Named consideration: {consideration}\n"
    "Question: In this specific context, does the named consideration support or oppose the action?\n"
    "{mapping_instruction}"
)
M1_SYMBOLS = ("A", "B")


def m1_mapping_for_item(item_id: str) -> dict[str, str]:
    reverse = bool(hashlib.sha256(f"answer-map:{item_id}".encode()).digest()[0] & 1)
    supports, opposes = (M1_SYMBOLS[1], M1_SYMBOLS[0]) if reverse else (M1_SYMBOLS[0], M1_SYMBOLS[1])
    name = f"primary:{'reversed' if reverse else 'standard'}"
    instruction = (
        f"Use {supports} when the named consideration supports the action in this situation, "
        f"and {opposes} when it opposes the action. Reply with only {M1_SYMBOLS[0]} or {M1_SYMBOLS[1]}."
    )
    return {"name": name, "supports": supports, "opposes": opposes, "instruction": instruction}


def m1_messages(situation: str, consideration: str, mapping: dict[str, str], chat_policy: str) -> list[dict[str, str]]:
    body = M1_JOINT_USER_TEMPLATE.format(situation=situation, consideration=consideration, mapping_instruction=mapping["instruction"])
    if chat_policy == "NATIVE_SYSTEM":
        return [{"role": "system", "content": M1_SYSTEM_TEMPLATE}, {"role": "user", "content": body}]
    if chat_policy == "FOLD_SYSTEM_INTO_USER":
        return [{"role": "user", "content": f"{M1_SYSTEM_TEMPLATE}\n\n{body}"}]
    raise ValueError(chat_policy)


def ids_sha256(ids: list[int]) -> str:
    """The frozen causal-baseline prompt hash: sha256 of 8-byte little-endian signed token ids."""
    return hashlib.sha256(b"".join(int(v).to_bytes(8, "little", signed=True) for v in ids)).hexdigest()
