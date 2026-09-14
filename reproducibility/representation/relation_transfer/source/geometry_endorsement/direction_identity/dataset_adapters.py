from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class RelationPair:
    row_id: str
    dataset_id: str
    split: str
    group_id: str
    source_text: str
    target_text: str
    semantic_label: str
    source_ref: str

    def public_record(self) -> dict[str, str]:
        value = asdict(self)
        value.pop("source_text")
        value.pop("target_text")
        value["source_text_sha256"] = hashlib.sha256(self.source_text.encode()).hexdigest()
        value["target_text_sha256"] = hashlib.sha256(self.target_text.encode()).hexdigest()
        return value


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def load_amperepp(root: str | Path) -> list[RelationPair]:
    base = Path(root)
    rows: list[RelationPair] = []
    seen_pairs: set[tuple[str,str,str]] = set()
    split_names = {"train": "train", "val": "validation", "test": "test"}
    for source_split, split in split_names.items():
        path = base / f"ampere_{source_split}.jsonl"
        if not path.is_file(): raise RuntimeError(f"PRIMARY_DATASET_BLOCKED: missing {path}")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                document_id = str(record["doc_id"])
                propositions = [str(text).strip() for text in record["text"]]
                for relation_number, relation in enumerate(record["relations"]):
                    label = str(relation["type"]).lower()
                    if label not in {"support", "attack"}: continue
                    source_index, target_index = int(relation["tail"]), int(relation["head"])
                    if source_index == target_index or min(source_index, target_index) < 0:
                        continue
                    try: source, target = propositions[source_index], propositions[target_index]
                    except IndexError as error: raise RuntimeError(f"PRIMARY_DATASET_BLOCKED: invalid relation in {path}:{line_number}") from error
                    if not source or not target: continue
                    pair=(source,target,label)
                    if pair in seen_pairs: continue
                    seen_pairs.add(pair)
                    row_id = _stable_id("AMPERE++", document_id, str(source_index), str(target_index), label)
                    rows.append(RelationPair(row_id, "AMPERE++", split, document_id,
                        source, target, label, f"{path.name}:{line_number}:relation:{relation_number}"))
    return rows


def load_abstrct(root: str | Path) -> list[RelationPair]:
    base = Path(root) / "AbstRCT_corpus" / "data"
    rows: list[RelationPair] = []
    seen_row_ids: set[str] = set()
    split_map = {"train": "train", "dev": "validation", "test": "test"}
    for source_split, split in split_map.items():
        for annotation in sorted((base / source_split).rglob("*.ann")):
            group_id = annotation.stem
            entities: dict[str, str] = {}
            relations: list[tuple[str, str, str, int]] = []
            for line_number, line in enumerate(annotation.read_text(encoding="utf-8").splitlines(), start=1):
                fields = line.split("\t")
                if fields[0].startswith("T") and len(fields) >= 3: entities[fields[0]] = fields[2].strip()
                elif fields[0].startswith("R") and len(fields) >= 2:
                    tokens = fields[1].split()
                    raw_label=tokens[0].lower() if tokens else ""
                    if len(tokens) >= 3 and raw_label in {"support", "attack", "partial-attack"}:
                        label="support" if raw_label=="support" else "attack"
                        relations.append((label, tokens[1].split(":",1)[1], tokens[2].split(":",1)[1], line_number))
            for label, source_id, target_id, line_number in relations:
                source, target = entities.get(source_id, ""), entities.get(target_id, "")
                if not source or not target: continue
                row_id = _stable_id("AbstRCT", group_id, source_id, target_id, label)
                if row_id in seen_row_ids: continue
                seen_row_ids.add(row_id)
                rows.append(RelationPair(row_id, "AbstRCT", split, group_id, source, target,
                    label, f"{annotation.relative_to(base)}:{line_number}"))
    return rows


def iter_aries_binary(csv_path: str | Path, dataset_id: str, aries_name: str) -> Iterator[RelationPair]:
    csv.field_size_limit(2**31 - 1)
    seen: set[tuple[str, str, str]] = set()
    with Path(csv_path).open(encoding="utf-8", newline="") as handle:
        for source_row, record in enumerate(csv.DictReader(handle)):
            if record["data_source"] != aries_name or record["relations"] not in {"1", "2"}: continue
            source, target = record["proposition_1"].strip(), record["proposition_2"].strip()
            if not source or not target or source == target: continue
            label = "support" if record["relations"] == "1" else "attack"
            pair_key = (source, target, label)
            if pair_key in seen: continue
            seen.add(pair_key)
            group_id = _stable_id(dataset_id, "group", record["argument"])
            row_id = _stable_id(dataset_id, str(source_row), source, target, label)
            yield RelationPair(row_id, dataset_id, "unassigned", group_id, source, target,
                label, f"data.csv:{source_row + 2}")


def assert_unique_pairs(rows: Iterable[RelationPair]) -> None:
    ids: set[str] = set()
    text_pairs: set[tuple[str, str, str]] = set()
    for row in rows:
        if row.row_id in ids: raise RuntimeError(f"Duplicate row ID: {row.row_id}")
        key = (row.source_text, row.target_text, row.semantic_label)
        if key in text_pairs: raise RuntimeError(f"Duplicate pair: {row.row_id}")
        ids.add(row.row_id); text_pairs.add(key)
