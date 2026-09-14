from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Iterable

from .dataset_adapters import RelationPair


def _order(seed: int, namespace: str, identifier: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{identifier}".encode()).hexdigest()


def deterministic_group_split(rows: Iterable[RelationPair], seed: int,
                              fractions: tuple[float, float, float] = (.7, .1, .2)) -> list[RelationPair]:
    values = list(rows)
    groups = sorted({row.group_id for row in values}, key=lambda g: _order(seed, "group", g))
    count = len(groups); train_end = round(count * fractions[0]); val_end = train_end + round(count * fractions[1])
    assigned = {g: ("train" if i < train_end else "validation" if i < val_end else "test") for i, g in enumerate(groups)}
    return [replace(row, split=assigned[row.group_id]) for row in values]


def balanced_cap(rows: Iterable[RelationPair], caps: dict[str, int], seed: int) -> list[RelationPair]:
    buckets: dict[tuple[str, str], list[RelationPair]] = defaultdict(list)
    for row in rows: buckets[(row.split, row.semantic_label)].append(row)
    selected: list[RelationPair] = []
    for split in ("train", "validation", "test"):
        positives = buckets[(split, "support")]; negatives = buckets[(split, "attack")]
        limit = min(caps[split], len(positives), len(negatives))
        for label, bucket in (("support", positives), ("attack", negatives)):
            bucket.sort(key=lambda r: _order(seed, f"{split}:{label}", r.row_id))
            selected.extend(bucket[:limit])
    return sorted(selected, key=lambda r: (r.split, r.semantic_label, r.row_id))


def assert_split_isolation(rows: Iterable[RelationPair]) -> None:
    seen: dict[str, str] = {}
    for row in rows:
        prior = seen.setdefault(row.group_id, row.split)
        if prior != row.split: raise RuntimeError(f"Group leakage: {row.group_id}: {prior}/{row.split}")


def freeze_summary(rows: Iterable[RelationPair]) -> dict:
    values = list(rows); assert_split_isolation(values)
    return {"row_count": len(values), "group_count": len({r.group_id for r in values}),
            "counts": {f"{s}:{l}": n for (s,l),n in sorted(Counter((r.split,r.semantic_label) for r in values).items())},
            "rows": [row.public_record() for row in values]}
