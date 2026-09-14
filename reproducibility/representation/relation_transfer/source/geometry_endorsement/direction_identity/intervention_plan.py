from __future__ import annotations

import hashlib
from typing import Iterable


FULL_DIRECTIONS = ("VALUEPRISM_DIM", "DIM_RESIDUAL", "D_PHYSICAL_TOKEN", "AMPEREPP_RELATION")
CONTROL_DIRECTIONS = ("FACTUAL_TRUE_FALSE", "SENTIMENT", "POOLED_EXTERNAL", "RANDOM_ORTHOGONAL", "NO_INTERVENTION")
FULL_DOSES = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
CONTROL_DOSES = (-1.0, 0.0, 1.0)


def content_hash_sample(ids: Iterable[str], count: int, namespace: str) -> list[str]:
    ordered = sorted(set(ids), key=lambda item: hashlib.sha256(f"{namespace}:{item}".encode()).hexdigest())
    if len(ordered) < count: raise RuntimeError(f"Insufficient {namespace} units: {len(ordered)} < {count}")
    return ordered[:count]


def intervention_grid() -> list[dict[str, object]]:
    rows=[]
    for direction in FULL_DIRECTIONS:
        rows.extend({"direction":direction,"dose":dose} for dose in FULL_DOSES)
    for direction in CONTROL_DIRECTIONS:
        rows.extend({"direction":direction,"dose":dose} for dose in CONTROL_DOSES)
    return rows
