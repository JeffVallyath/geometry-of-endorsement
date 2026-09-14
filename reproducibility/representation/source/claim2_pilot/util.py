from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


def immutable_write(path: str | Path, payload: bytes) -> str:
    target = Path(path)
    digest = sha256_bytes(payload)
    if target.exists():
        if sha256_file(target) != digest:
            raise RuntimeError(f"Immutable artifact differs: {target}")
        return digest
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with partial.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, target)
    if sha256_file(target) != digest:
        raise RuntimeError(f"Immutable artifact verification failed: {target}")
    return digest


def immutable_json(path: str | Path, value: Any) -> str:
    return immutable_write(path, canonical_json_bytes(value))


def require_exact_keys(value: dict[str, Any], required: Iterable[str], context: str) -> None:
    missing = set(required) - set(value)
    if missing:
        raise RuntimeError(f"{context} lacks required keys: {sorted(missing)}")
