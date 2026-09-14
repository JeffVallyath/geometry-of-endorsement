from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes_atomic(path: str | Path, value: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(value)
    os.replace(temporary, target)
    return target


def write_json(path: str | Path, value: Any) -> Path:
    return write_bytes_atomic(path, canonical_json_bytes(value))


def write_text(path: str | Path, value: str) -> Path:
    return write_bytes_atomic(path, (value.rstrip() + "\n").encode())


def artifact_record(path: str | Path, role: str) -> dict[str, Any]:
    target = Path(path).resolve()
    return {"absolute_path": str(target), "bytes": target.stat().st_size,
            "sha256": sha256_file(target), "role": role}


def verify_manifest(manifest: Mapping[str, Any]) -> int:
    checked = 0
    for record in manifest.get("files", []):
        path = Path(record["absolute_path"])
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"Artifact missing or size mismatch: {path}")
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Artifact hash mismatch: {path}")
        checked += 1
    return checked
