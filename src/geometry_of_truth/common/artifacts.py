from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def repository_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "artifacts").is_dir():
            return candidate
    installed = Path(__file__).resolve()
    for candidate in installed.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "artifacts").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the geometry-of-endorsement repository root")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_files(root: str | Path, expected: dict[str, str]) -> list[dict[str, Any]]:
    base = Path(root)
    rows = []
    for relative, digest in expected.items():
        path = base / relative
        actual = sha256_file(path) if path.is_file() else None
        rows.append({"file": relative, "expected SHA-256": digest, "actual SHA-256": actual, "pass": actual == digest})
    if not all(row["pass"] for row in rows):
        raise RuntimeError("Artifact verification failed")
    return rows
