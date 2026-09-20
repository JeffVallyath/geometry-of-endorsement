"""Portable paths and strict artifact loading for offline replay."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_path(relative: str, root: Path = ROOT) -> Path:
    posix = PurePosixPath(relative)
    if (not relative or posix.is_absolute() or ".." in posix.parts
            or "\\" in relative or ":" in relative or posix.as_posix() != relative):
        raise ValueError("Artifact paths must be normalized repository-relative paths")
    path = root.joinpath(*posix.parts)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes the repository")
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def new_output(path: Path) -> Path:
    path = path.resolve()
    if path == ROOT or path == Path(path.anchor) or ROOT.is_relative_to(path):
        raise ValueError("Replay output must not be a repository or filesystem root")
    if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / "reproduced"):
        raise ValueError("In-repository replay output must be under reproduced, never source or evidence")
    path.mkdir(parents=True, exist_ok=True)
    return path
