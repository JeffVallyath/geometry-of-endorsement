from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .data import sha256_file


class ChunkStore:
    """Atomic, hash-verified activation chunks with optional Drive mirroring."""

    def __init__(self, output_dir: str | Path, signature: str, persistent_dir: str | Path | None = None):
        self.output_dir = Path(output_dir)
        self.chunk_dir = self.output_dir / "activation_chunks"
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.persistent_dir = Path(persistent_dir) if persistent_dir else None
        self.persistent_chunks = self.persistent_dir / "activation_chunks" if self.persistent_dir else None
        if self.persistent_chunks:
            self.persistent_chunks.mkdir(parents=True, exist_ok=True)
        self.signature = signature
        self.manifest_path = self.output_dir / "cache_manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        candidates = [self.manifest_path]
        if self.persistent_dir:
            candidates.append(self.persistent_dir / "cache_manifest.json")
        for path in candidates:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("signature") != self.signature:
                    raise RuntimeError(f"Cache signature mismatch in {path}; use a new output directory.")
                return data
        return {"signature": self.signature, "chunks": {}}

    @staticmethod
    def filename(index: int) -> str:
        return f"chunk_{index:05d}.npz"

    def load(self, index: int, expected_record_ids: list[str]) -> dict[str, np.ndarray] | None:
        name = self.filename(index)
        entry = self.manifest["chunks"].get(name)
        if not entry:
            return None
        local = self.chunk_dir / name
        if not local.exists() and self.persistent_chunks:
            persisted = self.persistent_chunks / name
            if persisted.exists():
                shutil.copy2(persisted, local)
        if not local.exists() or sha256_file(local) != entry["sha256"]:
            raise RuntimeError(f"Recorded cache chunk is missing or corrupt: {name}")
        with np.load(local, allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        if str(data["signature"].item()) != self.signature:
            raise RuntimeError(f"Embedded signature mismatch: {name}")
        actual_ids = [str(value) for value in data["record_ids"]]
        if actual_ids != expected_record_ids:
            raise RuntimeError(f"Record order mismatch in cache chunk: {name}")
        return data

    def save(self, index: int, record_ids: list[str], **arrays: np.ndarray) -> Path:
        name = self.filename(index)
        target = self.chunk_dir / name
        partial = target.with_suffix(".npz.part")
        payload = {
            "signature": np.asarray(self.signature),
            "record_ids": np.asarray(record_ids),
            **arrays,
        }
        with partial.open("wb") as handle:
            np.savez(handle, **payload)
        os.replace(partial, target)
        digest = sha256_file(target)
        self.manifest["chunks"][name] = {
            "sha256": digest,
            "records": len(record_ids),
            "bytes": target.stat().st_size,
        }
        if self.persistent_chunks:
            persisted = self.persistent_chunks / name
            persisted_partial = persisted.with_suffix(".npz.part")
            shutil.copy2(target, persisted_partial)
            if sha256_file(persisted_partial) != digest:
                persisted_partial.unlink(missing_ok=True)
                raise RuntimeError(f"Persistent copy verification failed: {name}")
            os.replace(persisted_partial, persisted)
        self._write_manifest()
        return target

    def _write_manifest(self) -> None:
        payload = json.dumps(self.manifest, indent=2, sort_keys=True) + "\n"
        targets = [self.manifest_path]
        if self.persistent_dir:
            targets.append(self.persistent_dir / "cache_manifest.json")
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".part")
            partial.write_text(payload, encoding="utf-8")
            os.replace(partial, target)


def concatenate_chunks(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not chunks:
        raise ValueError("No chunks to concatenate.")
    keys = [key for key in chunks[0] if key not in {"signature"}]
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in keys}
