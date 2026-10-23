from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from .config import RunConfig
from .data import DatasetArtifact


def collect_environment(
    repo_root: str | Path,
    config: RunConfig,
    model_revision: str | None,
    datasets: list[DatasetArtifact],
    gpu: dict[str, Any] | None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config_path = Path(config.path).resolve()
    try:
        public_config_path = config_path.relative_to(root).as_posix()
    except ValueError:
        public_config_path = config_path.name
    packages = {}
    for name in ("torch", "transformers", "accelerate", "huggingface-hub", "numpy", "pandas", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "config": {"public_path": public_config_path, "sha256": config.digest, "mode": config.mode},
        "model": {
            "id": config.model["id"],
            "requested_revision": config.model["revision"],
            "resolved_revision": model_revision,
            "torch_dtype": config.model["torch_dtype"],
            "quantization": config.model["quantization"],
        },
        "datasets": [
            {"name": artifact.name, "sha256": artifact.sha256, "rows": artifact.rows}
            for artifact in datasets
        ],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "gpu": gpu,
        },
    }


def write_json_atomic(data: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    partial.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(partial, target)
