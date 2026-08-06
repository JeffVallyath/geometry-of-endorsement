from __future__ import annotations

from pathlib import Path
from typing import Any

from geometry_of_truth.common.artifacts import load_json, repository_root, verify_files


EXPECTED_DISPOSITION = "M1_VERTICAL_SLICE_SIGNAL_SUPPORTED"
EXPECTED_MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"


def load_status(root: str | Path | None = None) -> dict[str, Any]:
    repo = repository_root(root)
    artifact_root = repo / "artifacts" / "project"
    manifest = load_json(artifact_root / "manifest.json")
    checks = verify_files(artifact_root, manifest["public_files"])
    status = load_json(artifact_root / "status.json")
    if status["moral_relation_development"]["terminal_disposition"] != EXPECTED_DISPOSITION:
        raise RuntimeError("Development disposition mismatch")
    if status["moral_relation_development"]["model_revision"] != EXPECTED_MODEL_REVISION:
        raise RuntimeError("Development model revision mismatch")
    return {
        "root": repo,
        "artifact_root": artifact_root,
        "manifest": manifest,
        "checks": checks,
        "status": status,
    }
