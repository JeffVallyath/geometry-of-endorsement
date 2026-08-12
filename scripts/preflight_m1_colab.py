#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from datasets import load_dataset
from huggingface_hub import HfApi

from geometry_of_truth.m1.config import load_config
from geometry_of_truth.m1.manifests import file_sha256
from geometry_of_truth.m1.runner import _truth_control_summary
from geometry_of_truth.truth.reproduction.model import gpu_preflight


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--truth-results", required=True, type=Path)
    parser.add_argument("--drive-root", required=True, type=Path)
    parser.add_argument("--m0-dir", required=True, type=Path)
    parser.add_argument("--minimum-local-disk-gib", type=int, default=35)
    args = parser.parse_args()
    config = load_config(args.config)
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing from the environment.")
    gpu = gpu_preflight(int(config.raw["model"]["min_gpu_memory_mib"]))
    free_gib = shutil.disk_usage("/content").free / 1024**3
    if free_gib < args.minimum_local_disk_gib:
        raise RuntimeError(
            f"Only {free_gib:.1f} GiB is free under /content; "
            f"{args.minimum_local_disk_gib} GiB is required."
        )
    model_revision = HfApi(token=token).model_info(
        config.raw["model"]["id"],
        revision=config.raw["model"]["revision"],
    ).sha
    # Access check only: do not print or persist the licensed example.
    sample = load_dataset(
        config.raw["data"]["dataset"],
        config.raw["data"]["dataset_config"],
        split="train[:1]",
        token=token,
        revision=config.raw["data"]["dataset_revision"],
    )
    if len(sample) != 1:
        raise RuntimeError("ValuePrism access probe did not return exactly one row.")
    args.drive_root.mkdir(parents=True, exist_ok=True)
    probe = args.drive_root / f".m1-write-probe-{secrets.token_hex(8)}"
    payload = secrets.token_bytes(64)
    probe.write_bytes(payload)
    if probe.read_bytes() != payload:
        raise RuntimeError("Google Drive write/read verification failed.")
    probe.unlink()
    status = git("status", "--porcelain")
    truth = _truth_control_summary(
        args.truth_results,
        expected_model_id=str(config.raw["model"]["id"]),
    )
    expected_m0 = {
        "manifest_train_common.csv": "a674f40c5c47e48f523f8cc8f0a8d8fde4288e979294780dc4c78682e61bb20e",
        "manifest_confirmatory.csv": "af17c7e84a284406590a18ef4ab52dc6e919553b7b55cb6f1fa364d8ed6b0da5",
    }
    m0_hashes = {
        name: file_sha256(args.m0_dir / name)
        for name in expected_m0
        if (args.m0_dir / name).is_file()
    }
    if m0_hashes != expected_m0:
        raise RuntimeError(
            "The ValuePrism reconstruction directory does not match the frozen M0 manifests."
        )
    result = {
        "status": "PASS" if truth["terminal_disposition"] == "PASS" else "FAIL",
        "gpu": gpu,
        "free_local_disk_gib": free_gib,
        "repository": {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "working_tree_clean": status == "",
            "status_porcelain": status.splitlines(),
        },
        "model": {
            "id": config.raw["model"]["id"],
            "resolved_revision": model_revision,
        },
        "valueprism": {
            "access": "PASS",
            "rows_read_for_probe": 1,
            "content_printed_or_persisted": False,
        },
        "drive_write_read_delete": "PASS",
        "config_sha256": config.digest,
        "m0_manifest_hashes": m0_hashes,
        "truth_control": truth,
    }
    if not result["repository"]["working_tree_clean"]:
        raise RuntimeError("Execution checkout is dirty.")
    if result["status"] != "PASS":
        raise RuntimeError("The strict truth positive-control gate did not pass.")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
