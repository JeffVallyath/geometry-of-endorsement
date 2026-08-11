from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "review_templates"


def test_public_review_templates_have_headers_only() -> None:
    expected = {
        "claim1_first_stage.csv": 27,
        "claim1_second_stage.csv": 10,
        "claim2_rewrite.csv": 7,
    }
    for name, columns in expected.items():
        with (TEMPLATES / name).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        assert len(rows) == 1
        assert len(rows[0]) == columns


def test_public_review_package_contains_no_study_rows_or_answers() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in TEMPLATES.glob("*")
        if path.is_file()
    )
    forbidden = (
        "valueprism_relation",
        "native_margin",
        "raw_dim_margin",
        "candidate_id",
        "base_item_id",
        "board_id",
    )
    assert not any(token in text for token in forbidden)
    metadata = json.loads((TEMPLATES / "reviewer_metadata.json").read_text(encoding="utf-8"))
    assert metadata["freeze_id"] == "geometry_human_review_v2_2026-08-10"
    assert metadata["reviewer_pseudonym"] == ""
    assert metadata["session_log"] == []


def test_public_review_manifest_binds_every_public_artifact() -> None:
    manifest = json.loads((TEMPLATES / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FROZEN_READY_FOR_INDEPENDENT_REVIEW"
    assert manifest["contains_study_rows"] is False
    for relative, expected in manifest["public_files_sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected
