from __future__ import annotations

import json
from pathlib import Path

import nbformat

from scripts.build_reproduction_notebooks import build_all


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (ROOT / "artifacts" / "reproduction" / "contract.json").read_text(
        encoding="utf-8"
    )
)


def _source(document) -> str:
    return "\n\n".join(cell.source for cell in document.cells)


def test_generated_reproduction_notebooks_match_the_builder() -> None:
    built = build_all(CONTRACT)
    assert set(built) == set(CONTRACT["notebooks"])
    for name, expected in built.items():
        retained = nbformat.read(ROOT / "notebooks" / name, as_version=4)
        assert retained == expected


def test_reproduction_notebooks_are_clean_and_compile() -> None:
    for document in build_all(CONTRACT).values():
        for cell in document.cells:
            if cell.cell_type != "code":
                continue
            assert cell.execution_count is None
            assert cell.outputs == []
            compile(cell.source, "<notebook-cell>", "exec")


def test_pipeline_notebook_rebuilds_the_frozen_m0_inputs() -> None:
    source = _source(build_all(CONTRACT)["01_reproduce_valueprism_pipeline.ipynb"])
    assert 'RUN_MODE = "DEMO"' in source
    assert "('DEMO', 'FULL')" in source
    assert "geometry_of_truth.leakage.reproduce import reproduce" in source
    assert "valueprism-reproduction" in source
    assert "d439ca90825e5b4e5ef97798d9b5950e16ba7065" in source
    assert "run = reproduce(OUTPUT_ROOT)" in source
    assert 'run["comparison"]' in source


def test_m1_notebook_runs_smoke_before_full_and_keeps_confirmation_closed() -> None:
    source = _source(build_all(CONTRACT)["02_reproduce_llama_m1.ipynb"])
    assert "('DEMO', 'SMOKE', 'FULL')" in source
    assert source.index("m1_development_smoke.yaml") < source.index(
        'if RUN_MODE == "FULL"'
    )
    assert "scripts/run_m1_development.py" in source
    assert "reference_comparison.csv" in source
    assert '"model": reference["model"]' in source
    assert "Human-reviewed confirmation is not included" in source
    assert "GITHUB_TOKEN" not in source
    assert "claim2" not in source.lower()


def test_both_notebooks_pin_the_source_commit_and_public_origin() -> None:
    for document in build_all(CONTRACT).values():
        source = _source(document)
        assert CONTRACT["source_commit"] in source
        assert CONTRACT["public_repository"] in source
        assert '"git", "-C"' in source
        assert '"fetch"' in source
