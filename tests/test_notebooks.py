from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "00_project_status.ipynb",
    "geometry_of_truth.ipynb",
    "valueprism_leakage.ipynb",
)


def load(name: str) -> dict:
    return json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
def source(cell: dict) -> str:
    value = cell["source"]
    return "".join(value) if isinstance(value, list) else value



def test_source_notebooks_are_clean_and_parseable() -> None:
    for name in NOTEBOOKS:
        notebook = load(name)
        assert all(not cell.get("outputs") for cell in notebook["cells"])
        assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(source(cell))


def test_notebook_mode_contracts_reject_unknown_modes_before_setup() -> None:
    for name in NOTEBOOKS:
        notebook = load(name)
        setup = next(source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code")
        invalid = setup.replace("RUN_MODE = 'DEMO'", "RUN_MODE = 'INVALID'", 1)
        with pytest.raises(ValueError, match="RUN_MODE must be one of"):
            exec(compile(invalid, name, "exec"), {})


def test_truth_notebook_describes_the_implemented_estimator() -> None:
    text = "\n".join(source(cell) for cell in load("geometry_of_truth.ipynb")["cells"])
    assert "inside each subset" in text
    assert "all 32 transformer layers" in text
    assert "last nonpadding token of the rendered chat prompt" in text
    assert "2,000-group bootstrap" in text
    assert "Monte Carlo" in text
    assert "layers 8 through 24" not in text
    assert "Factual truth is easier and represents a different target" in text
    assert "requirements-truth-analysis.txt" in text
    assert "changes both the output symbols and the surrounding user phrasing" in text
    assert "direction-specific effects on the held-out test set" in text
    assert "rather than isolating answer symbols alone" in text
    assert "replaces A/B with 1/2" not in text
    assert "held-out subset effects" not in text
    assert "removing the original answer tokens" not in text
    assert "VALID_MODES = {'ANALYSIS', 'DEMO', 'FULL'}" in text


def test_current_frontier_is_explicit() -> None:
    status = "\n".join(source(cell) for cell in load("00_project_status.ipynb")["cells"])
    leakage = "\n".join(source(cell) for cell in load("valueprism_leakage.ipynb")["cells"])
    assert "moral-relation development test has now passed" in leakage
    assert "two L1-normalized collisions" in leakage
    assert "U3 is currently identical to U2" in leakage
    assert "layer 19" in status.lower()
    assert "0.780" in status
    assert "rephrasing-flip experiment has not run" in status
    assert "Rewarding demonstrated merit" in leakage
    assert "confirmatory row membership across all four board cells" in leakage
    assert "candidate ordering" not in leakage
    assert "VALID_MODES = {'DEMO', 'FULL'}" in leakage
    assert 'RUN_MODE == "ANALYSIS"' not in leakage
    assert "requirements-truth-analysis.txt" not in leakage
    assert "same board grouping, rank, primary or reserve assignment, and review order" in leakage
