from __future__ import annotations

from geometry_of_truth.leakage.contracts import load_bundle as load_leakage
from geometry_of_truth.project.contracts import load_status
from geometry_of_truth.truth.contracts import load_bundle as load_truth
from geometry_of_truth.truth.results import bootstrap_intervals, direction_method


def test_truth_method_and_intervals() -> None:
    bundle = load_truth()
    method = direction_method()
    fitting = method.loc[method["step"] == "fit", "data used"].item()
    assert "inside partition k" in fitting
    intervals = bootstrap_intervals(bundle["v2"]).set_index("answer symbols")
    assert intervals.loc["A/B", "CI low"] == 1.8929612278938293
    assert intervals.loc["1/2", "CI high"] == 1.874272894859314
    assert bundle["v2"]["selected_layer"] == 14
    assert len(bundle["v2"]["development_layer_sweep"]) == 32


def test_leakage_boundary_and_sensitivity() -> None:
    results = load_leakage()["results"]
    overlaps = results["strict_split"]["overlaps"]
    assert overlaps["l1_normalized"] == 2
    assert overlaps["l3_semantic_cluster"] == 0
    assert overlaps["situation"] == 0
    assert results["sensitivity"]["coverage"]["U1"]["rows"] == 7081
    assert results["sensitivity"]["coverage"]["U3"]["rows"] == 587
    assert results["sensitivity"]["human_adjudications"] == 0
    assert results["stress_test"]["training_injection_fraction"] == 0.3
    assert results["stress_test"]["situation_holdout_fraction"] == 0.35


def test_current_project_frontier() -> None:
    status = load_status()["status"]
    result = status["moral_relation_development"]
    assert result["selected_layer"] == 19
    assert result["evaluation_boards"] == 125
    assert result["difference_in_means"]["I_b"] == 1.6470105763501846
    assert result["logistic"]["I_b"] == 2.083584884709801
    assert result["controls"]["native_answer_margin_auroc"] == 0.721056
    assert result["logistic"]["relation_auroc"] == 0.779616
    stages = {row["stage"]: row["status"] for row in status["stages"]}
    assert stages["Human-audited confirmation"] == "Pending"
    assert stages["Rephrasing-flip prediction"] == "Not yet run"
