from __future__ import annotations

import numpy as np
import pytest

from geometry_endorsement.causal_fingerprint_v2.high_precision_endpoint import candidate_logit_margin
from geometry_endorsement.causal_fingerprint_v2.mapping_factorial import factorial_components
from geometry_endorsement.causal_fingerprint_v2.numerical_resolution import diagnostics, estimate_resolution_floor
from geometry_endorsement.causal_fingerprint_v2.pilot_gate_v2 import decide
from geometry_endorsement.causal_fingerprint_v2.realized_perturbation import realized_central_difference, realized_metrics
from geometry_endorsement.causal_fingerprint_v2.reliability_v2 import causal_gram_reliability, per_direction_fingerprint_similarity, stable_gradient_random_direction_fixture


def test_smooth_scalar_autograd_analogue_matches_central_difference() -> None:
    x = 0.3
    epsilon = 1e-4
    exact = 3.0 * x**2 + 2.0
    finite = (((x + epsilon) ** 3 + 2 * (x + epsilon)) - ((x - epsilon) ** 3 + 2 * (x - epsilon))) / (2 * epsilon)
    assert finite == pytest.approx(exact, rel=1e-7)


def test_quantized_output_hides_small_correct_gradient() -> None:
    x = 0.3
    quantized = lambda value: np.round(value / 0.125) * 0.125
    small = (quantized(x + 0.01) - quantized(x - 0.01)) / 0.02
    assert small == 0.0
    assert 1.0 != small


def test_larger_step_recovers_quantized_slope_before_nonlinearity() -> None:
    quantized = lambda value: np.round(value / 0.125) * 0.125
    recovered = (quantized(0.3 + 0.1) - quantized(0.3 - 0.1)) / 0.2
    nonlinear = ((0.3 + 1.0) ** 3 - (0.3 - 1.0) ** 3) / 2.0
    assert recovered == pytest.approx(0.625)
    assert abs(recovered - 1.0) < abs(0.0 - 1.0)
    assert nonlinear != pytest.approx(3 * 0.3**2)


def test_low_precision_cast_distorts_requested_delta() -> None:
    base = np.asarray([1024.0, 1.0], dtype=np.float16)
    requested = np.asarray([0.1, 0.001], dtype=np.float32)
    realized = (base + requested.astype(np.float16)).astype(np.float16).astype(np.float32)
    metrics = realized_metrics(base.astype(np.float32), realized, requested)
    assert metrics["realized_over_requested_norm"] < 0.1


def test_mapping_pair_recovers_planted_semantic_and_token_components() -> None:
    semantic = np.asarray([1.0, -2.0, 0.5])
    token = np.asarray([3.0, 1.0, -0.5])
    recovered_semantic, recovered_token = factorial_components(semantic + token, semantic - token)
    np.testing.assert_allclose(recovered_semantic, semantic)
    np.testing.assert_allclose(recovered_token, token)


def test_fixed_random_directions_can_be_reliable_without_semantics() -> None:
    result = stable_gradient_random_direction_fixture()
    assert result["semantic_structure_planted"] == 0.0
    assert result["random_direction_split_half_reliability"] > 0.95


def test_per_direction_fingerprint_reliability_recovers_stable_rows() -> None:
    rng = np.random.default_rng(7)
    planted = rng.normal(size=(12, 24))
    left = planted + rng.normal(scale=0.01, size=planted.shape)
    right = planted + rng.normal(scale=0.01, size=planted.shape)
    assert np.median(per_direction_fingerprint_similarity(left, right)) > 0.99
    assert causal_gram_reliability(left, right) > 0.99


def test_realized_denominator_controls_central_difference() -> None:
    base = np.zeros(2)
    direction = np.asarray([1.0, 0.0])
    slope, denominator = realized_central_difference(0.4, -0.2, base, np.asarray([0.2, 0.0]), np.asarray([-0.1, 0.0]), direction)
    assert denominator == pytest.approx(0.3)
    assert slope == pytest.approx(2.0)


def test_high_precision_candidate_head_uses_float32_rows() -> None:
    hidden = np.asarray([1.0, 2.0], dtype=np.float16)
    rows = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16)
    assert candidate_logit_margin(hidden, rows) == pytest.approx(-1.0)


def _passing_actor() -> dict:
    return {"correlation": 0.95, "sign_agreement": 0.95, "median_absolute_relative_error": 0.1, "saturation_fraction": 0.0, "median_per_direction_fingerprint_similarity": 0.8, "family_fingerprint_similarity": {"relation": 0.7, "truth": 0.7, "sentiment": 0.7, "language": 0.7}, "causal_gram_reliability": 0.8, "mapping_family_agreement": 0.8, "semantic_token_contrast_correlation": 0.2}


@pytest.mark.parametrize("field,value", [("correlation", 0.5), ("sign_agreement", 0.5), ("median_absolute_relative_error", 0.8), ("saturation_fraction", 0.2), ("median_per_direction_fingerprint_similarity", 0.1), ("causal_gram_reliability", 0.1), ("mapping_family_agreement", 0.1), ("semantic_token_contrast_correlation", 0.95)])
def test_v2_gate_rejects_each_invalid_measurement_class(field: str, value: float) -> None:
    actors = {"llama": _passing_actor(), "gemma": _passing_actor()}
    actors["llama"][field] = value
    assert decide(actors) == "CAUSAL_FINGERPRINT_MEASUREMENT_V2_INVALID"


def test_v2_gate_requires_both_actors() -> None:
    with pytest.raises(RuntimeError, match="BOTH_V2_ACTORS_REQUIRED"):
        decide({"llama": _passing_actor()})


def test_resolution_floor_finds_planted_lattice() -> None:
    changes = np.asarray([0.0, 0.125, 0.25, 0.375, 0.125, 0.25])
    assert estimate_resolution_floor(changes) == pytest.approx(0.125)


def test_numerical_gate_retains_original_thresholds() -> None:
    left = np.arange(1.0, 11.0)
    right = left * 1.02
    result = diagnostics(left, right, np.zeros(10, dtype=bool), 0.01)
    assert result.passed
