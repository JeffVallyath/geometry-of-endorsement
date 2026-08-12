from __future__ import annotations

import numpy as np
import pandas as pd

from geometry_of_truth.m1.manifests import _cluster_bucket, _public_rows
from geometry_of_truth.m1.metrics import (
    board_metrics,
    boards_from_manifest,
    paired_interaction_delta_ci,
    mirrored_pairwise,
)
from geometry_of_truth.m1.support.metrics import fit_scaler
from geometry_of_truth.m1.support.split_stress_test import CON, SIT, VAL


def board_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "row_id": ["r1", "r2", "r3", "r4"],
            SIT: ["s1", "s1", "s2", "s2"],
            CON: ["a", "b", "a", "b"],
            VAL: ["supports", "opposes", "opposes", "supports"],
            "label": [1, 0, 0, 1],
        }
    )
    manifest = pd.DataFrame(
        {
            "board_id": ["board-1"],
            "row_id_s1_A": ["r1"],
            "row_id_s1_B": ["r2"],
            "row_id_s2_A": ["r3"],
            "row_id_s2_B": ["r4"],
        }
    )
    return frame, manifest


def test_exact_m0_invariants_hold() -> None:
    frame, manifest = board_frame()
    boards = boards_from_manifest(manifest, frame)
    situation_only = np.asarray([2.0, 2.0, -1.0, -1.0])
    consideration_only = np.asarray([3.0, -2.0, 3.0, -2.0])
    additive = situation_only + consideration_only
    situation_metric = mirrored_pairwise(frame, situation_only)
    consideration_metric = mirrored_pairwise(frame, consideration_only)
    additive_metric = board_metrics(boards, additive, fit_scaler(additive))
    assert situation_metric["within_situation_exact_m0"] == 0.5
    assert consideration_metric["within_consideration_exact_m0"] == 0.5
    assert additive_metric["both_ways"] == 0.0
    assert abs(additive_metric["interaction_contrast_mean"]) < 1e-12
    assert additive_metric["signed_board_mean"] == 0.0
    tied_additive = np.zeros(4)
    tied_metric = board_metrics(boards, tied_additive, fit_scaler([0.0, 1.0]))
    assert tied_metric["interaction_contrast_mean"] == 0.0
    assert tied_metric["signed_board_mean"] == 0.0
    assert additive_metric["both_ways_chance"] == 0.258


def test_dyadic_delta_ib_is_endpoint_swap_invariant() -> None:
    frame, manifest = board_frame()
    boards = boards_from_manifest(manifest, frame)
    probe = board_metrics(boards, np.asarray([3.0, 0.0, 0.0, 3.0]), fit_scaler([0, 1, 2, 3]))
    baseline = board_metrics(boards, np.asarray([2.0, 0.0, 0.0, 2.0]), fit_scaler([0, 1, 2, 3]))
    original = paired_interaction_delta_ci(probe, baseline, confidence_level=0.95)
    for metric in (probe, baseline):
        for row in metric["_per_board"]:
            row["node_a"], row["node_b"] = row["node_b"], row["node_a"]
    swapped = paired_interaction_delta_ci(probe, baseline, confidence_level=0.95)
    assert original["standard_error"] == swapped["standard_error"]


def test_public_item_id_does_not_depend_on_valence() -> None:
    base = pd.DataFrame(
        {
            "row_id": ["label-bearing-row-id"],
            SIT: ["same situation"],
            CON: ["same consideration"],
            VAL: ["supports"],
            "vrd": ["Value"],
            "l3": [12],
            "split": ["pilot_train"],
            "board_id": [""],
            "cell_role": [""],
        }
    )
    opposite = base.copy()
    opposite[VAL] = "opposes"
    opposite["row_id"] = "different-label-bearing-row-id"
    assert _public_rows(base).iloc[0]["item_id"] == _public_rows(opposite).iloc[0]["item_id"]


def test_cluster_lane_is_deterministic_and_binary() -> None:
    values = [_cluster_bucket(str(index), 20260803) for index in range(100)]
    assert set(values) == {0, 1}
    assert values == [_cluster_bucket(str(index), 20260803) for index in range(100)]
