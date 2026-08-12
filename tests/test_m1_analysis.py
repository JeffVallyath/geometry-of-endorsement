from __future__ import annotations

import numpy as np
import pandas as pd

from geometry_of_truth.m1.analysis import analyze
from geometry_of_truth.m1.config import load_config
from geometry_of_truth.m1.probes import fit_difference_in_means, fit_logistic
from geometry_of_truth.m1.reporting import render_report, write_results
from geometry_of_truth.m1.support.split_stress_test import CON, SIT, VAL


def _rows_and_boards(split: str, count: int, start: int):
    rows = []
    boards = []
    for offset in range(count):
        number = start + offset
        board_id = f"{split}-board-{number}"
        situations = [f"{split}-s-{number}-1", f"{split}-s-{number}-2"]
        considerations = [f"{split}-c-{number}-a", f"{split}-c-{number}-b"]
        pattern = [
            (situations[0], considerations[0], 1),
            (situations[0], considerations[1], 0),
            (situations[1], considerations[0], 0),
            (situations[1], considerations[1], 1),
        ]
        ids = []
        for cell, (situation, consideration, label) in enumerate(pattern):
            row_id = f"{split}-r-{number}-{cell}"
            ids.append(row_id)
            rows.append(
                {
                    "item_id": f"{split}-item-{number}-{cell}",
                    "row_id": row_id,
                    "split": split,
                    "label": label,
                    "situation_id": situation,
                    "consideration_id": consideration,
                    "consideration_cluster_id": consideration,
                    "board_id": board_id,
                    "cell_role": str(cell),
                    SIT: situation,
                    CON: consideration,
                    VAL: "supports" if label else "opposes",
                    "vrd": "Value" if number % 2 else "Duty",
                }
            )
        boards.append(
            {
                "board_id": board_id,
                "row_id_s1_A": ids[0],
                "row_id_s1_B": ids[1],
                "row_id_s2_A": ids[2],
                "row_id_s2_B": ids[3],
            }
        )
    return rows, pd.DataFrame(boards)


def _synthetic_inputs():
    train_rows = []
    for index in range(10):
        situation = f"train-s-{index}"
        for label in (0, 1):
            consideration = f"train-c-{index}-{label}"
            train_rows.append(
                {
                    "item_id": f"train-item-{index}-{label}",
                    "row_id": f"train-r-{index}-{label}",
                    "split": "pilot_train",
                    "label": label,
                    "situation_id": situation,
                    "consideration_id": consideration,
                    "consideration_cluster_id": consideration,
                    "board_id": "",
                    "cell_role": "",
                    SIT: situation,
                    CON: consideration,
                    VAL: "supports" if label else "opposes",
                    "vrd": "Value",
                }
            )
    select_rows, select_boards = _rows_and_boards("pilot_select", 3, 100)
    eval_rows, eval_boards = _rows_and_boards("pilot_eval", 4, 200)
    frame = pd.DataFrame([*train_rows, *select_rows, *eval_rows])
    labels = frame["label"].to_numpy()
    rng = np.random.default_rng(7)
    features = {}
    for key in (
        "primary_joint",
        "primary_situation",
        "primary_consideration",
        "transfer_joint",
    ):
        array = rng.normal(0, 0.03, size=(len(frame), 3, 5)).astype(np.float32)
        for layer, strength in enumerate((0.2, 2.0, 1.6)):
            array[:, layer, 0] += (2 * labels - 1) * strength
        features[key] = array
    margins = (2 * labels - 1).astype(np.float32) * 2
    mappings = np.asarray(
        [
            f"primary:{'standard' if index % 2 == 0 else 'reversed'}"
            for index in range(len(frame))
        ]
    )
    transfer_mappings = np.asarray(
        [value.replace("primary:", "transfer:") for value in mappings]
    )
    return (
        frame,
        features,
        {"primary": margins, "transfer": margins},
        {"primary": mappings, "transfer": transfer_mappings},
        select_boards,
        eval_boards,
    )


def test_analysis_freezes_selection_and_enforces_exact_invariants(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "geometry_of_truth.m1.analysis.fit_sbert_interaction",
        lambda train, *frames, device: (
            [
                frame["label"].to_numpy(dtype=float)
                + np.linspace(0.0, 0.01, len(frame))
                for frame in frames
            ],
            {"name": "sbert_interaction", "encoder": "synthetic"},
        ),
    )
    frame, features, margins, mappings, select_boards, eval_boards = (
        _synthetic_inputs()
    )
    results, probes = analyze(
        frame=frame,
        features=features,
        native_margins=margins,
        mapping_names=mappings,
        select_board_manifest=select_boards,
        eval_board_manifest=eval_boards,
        truth_control={
            "terminal_disposition": "PASS",
            "source_terminal_disposition": "TRUTH_CONTROL_V2_PASS",
            "protocol": "TRUTH_CONTROL_V2_NEUTRAL_MAPPING",
            "selected_layer": 14,
            "primary_test_T": 1.9,
            "exact_permutation_p": 1 / 1001,
            "directional_consensus_C": 0.99,
            "scientific_commit": "b" * 40,
            "source_results_sha256": "c" * 64,
            "checks": {"synthetic_truth_gate": True},
        },
        pipeline_checks={"synthetic_contract": True},
        config=load_config("configs/m1_development_smoke.yaml"),
    )
    assert results["terminal_disposition"] == "M1_SMOKE_ONLY_NOT_EMPIRICAL"
    assert results["selection"]["pilot_eval_used_for_selection"] is False
    assert results["selection"]["transfer_used_for_selection"] is False
    assert all(results["exact_invariants"].values())
    assert results["evaluations"]["separate_encoding_additive"][
        "checkerboard_both_ways"
    ] == 0.0
    assert abs(results["evaluations"]["separate_encoding_additive"][
        "checkerboard_interaction_mean"
    ]) < 1e-12
    assert results["evaluations"]["separate_encoding_additive"][
        "checkerboard_signed_board_mean"
    ] == 0.0
    assert len(results["permutation_nulls"]["difference_in_means"][
        "checkerboard_interaction_mean"
    ]["values"]) == 2
    assert results["endpoint_contract"]["primary"].endswith("I_b")
    assert results["dyadic_delta_ib_over_sbert"]["difference_in_means"][
        "estimator"
    ] == "dyadic_robust"
    assert probes["difference_in_means_direction"].shape == (5,)
    results.update(
        {
            "runtime": {
                "repository_commit": "1" * 40,
                "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "model_revision": "2" * 40,
                "tokenizer_revision": "2" * 40,
                "chat_template_sha256": "3" * 64,
                "gpu_name": "synthetic",
                "gpu_memory_mib": 23034,
                "model_dtype": "bfloat16",
                "activation_dtype": "float16",
                "peak_vram_bytes": 1,
                "elapsed_seconds": 1.0,
                "cache_bytes": 1,
            },
            "manifest": {
                "pilot_manifest_sha256": "4" * 64,
                "m0_manifest_sha256": "5" * 64,
                "source_row_id_sha256": "6" * 64,
                "split_manifest_sha256": {},
                "board_manifest_sha256": {},
                "contains_source_text": False,
            },
            "prompt_contract_sha256": "7" * 64,
            "prompt_specification": load_config(
                "configs/m1_development_smoke.yaml"
            ).section("prompt"),
            "cache_index_sha256": "8" * 64,
            "cache_manifest_sha256": "9" * 64,
            "probe_parameters_sha256": "a" * 64,
            "failed_experiments": [],
            "reproduction_commands": ["python run.py"],
            "recommended_next_action": "Run the full frozen protocol.",
        }
    )
    report = render_report(results)
    for section in (
        "Truth positive control",
        "Full post-freeze DIM layer curve",
        "Permutation nulls",
        "Held-out 1/2 template",
        "Instrument validation method",
        "Error analysis",
        "Reproduction",
    ):
        assert section in report
    assert "chance 0.258" in report
    assert "Primary signed cross-fit T: 1.900000" in report
    assert "DESCRIPTIVE ONLY" in report
    assert "same situation" not in report
    write_results(results, tmp_path)
    assert (tmp_path / "m1_vertical_slice_results.json").is_file()
    assert (tmp_path / "M1_VERTICAL_SLICE_REPORT.md").is_file()
    assert (tmp_path / "m1_layerwise_probe_performance.png").is_file()
