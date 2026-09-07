"""Extract the plotted series from the retained experiment archives.

Writes artifacts/figures/figure_data.json: exactly the numbers the figures
draw, plus SHA-256 digests of every source archive and the commit each run
was produced at. scripts/make_figures.py reads only this file, so the figures
are reproducible without the archives present.

    python scripts/build_figure_data.py [--llama PATH] [--gemma PATH]

The source archives are large and stay outside the repository. Set GOE_ARCHIVE_DIR
to the directory holding them, or pass explicit paths. The recorded digests are
what tie a regenerated figure back to the run that produced it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "figures" / "figure_data.json"

ARCHIVE_DIR_ENV = "GOE_ARCHIVE_DIR"
_archives = Path(os.environ.get(ARCHIVE_DIR_ENV, "."))
DEFAULT_LLAMA = str(_archives / "M1_TIE_NEUTRAL_FINAL_AUDIT.zip")
DEFAULT_GEMMA = str(_archives / "GEMMA_M1_DEVELOPMENT_AUDIT.zip")
DEFAULT_PERM = str(ROOT / "artifacts" / "figures" / "permutation_null_llama_B10000.json")

SCORERS = [
    ("difference_in_means", "Support/opposition direction"),
    ("logistic", "Logistic activation probe"),
    ("native_answer_margin", "Model answer margin"),
    ("sbert_interaction", "Frozen MiniLM comparator"),
    ("deterministic_random", "Deterministic random item scores"),
    ("situation_only_activation", "Situation only"),
    ("consideration_only_activation", "Consideration only"),
    ("separate_encoding_additive", "Additive separate encoding"),
]


def _sd(values: list[float]) -> float:
    """Sample standard deviation, ddof=1, without a numpy dependency."""
    n = len(values)
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_block(results: dict, label: str, source: Path, member: str) -> dict:
    layerwise = [
        {
            "layer": e["layer"],
            "auroc": e["auroc"],
            "I_b": e["checkerboard_interaction_mean"],
            "balanced_accuracy": e["balanced_accuracy"],
            "within_situation": e["within_situation_macro"],
            "within_consideration": e["within_consideration_macro"],
        }
        for e in results["layerwise_difference_in_means"]
    ]

    evaluations = {}
    for key, name in SCORERS:
        e = results["evaluations"][key]
        ci = e.get("confidence_intervals", {}).get("checkerboard_interaction_mean", {})
        evaluations[key] = {
            "label": name,
            "auroc": e.get("auroc"),
            "I_b": e["checkerboard_interaction_mean"],
            "ci_low": ci.get("low"),
            "ci_high": ci.get("high"),
            "ci_estimator": ci.get("estimator"),
            "ci_level": ci.get("confidence_level"),
            "within_situation": e.get("within_situation_macro"),
            "within_consideration": e.get("within_consideration_macro"),
        }

    transfer = results["held_out_template_transfer"]["difference_in_means"]
    evaluations["held_out_template"] = {
        "label": "Held-out answer template",
        "auroc": transfer.get("auroc"),
        "I_b": transfer["checkerboard_interaction_mean"],
        "ci_low": None,
        "ci_high": None,
        "ci_estimator": None,
        "ci_level": None,
    }

    permutation = {}
    for key in ("difference_in_means", "logistic"):
        p = results["permutation_nulls"][key]["checkerboard_interaction_mean"]
        values = p["values"]
        permutation[key] = {
            "observed": p["observed"],
            "p": p["empirical_p_greater_equal"],
            "B": len(values),
            "k_ge_observed": sum(1 for v in values if v >= p["observed"]),
            "null_sd": _sd(values),
            "frac_beyond_one": sum(1 for v in values if abs(v) > 1.0) / len(values),
            "values": values,
        }

    selection = results["selection"]
    delta = results.get("dyadic_delta_ib_over_sbert", {})
    return {
        "label": label,
        "selected_layer": selection["selected_layer"],
        "selection_criterion": selection["criterion"],
        "pilot_eval_used_for_selection": selection["pilot_eval_used_for_selection"],
        "n_layers": len(layerwise),
        "sample_counts": results["sample_counts"],
        "layerwise": layerwise,
        "evaluations": evaluations,
        "permutation_null": permutation,
        "delta_over_text_baseline": delta,
        "truth_control_layer": results["truth_positive_control"]["selected_layer"],
        "truth_control_T": results["truth_positive_control"]["primary_test_T"],
        "terminal_disposition": results["terminal_disposition"],
        "source": {
            "archive": source.name,
            "archive_sha256": sha256(source),
            "member": member,
            "repository_commit": results["runtime"]["repository_commit"],
            "model_id": results["runtime"]["model_id"],
            "model_revision": results["runtime"]["model_revision"],
        },
    }


def truth_block() -> dict:
    path = ROOT / "artifacts" / "truth" / "v2_results.json"
    results = json.loads(path.read_text(encoding="utf-8"))
    selected = str(results["selected_layer"])
    confirmatory = results["confirmatory_layer_results"][selected]
    return {
        "llama": {
            "selected_layer": results["selected_layer"],
            "selection_rule": results.get("selection_rule"),
            "primary_T": confirmatory["test"]["primary"]["overall"]["T"],
            "transfer_T": confirmatory["test"]["transfer"]["overall"]["T"],
            "primary_ci": confirmatory["group_bootstrap_ci"]["primary"],
            "transfer_ci": confirmatory["group_bootstrap_ci"]["transfer"],
            "terminal_disposition": results["terminal_disposition"],
            "dev_sweep": [
                {
                    "layer": e["layer"],
                    "standard_T": e["primary_dev_by_mapping"]["standard"]["T"],
                    "reversed_T": e["primary_dev_by_mapping"]["reversed"]["T"],
                }
                for e in results["development_layer_sweep"]
            ],
            "source": {
                "artifact": "artifacts/truth/v2_results.json",
                "sha256": sha256(path),
            },
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama", default=DEFAULT_LLAMA)
    parser.add_argument("--gemma", default=DEFAULT_GEMMA)
    parser.add_argument("--permutation", default=DEFAULT_PERM)
    parser.add_argument("--allow-missing-permutation", action="store_true",
                        help="build without the large-B null instead of failing")
    args = parser.parse_args()

    llama_path = Path(args.llama)
    gemma_path = Path(args.gemma)
    llama_member = "m1_vertical_slice_results.json"
    gemma_member = "m1_vertical_slice_results.json"

    llama = json.loads(zipfile.ZipFile(llama_path).read(llama_member))
    gemma = json.loads(zipfile.ZipFile(gemma_path).read(gemma_member))

    data = {
        "models": {
            "llama": model_block(llama, "Llama-3.1-8B-Instruct", llama_path, llama_member),
            "gemma": model_block(gemma, "Gemma-2-9B-it", gemma_path, gemma_member),
        },
        "truth_control": truth_block(),
    }

    perm_path = Path(args.permutation) if args.permutation else None
    if perm_path and not perm_path.is_file():
        if not args.allow_missing_permutation:
            raise SystemExit(
                f"large-B permutation null not found: {perm_path}\n"
                "The figures would silently fall back to the B=200 source values. "
                "Regenerate it with scripts/rerun_permutation_null.py, or pass "
                "--allow-missing-permutation to build without it deliberately."
            )
        perm_path = None
    if perm_path:
        rerun = json.loads(perm_path.read_text(encoding="utf-8"))
        data["models"]["llama"]["permutation_null_large"] = {
            "null_definition": rerun["null_definition"],
            "seed": rerun["seed"],
            "probes": {
                name: {
                    "observed": block["observed"],
                    "p": block["p_empirical"],
                    "p_two_sided_sensitivity": block.get("p_two_sided_sensitivity"),
                    "k_ge_observed": block["k_ge_observed"],
                    "B": block["B"],
                    "null_sd": block.get("null_sd", _sd(block["values"])),
                    "frac_beyond_one": sum(1 for v in block["values"] if abs(v) > 1.0) / len(block["values"]),
                    "values": block["values"],
                }
                for name, block in rerun["probes"].items()
            },
            "inputs": rerun.get("inputs"),
            "p_value": rerun.get("p_value"),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes)")
    for name, block in data["models"].items():
        print(f"  {name}: layer {block['selected_layer']}, {block['source']['archive']}")


if __name__ == "__main__":
    main()
