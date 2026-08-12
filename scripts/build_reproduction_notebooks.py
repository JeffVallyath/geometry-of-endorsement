from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "artifacts" / "reproduction" / "contract.json"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def badge(filename: str, public_ref: str) -> str:
    target = (
        "https://colab.research.google.com/github/JeffVallyath/"
        f"geometry-of-endorsement/blob/{public_ref}/notebooks/{filename}"
    )
    image = "https://colab.research.google.com/assets/colab-badge.svg"
    return f"[![Open in Colab]({image})]({target})"


def setup_cell(
    contract: dict,
    valid_modes: tuple[str, ...],
    *,
    requirements_file: str | None = None,
) -> str:
    dependency_install = ""
    if requirements_file is not None:
        dependency_install = f'''
if RUN_MODE != "DEMO":
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_ROOT / {requirements_file!r})],
        check=True,
    )
'''
    return f'''
from pathlib import Path
import json
import os
import subprocess
import sys

RUN_MODE = "DEMO"
VALID_MODES = {valid_modes!r}
PUBLIC_REPOSITORY = {contract["public_repository"]!r}
SOURCE_COMMIT = {contract["source_commit"]!r}

if RUN_MODE not in VALID_MODES:
    raise ValueError(f"RUN_MODE must be one of {{VALID_MODES}}")

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    REPO_ROOT = Path("/content/geometry-of-endorsement-reproduction")
    if not (REPO_ROOT / ".git").is_dir():
        if REPO_ROOT.exists() and any(REPO_ROOT.iterdir()):
            raise RuntimeError("The checkout directory is not empty")
        REPO_ROOT.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(REPO_ROOT), "init"], check=True)
        subprocess.run(["git", "-C", str(REPO_ROOT), "remote", "add", "origin", PUBLIC_REPOSITORY], check=True)
    origin = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "remote", "get-url", "origin"],
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    if origin.rstrip("/").removesuffix(".git") != PUBLIC_REPOSITORY.rstrip("/").removesuffix(".git"):
        raise RuntimeError("The checkout has an unexpected origin")
    dirty = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        check=True, text=True, capture_output=True,
    ).stdout
    if dirty:
        raise RuntimeError("The checkout contains local changes")
    subprocess.run(["git", "-C", str(REPO_ROOT), "fetch", "--depth", "1", "origin", SOURCE_COMMIT], check=True)
    subprocess.run(["git", "-C", str(REPO_ROOT), "checkout", "--detach", SOURCE_COMMIT], check=True)
else:
    candidates = [Path.cwd(), *Path.cwd().parents]
    REPO_ROOT = next(path for path in candidates if (path / "pyproject.toml").is_file())

head = subprocess.run(
    ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
    check=True, text=True, capture_output=True,
).stdout.strip()
if IN_COLAB and head != SOURCE_COMMIT:
    raise RuntimeError(f"Expected source commit {{SOURCE_COMMIT}}, found {{head}}")

{dependency_install}
subprocess.run([sys.executable, "-m", "pip", "install", "-q", str(REPO_ROOT), "--no-deps"], check=True)
SOURCE_ROOT = str(REPO_ROOT / "src")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

print({{"mode": RUN_MODE, "source_commit": SOURCE_COMMIT, "repo_root": str(REPO_ROOT)}})
'''


def notebook(cells: list) -> nbf.NotebookNode:
    for cell in cells:
        material = f"{cell.cell_type}\0{cell.source}".encode("utf-8")
        cell["id"] = hashlib.sha256(material).hexdigest()[:16]
    document = nbf.v4.new_notebook(cells=cells)
    document["metadata"] = {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    return document


def pipeline_notebook(contract: dict) -> nbf.NotebookNode:
    filename = "01_reproduce_valueprism_pipeline.ipynb"
    cells = [
        markdown(f'''
# Reproduce the ValuePrism pipeline

{badge(filename, contract["public_ref"])}

The main dataset is ValuePrism. Each example gives us a situation, an action, and a moral reason. The dataset records whether that reason Supports or Opposes the action.

This notebook rebuilds the split and checkerboard inputs. The training and test sets share no situations and no exact reason labels.

`DEMO` checks the saved results. `FULL` rebuilds them from ValuePrism and saves the output to Google Drive.
'''),
        markdown('''
## Run instructions

1. Use `DEMO` for the quick check.
2. For `FULL`, accept the ValuePrism license and add `HF_TOKEN` to Colab Secrets.
3. Choose Runtime, then Run all.
'''),
        code(setup_cell(contract, ("DEMO", "FULL"))),
        code('''
from IPython.display import display

from geometry_of_truth.leakage.contracts import load_bundle
from geometry_of_truth.leakage.results import audit, overlap_checks, stress_draws

bundle = load_bundle(REPO_ROOT)
print("Retained aggregate integrity verified")
display(audit(bundle["results"]))
display(overlap_checks(bundle["results"]))
display(stress_draws(bundle["results"]))
'''),
        code('''
if RUN_MODE == "FULL":
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", f"{REPO_ROOT}[valueprism-full]"],
        check=True,
    )
    from google.colab import drive, userdata

    drive.mount("/content/drive")
    token = userdata.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Add HF_TOKEN to Colab Secrets")
    os.environ["HF_TOKEN"] = token
    OUTPUT_ROOT = Path("/content/drive/MyDrive/geometry-of-endorsement/valueprism-reproduction")
    from geometry_of_truth.leakage.reproduce import reproduce

    run = reproduce(OUTPUT_ROOT)
    display(run["comparison"])
    if not bool(run["comparison"]["pass"].all()):
        raise RuntimeError("The ValuePrism reconstruction differs from the retained aggregate")
else:
    OUTPUT_ROOT = None
    print("Set RUN_MODE to FULL to rebuild the licensed manifests")
'''),
        code('''
metadata = {
    "experiment_id": "2026-08-11-valueprism-pipeline",
    "hypothesis": "Withheld consideration identities create a larger leakage penalty than withheld situations alone.",
    "dataset": "allenai/ValuePrism",
    "dataset_revision": "d439ca90825e5b4e5ef97798d9b5950e16ba7065",
    "mode": RUN_MODE,
    "source_commit": SOURCE_COMMIT,
    "output_root": str(OUTPUT_ROOT) if OUTPUT_ROOT else None,
}
print(json.dumps(metadata, indent=2, sort_keys=True))
if OUTPUT_ROOT:
    (OUTPUT_ROOT / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
    )
'''),
        markdown('''
## Reading the result

`FULL` must match every saved count, hash, overlap check, and five-seed result. Use its output for the Llama notebook.
'''),
    ]
    return notebook(cells)


def m1_notebook(contract: dict) -> nbf.NotebookNode:
    filename = "02_reproduce_llama_m1.ipynb"
    cells = [
        markdown(f'''
# Reproduce the Llama M1 development experiment

{badge(filename, contract["public_ref"])}

This notebook reruns the Llama 3.1 8B Instruct development experiment.

The experiment used 1,500 rows to fit the activation measurements, 300 rows to choose the layer, and 500 rows for the final development evaluation. Layer 19 was selected.

`DEMO` checks the saved result. `SMOKE` checks the pipeline. `FULL` reruns the development experiment.
'''),
        markdown('''
## Run instructions

1. Run the ValuePrism notebook in `FULL` mode first.
2. Add `HF_TOKEN` to Colab Secrets and use a GPU with at least 23,000 MiB of memory.
3. Set `RUN_MODE`, then choose Runtime and Run all.
'''),
        code(setup_cell(
            contract,
            ("DEMO", "SMOKE", "FULL"),
            requirements_file="requirements-m1-reproduction.txt",
        )),
        code('''
from IPython.display import display
import pandas as pd

from geometry_of_truth.m1.reference import load_reference

reference = load_reference(REPO_ROOT)
display(pd.DataFrame([
    {"quantity": "selected layer", "value": reference["selected_layer"]},
    {"quantity": "DIM relation AUROC", "value": reference["metrics"]["difference_in_means_relation_auroc"]},
    {"quantity": "DIM I_b", "value": reference["metrics"]["difference_in_means_I_b"]},
    {"quantity": "logistic relation AUROC", "value": reference["metrics"]["logistic_relation_auroc"]},
    {"quantity": "logistic I_b", "value": reference["metrics"]["logistic_I_b"]},
    {"quantity": "SBERT I_b", "value": reference["metrics"]["sbert_interaction_I_b"]},
]))
print(reference["claim_boundary"])
'''),
        code('''
if RUN_MODE in {"SMOKE", "FULL"}:
    from google.colab import drive, userdata

    drive.mount("/content/drive")
    token = userdata.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Add HF_TOKEN to Colab Secrets")
    os.environ["HF_TOKEN"] = token
    DRIVE_ROOT = Path("/content/drive/MyDrive/geometry-of-endorsement")
    M0_ROOT = DRIVE_ROOT / "valueprism-reproduction"
    TRUTH_RESULTS = REPO_ROOT / "artifacts/truth/v2_results.json"
    subprocess.run([
        sys.executable, str(REPO_ROOT / "scripts/preflight_m1_colab.py"),
        "--config", str(REPO_ROOT / "configs/m1_development_smoke.yaml" if RUN_MODE == "SMOKE" else REPO_ROOT / "configs/m1_development.yaml"),
        "--truth-results", str(TRUTH_RESULTS),
        "--m0-dir", str(M0_ROOT),
        "--drive-root", str(DRIVE_ROOT),
    ], check=True)
else:
    DRIVE_ROOT = M0_ROOT = TRUTH_RESULTS = None
    print("Set RUN_MODE to SMOKE or FULL for a GPU replay")
'''),
        code('''
if RUN_MODE in {"SMOKE", "FULL"}:
    smoke_local = Path("/content/geometry-cache/m1-smoke")
    smoke_drive = DRIVE_ROOT / "m1-reproduction-smoke"
    smoke_manifests = DRIVE_ROOT / "m1-reproduction-smoke-manifests"
    subprocess.run([
        sys.executable, str(REPO_ROOT / "scripts/run_m1_development.py"),
        "--config", str(REPO_ROOT / "configs/m1_development_smoke.yaml"),
        "--truth-results", str(TRUTH_RESULTS),
        "--m0-dir", str(M0_ROOT),
        "--output-dir", str(smoke_local),
        "--persistent-dir", str(smoke_drive),
        "--manifest-dir", str(smoke_manifests),
        "--smoke-test",
    ], check=True)
    smoke = json.loads((smoke_local / "m1_vertical_slice_results.json").read_text())
    if smoke["terminal_disposition"] != "M1_SMOKE_ONLY_NOT_EMPIRICAL":
        raise RuntimeError("The smoke run returned an unexpected disposition")
    print(smoke["terminal_disposition"])
'''),
        code('''
if RUN_MODE == "FULL":
    full_local = Path("/content/geometry-cache/m1-development")
    full_drive = DRIVE_ROOT / "m1-reproduction-development"
    full_manifests = DRIVE_ROOT / "m1-reproduction-development-manifests"
    subprocess.run([
        sys.executable, str(REPO_ROOT / "scripts/run_m1_development.py"),
        "--config", str(REPO_ROOT / "configs/m1_development.yaml"),
        "--truth-results", str(TRUTH_RESULTS),
        "--m0-dir", str(M0_ROOT),
        "--output-dir", str(full_local),
        "--persistent-dir", str(full_drive),
        "--manifest-dir", str(full_manifests),
    ], check=True)
    display(pd.read_csv(full_local / "reference_comparison.csv"))
    result = json.loads((full_local / "m1_vertical_slice_results.json").read_text())
    print(result["terminal_disposition"])
else:
    full_local = full_drive = None
'''),
        code('''
metadata = {
    "experiment_id": "2026-08-11-llama-m1-development-replay",
    "hypothesis": "The joint pre-answer state carries a relation signal beyond the frozen SBERT interaction baseline.",
    "model": reference["model"],
    "dataset_revision": reference["data"]["revision"],
    "seed": reference["data"]["seed"],
    "mode": RUN_MODE,
    "source_commit": SOURCE_COMMIT,
    "output_root": str(full_drive) if full_drive else None,
}
print(json.dumps(metadata, indent=2, sort_keys=True))
if full_drive:
    (full_drive / "notebook_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
    )
'''),
        markdown('''
## Reading the result

`FULL` must match the saved layer, manifest hashes, model revisions, and reported measurements. Human-reviewed confirmation is not included.
'''),
    ]
    return notebook(cells)


def build_all(contract: dict | None = None) -> dict[str, nbf.NotebookNode]:
    selected = contract or json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    commit = selected["source_commit"]
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("source_commit must be a full lowercase Git commit")
    return {
        "01_reproduce_valueprism_pipeline.ipynb": pipeline_notebook(selected),
        "02_reproduce_llama_m1.ipynb": m1_notebook(selected),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "notebooks")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, document in build_all().items():
        nbf.write(document, args.output_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
