"""Package the figures with everything needed to regenerate them.

Bundles the rendered figures, their captions, both scripts, the frozen figure
data, and a manifest recording SHA-256 of every member plus the git commit and
the source archives the data came from.

    python scripts/package_figures.py [--out PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MEMBERS = [
    "figures/fig1_layer_sweep.pdf",
    "figures/fig1_layer_sweep.png",
    "figures/fig2_scorer_comparison.pdf",
    "figures/fig2_scorer_comparison.png",
    "figures/fig3_permutation_null.pdf",
    "figures/fig3_permutation_null.png",
    "figures/fig4_truth_control.pdf",
    "figures/fig4_truth_control.png",
    "figures/fig5_specificity.pdf",
    "figures/fig5_specificity.png",
    "figures/CAPTIONS.md",
    "scripts/build_figure_data.py",
    "scripts/make_figures.py",
    "artifacts/figures/figure_data.json",
    "artifacts/figures/permutation_null_llama_B10000.json",
    "artifacts/figures/m1_situation_groups.csv",
    "scripts/rerun_permutation_null.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=f"geometry-linear-probe-figures-{date.today()}.zip",
    )
    args = parser.parse_args()
    out = Path(args.out)

    missing = [m for m in MEMBERS if not (ROOT / m).is_file()]
    if missing:
        raise SystemExit("missing: " + ", ".join(missing))

    data = json.loads((ROOT / "artifacts/figures/figure_data.json").read_text(encoding="utf-8"))
    manifest = {
        "created": date.today().isoformat(),
        "repository": "JeffVallyath/geometry-of-endorsement",
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "regenerate": [
            "python scripts/build_figure_data.py   # source archives -> figure_data.json",
            "python scripts/make_figures.py        # figure_data.json -> figures/",
        ],
        "experiment_sources": {
            name: block["source"] for name, block in data["models"].items()
        },
        "truth_control_source": data["truth_control"]["llama"]["source"],
        "files": {m: sha256(ROOT / m) for m in MEMBERS},
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for m in MEMBERS:
            z.write(ROOT / m, m)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2) + "\n")

    print(f"{out}  ({out.stat().st_size:,} bytes)")
    for info in zipfile.ZipFile(out).infolist():
        print(f"  {info.file_size:>9,}  {info.filename}")


if __name__ == "__main__":
    main()
