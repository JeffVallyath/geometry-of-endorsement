#!/usr/bin/env python3
"""Render PAPER_MAP.md deterministically from the canonical paper_map.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def link(path):
    return f"[{path}]({path})"


def render(data):
    lines = [
        "# Paper and evidence map", "",
        "Generated from [paper_map.json](paper_map.json) by `python scripts/generate_paper_map.py`.", "",
        "Start with the [measured Ada and Bryn walkthrough](reproducibility/fresh_history_validation/examples/CASE_0017.md).",
        "Figures 1–5 belong to the main text; Appendix Figures 6–7 belong to the appendix.",
        "Additional repository analyses are identified separately; reproduction limits are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).", "",
        "## Manuscript appendix crosswalk", "",
        "A01–A13 are repository topic IDs, distinct from the manuscript appendix letters.", "",
        "| Repository topic | Manuscript location | Coverage |", "|---|---|---|",
    ]
    for row in data["appendix_crosswalk"]:
        lines.append(f"| [{row['topic']}]({row['page']}) | {row['location']} | {row['scope']} |")
    lines += ["", "## Figures", "", "| Figure | Manuscript location | Asset | Evidence guide |", "|---|---|---|---|"]
    figures = [e for e in data["entries"] if e.get("kind") in {"main_figure", "appendix_figure"}]
    figures.sort(key=lambda e: e["figure_number"])
    for e in figures:
        asset = e["figure_assets"][0]
        lines.append(f"| {e['title']} | {e.get('paper_location', '')} | [PDF]({asset}) · [PNG]({asset[:-4] + '.png'}) | [Guide]({e['page']}) · `{e['id']}` |")
    lines += ["", "## Mapped claims, tables, and examples", "", "| ID | Item | Type | Manuscript location | Guide | Study |", "|---|---|---|---|---|---|"]
    for e in data["entries"]:
        if e.get("kind") in {"main_figure", "appendix_figure"}:
            continue
        lines.append(f"| `{e['id']}` | {e.get('title', '')} | {e.get('kind', '')} | {e.get('paper_location', '')} | {link(e['page'])} | `{e.get('study', '')}` |")
    lines += ["", "## Study replay index", "", "Each study guide documents evidence, settings, commands, expected outputs, and reproduction limits.", ""]
    studies = {}
    for e in data["entries"]:
        if e.get("study") and e.get("commands") and e.get("study") not in studies:
            studies[e["study"]] = e
    lines += ["| Study | Guide | Reproduction level | Command |", "|---|---|---|---|"]
    for name, e in studies.items():
        cmd = e["commands"][0].replace("|", "\\|")
        lines.append(f"| `{name}` | {link(e['page'])} | {e.get('reproduction_level', '')} | `{cmd}` |")
    lines += ["", "## Figure rendering", "", "Render the five main-text and two appendix figures from retained measurements:", "", "```sh", "python -m repro.main_figures --output reproduced/main-figures", "```", ""]
    return "\n".join(lines).rstrip()


def main():
    data = json.loads((ROOT / "paper_map.json").read_text())
    (ROOT / "PAPER_MAP.md").write_text(render(data) + "\n")


if __name__ == "__main__":
    main()
