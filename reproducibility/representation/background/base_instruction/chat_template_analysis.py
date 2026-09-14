#!/usr/bin/env python3
"""C9 chat-template supplement (Unit 9, reported separately; CPU only).

The frozen C9 analysis (``c9_base_vs_instruct_v1.py analyze``) compares base
and instruct checkpoints on identical raw statement text with no chat template,
as the freeze prescribes for the geometry comparison. The freeze also records
that the instruct checkpoint is *additionally* extracted with its chat template
"as a separate condition". The extraction wrote ``activations_chat_template.npz``
for both instruct checkpoints, but the frozen analysis reads only the raw
condition. This script reports the template condition separately, using the
same probe, statements, splits, layers and metrics as the frozen analysis.

Per actor, task and layer it reports the AUROC grid over the three activation
sources {base_raw, instruct_raw, instruct_template} (probe fitted on the row
source, evaluated on the column source), and the cosine between the
instruct_template mean-difference direction and each of the two raw directions.

It changes no classification: the frozen verdict is fixed by the raw condition.
The template rows are descriptive and carry no gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from c9_base_vs_instruct_v1 import auroc, fit_probe, read_jsonl  # noqa: E402

SCHEMA = "C9_CHAT_TEMPLATE_SUPPLEMENT_V1"
SOURCES = ("base_raw", "instruct_raw", "instruct_template")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def write_create_only(path: Path, data: bytes) -> None:
    with path.open("xb") as f:
        f.write(data)


def load_sources(extract_root: Path, actor: str, layers: list[int], item_ids: list[str]) -> dict[str, dict[int, np.ndarray]]:
    files = {"base_raw": extract_root / f"{actor}_base" / "activations_raw.npz", "instruct_raw": extract_root / f"{actor}_instruct" / "activations_raw.npz", "instruct_template": extract_root / f"{actor}_instruct" / "activations_chat_template.npz"}
    out: dict[str, dict[int, np.ndarray]] = {}
    for name, path in files.items():
        with np.load(path) as z:
            assert list(z["item_ids"]) == item_ids, f"{path}: item order differs from the frozen statement set"
            out[name] = {L: z[f"layer_{L}"] for L in layers}
    return out


def supplement(freeze: dict[str, Any], statements: list[dict[str, Any]], acts_by_actor: dict[str, dict[str, dict[int, np.ndarray]]]) -> dict[str, Any]:
    results: dict[str, Any] = {"schema_version": SCHEMA, "condition": "instruct checkpoint with chat template (user turn = raw statement, no generation prompt), last token; compared with the raw condition used by the frozen analysis", "gate": "none — descriptive supplement; the frozen classification is fixed by the raw condition", "actors": {}}
    y = np.array([s["label"] for s in statements])
    for actor, cfg in freeze["actors"].items():
        acts = acts_by_actor[actor]
        per_task: dict[str, Any] = {}
        for task in ("relation", "truth", "sentiment"):
            fit = np.array([s["task"] == task and s["split"] == "fit" for s in statements])
            ev = np.array([s["task"] == task and s["split"] == "eval" for s in statements])
            per_layer: dict[int, Any] = {}
            for L in cfg["layers"]:
                probes = {src: fit_probe(acts[src][L][fit], y[fit]) for src in SOURCES}
                grid = {}
                for src in SOURCES:
                    sc, clf = probes[src]
                    for dst in SOURCES:
                        grid[f"{src}->{dst}"] = auroc(y[ev], clf.decision_function(sc.transform(acts[dst][L][ev])))
                dirs = {src: acts[src][L][fit][y[fit] == 1].mean(0) - acts[src][L][fit][y[fit] == 0].mean(0) for src in SOURCES}

                def cos(a: str, b: str) -> float:
                    return float(dirs[a] @ dirs[b] / (np.linalg.norm(dirs[a]) * np.linalg.norm(dirs[b]) + 1e-12))

                per_layer[L] = {"auroc_grid": grid, "cosine_template_vs_base_raw": cos("instruct_template", "base_raw"), "cosine_template_vs_instruct_raw": cos("instruct_template", "instruct_raw")}
            per_task[task] = {"per_layer": per_layer, "at_frozen_layer": per_layer[cfg["frozen_layer"]]}
        results["actors"][actor] = {"frozen_layer": cfg["frozen_layer"], "tasks": per_task}
    return results


def render(results: dict[str, Any]) -> str:
    lines = ["# C9 chat-template supplement (reported separately; no gate)", "", results["condition"], ""]
    for actor, r in results["actors"].items():
        L0 = r["frozen_layer"]
        lines += [f"## {actor} (frozen layer {L0})", "", "| task | layer | tmpl→tmpl | base_raw→tmpl | tmpl→base_raw | instr_raw→tmpl | tmpl→instr_raw | cos(tmpl, base_raw) | cos(tmpl, instr_raw) |", "|---|---|---|---|---|---|---|---|---|"]
        for task, t in r["tasks"].items():
            for L, v in t["per_layer"].items():
                g = v["auroc_grid"]
                mark = "*" if int(L) == L0 else ""
                lines.append(f"| {task} | {L}{mark} | {g['instruct_template->instruct_template']:.3f} | {g['base_raw->instruct_template']:.3f} | {g['instruct_template->base_raw']:.3f} | {g['instruct_raw->instruct_template']:.3f} | {g['instruct_template->instruct_raw']:.3f} | {v['cosine_template_vs_base_raw']:.3f} | {v['cosine_template_vs_instruct_raw']:.3f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze-root", type=Path, required=True)
    ap.add_argument("--extract-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.output_root.exists() and any(a.output_root.iterdir()):
        raise SystemExit("output root not empty")
    freeze = json.loads((a.freeze_root / "C9_FREEZE.json").read_text(encoding="utf-8"))
    statements = read_jsonl(a.freeze_root / "C9_STATEMENTS.jsonl")
    item_ids = [s["item_id"] for s in statements]
    acts = {actor: load_sources(a.extract_root, actor, cfg["layers"], item_ids) for actor, cfg in freeze["actors"].items()}
    results = supplement(freeze, statements, acts)
    results["source"] = {"freeze_sha256": sha256_file(a.freeze_root / "C9_FREEZE.json"), "script_sha256": sha256_file(Path(__file__).resolve()), "activation_files": {f"{actor}/{name}": sha256_file(a.extract_root / f"{actor}_{'base' if name == 'base_raw' else 'instruct'}" / ("activations_chat_template.npz" if name == "instruct_template" else "activations_raw.npz")) for actor in acts for name in SOURCES}}
    a.output_root.mkdir(parents=True)
    write_create_only(a.output_root / "C9_CHAT_TEMPLATE_SUPPLEMENT.json", canonical_json(results))
    write_create_only(a.output_root / "C9_CHAT_TEMPLATE_SUPPLEMENT.md", render(results).encode("utf-8"))
    files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(a.output_root.iterdir())]
    write_create_only(a.output_root / "artifact_manifest.json", canonical_json({"schema_version": "C9_CHAT_TEMPLATE_SUPPLEMENT_MANIFEST_V1", "files": files}))
    for actor, r in results["actors"].items():
        print(actor, json.dumps({t: v["at_frozen_layer"]["auroc_grid"]["instruct_template->instruct_template"] for t, v in r["tasks"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
