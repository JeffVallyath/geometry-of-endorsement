#!/usr/bin/env python3
"""GPU stage of RELATION_READOUT_REPORT_DISCRIMINATION_V1 (Arm B re-elicitation + Arm C extraction).

For one actor, loaded once in exact bfloat16 with the frozen Track A loader:

  1. F1 (C9_MAPPED_ENTAILMENT_8SHOT) candidate scoring on every development and
     held-out evaluation row under all four answer mappings; band activations are
     retained for the row's own frozen mapping.
  2. F2 (DIRECT_SEMANTIC_LABEL_8SHOT) candidate scoring, greedy exact-label
     generation, and band activations on the same rows.
  3. The open 125-board / 500-cell ValuePrism population under the exact M1
     prompt and answer-mapping contract, batch size 1, band activations.
  4. Frozen raw and jointly residualized relation directions projected at the
     primary layer (no refitting) for every retained activation.

``validate`` (CPU, tokenizer only) checks the freeze hashes, candidate
tokenization, and that every ValuePrism prompt's token ids hash to the frozen
batch-size-one causal-baseline value. ``run`` writes into a .partial output
root with an artifact manifest for verify_and_promote.sh. No checkpoint, no
scientific retry: a failed run restarts from a clean output root.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rrrd_v1_common as C  # noqa: E402
import run_relation_binding_factorial_v2_model as RUNNER  # noqa: E402

F2_PARSE = re.compile(r"^\s*(NOT_ENTAILED|ENTAILED)\b")
DIRECTION_KEYS = ("raw_relation_dim", "jointly_residualized_relation")


def load_freeze(freeze_root: Path) -> dict[str, Any]:
    manifest = C.read_json(freeze_root / "artifact_manifest.json")
    bad = [f["path"] for f in manifest["files"] if C.sha256_file(freeze_root / f["path"]) != f["sha256"]]
    if bad:
        raise RuntimeError(f"freeze manifest hash mismatch: {bad}")
    return {
        "prompts": C.read_jsonl(freeze_root / "TRACK_A_REELICITATION_PROMPTS.jsonl"),
        "cells": C.read_jsonl(freeze_root / "NATURALISTIC_CELL_INPUTS.jsonl"),
        "prompt_freeze": C.read_json(freeze_root / "TRACK_A_REELICITATION_PROMPT_FREEZE.json"),
        "contract": C.read_json(freeze_root / "STUDY_CONTRACT.json"),
        "input_manifest": C.read_json(freeze_root / "INPUT_MANIFEST.json"),
        "manifest": manifest,
    }


def load_directions(actor: str) -> dict[str, np.ndarray]:
    bundle = C.TRACK_A_FREEZE / "runtime" / actor / "competitor_directions_v2.npz"
    manifest = C.read_json(C.TRACK_A_FREEZE / "TRACK_A_COMPETITOR_DIRECTIONS.json")
    ent = next(a for a in manifest["actors"] if a["actor"] == actor)
    if C.sha256_file(bundle) != ent["bundle"]["sha256"]:
        raise RuntimeError("competitor direction bundle hash mismatch")
    out = {}
    with np.load(bundle, allow_pickle=False) as z:
        for key in DIRECTION_KEYS:
            arr = np.ascontiguousarray(z[key])
            if C.sha256_bytes(arr.tobytes()).lower() != ent["directions"][key]["sha256"]:
                raise RuntimeError(f"direction {key} hash mismatch")
            out[key] = np.asarray(arr, dtype=np.float64)
    if C.sha256_bytes(np.ascontiguousarray(out["raw_relation_dim"].astype(np.float32)).tobytes()) != C.ACTORS[actor]["raw_relation_dim_sha256"]:
        raise RuntimeError("raw relation DIM is not the frozen actor direction")
    return out


def tokenizer_only(actor: str, cache_dir: Path):
    from transformers import AutoTokenizer

    cfg = C.ACTORS[actor]
    return AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["model_revision"], cache_dir=str(cache_dir), token=os.environ.get("HF_TOKEN"))


def render_cell(tokenizer: Any, actor: str, cell: dict[str, Any]) -> dict[str, Any]:
    mapping = C.m1_mapping_for_item(cell["item_id"])
    if mapping["name"] != cell["mapping_name"]:
        raise RuntimeError(f"mapping drift for {cell['row_id']}")
    messages = C.m1_messages(cell["situation"], cell["consideration"], mapping, C.ACTORS[actor]["chat_policy"])
    ids, rendered = RUNNER.render_chat(tokenizer, messages)
    frozen = (cell.get("reference") or {}).get(actor, {}).get("bs1_prompt_token_ids_sha256")
    return {"prompt_ids": ids, "rendered": rendered, "ids_sha256": C.ids_sha256(ids), "frozen_ids_sha256": frozen, "supports": mapping["supports"], "opposes": mapping["opposes"], "mapping_name": mapping["name"]}


def raw_ids(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return [int(v) for v in ids]


def validate(actor: str, freeze_root: Path, cache_dir: Path, receipt_path: Path | None) -> dict[str, Any]:
    fz = load_freeze(freeze_root)
    directions = load_directions(actor)
    tok = tokenizer_only(actor, cache_dir)
    cand = {}
    for text in (" A", " B", " 1", " 2", " ENTAILED", " NOT_ENTAILED", "A", "B"):
        cand[text] = RUNNER.candidate_ids(tok, text)
    mismatches = []
    for cell in fz["cells"]:
        r = render_cell(tok, actor, cell)
        if r["ids_sha256"] != r["frozen_ids_sha256"]:
            mismatches.append(cell["row_id"])
    prompts = fz["prompts"]
    sha_bad = [p["prompt_id"] for p in prompts if C.sha256_text(p["prompt_text"]) != p["prompt_sha256"]]
    lengths = [len(raw_ids(tok, p["prompt_text"])) for p in prompts[:: max(1, len(prompts) // 50)]]
    receipt = {"schema_version": f"{C.SCHEMA_PREFIX}_VALIDATION_RECEIPT", "actor": actor, "mode": "VALIDATE_TOKENIZER_ONLY", "freeze_manifest_sha256": C.sha256_file(freeze_root / "artifact_manifest.json"), "prompts": len(prompts), "prompt_text_hash_mismatches": sha_bad[:5], "cells": len(fz["cells"]), "cell_prompt_hash_mismatches": len(mismatches), "cell_prompt_hash_mismatch_ids": mismatches[:10], "candidate_token_ids": cand, "sampled_prompt_token_lengths": {"min": min(lengths), "max": max(lengths), "n": len(lengths)}, "directions": {k: {"l2_norm": float(np.linalg.norm(v)), "dimension": int(v.shape[0])} for k, v in directions.items()}, "status": "PASS" if not mismatches and not sha_bad else "FAIL"}
    if receipt_path is not None:
        RUNNER.write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, default=float))
    if receipt["status"] != "PASS":
        raise SystemExit(2)
    return receipt


# ---------------------------------------------------------------- GPU core
class BandCapture:
    """Forward hooks on every band layer capturing the final prompt position."""

    def __init__(self, model: Any, torch: Any, layers: list[int]):
        self.torch = torch
        self.layers = layers
        self.stack = RUNNER.layer_stack(model)
        if max(layers) >= len(self.stack):
            raise RuntimeError("band exceeds decoder depth")
        self.position = 0
        self.enabled = True
        self.captured: dict[int, np.ndarray] = {}
        self.handles = [self.stack[L].register_forward_hook(self._hook(L)) for L in layers]

    def _hook(self, layer: int):
        def hook(_m: Any, _i: Any, output: Any) -> None:
            if not self.enabled:
                return  # generation runs incremental single-position steps; never capture there
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.dtype != self.torch.bfloat16:
                raise RuntimeError(f"layer {layer} activation is not exact bfloat16: {hidden.dtype}")
            self.captured[layer] = hidden[0, self.position, :].detach().float().cpu().numpy().astype(np.float32)

        return hook

    def close(self) -> None:
        for h in self.handles:
            h.remove()


def score_prompt(model: Any, torch: Any, capture: BandCapture, prompt_ids: list[int], candidates: list[list[int]], primary: int, want_activations: bool) -> dict[str, Any]:
    device = next(model.parameters()).device
    n = len(prompt_ids)
    capture.position = n - 1
    seq_scores, first_scores, acts, primary_check = [], [], None, None
    for k, cand in enumerate(candidates):
        capture.captured = {}
        ids = torch.tensor([list(prompt_ids) + list(cand)], dtype=torch.long, device=device)
        with torch.inference_mode():
            out = model(input_ids=ids, use_cache=False)
        logp = torch.log_softmax(out.logits[0].float(), dim=-1)
        total, first = 0.0, None
        for off, tid in enumerate(cand):
            v = float(logp[n - 1 + off, tid].detach().cpu())
            total += v
            if off == 0:
                first = v
        seq_scores.append(total)
        first_scores.append(first)
        if k == 0:
            acts = dict(capture.captured)
        else:
            delta = float(np.max(np.abs(capture.captured[primary] - acts[primary])))
            scale = float(max(1e-6, np.max(np.abs(acts[primary]))))
            # Candidates of different token length change bf16 kernel tiling, so the prompt-position
            # activation differs by a few bf16 quantization steps (Gemma residuals are O(100), where one
            # bf16 step is 1.0); a wrong capture position would differ by O(1) relative. Relative
            # tolerance 5e-2; the delta is recorded per row.
            if delta / scale > 5e-2:
                raise RuntimeError(f"prompt activation changed across candidates: {delta} (scale {scale})")
            primary_check = delta
        del out, logp
    return {"seq_logprobs": seq_scores, "first_token_logprobs": first_scores, "prompt_token_index": n - 1, "activations": acts if want_activations else None, "cross_candidate_max_abs_delta": primary_check}


def generate_label(model: Any, torch: Any, tokenizer: Any, prompt_ids: list[int], capture: BandCapture) -> dict[str, Any]:
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    capture.enabled = False
    try:
        with torch.inference_mode():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=C.F2_MAX_NEW_TOKENS, do_sample=False, num_beams=1, pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    finally:
        capture.enabled = True
    new = out[0, ids.shape[1] :].tolist()
    text = tokenizer.decode(new, skip_special_tokens=True)
    m = F2_PARSE.match(text)
    return {"generated_token_ids": [int(v) for v in new], "generated_text": text, "generated_label": m.group(1) if m else None}


def projections(act: np.ndarray, dirs: dict[str, np.ndarray]) -> dict[str, float]:
    a = np.asarray(act, dtype=np.float64)
    return {k: float(a @ v) for k, v in dirs.items()}


def save_activations(path: Path, prompt_ids: list[str], per_layer: dict[int, list[np.ndarray]]) -> dict[str, Any]:
    arrays = {f"layer_{L}": np.stack(rows).astype(np.float32) for L, rows in per_layer.items()}
    arrays["prompt_ids"] = np.array(prompt_ids)
    RUNNER.atomic_npz(path, **arrays)
    return {"path": path.name, "rows": len(prompt_ids), "layers": sorted(per_layer), "sha256": C.sha256_file(path), "bytes": path.stat().st_size}


def run(actor: str, freeze_root: Path, output_root: Path, cache_dir: Path) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("one-shot output root must be absent or empty")
    output_root.mkdir(parents=True, exist_ok=True)
    fz = load_freeze(freeze_root)
    dirs = load_directions(actor)
    cfg = C.ACTORS[actor]
    primary, band = cfg["primary_layer"], cfg["band"]
    token = os.environ.get("HF_TOKEN")
    tokenizer, model, torch = RUNNER.load_actor(actor, cache_dir, token)
    environment = {"schema_version": f"{C.SCHEMA_PREFIX}_ENVIRONMENT", "actor": actor, "model_id": cfg["model_id"], "model_revision": cfg["model_revision"], "primary_layer": primary, "band": band, "dtype": "bfloat16", "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda_version": torch.version.cuda, "device_name": torch.cuda.get_device_name(0), "transformers": __import__("transformers").__version__, "freeze_manifest_sha256": C.sha256_file(freeze_root / "artifact_manifest.json"), "started_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    RUNNER.write_json(output_root / "environment.json", environment)
    capture = BandCapture(model, torch, band)
    receipts: dict[str, Any] = {}
    try:
        # ---- Arm B: F1 then F2
        for fmt in ("F1", "F2"):
            rows = [p for p in fz["prompts"] if p["format"] == fmt]
            scored, act_ids, per_layer = [], [], {L: [] for L in band}
            for i, p in enumerate(rows):
                if C.sha256_text(p["prompt_text"]) != p["prompt_sha256"]:
                    raise RuntimeError(f"prompt text drift: {p['prompt_id']}")
                ids = raw_ids(tokenizer, p["prompt_text"])
                cands = [RUNNER.candidate_ids(tokenizer, c) for c in p["candidates"]]
                res = score_prompt(model, torch, capture, ids, cands, primary, p["retain_activations"])
                probs = RUNNER.normalized_probabilities(res["seq_logprobs"])
                winner = 0 if res["seq_logprobs"][0] > res["seq_logprobs"][1] else 1  # tie -> NOT_ENTAILED
                rec = {k: p[k] for k in p if k not in ("prompt_text", "context", "claim")}
                rec.update({"prompt_token_count": len(ids), "prompt_ids_sha256": C.ids_sha256(ids), "candidate_token_ids": cands, "seq_logprobs": res["seq_logprobs"], "first_token_logprobs": res["first_token_logprobs"], "native_semantic": p["candidate_semantics"][winner], "native_entailed": int(winner == 0), "native_probability": probs[winner], "margin_entailed_minus_not": res["seq_logprobs"][0] - res["seq_logprobs"][1], "prompt_token_index": res["prompt_token_index"], "cross_candidate_max_abs_delta": res["cross_candidate_max_abs_delta"]})
                if fmt == "F2":
                    rec.update(generate_label(model, torch, tokenizer, ids, capture))
                    rec["generated_matches_candidate"] = None if rec["generated_label"] is None else int(rec["generated_label"] == rec["native_semantic"])
                if res["activations"] is not None:
                    rec["dim_projection"] = projections(res["activations"][primary], dirs)
                    rec["activation_row"] = len(act_ids)
                    act_ids.append(p["prompt_id"])
                    for L in band:
                        per_layer[L].append(res["activations"][L])
                scored.append(rec)
                if i % 200 == 0:
                    print(f"{actor} {fmt} {i}/{len(rows)}", flush=True)
            RUNNER.write_jsonl(output_root / f"track_a_{fmt.lower()}_scores.jsonl", scored)
            receipts[fmt] = {"rows": len(scored), "activations": save_activations(output_root / f"track_a_{fmt.lower()}_activations.npz", act_ids, per_layer)}
        # ---- Arm C: naturalistic cells in the same model load
        cells = fz["cells"]
        scored, act_ids, per_layer = [], [], {L: [] for L in band}
        mism = 0
        for i, cell in enumerate(cells):
            r = render_cell(tokenizer, actor, cell)
            ok = r["ids_sha256"] == r["frozen_ids_sha256"]
            mism += int(not ok)
            cands = [RUNNER.candidate_ids(tokenizer, r["supports"]), RUNNER.candidate_ids(tokenizer, r["opposes"])]
            res = score_prompt(model, torch, capture, r["prompt_ids"], cands, primary, True)
            s_lp, o_lp = res["seq_logprobs"]
            margin = float(np.float32(s_lp - o_lp))
            rec = {"row_id": cell["row_id"], "item_id": cell["item_id"], "board_id": cell["board_id"], "cell_position": cell["cell_position"], "label": cell["label"], "stored_relation": cell["stored_relation"], "consensus_clear": cell["consensus_clear"], "mapping_name": r["mapping_name"], "supports_symbol": r["supports"], "opposes_symbol": r["opposes"], "prompt_token_count": len(r["prompt_ids"]), "prompt_ids_sha256": r["ids_sha256"], "prompt_ids_match_frozen_bs1": ok, "supports_logp": s_lp, "opposes_logp": o_lp, "semantic_margin": margin, "native_sign": 1 if margin > 0 else 0, "prompt_token_index": res["prompt_token_index"], "cross_candidate_max_abs_delta": res["cross_candidate_max_abs_delta"], "dim_projection": projections(res["activations"][primary], dirs), "activation_row": len(act_ids), "reference": cell["reference"][actor]}
            act_ids.append(cell["row_id"])
            for L in band:
                per_layer[L].append(res["activations"][L])
            scored.append(rec)
            if i % 100 == 0:
                print(f"{actor} naturalistic {i}/{len(cells)}", flush=True)
        RUNNER.write_jsonl(output_root / "naturalistic_scores.jsonl", scored)
        receipts["naturalistic"] = {"rows": len(scored), "prompt_hash_mismatches": mism, "activations": save_activations(output_root / "naturalistic_activations.npz", act_ids, per_layer)}
    finally:
        capture.close()
    receipt = {"schema_version": f"{C.SCHEMA_PREFIX}_GPU_RUN_RECEIPT", "actor": actor, "terminal_status": "FORWARD_COMPLETE_ANALYSIS_PENDING", "stages": receipts, "scientific_retries": 0, "checkpoint_used": False, "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())}
    RUNNER.write_json(output_root / "RUN_RECEIPT.json", receipt)
    files = C.manifest_for(output_root)
    RUNNER.write_json(output_root / "artifact_manifest.json", {"schema_version": f"{C.SCHEMA_PREFIX}_GPU_OUTPUT_MANIFEST", "actor": actor, "files": files})
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "activations"} | {"activation_rows": v["activations"]["rows"]} for k, v in receipts.items()}, indent=2))
    return receipt


def score_cell_baseline_path(model: Any, torch: Any, tokenizer: Any, prompt_ids: list[int], cands: list[list[int]]) -> tuple[float, float]:
    """Exact frozen causal-baseline scoring path (review_payoff.intervention.score_reason_with_intervention,
    family no_intervention): both candidate sequences right-padded into one batch of two, attention mask,
    float32 log-softmax, candidate token log-probabilities summed."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    seqs = [[*prompt_ids, *c] for c in cands]
    enc = tokenizer.pad({"input_ids": seqs}, padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(**enc, output_hidden_states=False, use_cache=False, return_dict=True)
        logp = torch.log_softmax(out.logits.float(), dim=-1)
    totals = []
    n = len(prompt_ids)
    for row, c in enumerate(cands):
        vals = torch.stack([logp[row, n + off - 1, tid].float() for off, tid in enumerate(c)])
        totals.append(float(vals.sum(dtype=torch.float32).item()))
    return totals[0], totals[1]


def prompt_only_activations(model: Any, torch: Any, capture: BandCapture, prompt_ids: list[int]) -> dict[int, np.ndarray]:
    """Exact frozen baseline activation path: prompt-only forward, batch size one, final prompt token."""
    device = next(model.parameters()).device
    capture.position = len(prompt_ids) - 1
    capture.captured = {}
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=False, use_cache=False, return_dict=True)
    return dict(capture.captured)


def rerun_naturalistic(actor: str, freeze_root: Path, output_root: Path, cache_dir: Path, limit: int | None, cells_file: Path | None = None) -> dict[str, Any]:
    """Naturalistic-only re-extraction on the frozen baseline's exact scoring and activation paths
    (technical recovery after the first extraction failed the margin-reproduction gate; the first
    extraction is preserved and no probe outcome was computed from it)."""
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("one-shot output root must be absent or empty")
    output_root.mkdir(parents=True, exist_ok=True)
    fz = load_freeze(freeze_root)
    dirs = load_directions(actor)
    cfg = C.ACTORS[actor]
    primary, band = cfg["primary_layer"], cfg["band"]
    tokenizer, model, torch = RUNNER.load_actor(actor, cache_dir, os.environ.get("HF_TOKEN"))
    RUNNER.write_json(output_root / "environment.json", {"schema_version": f"{C.SCHEMA_PREFIX}_ENVIRONMENT", "actor": actor, "mode": "NATURALISTIC_RERUN_BASELINE_SCORING_PATH", "model_id": cfg["model_id"], "model_revision": cfg["model_revision"], "primary_layer": primary, "band": band, "dtype": "bfloat16", "torch": torch.__version__, "cuda_version": torch.version.cuda, "device_name": torch.cuda.get_device_name(0), "transformers": __import__("transformers").__version__, "freeze_manifest_sha256": C.sha256_file(freeze_root / "artifact_manifest.json"), "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "scoring_path": "two candidate sequences right-padded in one batch (frozen causal baseline path); activations from a prompt-only batch-size-one forward (frozen cache path)"})
    capture = BandCapture(model, torch, band)
    cells = C.read_jsonl(cells_file) if cells_file else fz["cells"]
    cells = cells[:limit] if limit else cells
    scored, act_ids, per_layer, mism, within = [], [], {L: [] for L in band}, 0, 0
    try:
        for i, cell in enumerate(cells):
            r = render_cell(tokenizer, actor, cell)
            ok = (r["ids_sha256"] == r["frozen_ids_sha256"]) if r["frozen_ids_sha256"] else None
            mism += int(ok is False)
            cands = [RUNNER.candidate_ids(tokenizer, r["supports"]), RUNNER.candidate_ids(tokenizer, r["opposes"])]
            capture.enabled = False  # scoring forward: no capture (batch of two, stale position)
            try:
                s_lp, o_lp = score_cell_baseline_path(model, torch, tokenizer, r["prompt_ids"], cands)
            finally:
                capture.enabled = True
            margin = float(np.float32(s_lp - o_lp))
            acts = prompt_only_activations(model, torch, capture, r["prompt_ids"])
            ref = (cell.get("reference") or {}).get(actor, {})
            if "bs1_semantic_margin" in ref:
                within += int(abs(margin - ref["bs1_semantic_margin"]) <= 0.002)
            rec = {"row_id": cell["row_id"], "item_id": cell["item_id"], "board_id": cell["board_id"], "cell_position": cell["cell_position"], "label": cell["label"], "stored_relation": cell["stored_relation"], "consensus_clear": cell["consensus_clear"], "mapping_name": r["mapping_name"], "supports_symbol": r["supports"], "opposes_symbol": r["opposes"], "prompt_token_count": len(r["prompt_ids"]), "prompt_ids_sha256": r["ids_sha256"], "prompt_ids_match_frozen_bs1": ok, "supports_logp": s_lp, "opposes_logp": o_lp, "semantic_margin": margin, "native_sign": 1 if margin > 0 else 0, "prompt_token_index": len(r["prompt_ids"]) - 1, "cross_candidate_max_abs_delta": None, "dim_projection": projections(acts[primary], dirs), "activation_row": len(act_ids), "reference": ref, "scoring_path": "baseline_batch_of_two"}
            act_ids.append(cell["row_id"])
            for L in band:
                per_layer[L].append(acts[L])
            scored.append(rec)
            if i % 100 == 0:
                print(f"{actor} naturalistic-rerun {i}/{len(cells)} within_tol_so_far={within}", flush=True)
        RUNNER.write_jsonl(output_root / "naturalistic_scores.jsonl", scored)
        act_rec = save_activations(output_root / "naturalistic_activations.npz", act_ids, per_layer)
    finally:
        capture.close()
    receipt = {"schema_version": f"{C.SCHEMA_PREFIX}_GPU_RUN_RECEIPT", "actor": actor, "mode": "NATURALISTIC_RERUN_BASELINE_SCORING_PATH", "terminal_status": "FORWARD_COMPLETE_ANALYSIS_PENDING", "stages": {"naturalistic": {"rows": len(scored), "prompt_hash_mismatches": mism, "rows_within_0.002_of_frozen_bs1": within, "activations": act_rec}}, "scientific_retries": 0, "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}
    RUNNER.write_json(output_root / "RUN_RECEIPT.json", receipt)
    RUNNER.write_json(output_root / "artifact_manifest.json", {"schema_version": f"{C.SCHEMA_PREFIX}_GPU_OUTPUT_MANIFEST", "actor": actor, "mode": "NATURALISTIC_RERUN", "cells_file": str(cells_file) if cells_file else "freeze/NATURALISTIC_CELL_INPUTS.jsonl", "cells_file_sha256": C.sha256_file(cells_file) if cells_file else None, "files": C.manifest_for(output_root)})
    print(json.dumps(receipt["stages"]["naturalistic"] | {"activations": act_rec["rows"]}, indent=2))
    return receipt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["validate", "run", "rerun-naturalistic"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cells-file", type=Path, help="alternative population (rerun-naturalistic only)")
    ap.add_argument("--actor", required=True, choices=sorted(C.ACTORS))
    ap.add_argument("--freeze-root", type=Path, default=C.FREEZE_ROOT)
    ap.add_argument("--output-root", type=Path)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--receipt", type=Path)
    args = ap.parse_args(argv)
    if args.mode == "validate":
        validate(args.actor, args.freeze_root, args.cache_dir, args.receipt)
        return 0
    if args.output_root is None:
        raise SystemExit("--output-root required for run")
    if args.mode == "rerun-naturalistic":
        rerun_naturalistic(args.actor, args.freeze_root, args.output_root, args.cache_dir, args.limit, args.cells_file)
        return 0
    run(args.actor, args.freeze_root, args.output_root, args.cache_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
