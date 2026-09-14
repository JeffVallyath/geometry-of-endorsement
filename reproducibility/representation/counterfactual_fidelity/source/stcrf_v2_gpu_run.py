#!/usr/bin/env python3
"""GPU runner for STRONG_TARGET_COUNTERFACTUAL_RELATION_FIDELITY_V2 (one runner, both actors).

Stages (one actor, one protocol, one split; exact bfloat16; frozen Track A loader; frozen right-padded
batch-of-two candidate scoring path; interventions at the frozen primary layer, final prompt position):

  validate   tokenizer-only: freeze hashes, prompt re-rendering against the prompt manifest, token lengths.
  natural    unsteered base/counterfactual prompts Q1-Q6 (F2 realizations A and B, F1 realization A);
             primary-layer and witness-layer final-position activations captured (protocol selection,
             direction/witness fit).
  calibrate  natural + full-state patch + alpha-grid scans of every direction and the natural
             interpolation grid on Q1 realization A (fit worlds; no tau application).
  full       calibrate + tau-calibrated application of every arm to Q1-Q6 in both realizations, the
             natural interpolation arm, and the F1 mapped-format robustness arms (final worlds).

Per-world checkpoints (JSON + NPZ) are durable and resumable for operational recovery only; a resumed
world must reproduce its checkpoint margins. Nothing here chooses a multiple with Q2-Q6.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rrrd_v1_common as R  # noqa: E402
import run_relation_binding_factorial_v2_model as RUNNER  # noqa: E402
import stcrf_v2_build_corpus as B  # noqa: E402
import stcrf_v2_common as C  # noqa: E402

MARGIN_REPRODUCTION_TOLERANCE = 1e-6


def load_freeze(freeze_root: Path) -> dict[str, Any]:
    manifest = C.read_json(freeze_root / "artifact_manifest.json")
    bad = [f["path"] for f in manifest["files"] if C.sha256_file(freeze_root / f["path"]) != f["sha256"]]
    if bad:
        raise RuntimeError(f"freeze manifest hash mismatch: {bad}")
    worlds = C.read_jsonl(freeze_root / "CORPUS.jsonl")
    prompts = C.read_jsonl(freeze_root / "PROMPT_MANIFEST.jsonl")
    fresh = C.read_jsonl(freeze_root / "DEMOS_V2.jsonl")
    v1_demos = C.load_v1_demonstrations()
    return {"worlds": worlds, "by_world": {w["world_id"]: w for w in worlds}, "prompts": prompts, "by_id": {p["prompt_id"]: p for p in prompts}, "fresh_demos": sorted(fresh, key=lambda d: d["presentation_position"]), "v1_demos": v1_demos, "manifest_sha256": C.sha256_file(freeze_root / "artifact_manifest.json")}


def raw_ids(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return [int(v) for v in ids]


def tokenizer_only(actor: str, cache_dir: Path):
    from transformers import AutoTokenizer

    cfg = C.ACTORS[actor]
    return AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["model_revision"], cache_dir=str(cache_dir), token=os.environ.get("HF_TOKEN"))


def validate(actor: str, freeze_root: Path, cache_dir: Path, receipt_path: Path | None) -> dict[str, Any]:
    fz = load_freeze(freeze_root)
    tok = tokenizer_only(actor, cache_dir)
    cands = {t: RUNNER.candidate_ids(tok, t) for t in (" ENTAILED", " NOT_ENTAILED", " A", " B", " 1", " 2")}
    sample = fz["prompts"][:: max(1, len(fz["prompts"]) // 200)]
    lengths = []
    for p in sample:
        text, _ = B.render_from_manifest(fz["by_world"][p["world_id"]], p, fz["v1_demos"], fz["fresh_demos"])
        lengths.append(len(raw_ids(tok, text)))
    dirs = C.load_track_a_bundle(actor)
    receipt = {"schema_version": f"{C.SCHEMA_PREFIX}_VALIDATION_RECEIPT", "actor": actor, "mode": "VALIDATE_TOKENIZER_ONLY", "freeze_manifest_sha256": fz["manifest_sha256"], "worlds": len(fz["worlds"]), "prompts": len(fz["prompts"]), "rerendered_and_hash_checked": len(sample), "candidate_token_ids": cands, "sampled_prompt_token_lengths": {"min": min(lengths), "max": max(lengths), "n": len(lengths)}, "track_a_directions": {k: {"l2_norm": float(np.linalg.norm(v)), "dimension": int(v.shape[0])} for k, v in dirs.items()}, "status": "PASS"}
    if receipt_path is not None:
        RUNNER.write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, default=float))
    return receipt


# ---------------------------------------------------------------- GPU core
class SiteHook:
    """Forward hook on one decoder layer: optional intervention at one position (primary site only) and capture of every row."""

    def __init__(self, model: Any, torch: Any, layer: int, allow_intervention: bool):
        self.torch = torch
        stack = RUNNER.layer_stack(model)
        if layer >= len(stack):
            raise RuntimeError(f"layer {layer} exceeds decoder depth {len(stack)}")
        self.position = 0
        self.mode: tuple[str, Any] | None = None
        self.allow = allow_intervention
        self.captured: np.ndarray | None = None
        self.handle = stack[layer].register_forward_hook(self._hook)

    def _hook(self, _m: Any, _i: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.dtype != self.torch.bfloat16:
            raise RuntimeError(f"activation is not exact bfloat16: {hidden.dtype}")
        if self.mode is not None:
            if not self.allow:
                raise RuntimeError("intervention requested on a capture-only hook")
            kind, vec = self.mode
            v = self.torch.as_tensor(np.asarray(vec, dtype=np.float32), device=hidden.device)
            if v.dim() == 1:
                v = v[None, :].expand(hidden.shape[0], -1)
            if v.shape[0] != hidden.shape[0]:
                raise RuntimeError(f"intervention rows {v.shape[0]} != batch rows {hidden.shape[0]}")
            if kind == "add":
                hidden[:, self.position, :] = (hidden[:, self.position, :].float() + v).to(self.torch.bfloat16)
            elif kind == "set":
                hidden[:, self.position, :] = v.to(self.torch.bfloat16)
            else:
                raise ValueError(kind)
        self.captured = hidden[:, self.position, :].detach().float().cpu().numpy().astype(np.float32)
        return output

    def close(self) -> None:
        self.handle.remove()


def score_batch(model: Any, torch: Any, tokenizer: Any, prompt_ids: list[int], cands: list[list[int]], primary: SiteHook, witness: SiteHook, mode: tuple[str, Any] | None, rows_per_mode: int = 1) -> tuple[list[float], np.ndarray, np.ndarray]:
    """Frozen right-padded batch scoring: rows_per_mode copies of the candidate pair. For per-row interventions pass a
    (rows*2, hidden) matrix. Returns candidate-sequence log-probabilities in row order and the captured row activations
    (primary site post-intervention, witness layer) for the first candidate row of each mode copy."""
    n = len(prompt_ids)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    seqs = [[*prompt_ids, *c] for _ in range(rows_per_mode) for c in cands]
    L = max(len(s) for s in seqs)
    device = next(model.parameters()).device
    input_ids = torch.full((len(seqs), L), int(pad), dtype=torch.long)
    attn = torch.zeros((len(seqs), L), dtype=torch.long)
    for r, s in enumerate(seqs):
        input_ids[r, : len(s)] = torch.tensor(s, dtype=torch.long)
        attn[r, : len(s)] = 1
    input_ids, attn = input_ids.to(device), attn.to(device)
    primary.position = n - 1
    witness.position = n - 1
    primary.mode = mode
    primary.captured = None
    witness.captured = None
    keep = L - n + 1
    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False, return_dict=True, logits_to_keep=keep)
            logp = torch.log_softmax(out.logits.float(), dim=-1)
    finally:
        primary.mode = None
    totals = []
    for r, s in enumerate(seqs):
        c = s[n:]
        vals = torch.stack([logp[r, off, tid].float() for off, tid in enumerate(c)])
        totals.append(float(vals.sum(dtype=torch.float32).item()))
    cp, cw = primary.captured[0::2], witness.captured[0::2]
    del out, logp
    return totals, cp, cw


def margin_of(pair: list[float]) -> float:
    return float(np.float32(pair[0] - pair[1]))


# ---------------------------------------------------------------- prefix-cached scan path (search only; never a measured row)
class PrefixScan:
    """Prefix KV cache for one prompt, batch-expanded to SCAN_POINTS_PER_FORWARD x 2 rows; each scan forward feeds only
    the final prompt token and the candidate tokens, applies the intervention at that final position, then crops the
    cache back to the prefix. Used only to search the alpha / interpolation grids; every selected multiple is
    re-measured on the frozen full-forward path."""

    def __init__(self, ctx: dict[str, Any], prompt_ids: list[int], cands: list[list[int]]):
        import copy

        from transformers import DynamicCache

        self.ctx = ctx
        torch, model = ctx["torch"], ctx["model"]
        self.device = next(model.parameters()).device
        self.prompt_ids, self.cands = prompt_ids, cands
        self.points = C.SCAN_POINTS_PER_FORWARD
        self.rows = self.points * len(cands)
        prefix = torch.tensor([prompt_ids[:-1]], dtype=torch.long, device=self.device)
        ctx["primary"].position = 0  # capture hooks stay armed; point them inside the prefix
        ctx["witness"].position = 0
        ctx["primary"].mode = None
        with torch.inference_mode():
            out = model(input_ids=prefix, attention_mask=torch.ones_like(prefix), use_cache=True, return_dict=True, past_key_values=DynamicCache(), logits_to_keep=1)
        cache = out.past_key_values
        del out
        self.past_len = len(prompt_ids) - 1
        cache.batch_repeat_interleave(self.rows)
        self.cache = cache
        pad = ctx["tokenizer"].pad_token_id if ctx["tokenizer"].pad_token_id is not None else ctx["tokenizer"].eos_token_id
        suffixes = [[prompt_ids[-1], *c] for _ in range(self.points) for c in cands]
        self.L = max(len(s) for s in suffixes)
        ids = torch.full((self.rows, self.L), int(pad), dtype=torch.long)
        attn = torch.ones((self.rows, self.past_len + self.L), dtype=torch.long)
        for r, s in enumerate(suffixes):
            ids[r, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn[r, self.past_len + len(s) :] = 0
        self.input_ids, self.attn = ids.to(self.device), attn.to(self.device)
        self.suffixes = suffixes

    def scan(self, vecs: np.ndarray, kind: str) -> list[float]:
        """vecs: (P, hidden) one intervention vector per grid point (P <= points). Returns P semantic margins."""
        torch, model, primary, witness = self.ctx["torch"], self.ctx["model"], self.ctx["primary"], self.ctx["witness"]
        P = vecs.shape[0]
        full = np.repeat(np.concatenate([vecs, np.repeat(vecs[-1:], self.points - P, axis=0)], axis=0), len(self.cands), axis=0).astype(np.float32)
        primary.position = 0
        witness.position = 0
        primary.mode = (kind, full)
        try:
            with torch.inference_mode():
                out = model(input_ids=self.input_ids, attention_mask=self.attn, past_key_values=self.cache, use_cache=True, return_dict=True, logits_to_keep=0)
                logp = torch.log_softmax(out.logits.float(), dim=-1)
        finally:
            primary.mode = None
            self.cache.crop(self.past_len)
        totals = []
        for r, s in enumerate(self.suffixes):
            c = s[1:]
            totals.append(float(torch.stack([logp[r, k, tid].float() for k, tid in enumerate(c)]).sum(dtype=torch.float32).item()))
        del out, logp
        return [margin_of(totals[2 * k : 2 * k + 2]) for k in range(P)]

    def scan_grid(self, vec_for: Any, grid: list[float], kind: str) -> list[float]:
        margins: list[float] = []
        for k0 in range(0, len(grid), self.points):
            chunk = grid[k0 : k0 + self.points]
            margins += self.scan(np.stack([vec_for(m) for m in chunk]), kind)
        return margins

    def close(self) -> None:
        del self.cache, self.input_ids, self.attn
        self.ctx["torch"].cuda.empty_cache()


class World:
    """One world under one actor/protocol: prompt cache, natural states, rows and activations."""

    def __init__(self, ctx: dict[str, Any], world: dict[str, Any], protocol: str):
        self.ctx = ctx
        self.world = world
        self.wid = world["world_id"]
        self.protocol = protocol
        self.ids: dict[str, list[int]] = {}
        self.cands: dict[str, list[list[int]]] = {}
        self.rows: list[dict[str, Any]] = []
        self.acts_primary: dict[str, np.ndarray] = {}
        self.acts_witness: dict[str, np.ndarray] = {}
        self.natural: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        self.summary: dict[str, Any] = {"world_id": self.wid, "protocol": protocol, "split": world["split"], "direction": world["factors"]["direction"], "arms": {}}

    def prep(self, fmt: str, real: str, state: str, q: str) -> str:
        fz, tok, cache = self.ctx["fz"], self.ctx["tokenizer"], self.ctx["cand_cache"]
        pid = C.prompt_id(self.wid, self.protocol, fmt, real, state, q)
        if pid not in self.ids:
            row = fz["by_id"][pid]
            text, cands = B.render_from_manifest(self.world, row, fz["v1_demos"], fz["fresh_demos"])
            self.ids[pid] = raw_ids(tok, text)
            self.cands[pid] = [cache.setdefault(c, RUNNER.candidate_ids(tok, c)) for c in cands]
        return pid

    def score(self, pid: str, mode: tuple[str, Any] | None, rows_per_mode: int = 1) -> tuple[list[float], np.ndarray, np.ndarray]:
        c = self.ctx
        return score_batch(c["model"], c["torch"], c["tokenizer"], self.ids[pid], self.cands[pid], c["primary"], c["witness"], mode, rows_per_mode)

    def key(self, fmt: str, real: str, state: str, q: str, intervention: str, tau: float | None) -> str:
        return f"{fmt}|{real}|{state}|{q}|{intervention}|{'' if tau is None else tau}"

    def record(self, fmt: str, real: str, state: str, q: str, intervention: str, tau: float | None, lp: list[float], ap: np.ndarray, aw: np.ndarray, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        pid = C.prompt_id(self.wid, self.protocol, fmt, real, state, q)
        row = {"world_id": self.wid, "protocol": self.protocol, "format": fmt, "realization": real, "state": state, "query": q, "intervention": intervention, "tau": tau, "seq_logprobs": lp, "semantic_margin": margin_of(lp), "prompt_id": pid, "label": self.ctx["fz"]["by_id"][pid]["label"]}
        if extra:
            row |= extra
        if intervention != "NO_INTERVENTION":
            row["delta_margin"] = row["semantic_margin"] - self.natural[(fmt, real, "base", q)]["margin"]
        self.rows.append(row)
        k = self.key(fmt, real, state, q, intervention, tau)
        self.acts_primary[k] = ap[0]
        self.acts_witness[k] = aw[0]
        return row

    # ---- stages
    def run_natural(self, fmt: str) -> None:
        reals = list(C.REALIZATIONS) if fmt == "F2" else ["A"]
        for real in reals:
            for state in ("base", "counterfactual"):
                for q in C.QUERIES:
                    pid = self.prep(fmt, real, state, q)
                    lp, ap, aw = self.score(pid, None)
                    rec = self.record(fmt, real, state, q, "NO_INTERVENTION", None, lp, ap, aw, {"prompt_token_count": len(self.ids[pid]), "prompt_ids_sha256": R.ids_sha256(self.ids[pid])})
                    self.natural[(fmt, real, state, q)] = {"margin": rec["semantic_margin"], "act": ap[0].astype(np.float64)}
        base_q1, cf_q1 = self.natural[(fmt, "A", "base", "Q1")]["margin"], self.natural[(fmt, "A", "counterfactual", "Q1")]["margin"]
        self.summary.setdefault("natural", {})[fmt] = {"base_q1_A": base_q1, "cf_q1_A": cf_q1, "natural_q1_change_A": cf_q1 - base_q1, "targets": {str(t): base_q1 + t * (cf_q1 - base_q1) for t in C.TAUS}, "natural_full_state_l2_q1_A": float(np.linalg.norm(self.natural[(fmt, "A", "counterfactual", "Q1")]["act"] - self.natural[(fmt, "A", "base", "Q1")]["act"]))}

    def run_patch(self, fmt: str) -> None:
        reals = list(C.REALIZATIONS) if fmt == "F2" else ["A"]
        for real in reals:
            for q in C.QUERIES:
                pid = self.prep(fmt, real, "base", q)
                lp, ap, aw = self.score(pid, ("set", self.natural[(fmt, real, "counterfactual", q)]["act"].astype(np.float32)))
                self.record(fmt, real, "base", q, C.FULL_PATCH, None, lp, ap, aw)

    def scanner(self, fmt: str) -> PrefixScan:
        key = f"scan|{fmt}"
        if key not in self.ctx.setdefault("scanners", {}):
            pid = self.prep(fmt, "A", "base", "Q1")
            self.ctx["scanners"][key] = PrefixScan(self.ctx, self.ids[pid], self.cands[pid])
        return self.ctx["scanners"][key]

    def close_scanners(self) -> None:
        for s in self.ctx.get("scanners", {}).values():
            s.close()
        self.ctx["scanners"] = {}

    def scan_direction(self, fmt: str, arm: str, u: np.ndarray, taus: tuple[float, ...]) -> dict[str, Any]:
        """Alpha-grid scan on Q1 realization A along both rays (prefix-cached path); frozen root rule v2; no application."""
        hb, hc = self.natural[(fmt, "A", "base", "Q1")]["act"], self.natural[(fmt, "A", "counterfactual", "Q1")]["act"]
        nat = self.summary["natural"][fmt]
        r = nat["natural_full_state_l2_q1_A"]
        proj = float(u @ (hc - hb))
        s_nat = 1.0 if proj >= 0 else -1.0
        pid = self.prep(fmt, "A", "base", "Q1")
        grid = list(C.ALPHA_GRID)
        sc = self.scanner(fmt)
        m_plus = sc.scan_grid(lambda m: m * r * u, grid, "add")
        m_minus = sc.scan_grid(lambda m: -m * r * u, grid, "add")
        roots = {str(t): C.select_signed(grid, m_plus, m_minus, nat["base_q1_A"], nat["targets"][str(t)], s_nat) for t in taus}
        rec = {"arm": arm, "format": fmt, "natural_sign": s_nat, "natural_projection": proj, "r_l2": r, "grid": grid, "grid_margins_plus": m_plus, "grid_margins_minus": m_minus, "roots": roots, "applied": {}}
        self.rows.append({"world_id": self.wid, "protocol": self.protocol, "format": fmt, "realization": "A", "state": "base", "query": "Q1", "intervention": f"{arm}__GRID", "tau": None, "grid": grid, "grid_margins_plus": m_plus, "grid_margins_minus": m_minus, "natural_sign": s_nat, "r_l2": r, "prompt_id": pid, "scan_path": "prefix_cached"})
        self.summary["arms"].setdefault(fmt, {})[arm] = {k: v for k, v in rec.items() if k not in ("grid", "grid_margins_plus", "grid_margins_minus")}
        return rec

    def apply_direction(self, fmt: str, arm: str, u: np.ndarray, rec: dict[str, Any], taus: tuple[float, ...]) -> None:
        reals = list(C.REALIZATIONS) if fmt == "F2" else ["A"]
        nat = self.summary["natural"][fmt]
        for t in taus:
            root = rec["roots"][str(t)]
            if root["multiple"] is None:
                self.summary["arms"][fmt][arm]["applied"][str(t)] = {"status": root["status"], "multiple": None}
                continue
            m = float(root["multiple"])
            vec = (m * rec["r_l2"] * root["sign"] * u).astype(np.float32)
            for real in reals:
                for q in C.QUERIES:
                    pid = self.prep(fmt, real, "base", q)
                    lp, ap, aw = self.score(pid, ("add", vec))
                    self.record(fmt, real, "base", q, arm, t, lp, ap, aw, {"multiple": m, "intervention_l2": float(m * rec["r_l2"])})
            ach = self.natural[(fmt, "A", "base", "Q1")]["margin"] + next(r_["delta_margin"] for r_ in self.rows if r_["format"] == fmt and r_["realization"] == "A" and r_["query"] == "Q1" and r_["intervention"] == arm and r_["tau"] == t)
            target = nat["targets"][str(t)]
            self.summary["arms"][fmt][arm]["applied"][str(t)] = {"status": "APPLIED", "multiple": m, "sign": root["sign"], "sign_agrees_with_natural": root.get("sign_agrees_with_natural"), "intervention_l2": float(m * rec["r_l2"]), "norm_ratio": m, "achieved_q1_A": ach, "target_q1_A": target, "scan_minus_frozen_q1": float(target - ach), "within_tolerance": C.achieved_ok(ach, target, nat["natural_q1_change_A"]), "hard_answer_counterfactual": bool(ach > 0), "signed_margin_ge_1": bool(ach >= 1.0), "saturated": bool(root.get("saturated", False)), "finite": bool(all(np.isfinite(r_["semantic_margin"]) for r_ in self.rows if r_["format"] == fmt and r_["intervention"] == arm and r_["tau"] == t))}

    def scan_interpolation(self, fmt: str) -> dict[str, Any]:
        hb, hc = self.natural[(fmt, "A", "base", "Q1")]["act"], self.natural[(fmt, "A", "counterfactual", "Q1")]["act"]
        nat = self.summary["natural"][fmt]
        pid = self.prep(fmt, "A", "base", "Q1")
        grid = list(C.INTERPOLATION_GRID)
        margins = self.scanner(fmt).scan_grid(lambda f: hb + f * (hc - hb), grid, "set")
        roots = {str(C.PRIMARY_TAU): C.select_multiple(grid, margins, nat["base_q1_A"], nat["targets"][str(C.PRIMARY_TAU)])}
        rec = {"arm": C.NATURAL_INTERPOLATION, "format": fmt, "grid": grid, "grid_margins": margins, "roots": roots, "applied": {}}
        self.rows.append({"world_id": self.wid, "protocol": self.protocol, "format": fmt, "realization": "A", "state": "base", "query": "Q1", "intervention": f"{C.NATURAL_INTERPOLATION}__GRID", "tau": None, "grid": grid, "grid_margins": margins, "prompt_id": pid, "scan_path": "prefix_cached"})
        self.summary["arms"].setdefault(fmt, {})[C.NATURAL_INTERPOLATION] = {k: v for k, v in rec.items() if k not in ("grid", "grid_margins")}
        return rec

    def apply_interpolation(self, fmt: str, rec: dict[str, Any]) -> None:
        t = C.PRIMARY_TAU
        root = rec["roots"][str(t)]
        nat = self.summary["natural"][fmt]
        if root["multiple"] is None:
            self.summary["arms"][fmt][C.NATURAL_INTERPOLATION]["applied"][str(t)] = {"status": root["status"], "fraction": None}
            return
        f = float(root["multiple"])
        reals = list(C.REALIZATIONS) if fmt == "F2" else ["A"]
        for real in reals:
            for q in C.QUERIES:
                hb, hc = self.natural[(fmt, real, "base", q)]["act"], self.natural[(fmt, real, "counterfactual", q)]["act"]
                pid = self.prep(fmt, real, "base", q)
                lp, ap, aw = self.score(pid, ("set", (hb + f * (hc - hb)).astype(np.float32)))
                self.record(fmt, real, "base", q, C.NATURAL_INTERPOLATION, t, lp, ap, aw, {"fraction": f, "intervention_l2": float(f * np.linalg.norm(hc - hb))})
        ach = self.natural[(fmt, "A", "base", "Q1")]["margin"] + next(r_["delta_margin"] for r_ in self.rows if r_["format"] == fmt and r_["realization"] == "A" and r_["query"] == "Q1" and r_["intervention"] == C.NATURAL_INTERPOLATION and r_["tau"] == t)
        target = nat["targets"][str(t)]
        self.summary["arms"][fmt][C.NATURAL_INTERPOLATION]["applied"][str(t)] = {"status": "APPLIED", "fraction": f, "norm_ratio": f, "achieved_q1_A": ach, "target_q1_A": target, "within_tolerance": C.achieved_ok(ach, target, nat["natural_q1_change_A"]), "hard_answer_counterfactual": bool(ach > 0), "signed_margin_ge_1": bool(ach >= 1.0), "saturated": bool(root.get("saturated", False)), "finite": True}

    def logit_controls(self, fmt: str) -> None:
        nat = self.summary["natural"][fmt]
        self.summary["arms"].setdefault(fmt, {})[C.LOGIT_CONTROL] = {"arm": C.LOGIT_CONTROL, "format": fmt, "applied": {str(t): {"status": "ANALYTIC", "bias": nat["targets"][str(t)] - nat["base_q1_A"], "achieved_q1_A": nat["targets"][str(t)], "target_q1_A": nat["targets"][str(t)], "within_tolerance": True} for t in C.TAUS}}


def run_world(ctx: dict[str, Any], world: dict[str, Any], protocol: str, stage: str, dirs: dict[str, np.ndarray] | None) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    W = World(ctx, world, protocol)
    W.run_natural("F2")
    W.run_natural("F1")
    if stage == "natural":
        W.summary["rows"] = W.rows
        return W.summary, W.acts_primary, W.acts_witness
    assert dirs is not None
    W.run_patch("F2")
    W.run_patch("F1")
    W.logit_controls("F2")
    W.logit_controls("F1")
    arms = [a for a in C.DIRECTION_ARMS if a in dirs]
    recs: dict[str, dict[str, Any]] = {}
    for arm in arms:
        taus = C.TAUS if arm in C.ALL_TAU_ARMS else (C.PRIMARY_TAU,)
        recs[arm] = W.scan_direction("F2", arm, dirs[arm], taus)
    interp = W.scan_interpolation("F2")
    W.close_scanners()
    if stage == "full":
        for arm in arms:
            taus = C.TAUS if arm in C.ALL_TAU_ARMS else (C.PRIMARY_TAU,)
            W.apply_direction("F2", arm, dirs[arm], recs[arm], taus)
        W.apply_interpolation("F2", interp)
        recs1 = {arm: W.scan_direction("F1", arm, dirs[arm], C.TAUS) for arm in C.ALL_TAU_ARMS if arm in dirs}
        W.close_scanners()
        for arm, rec1 in recs1.items():
            W.apply_direction("F1", arm, dirs[arm], rec1, C.TAUS)
    W.summary["rows"] = W.rows
    return W.summary, W.acts_primary, W.acts_witness


def scan_path_check(ctx: dict[str, Any], world: dict[str, Any], protocol: str, dirs: dict[str, np.ndarray], output_root: Path) -> dict[str, Any]:
    """Receipt: the prefix-cached scan path must reproduce the frozen full-forward path on the first world."""
    W = World(ctx, world, protocol)
    W.run_natural("F2")
    hb, hc = W.natural[("F2", "A", "base", "Q1")]["act"], W.natural[("F2", "A", "counterfactual", "Q1")]["act"]
    r = W.summary["natural"]["F2"]["natural_full_state_l2_q1_A"]
    u = dirs["ORIGINAL_FROZEN_DIM"]
    pid = W.prep("F2", "A", "base", "Q1")
    points = [0.05, 0.5, 1.0, 4.0]
    frozen = []
    for m in points:
        lp, _, _ = W.score(pid, ("add", (m * r * u).astype(np.float32)))
        frozen.append(margin_of(lp))
    lp_set, _, _ = W.score(pid, ("set", (hb + 0.5 * (hc - hb)).astype(np.float32)))
    frozen_set = margin_of(lp_set)
    sc = PrefixScan(ctx, W.ids[pid], W.cands[pid])
    fast = sc.scan(np.stack([m * r * u for m in points]), "add")
    fast_set = sc.scan(np.stack([hb + 0.5 * (hc - hb)]), "set")[0]
    fast_nat = sc.scan(np.zeros((1, u.shape[0])), "add")[0]
    sc.close()
    deltas = [abs(a - b) for a, b in zip(frozen, fast)] + [abs(frozen_set - fast_set), abs(fast_nat - W.natural[("F2", "A", "base", "Q1")]["margin"])]
    rec = {"schema_version": f"{C.SCHEMA_PREFIX}_SCAN_PATH_CHECK", "world_id": world["world_id"], "points": points, "frozen_margins": frozen, "prefix_cached_margins": fast, "frozen_set_margin": frozen_set, "prefix_cached_set_margin": fast_set, "natural_base_margin_frozen": W.natural[("F2", "A", "base", "Q1")]["margin"], "natural_base_margin_prefix_cached": fast_nat, "max_abs_delta_nats": float(max(deltas)), "threshold": C.SCAN_PATH_MAX_ABS_DELTA_NATS, "pass": bool(max(deltas) <= C.SCAN_PATH_MAX_ABS_DELTA_NATS)}
    RUNNER.write_json(output_root / "SCAN_PATH_CHECK.json", rec)
    if not rec["pass"]:
        raise RuntimeError(f"prefix-cached scan path deviates from the frozen path: {rec}")
    return rec


def checkpoint_paths(output_root: Path, wid: str) -> tuple[Path, Path]:
    d = output_root / "worlds"
    return d / f"{wid}.json", d / f"{wid}.npz"


def checkpoint_complete(output_root: Path, wid: str) -> bool:
    j, n = checkpoint_paths(output_root, wid)
    if not (j.exists() and n.exists()):
        return False
    try:
        rec = C.read_json(j)
        return rec.get("complete") is True and C.sha256_file(n) == rec.get("npz_sha256")
    except Exception:
        return False


def load_dirs(direction_file: Path | None, actor: str) -> dict[str, np.ndarray] | None:
    if direction_file is None:
        return None
    meta = C.read_json(direction_file.with_suffix(".json"))
    if meta["actor"] != actor:
        raise RuntimeError("direction file actor mismatch")
    d = C.load_direction_file(direction_file, meta["sha256"])
    return {k: C.unit(v) for k, v in d.items()}


def run(actor: str, protocol: str, split: str, stage: str, freeze_root: Path, output_root: Path, cache_dir: Path, direction_file: Path | None, limit: int | None, reproduce_check: int, world_ids: list[str] | None) -> dict[str, Any]:
    fz = load_freeze(freeze_root)
    dirs = load_dirs(direction_file, actor)
    if stage != "natural" and dirs is None:
        raise SystemExit("--direction-file required for calibrate/full")
    cfg = C.ACTORS[actor]
    primary_layer, witness_layer = cfg["primary_layer"], C.WITNESS_LAYER[actor]
    worlds = [w for w in fz["worlds"] if w["split"] == split and (world_ids is None or w["world_id"] in world_ids)]
    if limit:
        worlds = worlds[:limit]
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "worlds").mkdir(exist_ok=True)
    tokenizer, model, torch = RUNNER.load_actor(actor, cache_dir, os.environ.get("HF_TOKEN"))
    env = {"schema_version": f"{C.SCHEMA_PREFIX}_ENVIRONMENT", "actor": actor, "protocol": protocol, "split": split, "stage": stage, "model_id": cfg["model_id"], "model_revision": cfg["model_revision"], "primary_layer": primary_layer, "witness_layer": witness_layer, "dtype": "bfloat16", "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda_version": torch.version.cuda, "device_name": torch.cuda.get_device_name(0), "transformers": __import__("transformers").__version__, "freeze_manifest_sha256": fz["manifest_sha256"], "direction_file": str(direction_file) if direction_file else None, "direction_file_sha256": C.sha256_file(direction_file) if direction_file else None, "direction_arms": sorted(dirs) if dirs else [], "alpha_grid": list(C.ALPHA_GRID), "interpolation_grid": list(C.INTERPOLATION_GRID), "taus": list(C.TAUS), "root_rule": C.ROOT_RULE, "scan_path_rule": C.SCAN_PATH_RULE, "scan_points_per_forward": C.SCAN_POINTS_PER_FORWARD, "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "scoring_path": "right-padded batch of candidate sequences, attention mask, float32 log-softmax on logits_to_keep tail, candidate token log-probabilities summed; intervention hook at the primary layer final prompt position; capture hooks at the primary and witness layers"}
    RUNNER.write_json(output_root / f"environment_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json", env)
    primary = SiteHook(model, torch, primary_layer, True)
    witness = SiteHook(model, torch, witness_layer, False)
    ctx = {"fz": fz, "tokenizer": tokenizer, "model": model, "torch": torch, "primary": primary, "witness": witness, "cand_cache": {}}
    resumed, done, reproduced = [], [], []
    t0 = time.time()
    try:
        # logits_to_keep self-check on the first prompt (must equal the full-logit computation)
        w0 = worlds[0]
        p0 = fz["by_id"][C.prompt_id(w0["world_id"], protocol, "F2", "A", "base", "Q1")]
        text0, c0s = B.render_from_manifest(w0, p0, fz["v1_demos"], fz["fresh_demos"])
        ids0 = raw_ids(tokenizer, text0)
        c0 = [RUNNER.candidate_ids(tokenizer, c) for c in c0s]
        lp_keep, _, _ = score_batch(model, torch, tokenizer, ids0, c0, primary, witness, None)
        with torch.inference_mode():
            pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            seqs = [[*ids0, *c] for c in c0]
            L = max(map(len, seqs))
            ii = torch.full((2, L), int(pad), dtype=torch.long)
            am = torch.zeros((2, L), dtype=torch.long)
            for r_, s in enumerate(seqs):
                ii[r_, : len(s)] = torch.tensor(s)
                am[r_, : len(s)] = 1
            dev = next(model.parameters()).device
            out = model(input_ids=ii.to(dev), attention_mask=am.to(dev), use_cache=False, return_dict=True)
            full = torch.log_softmax(out.logits.float(), dim=-1)
            lp_full = [float(sum(full[r_, len(ids0) - 1 + off, tid] for off, tid in enumerate(c))) for r_, c in enumerate(c0)]
        keep_check = {"logits_to_keep_margin": margin_of(lp_keep), "full_logits_margin": margin_of(lp_full), "abs_delta": abs(margin_of(lp_keep) - margin_of(lp_full))}
        RUNNER.write_json(output_root / "LOGITS_TO_KEEP_CHECK.json", keep_check)
        if keep_check["abs_delta"] > 1e-3:
            raise RuntimeError(f"logits_to_keep path deviates from full logits: {keep_check}")
        if stage != "natural":
            chk = scan_path_check(ctx, w0, protocol, dirs, output_root)
            print(f"scan path check: max abs delta {chk['max_abs_delta_nats']:.4f} nats (pass={chk['pass']})", flush=True)
        for i, w in enumerate(worlds):
            wid = w["world_id"]
            jpath, npath = checkpoint_paths(output_root, wid)
            if checkpoint_complete(output_root, wid):
                if len(reproduced) < reproduce_check:
                    summary, _, _ = run_world(ctx, w, protocol, stage, dirs)
                    old = C.read_json(jpath)
                    key = lambda r_: (r_["format"], r_["realization"], r_["state"], r_["query"], r_["intervention"], str(r_.get("tau")))
                    old_m = {key(r_): r_["semantic_margin"] for r_ in old["rows"] if "semantic_margin" in r_}
                    new_m = {key(r_): r_["semantic_margin"] for r_ in summary["rows"] if "semantic_margin" in r_}
                    diff = max(abs(old_m[k] - new_m[k]) for k in old_m)
                    reproduced.append({"world_id": wid, "max_abs_margin_diff": diff, "identical": diff <= MARGIN_REPRODUCTION_TOLERANCE})
                    if diff > MARGIN_REPRODUCTION_TOLERANCE:
                        raise RuntimeError(f"resumed world {wid} does not reproduce its checkpoint: {diff}")
                resumed.append(wid)
                continue
            summary, ap, aw = run_world(ctx, w, protocol, stage, dirs)
            keys = sorted(ap)
            RUNNER.atomic_npz(npath, keys=np.array(keys), primary=np.stack([ap[k] for k in keys]).astype(np.float32), witness=np.stack([aw[k] for k in keys]).astype(np.float32))
            summary.update({"complete": True, "stage": stage, "npz_sha256": C.sha256_file(npath), "finished_at": dt.datetime.now(dt.timezone.utc).isoformat()})
            RUNNER.write_json(jpath, summary)
            done.append(wid)
            el = time.time() - t0
            nat = summary["natural"]["F2"]
            extra = ""
            if stage != "natural" and "REGIME_MATCHED_DIM" in summary["arms"].get("F2", {}):
                rr = summary["arms"]["F2"]["REGIME_MATCHED_DIM"]["roots"].get("0.75", {})
                extra = f" regime@0.75={rr.get('status')}:{rr.get('multiple')}"
            print(f"{actor} {protocol} {split} {stage} {i + 1}/{len(worlds)} {wid} elapsed={el:.0f}s per_world={el / max(1, len(done)):.1f}s natQ1={nat['natural_q1_change_A']:.2f} r={nat['natural_full_state_l2_q1_A']:.1f}{extra}", flush=True)
    finally:
        primary.close()
        witness.close()
    receipt = {"schema_version": f"{C.SCHEMA_PREFIX}_GPU_RUN_RECEIPT", "actor": actor, "protocol": protocol, "split": split, "stage": stage, "worlds": len(worlds), "completed_this_attempt": done, "resumed_from_checkpoint": resumed, "reproduction_checks": reproduced, "scientific_retries": 0, "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "terminal_status": "FORWARD_COMPLETE_ANALYSIS_PENDING"}
    RUNNER.write_json(output_root / f"RUN_RECEIPT_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json", receipt)
    compiled = [C.read_json(checkpoint_paths(output_root, w["world_id"])[0]) for w in worlds]
    RUNNER.write_jsonl(output_root / "world_summaries.jsonl", compiled)
    RUNNER.write_json(output_root / "artifact_manifest.json", {"schema_version": f"{C.SCHEMA_PREFIX}_GPU_OUTPUT_MANIFEST", "actor": actor, "protocol": protocol, "split": split, "stage": stage, "files": C.manifest_for(output_root)})
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("completed_this_attempt", "resumed_from_checkpoint")} | {"completed": len(done), "resumed": len(resumed)}, indent=2))
    return receipt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["validate", "run"])
    ap.add_argument("--actor", required=True, choices=sorted(C.ACTORS))
    ap.add_argument("--protocol", choices=C.PROTOCOLS, default="P1")
    ap.add_argument("--split", choices=C.SPLITS, default="selection")
    ap.add_argument("--stage", choices=["natural", "calibrate", "full"], default="natural")
    ap.add_argument("--freeze-root", type=Path, default=C.FREEZE_ROOT)
    ap.add_argument("--output-root", type=Path)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--direction-file", type=Path, help="frozen per-actor unit-direction npz (with sibling .json hashes)")
    ap.add_argument("--limit-worlds", type=int)
    ap.add_argument("--world-ids", help="comma list (smoke tests only)")
    ap.add_argument("--reproduce-check", type=int, default=1)
    ap.add_argument("--receipt", type=Path)
    args = ap.parse_args(argv)
    if args.mode == "validate":
        validate(args.actor, args.freeze_root, args.cache_dir, args.receipt)
        return 0
    if args.output_root is None:
        raise SystemExit("--output-root required for run")
    wids = [w.strip() for w in args.world_ids.split(",")] if args.world_ids else None
    run(args.actor, args.protocol, args.split, args.stage, args.freeze_root, args.output_root, args.cache_dir, args.direction_file, args.limit_worlds, args.reproduce_check, wids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
