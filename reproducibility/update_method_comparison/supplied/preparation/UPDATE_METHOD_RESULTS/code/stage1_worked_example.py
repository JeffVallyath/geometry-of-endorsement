"""START_HERE.md Step 4 / COMPARISON_SPEC.md Stage 1 -- external implementation check.

Runs the RELEASED account-role worked example on Qwen3-8B. This is an
implementation anchor, NOT a statistical replication of the paper, and NOT part
of the matched Stage 2 comparison.

Why this file exists instead of `python -m editkv.example Qwen/Qwen3-8B`
-----------------------------------------------------------------------
Two minimal, documented compatibility changes are required. Both are recorded in
notes/IMPLEMENTATION_NOTES.md and neither alters the scientific operation.

  PATCH 1  editkv/example.py prints STALE, IN_PLACE, ERRATUM, FULL_REPREFILL,
           the diagnostic, and AUTO. The spec's Stage 1 requires stale, clean
           recomputation, erratum, AND field-plus-erratum. Mode.FIELD_PLUS_ERRATUM
           exists in editkv/core.py but the released example never calls it.
           We add that one condition. Nothing else changes.

  PATCH 2  editkv/example.py hardcodes device_map="cuda". Kept as-is on a CUDA
           host; exposed as --device only so the failure is legible elsewhere.

The published POLICY / CONVO / DECISION strings and the driver's decoding mode
are preserved verbatim by importing them from the pinned checkout rather than
retyping them.

AUTO mode and the per-question diagnostic ARE exercised here, because Stage 1 is
the external anchor and the paper's own example reports them. They are recorded
and then EXCLUDED from Stage 2, where they would give the editor access to future
answers that the learned updater does not have.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="pinned programmable-kv checkout")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from editkv import EditableContext, Mode
    from editkv.core import DEFAULT_TRIGGER
    from editkv.diagnostics import needs_erratum
    from editkv.example import POLICY, CONVO, DECISION, build

    record = dict(schema="STAGE1_WORKED_EXAMPLE_V1", model=args.model,
                  device=args.device, trigger_template=DEFAULT_TRIGGER,
                  patches=["FIELD_PLUS_ERRATUM condition added to the released example",
                           "device_map exposed as a flag"],
                  source="released editkv/example.py at the pinned commit",
                  fresh_result=True, released_json=False, started=time.time())

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="sdpa", trust_remote_code=True).eval()

    ctx = build(model, tok)
    record["field_span"] = list(ctx.fields["account_role"].span)

    def g(value, mode):
        t0 = time.perf_counter()
        text = ctx.generate("account_role", value, mode,
                            decision_prompt=DECISION, max_new_tokens=6).strip()
        return text, time.perf_counter() - t0

    conditions = {}
    # the four conditions the spec names, in its order
    conditions["stale"] = g("verified_admin", Mode.STALE)
    conditions["full_reprefill"] = g("suspended_user", Mode.FULL_REPREFILL)
    conditions["erratum"] = g("suspended_user", Mode.ERRATUM)
    conditions["field_plus_erratum"] = g("suspended_user", Mode.FIELD_PLUS_ERRATUM)  # PATCH 1
    # released example also reports these; recorded, excluded from Stage 2
    conditions["in_place"] = g("suspended_user", Mode.IN_PLACE)
    conditions["auto"] = g("suspended_user", Mode.AUTO)

    record["conditions"] = {k: dict(decision=v[0], seconds=v[1]) for k, v in conditions.items()}

    d = needs_erratum(ctx, "account_role", "suspended_user", probe=DECISION)
    record["diagnostic"] = dict(
        needs_erratum=bool(d.needs_erratum), in_place=d.in_place_decision,
        erratum=d.erratum_decision, stale=d.stale_decision,
        logit_drift=float(d.logit_drift), note=getattr(d, "note", None),
        excluded_from_stage2="gives access to future answers the learned updater lacks")

    # The documented expectation: the erratum path recovers 'escalate' where the
    # stale cache stays on 'refund'. We record whether that is recovered; we do
    # not assert the paper fails if it is not.
    record["expected_tool_decision_recovered"] = (
        conditions["field_plus_erratum"][0].lower().startswith("escalate")
        and conditions["stale"][0].lower().startswith("refund"))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(json.dumps(record["conditions"], indent=2))
    print("recovered:", record["expected_tool_decision_recovered"])
    print("EXAMPLE_OK")


if __name__ == "__main__":
    raise SystemExit(main())
