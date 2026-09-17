"""LATEST_ERRATUM and FIELD_PLUS_LATEST_ERRATUM for the supplied fixed comparison.

`inputs.prepare` raises NotImplementedError for both of these methods:

    if method in ('LATEST_ERRATUM','FIELD_PLUS_LATEST_ERRATUM'):
        raise NotImplementedError('External Stage 1 adapter and stock-equivalence/
        token compatibility checks are required; no replacement implementation supplied')

This module is that adapter. It does NOT modify inputs.py, canonical/, the fixed
cases, or the learned-editor checkpoints. It reuses inputs.validate / inputs.normalize
for the source/command boundary and Backbone.compile for cache construction, so the
serving path (serve_questions / metrics.score_case) is byte-identical to every other arm.

Provenance of the update pattern
--------------------------------
The trigger template is copied verbatim from the released implementation at the
pinned commit (see external_reproduction/commit.txt):

    editkv/core.py :: DEFAULT_TRIGGER

Adaptation, per COMPARISON_SPEC.md "Fixed alternatives":
  - Published operation addresses a single named *field* (`account_role`).
  - The supplied task addresses whole *record clauses* of the canonical form
    "<Actor> is in favor of|opposed to the <Project> proposal."
  - We substitute field identifiers and requested values ONLY. We do not supply
    logical consequences, recommended answers, or any task-specific reasoning.

This is a documented adaptation from a single field to the explicitly addressed
clauses. It is NOT a claim that the paper evaluated the supplied task.

Scientific cautions preserved from the released README
------------------------------------------------------
  - Single erratum for the CURRENT value, never a stacked history (a non-monotonic
    history such as A->B->A lets a salient intermediate state dominate). This is
    why both erratum arms consume inputs.normalize(commands), not the raw sequence.
  - IN_PLACE / field refresh requires the replacement to be length-preserving in
    TOKENS. Stock build_cache SILENTLY swallows LengthChangeError for
    FIELD_PLUS_ERRATUM. The spec forbids silent degradation, so this module catches
    it explicitly and marks the row (`refresh_fallback=True`).

UNVERIFIED AGAINST A REAL BACKBONE. This file was written without GPU/tokenizer
access. code/preflight.py is the gate that must pass before any measurement.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import replace

import inputs as INP
from inputs import AddressedUpdate, UpdatedContext, digest

# ---------------------------------------------------------------- fixed template

# Verbatim from editkv/core.py::DEFAULT_TRIGGER at the pinned commit.
PUBLISHED_TRIGGER = ("[STATE UPDATE] {label} has changed to {value}; this overrides any "
                     "earlier value AND any earlier conclusion. Apply the current value.")

# Clause-level identifier binding. Field identifiers and requested values only.
#
# FROZEN BEFORE MEASUREMENT. The binding is hashed into template_fingerprint()
# and logged by preflight before any evaluation runs. Wording is part of what this
# comparison tests (LATEST_SAME_WORDING vs LATEST_ERRATUM isolates exactly that),
# so changing it after seeing any model outcome would invalidate the arm.
LABEL_PATTERN = "{actor}'s position on the {project} proposal"
VALUE_TRUE = "in favor"
VALUE_FALSE = "opposed"

STANCE = {1: VALUE_TRUE, 0: VALUE_FALSE}
# stance text as it appears INSIDE the canonical clause (see inputs.CLAUSE / inputs.rewrite).
# NOTE these differ in length: "in favor of" is 11 chars, "opposed to" is 10.
CLAUSE_STANCE = {1: "in favor of", 0: "opposed to"}
TAIL = "End of records.\n"


def template_fingerprint() -> dict:
    """Exact adapter template, hashed. MUST be logged before any evaluation."""
    blob = "\n".join([PUBLISHED_TRIGGER, LABEL_PATTERN, VALUE_TRUE, VALUE_FALSE])
    return dict(
        schema="ERRATUM_TEMPLATE_V1",
        published_trigger=PUBLISHED_TRIGGER,
        label_pattern=LABEL_PATTERN,
        value_true=VALUE_TRUE,
        value_false=VALUE_FALSE,
        source="editkv/core.py::DEFAULT_TRIGGER",
        adaptation="single named field -> explicitly addressed record clauses",
        stacked_history=False,
        sha256=hashlib.sha256(blob.encode("utf8")).hexdigest(),
    )


# ---------------------------------------------------------------- text rendering

def _parse_clause(source_text: str, c: AddressedUpdate) -> re.Match:
    match = INP.CLAUSE.fullmatch(source_text[c.start:c.end])
    if match is None:
        raise ValueError("Addressed span is not a canonical record clause")
    return match


def erratum_line(source_text: str, c: AddressedUpdate) -> str:
    m = _parse_clause(source_text, c)
    label = LABEL_PATTERN.format(actor=m["actor"], project=m["project"])
    return PUBLISHED_TRIGGER.format(label=label, value=STANCE[c.value])


def erratum_block(source_text: str, commands):
    """Rendered erratum lines + normalized commands, in stable source-record order.

    Lines are ALWAYS rendered against the ORIGINAL source text, where the supplied
    command offsets are valid. They are then appended verbatim to whichever prefix
    needs them. Re-parsing commands against a REBUILT text is incorrect: the stance
    swap is not length-preserving in characters, so every later offset shifts.
    """
    active = INP.normalize(commands)
    ordered = tuple(sorted(active, key=lambda c: c.start))
    lines = "".join(erratum_line(source_text, c) + "\n" for c in ordered)
    return lines, ordered


def append_block(text: str, lines: str) -> str:
    """Insert the block immediately before the 'End of records.' tail.

    Identical insertion point to inputs.rewrite for EXISTING_CORRECTION and
    LATEST_SAME_WORDING, so the ONLY difference between LATEST_SAME_WORDING and
    LATEST_ERRATUM is the wording itself.
    """
    if not text.endswith(TAIL):
        raise ValueError("Expected supplied source text ending with End of records.")
    return text[:-len(TAIL)] + lines + TAIL


def erratum_text(source_text: str, commands):
    lines, ordered = erratum_block(source_text, commands)
    return append_block(source_text, lines), ordered


def fresh_clause_spans(source_text: str, ordered):
    """Original-text clause spans mapped to their spans in the REBUILT text.

    REBUILD swaps 'in favor of' <-> 'opposed to', a +/-1 character change, so
    offsets shift cumulatively. Returns [(stale_span, fresh_span, char_delta), ...]
    in ascending source-record order.
    """
    offset, out = 0, []
    for c in sorted(ordered, key=lambda x: x.start):
        m = _parse_clause(source_text, c)
        delta = len(CLAUSE_STANCE[c.value]) - len(m["stance"])
        out.append(((c.start, c.end), (c.start + offset, c.end + offset + delta), delta))
        offset += delta
    return out


def rebuilt_text(source_text: str, commands) -> str:
    """Final-state text. Used only as the SOURCE OF TOKENS for the clause refresh.

    This is the same operation inputs.rewrite performs for REBUILD. It is an
    ordinary textual construction from the supplied commands; it uses no gold
    label, origin, or evaluator final-world dictionary.
    """
    return INP.rewrite(source_text, commands, "REBUILD")


# ---------------------------------------------------------------- refresh feasibility

def refresh_plan(backbone, stale_prefix: str, fresh_prefix: str, span_pairs):
    """Token-level feasibility of the published field refresh, per addressed clause.

    Checks the ACTUAL tokenizer and cache interface, never string lengths. Never
    pads, aligns, or splices unequal-length spans.

    Returns (plan, report). plan is [] when the refresh is infeasible, in which case
    the caller MUST fall back to erratum-only and mark the row.
    """
    stale_ids = backbone.prefix_ids(stale_prefix)
    fresh_ids = backbone.prefix_ids(fresh_prefix)
    report = dict(stale_tokens=len(stale_ids), fresh_tokens=len(fresh_ids),
                  char_deltas=[d for _, _, d in span_pairs], clauses=[])

    if len(stale_ids) != len(fresh_ids):
        report["reason"] = "PREFIX_TOKEN_LENGTH_CHANGE"
        return [], report

    plan = []
    for stale_span, fresh_span, delta in span_pairs:
        s_pos = backbone.clause_positions(stale_prefix, stale_span)
        f_pos = backbone.clause_positions(fresh_prefix, fresh_span)
        entry = dict(stale_char=list(stale_span), fresh_char=list(fresh_span),
                     char_delta=delta,
                     stale_token_span=[s_pos[0], s_pos[-1] + 1],
                     fresh_token_span=[f_pos[0], f_pos[-1] + 1])
        if s_pos != f_pos:
            entry["ok"] = False
            entry["reason"] = "CLAUSE_TOKEN_SPAN_CHANGE"
            report["clauses"].append(entry)
            report["reason"] = "CLAUSE_TOKEN_SPAN_CHANGE"
            return [], report
        entry["ok"] = True
        report["clauses"].append(entry)
        plan.append((s_pos[0], s_pos[-1] + 1))

    plan.sort()
    report["order"] = "ascending_source_record_offset"
    report["cached_prefix_per_clause"] = [
        dict(span=[a, b], attends_to_refreshed_clauses=[list(x) for x in plan[:i]])
        for i, (a, b) in enumerate(plan)
    ]
    return plan, report


# ---------------------------------------------------------------- cache surgery

def _refresh_clauses(backbone, art, fresh_prefix: str, plan):
    """Overwrite each addressed clause's K/V with its exact recomputed K/V.

    Ported from editkv/core.py::EditableContext._refresh_field_inplace at the
    pinned commit, retargeted from a Field span to a canonical record clause span
    and from editkv's own DynamicCache to the canonical Backbone artifact cache.
    Stock cache mechanics (DynamicCache layer keys/values, cache_position slice
    forward) are reused unchanged.
    """
    t = backbone.t
    cache = art["cache"]
    fresh_ids = backbone.prefix_ids(fresh_prefix)
    ids = t.tensor([list(fresh_ids)], device=backbone.device)

    for (s, e) in plan:
        prefix = backbone.clone_cache(cache)
        # truncate the clone to the tokens strictly before the clause
        for layer in prefix.layers:
            layer.keys = layer.keys[:, :, :s, :].contiguous()
            layer.values = layer.values[:, :, :s, :].contiguous()
        with t.no_grad():
            out = backbone.model(
                input_ids=ids[:, s:e],
                past_key_values=prefix,
                cache_position=t.arange(s, e, device=backbone.device),
                use_cache=True,
            )
        for i in range(len(cache.layers)):
            cache.layers[i].keys[:, :, s:e, :] = out.past_key_values.layers[i].keys[:, :, s:e, :]
            cache.layers[i].values[:, :, s:e, :] = out.past_key_values.layers[i].values[:, :, s:e, :]

    # the artifact hash is part of the serving contract and must be recomputed
    art["hash"] = backbone.cache_hash(cache)
    art["bytes"] = backbone.cache_bytes(cache)
    return art


# ---------------------------------------------------------------- public entrypoint

def prepare_erratum(source_text, commands, *, method, backbone, actor):
    """Build an UpdatedContext for LATEST_ERRATUM or FIELD_PLUS_LATEST_ERRATUM.

    Question-blind: no question, gold label, origin, or final-world dictionary
    is accepted or consulted. Mirrors inputs.prepare's contract exactly.
    """
    if method not in ("LATEST_ERRATUM", "FIELD_PLUS_LATEST_ERRATUM"):
        raise ValueError("Undeclared method for this adapter")
    if backbone is None:
        raise ValueError("Erratum methods require a supplied backbone")
    if actor != backbone.cfg["key"]:
        raise ValueError("Backbone actor mismatch")

    started = time.perf_counter()
    INP.validate(source_text, commands)
    lines, ordered = erratum_block(source_text, commands)
    text = append_block(source_text, lines)
    pre_seconds = time.perf_counter() - started

    telemetry = dict(method=method, refresh_attempted=False, refresh_fallback=False,
                     refresh_report=None, refresh_seconds=None,
                     template_sha256=template_fingerprint()["sha256"])

    INP._sync(backbone)
    t0 = time.perf_counter()
    ids = tuple(backbone.prefix_ids(text))
    art = backbone.compile(ids, grad=False)
    INP._sync(backbone)
    prefill_seconds = time.perf_counter() - t0

    if method == "FIELD_PLUS_LATEST_ERRATUM":
        telemetry["refresh_attempted"] = True
        # same erratum block appended to the REBUILT source; char offsets remapped
        fresh_prefix = append_block(rebuilt_text(source_text, commands), lines)
        span_pairs = fresh_clause_spans(source_text, ordered)
        plan, report = refresh_plan(backbone, text, fresh_prefix, span_pairs)
        telemetry["refresh_report"] = report
        if not plan:
            # Library-documented erratum-only fallback, as a FIXED policy.
            # Marked, counted, and reported separately. Never silent.
            telemetry["refresh_fallback"] = True
        else:
            INP._sync(backbone)
            t1 = time.perf_counter()
            art = _refresh_clauses(backbone, art, fresh_prefix, plan)
            INP._sync(backbone)
            telemetry["refresh_seconds"] = time.perf_counter() - t1

    if backbone.cache_hash(art["cache"]) != art["hash"]:
        raise RuntimeError("Compiled cache hash mismatch")

    context = UpdatedContext(
        method=method, text=text,
        source_sha256=digest(source_text), updated_text_sha256=digest(text),
        commands=ordered, artifact_sha256=art["hash"],
        preprocess_seconds=pre_seconds,
        compile_seconds=prefill_seconds + (telemetry["refresh_seconds"] or 0.0),
        selected_weight_sha256=None, _artifact=art,
    )
    telemetry["source_prefill_seconds"] = prefill_seconds
    return context, telemetry
