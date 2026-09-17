"""Single dispatch point for every declared arm of the fixed comparison.

Twelve runs per plan row:

  deterministic (seed=None)          SOURCE
                                     REBUILD
                                     EXISTING_CORRECTION
                                     LATEST_SAME_WORDING
                                     LATEST_ERRATUM               <- adapter
                                     FIELD_PLUS_LATEST_ERRATUM    <- adapter
  learned (seed in {0,1})            INV_PAIR_NLL
                                     FREE_PAIR_CONSISTENCY
  canonical replay (seed in {0,1})   CANONICAL_REPLAY

Nothing here selects, filters, reweights, or drops a condition. Every declared
arm runs on every fixed case. A condition that performs badly still reports.
"""
from __future__ import annotations

import time

import inputs as INP
from inputs import RECIPES
import erratum_adapter as EA

ADAPTER_METHODS = ("LATEST_ERRATUM", "FIELD_PLUS_LATEST_ERRATUM")
DETERMINISTIC = INP.TEXT_METHODS + ADAPTER_METHODS
LEARNED = RECIPES + ("CANONICAL_REPLAY",)

# The spec's primary contrasts are the two learned recipes vs FIELD_PLUS_LATEST_ERRATUM.
# Everything else is explicitly secondary and must remain visible regardless of outcome.
PRIMARY_BASELINE = "FIELD_PLUS_LATEST_ERRATUM"


def arms():
    """Fixed, outcome-independent enumeration. Declared before measurement."""
    out = [(m, None) for m in DETERMINISTIC]
    out += [(m, s) for m in LEARNED for s in (0, 1)]
    return tuple(out)


def condition_name(method, seed):
    learned = method in RECIPES or method == "CANONICAL_REPLAY"
    return method + (f"_s{seed}" if learned else "")


def build_context(source_text, commands, *, method, seed, backbone, actor):
    """Returns (UpdatedContext, telemetry dict). Question-blind on every path."""
    if method in ADAPTER_METHODS:
        if seed is not None:
            raise ValueError("Deterministic baselines carry seed=None")
        return EA.prepare_erratum(source_text, commands, method=method,
                                  backbone=backbone, actor=actor)

    learned = method in LEARNED
    if learned and seed not in (0, 1):
        raise ValueError("Original writer seeds only")
    if not learned and seed is not None:
        raise ValueError("Deterministic baselines carry seed=None")

    INP._sync(backbone)
    t0 = time.perf_counter()
    context = INP.prepare(source_text, commands, method=method, backbone=backbone,
                          actor=actor, seed=seed)
    INP._sync(backbone)
    wall = time.perf_counter() - t0

    return context, dict(
        method=method, seed=seed,
        refresh_attempted=False, refresh_fallback=False, refresh_report=None,
        refresh_seconds=None,
        # inputs.prepare recompiles the source prefix for learned/replay arms and
        # charges that full work. We do not discount it. Reported as measured.
        source_prefill_seconds=context.compile_seconds,
        prepare_wall_seconds=wall,
        template_sha256=None,
    )


def retained_state(method):
    """Information and resource privileges each arm actually holds.

    COMPARISON_SPEC.md: "Readable source text, command history, and source backup
    are real information/resource privileges: report them for each method rather
    than pretending every method has identical retained state."
    """
    learned = method in LEARNED
    return dict(
        method=method,
        readable_source_text=True,
        full_command_history=method in ("EXISTING_CORRECTION",) or learned,
        normalized_last_value_only=method in ("LATEST_SAME_WORDING", "LATEST_ERRATUM",
                                              "FIELD_PLUS_LATEST_ERRATUM",
                                              "CANONICAL_REPLAY"),
        source_backup_retained=method in ("SOURCE", "REBUILD"),
        stale_cache_semantics=method in ADAPTER_METHODS,
        recompiles_source_prefix=method not in ADAPTER_METHODS,
        learned_parameters=learned and method != "CANONICAL_REPLAY",
        # Explicitly excluded by the spec for every arm:
        answer_key=False, origin_label=False, final_world_dictionary=False,
        future_question_access=False, per_question_diagnostic=False, auto_mode=False,
    )
