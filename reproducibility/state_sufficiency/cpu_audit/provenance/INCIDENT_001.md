# Technical recovery: floating-point equality in the extra audit

2026-09-18 04:19:20Z: LOCAL attempt `cpu-attempt-001-c3c9bdb4` started under
capsule `3d1038d91f1a212c39a5da7c009d00523d7344de02cdf9a69b96ebdf808bef73`.
All 1,383 original members and 418 frozen scientific files passed integrity checks.
The unmodified independent reconstruction completed and its original comparison
passed. At 04:19:56Z an additional bit-for-bit audit assertion stopped the wrapper
before certification or outcome joining. The runner recorded FAILED and released
its CPU lease. Its reconstruction outputs and stderr/state records are retained.

Cause: the wrapper mistakenly imposed exact binary floating-point equality on
recomputed probability arrays, beyond the upstream verifier's existing
`atol=1e-12, rtol=0` numerical contract. Exactly 82 joint-probability rows and
132 direct-probability rows differ, by at most 2.220446049250313e-16 and
5.551115123125783e-17 respectively. All 16,128 rows' classifications/gold/hard
answers and all 896 root rows agree exactly. No scientific discrepancy was found.

Remediation: exact equality remains mandatory for every classification, hard
answer, identity, denominator and count. Probability arrays use the unchanged
upstream numerical tolerance. A regression exercises one-ULP drift, a changed
witness flag, and excessive probability drift. Attempt 002 resumes the hashed
completed reconstruction, preserving outputs instead of repeating classification.
The GATE_SPEC SHA-256 remains
`2e3c9bdfdf9a20bfa3042211ddeea5fec4af48813c2937d35561a4440727faf2`.
This is bounded technical recovery of the same scientific attempt, not a retry
after observing a new scientific result. No original artifact or threshold changed.

Rollback: retain/use the original capsule and code commit `d51e7b1` for diagnosis;
do not execute it again expecting a different result. Original stderr is at
`D:/research-agent-platform/runs/cpu-attempt-001-c3c9bdb4/logs/stderr.log`.
The only remaining numerical uncertainty is ordinary sub-tolerance floating
roundoff; an independent scalar-score checker validates downstream classifications.
