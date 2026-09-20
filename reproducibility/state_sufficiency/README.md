# Do current facts determine later answers?

The CPU audit's private input location and local interpreter paths are removed
in [documented privacy derivatives](../privacy_derivatives.json). Original
identities remain separately bound. Score rows, gates and numerical results
are unchanged.

Exact study IDs: prospective confirmation `V10`, retrospective census `V9`,
archival corroboration `V6`, and combined-update evidence `V7`.

A benchmark case (`root` in saved records) groups starting histories that lead
to the same intended facts. Individual-fact questions check those facts; joint
questions combine them. A qualifying history-dependent question, called a
`witness`, passes the specified factual and reference checks but receives
different answers across histories.

This package separates retrospective recurrence, archival corroboration and
prospective confirmation. The quantitative interpretation is in
[Results and Claims](../../docs/RESULTS_AND_CLAIMS.md).

| Scientific role | Exact directory | What can be checked here |
|---|---|---|
| Prospective matched-history result | `v10` | Primary rates, case counts, stronger probability-margin checks, bootstrap intervals and selected example |
| Independent score-level verification | `independent_verification` | Factual/reference checks, qualifying questions and intervals reconstructed without original analysis imports |
| Retrospective history-dependence census | `v9` | Aggregate counts and denominators, plus the adverse control using current facts from the start |
| Fixed-criterion archival corroboration | `v6` | Aggregate identities and denominators using both answer formats |
| Combined-update capability | `v7` | Triple-update rates and all pass/fail flags, including failed combined requirements |

Run from the repository root with the normal CPU dependencies:

```sh
python -m repro verify
python -m repro evidence
python -m repro.state_sufficiency --output reproduced
python -m pytest tests/test_state_sufficiency.py
```

The ordinary `python -m repro tables` command includes these checks. The dedicated
command exports compact CSV tables and refuses to overwrite existing files.
The checked `headline_values.json` is derived by `repro.state_sufficiency.reconstruct`.
It provides numeric fields for the evidence registry, not an additional experiment.

## Independent per-example reconstruction

The aggregate replay above begins with saved counts. The
[collaborator package](independent_verification/README.md) instead begins with
saved model scores and semantic records. Its calculation never opens reported
expected outputs; comparison is a separate command. A fresh neural rerun would
generate new outputs and is not required or performed here.

```sh
python reproducibility/state_sufficiency/independent_verification/reconstruct.py analyze --output reproduced/independent
python reproducibility/state_sufficiency/independent_verification/reconstruct.py compare --output reproduced/independent
python -m pytest tests/test_independent_state_sufficiency.py
```

The independent script reconstructs complete intended final tables, direct
factual checks, answer mappings, response validity, reference eligibility,
primary/strong witnesses, benchmark case counts and the reported bootstrap intervals.
Its [examples](independent_verification/examples/EXAMPLES.md) expose the full
source histories and scores behind positive, non-witness and excluded cases.

## Provenance and limits

`provenance.json` identifies each original archive and selected member by SHA-256,
records the selection or transformation, and hashes the compact derivative.
Selected members were checked against their supplied archive manifests before
curation. The retrospective and archival source CSV fields remain strings in
the selected summaries; the headline projection converts count fields to integers.
The prospective study's primary values
are reconstructed from saved benchmark case counts, with seeds averaged within benchmark cases and
case prevalence counting every case that qualifies under at least one seed. Its full opportunity denominator
is retained even for ineligible questions.

The combined-update, retrospective and archival entries remain aggregate summaries. The original `v10/` directory
supports saved-count reconstruction. The added `independent_verification/`
directory provides the prospective study's lower-level saved outputs needed to reconstruct
eligibility and witnesses independently. Its own manifest binds every derivative
to the original archives and records field selection, hashes and row counts.
The examples are illustrations, not prevalence estimates.

No primary classification input is missing at the saved-output level. Recorded
validity flags are cross-checked, not trusted as classifications. Candidate score
mass suffices to establish argmax membership for all retained responses. Full
model logits, cache tensors, training packages and fresh neural execution are
not included. Saved repeat scores are independently compared only where the
original study actually recorded them; textual-correction repeats are absent.

Original archives and reviews have no public retrieval location recorded here.
Their hashes identify the sources but do not make omitted evidence available to
an external reproducer. Review-supported prospective chronology, exact-input
nonoverlap and shared-checkpoint identity are not newly audited by these tests.
No neural inference, experiment allocation or private operations logs are part
of this package. The separate [public-data in-context study](../in_context_updates/README.md)
is completed and replayable. Conventional parameter-editor validation remains
unresolved; it must not be conflated with the early unexecuted AlphaEdit route.

## Post-hoc CPU robustness audit

The canonical audit is `V10_CPU_AUDIT_RESULTS_20260917.zip`. Its four members
are retained byte-for-byte in [cpu_audit/](cpu_audit/). This is a post-hoc
reanalysis of the original saved V10 runs. The prospective classifications
remain unchanged.

```sh
python -m repro.cpu_robustness --output reproduced/cpu-audit
python -m pytest tests/test_cpu_robustness.py
```

The replay reads the existing `independent_verification/inputs/` files. It derives
logical answers from the case designs and model answers from saved scores.
It varies which history pairs and additional correctness controls qualify;
the facts, questions, scores and original witness definition stay fixed.
It then compares every exported control flag and matching pair against the
canonical CSV. A surviving disagreement is a history-dependent answer under
the added controls. No model or tokenizer is loaded.

| Canonical artifact | Role |
|---|---|
| [GATE_SPEC.json](cpu_audit/GATE_SPEC.json) | Fixed same-pair and cross-pair question mappings, matching rules and post-hoc status |
| [RESULTS.csv](cpu_audit/RESULTS.csv) | One row per coded opportunity, including eligibility, matching pairs and control intersections |
| [SURVIVORS.json](cpu_audit/SURVIVORS.json) | Surviving examples, full response panels and manual-review selections |
| [CPU_AUDIT_REPORT.md](cpu_audit/CPU_AUDIT_REPORT.md) | Unmodified source report, detailed tables and historical commands |
| [MANIFEST.json](cpu_audit/MANIFEST.json) | Archive/member hashes, input identity, successful-run binding and verifier provenance |

The portable replay adapts the supplied scalar-score verifier rather than
rerunning its historical execution wrapper. It exports `summary.json`,
`by_condition.csv` and `verification.json` into a new directory. The
[paper guide](../../PAPER_GUIDE.md#post-hoc-robustness) maps each claim to exact
CSV and replay fields. [Results and Claims](../../docs/RESULTS_AND_CLAIMS.md#post-hoc-controls-on-the-same-saved-runs)
gives the canonical interpretation and limitations.

The newer audit distinguishes the same-pair logical consistency control from
the cross-pair propagation control. It supersedes the earlier
`V10_SAME_INSTANCE_CPU_RESULTS_20260917.zip` for the G2 interpretation.
That earlier analysis used the same outputs and remains provenance, not an
independent replication. Its [receipt](cpu_audit/provenance/prior_receipt.json)
and technical recovery records are retained without importing its headline
tables as additional evidence.

Both stronger controls require correct answers to all measured direct factual
questions and the original references. The same-pair control checks the other
two logically sufficient compositions of the target relation pair, under the
target answer-code draw. The cross-pair control checks all three compositions
on each of the other two pairs under that draw. The latter therefore checks
propagation elsewhere in the same relational system.

Each model/condition has 1,152 coded opportunities. Learned-method summaries
combine two seeds, giving 2,304 opportunities; shared designs are counted once
within each reported union. These pooled counts are descriptive. Cross-code
fields are counted only where `semantic_representative=True`, giving nine
semantic opportunities per design rather than eighteen coded opportunities.
Both codes must show disagreement on the same matched pair in the same direction.

### Floating-point recovery

An extra equality check initially stopped the earlier audit on a probability
discrepancy of about 2.2e-16. The intended `atol=1e-12, rtol=0` tolerance was
restored for probabilities. Scientific classifications, hard answers, identities
and counts continued to require exact agreement. A regression test covered
one-ULP drift, a changed witness flag and excessive numerical drift. The recovery
reused the hash-bound completed reconstruction.

This was a technical check failure, separate from the scientific result.
The original [incident record](cpu_audit/provenance/INCIDENT_001.md) and
[regression source](cpu_audit/provenance/test_analysis.py) are retained unchanged.
The portable replay tests the same numerical-versus-classification boundary in
[test_cpu_robustness.py](../../tests/test_cpu_robustness.py).
