# Do current facts determine later answers?

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
