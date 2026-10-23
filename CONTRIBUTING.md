# Find the observation, then the implementation

Start from an [example](examples/README.md) or a result in the
[paper map](history_updates/docs/paper_map.md). The table below points to the
implementation. Keep changes on a branch and preserve the recorded observations.

## Code navigation

Study implementations are under `history_updates/experiments/`; compact example
inputs are under `examples/records/`. The links below go to actual files, including
shared code used by more than one study.

| Job | File and function |
|---|---|
| Generate fresh synthetic cases and controls | [prepare_data.py](research_sources/fresh_history/prepare_data.py): `build_root`, `gate_mapping` |
| Recount the examples | [review_examples.py](tools/review_examples.py): `synthetic`, `public` |
| Decode synthetic scores and apply controls | [analyze.py](history_updates/experiments/matched_histories/v3/studies/original_public_history/reproducibility/fresh_history_validation/source/analyze.py): `decode`, `truth`, `primary` |
| Bind scores to questions, tokens and cache lengths | [replay.py](history_updates/experiments/matched_histories/v3/studies/original_public_history/reproducibility/fresh_history_validation/source/replay.py): `check_case`; [panels.py](history_updates/experiments/matched_histories/v3/studies/original_public_history/reproducibility/fresh_history_validation/source/panels.py): `panel_index` |
| Render and apply update requests | [inputs.py](history_updates/experiments/matched_histories/v3/studies/original_public_history/reproducibility/learned_editors/inputs.py): `rewrite`, `prepare`, `answer` |
| Apply activation hooks | [adapter.py](history_updates/experiments/matched_histories/v3/studies/original_public_history/reproducibility/learned_editors/canonical/adapter.py): `Backbone._hook`, `Backbone.compile` |
| Address sentence tokens and clone the cache | [base_adapter.py](history_updates/experiments/matched_histories/v3/studies/original_public_history/reproducibility/learned_editors/canonical/base_adapter.py): `Backbone.clause_positions`, `Backbone.clone_cache` |
| Score public answers and guards | [mqmh.py](history_updates/experiments/interpreter_checks/v3/studies/history_exposure/sources/ppr1/reference/mqmh.py): `parse_response`, `score`, `endpoints`, `summarize` |
| Build equal current records and verify recorded execution | [example.py](history_updates/experiments/external_state/v3/src/repro/example.py): `current_records`, `verify_pair`, `build` |
| Apply updates and enumerate current-state paths | [state_grounding.py](history_updates/experiments/interpreter_checks/v3/studies/history_exposure/runtime/payload/vendor/state_grounding.py): `materialize`, `current_payload`, `enumerate_paths` |
| Score and choose a candidate operation | [ranker_contract.py](history_updates/experiments/interpreter_checks/v3/studies/history_exposure/runtime/payload/vendor/ranker_contract.py): `request`, `choose` |
| Store versions and execute a dictionary operation | [pst1_core.py](history_updates/experiments/interpreter_checks/v3/studies/history_exposure/runtime/payload/vendor/pst1_core.py): `Store.apply`, `Store.query`, `execute` |
| Train and serve the QA/PROGRAM readers | [model_runtime.py](research_sources/readers/model_runtime.py): `train`, `Engine`; [worker.py](research_sources/readers/worker.py): `evaluate_case` |
| Train and score the ranking interpreter | [qsa_model.py](research_sources/readers/qsa_model.py): `training`, `Scorer`, `score_arm` |
| Apply corrected GMeLLo graph edits and queries | [gmello.py](history_updates/experiments/gmello_comparison/gmello/inference/gmello.py): `parse_generated`, `graph_for_history`, `query_generated`; [evaluate.py](history_updates/experiments/gmello_comparison/gmello/evaluate.py): `evaluate` |
| Full section orchestration | [submitted reproduce.py](history_updates/reproduce.py): `commands`; [root wrapper](reproduce.py): `command` |
| Resolve archived paths and preserve hashes | [release.py](history_updates/experiments/_shared/release.py): `verify`, `mount`; [transport.py](history_updates/experiments/_shared/transport.py): `restore` |
| Export paper tables | [paper_outputs.py](history_updates/experiments/_shared/paper_outputs.py): `export` |
| Plot fresh synthetic results | [result_figures.py](history_updates/experiments/matched_histories/v3/studies/original_public_history/reproducibility/fresh_history_validation/source/result_figures.py) |

The [asset inventory](HANDOFF.md#available-assets-and-gaps) lists the dependencies
needed before executing the generation or training code. For filename and symbol
searches, use the examples in [AGENTS.md](AGENTS.md#find-files-quickly).

If a result names a logical `v3/...` path that is not present in the checkout,
look it up in [study_index.json](history_updates/metadata/study_index.json).
For example, use the `v3` entry `studies/original_public_history/src/repro/mqmh.py`;
its `included` field gives the path relative to `history_updates/`.
Shared code can be stored under another study's
directory. Entries marked `inline` are stored in the index itself.

## Minimal contribution workflow

For a documentation or tooling change, create a branch, edit the relevant files,
and run the affected checks. The onboarding suite checks the examples, source
bindings, CLI behavior and local links:

```text
python -B -X utf8 -m unittest discover -s tests_onboarding -v
```

Open a pull request with the reason for the change and the checks you ran. A
prose-only change does not need a full replay of the scientific analyses.

For a new experiment, identify the question and code entry point first. Use a new
study ID and output directory. Record the inputs, model revisions, controls,
compute allowance and stopping rule before execution. Compare with the existing
results, including changed eligibility, refusals and errors. Keep the original
observations, manifests and scoring definitions available as the baseline.
