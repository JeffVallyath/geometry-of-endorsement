# Working in this repository

Start with [README.md](README.md) and use the file map below for the task at hand.
[HANDOFF.md](HANDOFF.md) covers commands and model/data availability.
Do not read the whole repository or its history before locating the relevant code.

## Find files quickly

| Task | Start here |
|---|---|
| Find a paper result | [Current paper map](history_updates/docs/paper_map.md), then [tables](history_updates/results/tables) |
| Inspect a measured example | [Synthetic](examples/synthetic/README.md), [public](examples/public/README.md), [compact records](examples/records) |
| Change example replay or presentation | [review_examples.py](tools/review_examples.py), [review_support.py](tools/review_support.py), [review_views.py](tools/review_views.py) |
| Find an experiment's implementation | [File/function map](CONTRIBUTING.md#code-navigation) |
| Find generation or training source | [Generator](research_sources/fresh_history/README.md), [readers](research_sources/readers/README.md) |
| Check an onboarding change | [tests_onboarding/](tests_onboarding) |
| Locate an archived logical path | [study_index.json](history_updates/metadata/study_index.json) |

Use `rg --files` for filenames, then `rg -n` for a symbol in the relevant area.
For example, from the Git root:

```text
rg --files history_updates/experiments -g mqmh.py
rg -n "^def (parse_response|score|endpoints)\(" history_updates/experiments -g mqmh.py
rg --files research_sources -g prepare_data.py
rg -n "^def (build_root|gate_mapping)\(" research_sources/fresh_history/prepare_data.py
```

A logical source path can map to shared code in another study's directory.
Query the index instead of reconstructing the data tree just to find a file:

```text
python -c "import json; i=json.load(open('history_updates/metadata/study_index.json', encoding='utf-8')); print(i['mounts']['v3']['studies/original_public_history/src/repro/mqmh.py']['included'])"
```

The printed path is relative to `history_updates/`. Data paths may require the
separate software/data archives. Entries with `inline` contain their data in the index.
Use filename filters or `rg -l` before reading large result files.

## Changes and checks

Work on a branch. For documentation and tooling changes, edit the relevant
files and run the affected checks:

```text
python -B -X utf8 -m unittest discover -s tests_onboarding -v
```

For example/scoring changes, also run:

```text
python -B -X utf8 tools/review_examples.py --all --output "../review examples"
```

Do not run the full scientific replay for prose-only changes. Report checks
actually run and distinguish them from earlier results.

## Boundaries

- Preserve the recorded observations, scientific source and immutable input
  manifests. Put new experiments and outputs under a new study ID.
- Keep the exact checker in `history_updates/experiments/_shared/release.py`
  intact. Full replay uses a separate software/data extraction.
- Do not silently change guards, denominators, seeds, prompt wrappers or result
  selection. Use the paper and recorded inputs to establish scientific claims.
- Reading and saved-output replay do not authorize fresh inference, training,
  model downloads, paid compute, manuscript changes or publication.
- Keep outputs outside the Git checkout and submitted inputs. Do not overwrite
  existing outputs or discard failed diagnostics.
- Keep names and account links that identify contributors, private conversations,
  task requests, credentials, personal paths and provider identifiers out of Git,
  including commit messages and document metadata. Preserve scientific citations,
  public benchmark entities, license and disclosure notices.
- Use finite subprocess timeouts and hidden Windows child processes.
- Keep one primary agent and integrator. Do not delegate by default or force-push.

Publication, repository visibility and access changes require explicit authorization.
