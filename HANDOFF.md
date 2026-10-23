# Working handoff

## Objective and current state

The project tests whether a model's answers still depend on earlier facts after
an update, and compares internal interventions with external-state repair.
Start with the [README](README.md), then use the
[paper map](history_updates/docs/paper_map.md) for the result you want to inspect.

The small CPU replay checks the synthetic example and recounts the public
questions from saved outputs. It needs no model weights. Full replay uses the
separately distributed software and data archives.

## Directory map

| Need | Location |
|---|---|
| Paper and synthetic-method slides | [papers/](papers/README.md) |
| Exact examples, prompts and checks | [examples/](examples/README.md) |
| Scientific implementations | [history_updates/experiments/](history_updates/experiments) |
| Result tables and claim calculations | [results/tables/](history_updates/results/tables) |
| Logical-to-physical file paths | [study_index.json](history_updates/metadata/study_index.json) |
| Small replay, readable views and input checks | [tools/](tools) |
| Generation and reader-training source | [research_sources/](research_sources) |

[AGENTS.md](AGENTS.md#find-files-quickly) has filename and symbol searches.
Use those or the code map before scanning a whole source tree.

## Commands

From the Git root, in the Python 3.13 environment described in the README:

```text
python -B -X utf8 tools/review_examples.py --all --output "../review examples"
python -B -X utf8 -m unittest discover -s tests_onboarding -v
python reproduce.py --help
```

The example should report 160/160 synthetic checks and public counts of 94/384
and 101/384. Outputs must be new directories outside the checkout.

For the complete analyses, follow the
[full-replay setup](history_updates/docs/reproduction.md#archive-handoff-for-full-replay).
Keep the extracted inputs separate from Git. The root wrapper verifies them and
runs the original analyses; it also resolves relative output paths.

## Available assets and gaps

| Asset | Where to look / what is needed |
|---|---|
| Synthetic model revisions | [Gemma-2-9B and Qwen3-8B pins](examples/records/synthetic/model-pins.json); obtain base weights separately |
| Public diagnostic models | [Qwen2.5-7B and Gemma-2-9B pins](examples/records/public/SOURCE_HASHES.json); message plans and responses are included |
| Repair-model revisions | [7B](examples/records/public/repair-qwen7-model.json), [32B](examples/records/public/repair-qwen32-config.json), [27B immutable lock](examples/records/public/repair-gemma27-lock.json) |
| Synthetic setters | Selected weights and initialization files are in `data.zip`, located through the study index |
| 7B repair and QA/PROGRAM adapters | Six adapters are stored separately; see [reader source and checkpoint names](research_sources/readers/README.md) |
| 32B/27B repair adapters | Final adapters still need to be supplied before fresh execution |
| Training data and sample order | `data.zip` includes training pairs, reader joins, sample orders and training records through the study index |
| Case generation | [Generator source](research_sources/fresh_history/README.md); execution also needs the original inventory, path bindings and model metadata |
| Training and serving | [Reader source](research_sources/readers/README.md); model files, adapters and relative runtime dependencies must be assembled before use |

The Git checkout supports reading and saved-output replay. These source files
are entry points for extending the work, not a complete fresh-inference setup.

## Code entry points for an extension

The [code map](CONTRIBUTING.md#code-navigation) links the functions for case
generation, updates, activation hooks, cache cloning, scoring, current-state
construction, interpretation, execution and training. The
[methods](history_updates/docs/methods.md) define the checks and denominators.

For an ordinary documentation or tooling change, follow the
[contribution workflow](CONTRIBUTING.md#minimal-contribution-workflow).
For a new experiment, record the question, controls, inputs and compute limits
under a new study ID. Keep the existing observations and scoring definitions as
the comparison baseline. The [review questions](docs/review_and_extensions.md)
offer possible starting points.

## Practical notes

On Windows, use a short clone path and enable repository-local
`core.longpaths`; the replay guide gives the clone command.
Use the root wrapper for full replay. Direct calls to the extracted runner
should use absolute output paths.

Keep experimental prompts and source bindings with new results, but leave
private conversations and credentials out of Git. The existing
[rights and disclosure](history_updates/docs/rights.md) continue to apply.
