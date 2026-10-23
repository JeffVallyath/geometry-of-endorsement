# Replaying the saved results

The small example runs from the Git checkout with the
[README setup](../../README.md#run-the-saved-example-check). The instructions below
run the complete analyses from the software and data archives.

## Clone and locate the code

On Windows, use a short checkout path and enable long-path support:

```text
git clone --config core.longpaths=true --branch BRANCH REPOSITORY_URL geometry-review
```

Replace `BRANCH` and `REPOSITORY_URL` with the branch and clone URL supplied with
the repository. For an existing checkout, run `git config --local core.longpaths true`.
The [code map](../../CONTRIBUTING.md#code-navigation) links each implementation.
Run `python reproduce.py --help` from the Git root to inspect the replay options.

## Archive handoff for full replay

Obtain the separately distributed `software.zip` and `data.zip`, then extract
them into one new directory outside the Git checkout. From the Git root, replace
`PATH/TO` with the directory containing the ZIPs:

```text
python -m zipfile -e "PATH/TO/software.zip" "../submitted"
python -m zipfile -e "PATH/TO/data.zip" "../submitted"
```

Together they create `../submitted/history_updates/`. Python's ZIP reader
supports the LZMA compression used by the data archive.

Keep this input tree separate from Git and from replay outputs. The runner
checks every input against the original manifest, so extra or changed files
inside it are errors.

## Set up the environments

Use Python 3.13. The setup script creates three environments alongside the
extracted input tree: core analysis, historical tensor checks and corrected GMeLLo.

```text
python ../submitted/history_updates/environments/setup.py --plan
python ../submitted/history_updates/environments/setup.py
```

Installation uses package indexes and the pinned dependency files. The script
refuses to overwrite an existing environment.

Activate the core environment from the Git root:

```powershell
..\submitted\.venv-core\Scripts\Activate.ps1
```

On Linux/macOS, the corresponding command is
`source ../submitted/.venv-core/bin/activate`. Windows with Python 3.13.12 is
tested; native Linux/macOS execution has not been verified.
If activation is unavailable, invoke that environment's Python executable
directly in place of `python` below.

## Run all results or one section

With the core environment active, run from the Git root:

```text
python reproduce.py --snapshot "../submitted/history_updates" --verify
python reproduce.py --snapshot "../submitted/history_updates" --all --output "../full replay"
```

The wrapper verifies the inputs and calls the original analysis runner.
It finds the two other sibling environments automatically. Use
`--historical-python PATH` and `--gmello-python PATH` for environments elsewhere.

For one section, replace `--all` with `--section NAME`:

| Name | Analyses |
|---|---|
| `matched_histories` | Synthetic and public history-dependence measurements |
| `cache_interventions` | Cache interventions, refresh and their controls |
| `external_state` | Current-state repair and interpreter results |
| `interpreter_checks` | History exposure and trained QA/PROGRAM comparisons |
| `gmello_comparison` | Corrected graph-edit comparison and contrasts |

Choose a new output directory outside both the checkout and input tree.
`--test --output "../test results"` runs the submitted protection tests.
Direct calls to the extracted runner, rather than the root wrapper, should use
absolute output paths.

## Read the outputs

`RESULTS.md` points to the generated tables and observations.
`RUN_STATUS.json` records which stages completed.
The [paper map](paper_map.md) connects those outputs to the current paper.
A failed stage keeps its diagnostic log; add `--diagnostics` to retain successful
stage logs too.

Replay recomputes counts, checks and figures from recorded model outputs.
It does not train models or obtain new model responses. The source files for
generation, training and serving are linked in the
[code map](../../CONTRIBUTING.md#code-navigation); the
[asset inventory](../../HANDOFF.md#available-assets-and-gaps) lists the model,
adapter and runtime dependencies needed for new experiments.
