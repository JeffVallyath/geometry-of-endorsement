# Start here

This is an input package for the fixed comparison, not comparison results.
It contains the selected small editors, fixed cases, original scientific
functions, a source/update wrapper, independent scoring code, and CPU tests.
It does not contain pretrained backbones or the external programmable-KV adapter.

## 1. Set up the CPU environment

Extract the ZIP and open a terminal in its `UPDATE_METHOD_INPUTS` directory.
Use Python 3.11.15. These commands create a separate environment; they do not
download a model or call a model API.

Windows PowerShell:

```powershell
uv python install 3.11.15
uv venv --python 3.11.15 .venv
uv pip install --python .venv/Scripts/python.exe --index-strategy unsafe-best-match -r requirements-cpu.lock.txt
.venv/Scripts/python.exe verify_package.py
```

Linux/macOS shell (CPU wheel availability depends on platform):

```sh
uv python install 3.11.15
uv venv --python 3.11.15 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements-cpu.lock.txt
.venv/bin/python verify_package.py
```

`uv` must already be installed. The CPU lock pins the tested dependency closure,
including the CPU-only PyTorch build. The preparation was tested on Windows;
other platforms have not been verified. No authentication information is supplied.

## 2. Run the CPU checks

```powershell
.venv/Scripts/python.exe check_cpu.py
```

On Linux, replace `.venv/Scripts/python.exe` with `.venv/bin/python`.
This writes `CPU_CHECK.json`. To repeat it, use a new filename, for example
`check_cpu.py --output CPU_CHECK_2.json`. The shipped `verification/CPU_CHECK.json`
records the checks actually run during preparation, not model-quality results.

## 3. Try one supplied case without model inference

```powershell
.venv/Scripts/python.exe run_case.py --case cases/fixed/example_input.json --method REBUILD --output example_output
```

Expected files:

- `example_output/updated_context.txt`: source text with the explicit update applied.
- `example_output/updated_context.json`: immutable-context identity and wrapper metadata.
- `example_output/answer.json`: `PENDING_NO_MODEL`, with the separately loaded question.

The command updates text and exercises the wrapper. It does not invent an answer
or score. Choose a new output directory to run it again. The question file is
opened only after the updated context has been constructed.

## 4. Files you need to touch

- `run_case.py`: one-case entrypoint; use its arguments, normally without editing it.
- `inputs.py`: `prepare(source_text, commands, ...)` then
  `answer(context, question, answer_labels, backbone=...)`.
- `cases/fixed/inputs.jsonl`: updater inputs for both models. Pass only `source_text`
  and `commands` into `prepare`, never the whole evaluation row.
- `cases/fixed/plan.jsonl`: evaluator-side IDs, panels, programs and grouping.
- `cases/fixed/evaluation.jsonl`: later questions, public answer symbols and gold
  labels. Keep this file on the scoring side. `metrics.py` joins labels afterward.
- `models.json`, `weights/SELECTED.json`: exact models and selected factor bindings.
- `METHODS.md`: operations, score conventions, retained information and limitations.
- `requirements-cpu.lock.txt`, `requirements-inference.txt`: separate environments.

The `canonical/` directory is the small source-derived dependency closure; do not
replace it with an older experiment tree. You do not need to edit weights, prompts,
manifests or generators. To reproduce generation in a *new* directory, run
`generate_cases.py --output regenerated_cases`. Do not substitute new cases after
looking at model outcomes. `verify_package.py` rechecks source collisions before
the wrapper permits inference.

## 5. What requires GPU inference

Setup, hashes, population generation, text updating, independent labels and all
CPU tests need no GPU or paid API. CPU tests use synthetic tokens/cache calls;
they do not certify either pretrained tokenizer or model.

Learned editing, real cache compilation, answers, external-method compatibility,
timing and memory measurements require a separately authorized inference run.
Do not use the CPU environment for the model comparison. On a suitable Linux GPU
host, the recorded scientific runtime uses Python 3.12.3 and PyTorch
2.11.0+cu128. For the minimal inference dependencies:

```sh
uv python install 3.12.3
uv venv --python 3.12.3 .venv-gpu
uv pip install --python .venv-gpu/bin/python --index-strategy unsafe-best-match -r requirements-inference.txt
```

This inference setup has **not** been run by the preparation task. Its direct
scientific pins come from the completed study; see `ENVIRONMENT.json`. Before
using it, verify real-tokenizer boundaries, stock SOURCE scores, clone immutability,
precision, both selected models, and the external implementation as required by
the comparison brief. Model access/licensing and downloading the exact snapshots
must be handled separately; the runner only accepts existing local snapshots.

Once those prerequisites are satisfied, an explicit one-case inference command is:

```sh
.venv-gpu/bin/python run_case.py --case cases/fixed/example_input.json --method INV_PAIR_NLL --seed 0 --model-inference --cache /path/to/huggingface/hub --output model_example
```

The `/path/to/huggingface/hub` value is a placeholder, not a private machine path.
`model_example/answer.json` then contains the forced-choice answer and full-vocabulary
normalized label log probabilities. This one-case command is not the full comparison
or a timing benchmark. All fixed models, seeds, panels and alternatives still have
to be measured; do not choose a favorable subset.
