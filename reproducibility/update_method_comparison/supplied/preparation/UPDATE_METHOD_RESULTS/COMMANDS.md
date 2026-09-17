# Exact commands

Run in order. Every command writes a retained receipt. Later stages refuse to run
if an earlier receipt is missing.

Target runtime: **Colab A100 40GB** (matches the recorded scientific runtime in
`ENVIRONMENT.json`). L4 24GB works. **T4 will not** — 16GB cannot hold
Gemma-2-9B in bf16, and `base_adapter.Backbone` hard-requires unmodified bf16 on
GPU with no offload.

Placeholders: `$PKG` = extracted `UPDATE_METHOD_INPUTS`, `$OUT` = this results
directory, `$HF` = local HF snapshot cache, `$REPO` = pinned programmable-kv
checkout.

---

## 0. Layout

`code/*.py` import `inputs`, `metrics`, and `canonical` directly, so they must sit
next to them:

```sh
cp $OUT/code/*.py $PKG/
cd $PKG
```

## 1. Model access (do this yourself — no credentials belong in this package)

`google/gemma-2-9b-it` is a gated repository. Accept the licence on Hugging Face
under your own account, then put a read token in **Colab Secrets** as `HF_TOKEN`
and enable notebook access. Do not paste a token into a cell, a file, or this
package.

```python
from google.colab import userdata
import os; os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

## 2. Environment

The recorded scientific runtime is Python 3.12.3 / torch 2.11.0+cu128. Colab's
preinstalled torch will usually differ. Record whatever you actually get — a
deviation is an integration fact to report, not something to hide.

```sh
pip install -q -r $PKG/requirements-inference.txt
python -c "import torch,transformers,sys; print(sys.version, torch.__version__, torch.version.cuda, transformers.__version__)"
```

## 3. Download the pinned snapshots (revisions are fixed in models.json)

`Backbone` loads with `local_files_only=True`, so snapshots must exist first.

```python
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-8B",       revision="b968826d9c46dd6066d109eabc6255188de91218", cache_dir="$HF")
snapshot_download("google/gemma-2-9b-it", revision="11c9b309abf73637e4b6f9a3fa1e92e615547819", cache_dir="$HF")
```

## 4. Package verification and CPU checks (Step 5)

```sh
cd $PKG
python verify_package.py
python check_cpu.py --output $OUT/raw/CPU_CHECK_colab.json
python preflight.py --stage cpu --package $PKG --output $OUT/raw/PREFLIGHT_CPU.json
```

The CPU preflight logs the frozen erratum template hash. Expected:
`2d88092b1c5dfa5da26f59ed7262713fb1b28d4f920579aa991ca5d5d0467c68`.
If it differs, the template was edited — stop and re-freeze deliberately.

## 5. External anchor — Stage 1 worked example (Step 4)

```sh
git clone https://github.com/19PINE-AI/programmable-kv $REPO
cd $REPO && git checkout e9085eafcc6c83e60c548de69060c4bd5b210c96 && git rev-parse HEAD
pip install -q -e $REPO

cd $PKG
python stage1_worked_example.py \
  --repo $REPO --model Qwen/Qwen3-8B \
  --output $OUT/external_reproduction/outputs/worked_example.json \
  > $OUT/external_reproduction/stdout.txt \
  2> $OUT/external_reproduction/stderr.txt
```

If this fails for a concrete compatibility reason, record the exact error and
**stop the external-method branch**. Do not substitute an invented method. A
missing dependency or unsupported operation is an integration issue, not an
unsuccessful scientific result.

## 6. Model preflight — the gate (Step 7)

```sh
python preflight.py --stage model --package $PKG --cache $HF --actor qwen  --output $OUT/raw/PREFLIGHT_MODEL_qwen.json
python preflight.py --stage model --package $PKG --cache $HF --actor gemma --output $OUT/raw/PREFLIGHT_MODEL_gemma.json
```

**Read `refresh_census` in both before continuing.** If `fallback_rate` is high,
`FIELD_PLUS_LATEST_ERRATUM` is collapsing toward `LATEST_ERRATUM` and the headline
must say so. This does not change the plan — all arms still run — it changes how
the primary contrast is reported.

## 7. Smoke run on DEV only (Step 7, implementation errors only)

DEV is for adapter checks, never efficacy selection. Do not read it as a result.

```sh
python run_comparison.py --package $PKG --raw $OUT/raw/dev --cache $HF \
  --panel DEV --limit 2
```

Freeze the implementation here. Record hashes before the fixed evaluation:

```sh
python build_manifest.py --root $OUT --output $OUT/MANIFEST.json --stage frozen
```

## 8. Full fixed comparison (Step 8)

19,392 context builds, ~531k scored answers. Sharded by
`(actor, panel, method, seed)`; finished shards are skipped on re-run. Run this
cell repeatedly across sessions until no shards remain.

```sh
python run_comparison.py --package $PKG --raw $OUT/raw --cache $HF --max-seconds 36000
```

Resume after any disconnect with the identical command. Check what is left:

```sh
python run_comparison.py --package $PKG --raw $OUT/raw --report-gaps
```

`--report-gaps` is the record of identified missing cells if compute runs short.
Never substitute a favourable subset for the missing shards.

Optional narrower shards if sessions are short:

```sh
python run_comparison.py --package $PKG --raw $OUT/raw --cache $HF --actor qwen  --panel ORIGIN
python run_comparison.py --package $PKG --raw $OUT/raw --cache $HF --actor gemma --panel ORIGIN
```

`ORIGIN` carries the primary endpoint, so run it first if compute is tight.
`SOURCE` is always scheduled first within each `(actor, panel)` because every
other arm is scored against the original same-interface SOURCE answers.

## 9. Tables and figure (Step 9)

```sh
python analysis.py --package $PKG --raw $OUT/raw --tables $OUT/tables
python figure.py --tables $OUT/tables --output $OUT/figure/comparison.png
python build_manifest.py --root $OUT --output $OUT/MANIFEST.json --stage final
```

## 10. Before sharing

`build_manifest.py` refuses to finalize if it finds a token-like string. Also
confirm by eye that no `/content/drive/...` paths, account data, or unrelated
logs are present in `raw/` or `external_reproduction/`.
