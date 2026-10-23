# Start with the observed answers

Two starting histories can receive the same updates and pass the required fact
checks, yet give different answers to the same downstream question.

1. [Synthetic walkthrough](synthetic/README.md): the measured H5/H6 example from Figure 1.
2. [Public walkthrough](public/README.md): a two-hop MQuAKE example, followed by the paper's Levada case.
3. Recount their checks from the saved observations:

```text
python -m pip install -r requirements-onboarding.lock
python -B -X utf8 tools/review_examples.py --all --output "../review examples"
```

Run from the repository root with Python 3.13. The small route needs NumPy,
not PyTorch, model weights, tokens or a GPU. It blocks network and model-library
imports. It refuses existing output directories and writes outside the inputs.
`REPORT.md` is the short result; `results.json` contains all checks and decisions.

The [compact records](records/BINDINGS.json) retain source hashes, case IDs,
model identities, seeds, token bindings and original responses. The export uses
the existing XZ-to-gzip resolver. The replay calls the submitted synthetic
decoder and gates, and the submitted MQuAKE parser and eligibility functions.
It also recounts all 384 public questions for each original model.

To compare the compact records with a complete local release, add
`--verify-source "PATH/TO/history_updates"`. To also check the selected saved
native prompt plans against their original files, add
`--native-source "PATH/TO/original-mqmh-source"`. Neither is needed for the small
replay. [Export code](../tools/export_review_examples.py) records the selection
and verifies the full original inputs before exporting.

This command reuses recorded model outputs. It does not run new inference or
claim to reproduce training. The [full reproduction guide](../history_updates/docs/reproduction.md)
describes the separate frozen-snapshot routes and their environments.
