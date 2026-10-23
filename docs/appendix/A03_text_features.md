# Text-only baseline and training data

Repository topic A03. See manuscript Section 2.1 and Appendix A.3.

The comparator fits a logistic classifier to frozen sentence embeddings. It reads the input text and never reads model activations. The exact [feature implementation](../../src/relation_study/m1/text_baseline.py) is retained.

## Features and fit

The encoder identifier is `sentence-transformers/all-MiniLM-L6-v2`. The code supplies
no immutable revision to its constructor. The Gemma run record specifies revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; that observation does not make the original
constructor pinned. A matching fitted record for the current Llama artifact remains
unestablished.

Situation and consideration fields are separately converted to strings and embedded in batches of 256. `normalize_embeddings=True` normalizes each embedding. The source applies no additional text cleaning in this function. It concatenates four feature blocks in this order:

`[situation, consideration, situation * consideration, abs(situation - consideration)]`.

The product and absolute difference are elementwise. No further feature standardization is applied. The classifier is `LogisticRegression(C=1.0, max_iter=3000)` with remaining library defaults. Supports is encoded as one. Evaluation scores are class-one probabilities. The encoder stays frozen and the classifier is fitted once before scoring the supplied evaluation frames.

<a id="training-provenance"></a>

## Training data

The Gemma record reports 107,368 fitted comparator rows. Its activation
probe uses 1,500 training rows. These are different training populations even
though evaluation uses the same boards.

The full-run implementation selects the original common-training manifest; smoke
mode uses the smaller pilot training frame. Manifest choice alone does not identify
the rows actually fitted. The supplied record establishes the Gemma fitted count
and revision, but not its exact MiniLM membership list. Separate membership-only
files identify its activation-probe splits. These findings are documented with the
[retained parameters](../../artifacts/probe_weights/README.md#recovered-gemma-provenance).
The supplied comparator scores are the recorded run values.

## Evaluation boundary

Selection and evaluation use the same fixed development frames as the activation analysis. Selection scores determine the comparator's standardization for the checkerboard contrast. The classifier itself has no additional regularization search. The reported Llama comparator interaction is 0.2842197585 with AUROC 0.536096.

Synthetic unit tests verify the feature block order, normalized-embedding request,
input strings and classifier settings without loading an encoder. The
[reproduction matrix](../../REPRODUCIBILITY.md#reproduction-levels) records the remaining
inputs needed for refitting. Aggregate replay does not substitute for that refit.
