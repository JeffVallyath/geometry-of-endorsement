# Fitted relation readouts

These small parameter files contain the fitted difference-in-means direction and
logistic probe used to read support or opposition from activations. They contain
no pretrained backbone weights or dataset text.

| File | Model | Selected layer | Direction dimension |
|---|---|---:|---:|
| [Llama parameters](m1_relation/llama_layer19.npz) | Meta-Llama-3.1-8B-Instruct | 19 | 4,096 |
| [Gemma parameters](m1_relation/gemma_layer27.npz) | Gemma-2-9B-it | 27 | 3,584 |

The files preserve fitted coefficients, intercepts, selection-split scaling,
model/tokenizer revisions and scientific split/template hashes. The original
repository commit field has been removed. All retained array bytes are unchanged.
Container timestamps use a neutral packaging epoch, not an experiment date.

The `difference_in_means_direction` arrays also match the counterfactual study's
recorded original frozen directions. Their presence supplies that specific
parameter dependency; other required fitted components and fresh model execution
need separate qualification.

Load with `numpy.load(path, allow_pickle=False)`. The direction score subtracts
`difference_in_means_midpoint` from the activation dot product. Standardize with
the stored `dim_scaler_mu` and `dim_scaler_sigma` when reproducing standardized
interaction scores. The logistic parameters have their own stored scaling.

[manifest.json](manifest.json) hashes the actual sanitized parameter files.

<a id="recovered-gemma-provenance"></a>

## Gemma training data

The [recorded fit metadata](m1_relation/provenance/training_provenance.json)
establishes 107,368 fitted rows for Gemma's frozen MiniLM comparator and records
the encoder revision. This population differs from its 1,500-row activation-probe
training split. The [membership-only manifests](m1_relation/provenance/probe_membership.json)
identify the Gemma activation probe's 1,500 training, 300 selection and 500 evaluation
identifiers. Source text, reference labels and cell-role labels are withheld.

Exact MiniLM fitted-row membership and current Llama fit provenance remain
unestablished. The exact Gemma aggregate split manifest and full
configuration are also unavailable. See the [reproduction matrix](../../REPRODUCIBILITY.md#reproduction-levels)
for the resulting execution limits.
