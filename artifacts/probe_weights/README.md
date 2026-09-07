# Probe weights

```
m1_relation/
  llama_layer19.npz    Meta-Llama-3.1-8B-Instruct, layer 19, 4096-d
  gemma_layer27.npz    google/gemma-2-9b-it,        layer 27, 3584-d
```

## What these are

A probe here is just **one vector the length of the model's hidden state**.

Run the model on a situation plus a reason, grab its internal state at the layer
named above, and take the dot product with the vector. You get one number. The
sign says whether the model is treating that reason as supporting or opposing
the action; positive means supports. That number is the "support/opposition
direction" everything in Part I of
[Results and Claims](../../docs/RESULTS_AND_CLAIMS.md) is built on.

Each file has two probes fitted on the same data:

- **difference-in-means** — the simple one. Average the internal states of all
  the Supports examples, average the Opposes ones, subtract. That difference,
  normalized, is the direction.
- **logistic** — a trained linear classifier over the same states. Usually a bit
  more accurate, less interpretable.

Both were fitted on 1,500 training examples and scored on 500 held-out ones.
They are fitted parameters, not model weights — you still need the model itself
to produce the internal states they get applied to.

## Using them

```python
import numpy as np

probe = np.load("artifacts/probe_weights/m1_relation/llama_layer19.npz")
direction = probe["difference_in_means_direction"]      # (4096,)
midpoint = float(probe["difference_in_means_midpoint"])
layer = int(probe["selected_layer"])                    # 19

# h = the model's residual stream at `layer`, just before it answers, shape (n, 4096)
score = h.astype(np.float32) @ direction - midpoint     # > 0 means Supports
```

To compare scores across examples, put them on a common scale first:

```python
z = (score - float(probe["dim_scaler_mu"])) / float(probe["dim_scaler_sigma"])
```

Those two constants were computed once on a held-out slice and frozen, so the
scale never depends on whatever you are currently measuring. Use the same
pattern with `logistic_coef`, `logistic_intercept`, and the matching
`logistic_scaler_*` for the classifier.

## Keys

| Key | What it is |
|---|---|
| `difference_in_means_direction` | The direction vector, unit length |
| `difference_in_means_midpoint` | Subtract this from the dot product |
| `logistic_coef`, `logistic_intercept` | The classifier's weights and bias |
| `dim_scaler_mu`, `dim_scaler_sigma` | Frozen scaling for the direction score |
| `logistic_scaler_mu`, `logistic_scaler_sigma` | Frozen scaling for the classifier score |
| `selected_layer` | Which layer to pull activations from |
| `logistic_c` | Regularization strength used when fitting |

The remaining string keys are hashes of the data split, model revision, and
prompt format used at fit time, so a rerun can confirm it matched setups.
`manifest.json` holds SHA-256 digests of both files.

No dataset text is stored in these files.
