# Scientific inputs and interface

## Frozen learned recipes

Use `INV_PAIR_NLL` and `FREE_PAIR_CONSISTENCY`, each at original seeds 0 and 1
on both backbones. `weights/SELECTED.json` binds all eight selected files, their
original selected epochs, CAL metadata, hashes and original freeze identities.
Two extra `INV_PAIR_CONSISTENCY` seed-1 files are required **only** by the original
canonical replay selection. They are not additional selected comparison recipes.
Private provenance locations inside META/selection records are replaced by
`provenance/sha256-...` identifiers. These are intentionally opaque, not missing
runtime dependencies. Load the real relative `path` at the top level of each
SELECTED.json entry. Original metadata hashes and portable metadata hashes are both
recorded; the private original-to-portable mapping is not part of this package.

Canonical replay uses INV_PAIR_NLL for seed 0 and INV_PAIR_CONSISTENCY for seed 1
on both models. It retains original source backup and full command history,
normalizes last value per token address in source order, and recompiles that source
using the selected setter. No final-world or answer dictionary is a replay donor.

Gemma FREE_PAIR_CONSISTENCY seed 0 selected V7 epoch 0: its bytes are the retained
V4 checkpoint. Its META still correctly says V4 epoch 2. Do not call retention new
training, or rewrite that historical META epoch to zero. All other requested
learned selections are recorded individually in SELECTED.json.

Both setters share rank 16 and two value-conditioned heads. The invariant setter
uses only the orthogonal complement; the free setter also reads mutable coordinates.
The original CoordinateSetter forward body and saved realized bases are retained.
No scale or clipping is added. Each explicit command acts in the FP32 prefix
workspace; BF16 casting occurs on writeback at the addressed post-block site.
Saved `basis_realized` is used for inference. The canonical loader also computes
a QR *diagnostic* but does not substitute that diagnostic basis.

INV_PAIR_NLL used paired full-vocabulary label NLL, kappa 0, learning rate 0.0003.
FREE_PAIR_CONSISTENCY used the same paired NLL plus symmetric KL on the conditional
two-label distribution, kappa 1, learning rate 0.001. Both used beta 0.01 norm
regularization, matched one-epoch fitting, frozen backbones and CAL-only selection.
The selected weights are used as supplied; there is no training entrypoint here.

## Source/update versus scoring boundary

Updater input is source text and a tuple of `AddressedUpdate(start, end, value)`.
Offsets are half-open Unicode character positions of complete supplied source
clauses. Values are explicit bits, never inferred from gold. The original tokenizer
converts those character spans to overlapping token positions and checks contiguity.

`prepare` receives no question, evaluator label, terminal dictionary or origin.
Its learned writer receives exactly prefix token IDs, addressed token positions
and requested bits. Learned recipes preserve every command in sequence, including
no-ops, repetitions and restorations. They do not silently normalize their history.

SOURCE leaves text unchanged. REBUILD applies actual commands to source clauses.
EXISTING_CORRECTION retains every original correction sentence and its sequence.
LATEST_SAME_WORDING uses the original sentence after last-write normalization in
source-record order. Restoration and no-op commands remain present. No method is
allowed to inspect evaluator gold to choose which command executes.

`UpdatedContext` is a frozen wrapper around text and, when a backbone is supplied,
a hash-bound canonical cache. Each answer uses the canonical cache-clone serving
path. Python is not a security sandbox: application code must not bypass the typed
boundary or directly mutate private `_artifact`. Hash checks reject mutations.

`answer_labels` are the public answer symbols already specified in the question,
ordered [false, true], not the evaluator gold index. Gold enters only `metrics.py`.
For complete cases, `metrics.score_case` accepts an already compiled context,
its separately loaded evaluation rows and original SOURCE answers keyed by
query_id. It sends only text/public symbols to the serving helper, then joins gold
after answers return. Its rows carry the canonical metric fields and physical
batch/slot identities; deterministic baselines have no writer seed.
All record names in the scientific prompts are fictional generator identifiers;
they are intentionally preserved, not personal identity metadata.

## Models and score convention

- Gemma: `google/gemma-2-9b-it` revision
  `11c9b309abf73637e4b6f9a3fa1e92e615547819`, post-block site 15, procedure C1.
- Qwen: `Qwen/Qwen3-8B` revision
  `b968826d9c46dd6066d109eabc6255188de91218`, post-block site 13, procedure T4,
  `enable_thinking=False`.

R0 wording, original FIT demonstrations and public answer mappings are unchanged.
Backbones are BF16 with a detached FP32 output-head copy. The original Gemma final
logit softcap is 30; Qwen has no final softcap. Score full-vocabulary log-softmax,
then choose the larger of the two supplied answer-token log probabilities. Ties
and nonfinite scores are invalid; vocabulary argmax outside the label set is
reported separately and does **not** invalidate historical forced-choice scoring.

The one-question wrapper is a convenience interface. The inherited production
batch chunk is 12, with exact-input reuse only within one case and physical
batch/slot identity retained. Use the supplied `serve_questions` helper for the
fixed question batches. Verify batched versus single-suffix numerical behavior
with both pinned models before a comparison. Do not treat batch changes as exact
floating-point equivalence without that gate.

## Fixed populations and metrics

The original V7 candidate generator runs with seed 260913901 and namespace
UPDATE_COMPARISON. Both models use the same 64 balanced main scenes and 64
balanced terminal roots with four origins. The two writer seeds share cases.
There are 32 size/project/three-bit cells, two scenes or roots each. The program
subset selects the lowest source-text SHA256 in each cell (32 scenes, all 13
inherited programs). Eight separate technical-development scenes cover the eight
size/project/direction cells; they are not efficacy-selection data.

The main bundle has 24 original questions, two answer-mapping draws. Each origin
has 34 questions: the original 24 plus ten marked additions. Program coverage
additions remain marked and never redefine all24. Original query-generator seeds
260912501 (main/program) and 260912601 (origins) preserve inherited answer-mapping
procedures; only population generation takes the new population seed.

Before model loading, verify all manifest hashes and exact source-text collisions
against the supplied hash-only prior-data index. No prior outcomes are in the ZIP.
`verification/ORIGINAL_EQUIVALENCE.json` records the direct source-side check of
all supplied requests/questions against the completed-study implementation.

The primary endpoint is complete 34-question correctness across all four origins.
The independent unit is a terminal root, not 136 rows, four origins or two seeds.
`paired_primary` forms the seed mean *inside* each root, subtracts the deterministic
FIELD_PLUS_LATEST_ERRATUM baseline once, and uses 10,000 paired root bootstrap draws
with seed 260913902 and 98.75% intervals for the four fixed contrasts. Missing roots
or seed duplication of that baseline fail closed.

`origin_rates`, `root_statistics`, `scene_summary`, `program_retention`,
`recomposition`, and `fixed_criteria` retain secondary measurement definitions.
Invariant harm counts new errors among SOURCE-correct invariant questions divided
by **all invariant questions**; conditional harm uses only SOURCE-correct invariant
questions as its denominator. Required-change accuracy uses all required-change
questions. Undefined denominators remain null, never zero. Original SOURCE must
remain the shared denominator; any interface-specific SOURCE is a separate result.

## Retained information and verification limits

All text baselines have readable source and explicit command history. REBUILD
constructs final text using those commands; this is not an oracle answer. Learned
and replay implementations currently recompile the original prefix with hooks;
their real cost must include that work. Canonical replay additionally normalizes
the retained history. Per-question serving clones the updated cache. The package
does not assert equal storage privileges or an in-place, low-cost cache update.

EXISTING_CORRECTION follows the historical full-prefix compilation path. Before
any speed comparison, validate incremental correction serving and account for all
actual preprocessing, shared source prefill, update, suffix, warmup, synchronization
and retained/transient memory. The wrapper's diagnostic timings are not a speedup
claim. The new public field-refresh/erratum alternatives are not implemented here:
they require Stage 1 stock-code inspection, a pinned external commit, unchanged
worked-example reproduction, template freeze, token-length compatibility checks
and the specified erratum-only fallback. No AUTO or question-aware diagnostic may
be used to choose an update. These are full-comparison prerequisites, not CPU passes.
