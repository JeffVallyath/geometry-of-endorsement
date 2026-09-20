# Does SHS change a method-selection decision? Saved-data V10 reanalysis

## Verdict

**Not established.** Qwen's learned procedures have lower SHS witness density,
but they are not practically equivalent to the textual procedures across the
recorded conventional metrics. Ordinary locality/preservation and downstream
accuracy already distinguish them. None of the ten planned model/comparison
claims satisfies both conventional equivalence and material SHS separation at
any declared 1, 2 or 5 percentage-point (pp) margin. Even the deliberately loose
10pp stress test establishes no all-metric equivalence. Gemma has no supported
SHS separation and is not a cross-model replication of Qwen's pattern.

This analysis is retrospective. V10's experiment/SHS definitions were frozen
prospectively, but the new margins and paired contrasts were not preregistered
before that experiment. The public headline rates were already known. The
present ANALYSIS_PLAN.md and code were hash-bound before calculating the new
ordinary-metric effects or paired SHS contrasts; they were not changed afterward.

## Sources and genuinely comparable measurements

V10/SSC1 evaluates the same 64 generated final cases per model, eight matched
histories and 34 questions per history. The complete intended current tables
agree across histories. Measurements are hard-answer correctness reconstructed
from retained candidate log probabilities, not new inference. Two answer-code
draws and histories remain nested within case. The two learned seeds are
averaged within case; n is 64, not 128 or the number of questions.

The six executed updating conditions are INV_PAIR_NLL_s0/s1,
FREE_PAIR_CONSISTENCY_s0/s1, EXISTING_CORRECTION and LATEST_SAME_WORDING.
Recipe-level means average their two fixed seeds. LEARNED_MEAN averages the two
recipes equally; TEXTUAL_MEAN averages the two textual conditions equally.
These aggregates compare those specified mixtures, not learned/textual methods
in general. SOURCE is an unupdated control; NATIVE_FINAL is a rebuilt final-state
reference, not a competing learned updater.

| Metric | Exact saved-data definition per root | Interpretation |
|---|---|---|
| updated_direct | Correct addressed-fact reads / 48 | Direct efficacy on three commanded records, including already-correct assignments |
| changed_direct | Correct actually changed addressed-fact reads / 24 | Efficacy excluding semantic no-ops, selected from source/final tables, not answers |
| unchanged_direct | Correct unaddressed-fact reads / 48 | Locality accuracy on three measured untouched records |
| locality_retention | Retained initially correct unaddressed reads / initially correct SOURCE reads | Zero-credit-for-gains preservation rate; masks shared across methods |
| opposite_wording | Correct opposite-wording reads / 32 | Bounded generalization to four recorded 'opposes' queries, not an external paraphrase benchmark |
| joint_accuracy | Correct downstream joint answers / 144 | Ordinary consequence accuracy; no history-agreement requirement or eligibility filtering |
| all_question_accuracy | Correct answers / 272 | Aggregate pointwise quality across the full 34-question panel |

Each rate is computed per root, then averaged across the 64 roots. Locality
retention denominators can vary across roots; this is a mean of root retention
rates, not a pooled answer-level fraction. No root has an empty retention mask.
Both answer codes are included. Recorded no-op origin-zero joint competence and
native-final competence are reported separately as **study-specific controls**.
The native reference is identical across methods and cannot establish updater
equivalence merely by being shared.

These are operational analogues of familiar efficacy, preservation and consequence
metrics in a synthetic relation task. Broad real-world unrelated-fact locality,
fluency, realistic deployment utility, and complete method cost are not measured
by V10. We do not call this a complete standard editing benchmark battery.

The earlier update-method comparison was separately replayed: 208,896 score rows,
64 different cases/model, four histories, and additional methods. Its baseline
FIELD_PLUS_LATEST_ERRATUM uses recorded fallback in 192/256 contexts/model.
Its Qwen learned-vs-baseline whole-bundle advantages are +31.250pp and +39.844pp
(98.75% CIs [20.312,43.750] and [28.906,50.781]); Gemma is mixed. That context
already concerns ordinary downstream performance. Its cases, baseline and
endpoints are **not pooled with V10**, and its conventional metrics are not
attached to V10 SHS rates as though measured on the same observations.

## Equivalence margins and uncertainty

No application-specific loss function justifies one universal tolerance. The
locked sensitivity grid uses symmetric **1, 2 and 5pp** margins on every required
conventional metric: tolerating at most one, two or five additional wrong/lost
answers per 100 scored opportunities. Five points is permissive and application
dependent. Ten points is a stress test only. The data-dependent minimum CI-enclosing
margin is reported for transparency, not adopted as a new equivalence threshold.

Paired case bootstrap: 100,000 draws, seed 2609181101, all histories/questions/
codes/seeds clustered within root. Tables supply 95% pointwise CIs and 99% CIs
for the ten planned claims (two models x four recipe/text contrasts plus one
class-average contrast). An equivalence pass requires the entire 99% CI strictly
inside (-margin,+margin). A joint decision claim requires **all seven** conventional
metrics and material SHS separation; this intersection-union requirement is
conservative with Bonferroni alpha=.05/10 across comparisons. These intervals
are not claimed to cover every displayed endpoint simultaneously. Per-seed
contrasts are supplementary and cannot select the headline. This bootstrap
describes uncertainty over the generated-case sampling scheme, not deployment.

Failing equivalence is not automatically proof of a practically important
difference; distinguish wide uncertainty from a clearly separated ordinary
metric. Conversely, a statistically nonzero difference can be practically
equivalent: Gemma's aggregate updated-direct difference has 99% CI
[-1.383,-0.285]pp, entirely within the 2pp margin despite excluding zero.

## Ordinary metrics: aggregate learned minus textual comparison

Means below are percentages; differences and intervals are percentage points.
Complete per-condition means/95% CIs, all pairwise effects/95% and 99% CIs, and
every margin decision are in METHOD_SELECTION_TABLE.csv and EQUIVALENCE_ANALYSIS.csv/json.

| Model | Metric | Learned | Textual | Difference [99% paired CI] | Equivalent at 5pp? |
|---|---|---:|---:|---|---|
| Qwen | Updated direct | 96.851 | 97.738 | -0.887 [-3.158,1.245] | Yes |
| Qwen | Actually changed direct | 95.817 | 96.126 | -0.309 [-3.874,3.190] | Yes |
| Qwen | Unchanged direct | 96.419 | 86.133 | +10.286 [5.688,15.234] | No |
| Qwen | Locality retention | 97.171 | 89.060 | +8.111 [3.903,12.743] | No |
| Qwen | Opposite wording | 93.958 | 95.825 | -1.868 [-6.030,2.185] | No |
| Qwen | Joint accuracy | 93.802 | 78.190 | +15.611 [11.979,19.404] | No |
| Qwen | All-question accuracy | 94.820 | 85.116 | +9.704 [7.634,11.811] | No |
| Gemma | Updated direct | 99.211 | 99.967 | -0.757 [-1.383,-0.285] | Yes |
| Gemma | Actually changed direct | 98.926 | 100.000 | -1.074 [-2.083,-0.326] | Yes |
| Gemma | Unchanged direct | 99.382 | 97.656 | +1.725 [0.342,3.304] | Yes |
| Gemma | Locality retention | 99.382 | 97.646 | +1.735 [0.350,3.326] | Yes |
| Gemma | Opposite wording | 96.863 | 99.585 | -2.722 [-4.724,-0.977] | Yes |
| Gemma | Joint accuracy | 95.074 | 87.375 | +7.699 [4.506,11.168] | No |
| Gemma | All-question accuracy | 96.774 | 92.848 | +3.926 [2.202,5.778] | No |

No planned comparison passes the complete conventional vector at any grid
margin. At 5pp, every Qwen recipe/text comparison fails unchanged accuracy,
retention, joint accuracy and overall accuracy; other failures vary. Every
Gemma recipe/text comparison fails joint and overall accuracy. These failures
are not created solely by including a redundant aggregate metric: joint accuracy
alone rules out the all-metric equivalence claim at every declared margin.
Qwen's ordinary joint difference is well above 5pp even at its 99% lower bound.
Class-average CI-enclosing margins would have to exceed 19.404pp (Qwen) or
11.168pp (Gemma); these were not adopted or justified as equivalence margins.

Native joint competence is 77.170% Qwen / 84.809% Gemma, shared across methods.
Same-method no-op joint accuracy is 94.748% vs 81.207% in Qwen (+13.542pp,
99% CI [9.418,17.969]) and 96.441% vs 89.236% in Gemma (+7.205pp,
[3.841,11.068]). These study-specific controls also do not suggest an otherwise
indistinguishable set of updaters.

## SHS separation: frozen endpoint retained

SHS witness density is W/18 per root: both direct operands correct across all
eight histories, native and same-method no-op reference correct, and differing
valid joint answers across histories. Ineligible opportunities remain in the
fixed denominator. This is not an individual-answer error rate or a general
prevalence estimate. All eight original primary means and multiplicity-adjusted
99.375% intervals were exactly reproduced, as were all 896 root-condition rows,
16,128 query classifications and the fixed repeat checks.

| Model | INV_PAIR_NLL | FREE_PAIR_CONSISTENCY | Existing correction | Latest same wording |
|---|---:|---:|---:|---:|
| Qwen | 2.474% | 2.387% | 6.163% | 6.337% |
| Gemma | 6.337% | 5.122% | 5.990% | 5.556% |

New planned paired contrasts below are learned minus textual, in pp. Materiality
requires the entire 99% CI beyond a fixed 1pp threshold; sensitivity results
for 0, 0.5 and 2pp are retained in DECISION_GRID.json. This threshold describes
one fewer guarded witness per 100 fixed opportunities, not demonstrated utility.

| Model | Learned / textual comparison | SHS difference [99% paired CI] | Material at 1pp? |
|---|---|---|---|
| Qwen | Class averages | -3.819 [-6.293,-1.454] | Yes |
| Qwen | INV / existing | -3.689 [-6.337,-1.042] | Yes |
| Qwen | INV / latest | -3.863 [-6.510,-1.215] | Yes |
| Qwen | FREE / existing | -3.776 [-6.424,-1.172] | Yes |
| Qwen | FREE / latest | -3.950 [-6.554,-1.476] | Yes |
| Gemma | Class averages | -0.043 [-2.648,2.431] | No |
| Gemma | INV / existing | +0.347 [-2.431,3.082] | No |
| Gemma | INV / latest | +0.781 [-2.040,3.559] | No |
| Gemma | FREE / existing | -0.868 [-3.906,1.953] | No |
| Gemma | FREE / latest | -0.434 [-3.472,2.387] | No |

Qwen class means are 2.431% vs 6.250%; ratio 0.389 (descriptive only).
Gemma means are 5.729% vs 5.773%. None of the Qwen 99% intervals establishes
a difference larger than 2pp, despite point differences near 4pp. Gemma's
non-significance is not evidence of SHS equivalence.
The INV/existing 1pp pass is particularly close to the threshold (upper endpoint
-1.042pp); treat that discrete-bootstrap classification cautiously, not as a
sharp utility boundary. The aggregate contrast is farther from the threshold.

### Could eligibility alone explain lower Qwen witness density?

Eligibility AC/18 is 65.755% learned vs 63.368% textual in Qwen and 79.405%
vs 78.038% in Gemma. The learned methods do not simply have fewer eligible
opportunities in aggregate. The stricter common-support diagnostic requires
eligibility under all four learned seeds and both textual conditions, leaving
500/1,152 Qwen opportunities across 59 roots. On that same subset, the learned
minus textual witness-fraction difference is -6.650pp (99% root-bootstrap CI
[-10.828,-2.681]). Gemma has 740 common opportunities across 63 roots and
-0.203pp [-3.630,3.001]. All pair-specific common-support results are retained.
This sensitivity supports a real Qwen SHS contrast, not a revised primary
endpoint or a correction for every possible selection effect.

## Decision consequence and paper framing

There is **no validated example here of conventional equivalence plus material
SHS separation**. Comparing direct efficacy alone at a permissive 5pp margin
would conceal recorded preservation and downstream differences. A researcher
already examining those ordinary metrics has reasons to distinguish Qwen's
methods before seeing SHS. SHS reinforces a preference for the measured learned
procedures in history-screening applications, but this dataset does not show
that adding SHS changes the method choice after ordinary metrics have done
their job. No application utility function was supplied; actual choice reversal
or positive cost-benefit is not measured. Learned methods also have weaker
direct/opposite-wording performance in some comparisons, so 'universally better'
would ignore a real tradeoff.

**What should anybody do differently?** For a system intended to reuse current
information independently of superseded history, report matched-history
dependence as a separate validation property alongside correctness and
preservation. These results do not justify the stronger claim that SHS breaks
a conventional-metric tie or is always worth its evaluation cost.

**Strongest defensible sentence:** In V10's controlled Qwen task, the measured
learned updates have lower guarded source-history witness density than the
textual updates, including on common eligible opportunities, but they also
already differ on ordinary preservation and downstream accuracy; this reanalysis
does not establish an SHS-induced method-selection reversal among otherwise
practically equivalent methods.

**Tempting unsupported overclaim:** Conventional evaluation treats these methods
as interchangeable, and only SHS reveals which updater researchers should choose.

The companion formal observation is a measurement non-implication, not an
empirical full-battery counterexample. It remains valid without manufacturing a
decision consequence from these data. No CCB1 failure, V9 witness or other
experiment is relabeled as support for the missing equivalence premise.

## Audit and reproduction

New analysis: src/repro/method_selection.py; ten focused tests in
tests/test_method_selection.py. Reused semantic/score/alias verification:
reproducibility/state_sufficiency/independent_verification/reconstruct.py.
Inputs and new protocol/code/tests are bound by RUN_CAPSULE.json, SHA256
ff7a290b832bec3b58a9556849c521dc208fc8a17632406f43bec4e091f7709c.
All new machine outputs live in reproduced/method_selection_v1. No pretrained
inference, provider allocation or changes to frozen experiment artifacts occurred.
The supplied snapshot is not a Git worktree; the installed rr capsule builder
requires one, so this local-only analysis uses the existing Python replay
interface with an immutable hash manifest, not a fabricated rr execution receipt.

From the repository root with its CPU dependencies and src on PYTHONPATH:

```text
python -m pytest tests/test_method_selection.py -q
python -m repro.method_selection freeze --output reproduced/method_selection_repeat
python -m repro.method_selection analyze --output reproduced/method_selection_repeat
```

Use a new output path; never overwrite the retained capsule or frozen results.
ROOT_METRICS.json preserves per-case numerators and denominators, the CSV tables
retain every individual update condition and seed diagnostics, and
REPLAY_VERIFICATION.json records the original full replay checks.
