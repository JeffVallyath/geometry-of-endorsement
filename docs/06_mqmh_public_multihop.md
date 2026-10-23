# Public multi-hop matched histories

On 128 MQuAKE-derived source cases, matched histories that converge to the same
current record can still yield different answers to the same multi-hop question
despite correct current-fact reports and successful clean references. This occurs
in both Qwen2.5-7B-Instruct and Gemma-2-9B-it. Relevant obsolete records also produce
more guarded downstream errors than matched irrelevant records, and the effect
survives an independent remaining-chain check on longer questions.

## Two model evaluations, one frozen cohort

Both models use the same 128 source subjects, endpoint records and three original
public question phrasings per subject. Target-level factual, reference, matching,
parser and robustness criteria are identical. The results describe recurrence
across two model families on this shared cohort. They are not pooled and do not
establish that either model is more susceptible than the other.

| Model | Common guard | Qualified history dependence (98.75% interval) | Relevant-minus-irrelevant error (98.75% interval) | Strong long-chain |
|---|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | 242/384 | 94/384 = 24.48% [17.71, 31.77]% | +12.50 pp [+8.33, +17.45] pp | 32/195 |
| Gemma-2-9B-it | 279/384 | 101/384 = 26.30% [18.62, 34.38]% | +15.63 pp [+10.81, +20.96] pp | 65/195 |

384 is the full number of target phrasings for **each** model, not the number
passing the guard and not a cross-model denominator. 195 is the subset of target
phrasings with three or four hops. Failed guards remain in both full denominators.
The intervals resample whole subjects, retaining their three phrasings together:
10,000 bootstrap draws, seed 20260923, 98.75% coverage level. They are approximate
constructed-sample intervals; shared intermediates and relation families can
remain dependent across subjects.

## What is compared

H0 and H1 put an obsolete first-hop fact on the queried entity. S0 and S1 put the
matched obsolete fact on an irrelevant entity instead. The two orderings swap
which first-hop record appears first or last around the shared remaining-chain
records. All four histories receive the same correction block and terminate in
the same current record. They are not identical full prompts: the obsolete
material remains in context.

Required current facts are queried independently, without feeding their answers
into the multi-hop question. For each target phrasing, the common guard requires
every needed current-path fact to be correct under all four histories, and the
original multi-hop question to be answered correctly by both references:

- C: the current record is present from the start, without obsolete material;
- C+: an already-correct record receives the same corrections as a no-op update.

A guarded target is history-dependent when a pair among the four histories gives
distinct valid entity answers, with exactly one correct. Answers use the same
case-sensitive, first-line exact-alias parser for both models. It permits the
specified optional answer prefix, quotation marks and final period; it never
uses the gold answer to choose an interpretation. Unrecognized, ambiguous,
truncated or invalid-control-token outputs cannot serve as valid witnesses.

For each target, guarded relevance error is the mean error of H0/H1 minus the mean
error of S0/S1, multiplied by the common-guard indicator. The reported contrast
averages these differences over all 384 targets. Invalid outputs count as errors
in this contrast, but failed guards contribute zero rather than disappearing.

For longer questions, an independent remaining-chain question starts from the
current first-hop answer and asks the model to traverse the rest of the chain.
The strong long-chain count additionally requires this control to succeed under
all four histories. This checks remaining-chain competence without replacing the
original public target phrasing with a decomposed or rewritten question.

## Robustness and repeats

Gemma's relevance contrast has zero invalid-output contribution; Qwen's has a
+0.39 pp contribution. Under the stricter canonical-label-only parser, qualified
targets remain 87/384 for Qwen and 98/384 for Gemma, with strong long-chain counts
29/195 and 63/195 respectively. These are sensitivity analyses of the same saved
responses, not additional samples.

Long-chain plus cross-phrasing checks hold for 11 Qwen subjects and 21 Gemma
subjects: the same ordered pair of distinct answers recurs across at least two
qualified phrasings, with the remaining-chain control correct.

All 12 complete-case fresh-process repeats are semantically stable for each
model, including case 1617. Each repeat set comprises eight preselected cases
and four qualified-positive cases selected by the same fixed ordering rule.
Stability means agreement in validity, parsed entity and correctness. These
repeats do not increase the cohort or establish that every positive case was
repeated.

## Worked example: source case 1617

These records are **benchmark counterfactual stipulations, not biographical
claims** about the named people. This illustrative example is selected from the
saved outputs; it is not a separate prevalence sample.

The current stipulated path is William Levada → Church of Scotland → John Knox
→ Edinburgh. The obsolete path instead goes through Catholic Church and Jesus
Christ to Galilee. Thus the current terminal is **Edinburgh**, and the obsolete
terminal is **Galilee**. The matched irrelevant entity is Albert of Saxony; its
corrected first-hop record also names Church of Scotland.

The original public target phrasing shown here (J2) is:

> What place did the person who founded William Levada's religion work in?

For **both** models, all required current facts are correct under all four
histories: the religion is Church of Scotland, its founder is John Knox, and the
founder's work location is Edinburgh. The irrelevant-entity fact is also correct.
The independent remaining-chain question, starting from Church of Scotland and
following “founded by” then “work location,” returns Edinburgh under every history.

| Context | Qwen2.5-7B-Instruct answer | Gemma-2-9B-it answer |
|---|---|---|
| H0: relevant obsolete record first | Galilee | Edinburgh |
| H1: relevant obsolete record last | Edinburgh | Galilee |
| S0: irrelevant obsolete record first | Edinburgh | Edinburgh |
| S1: irrelevant obsolete record last | Edinburgh | Edinburgh |
| C: clean current-record reference | Edinburgh | Edinburgh |
| C+: no-op reference | Edinburgh | Edinburgh |

Both models therefore exhibit a qualified history-dependent answer under the
same endpoint record and the same question. Each complete case, including its
fact, reference, remaining-chain and target answers, passes its fresh-process
semantic repeat. The differing failing orderings do **not** establish a universal
positional mechanism.

Qualification is target-specific. Qwen's J0 phrasing of this case fails the clean
reference guard; it is not counted as a qualified witness and still remains in
the full denominator. All phrasings and responses, not just the displayed J2
example, are retained in [the compact evidence](../reproducibility/mqmh/MANIFEST.json).
[The worked-example data](../reproducibility/mqmh/case_1617.json) contain the exact
stipulated contexts, questions and parsed answers for both models.

## Reproduce and interpret

Using the [documented CPU environment](../REPRODUCIBILITY.md):

```sh
python -m repro.mqmh --output reproduced/mqmh
```

The command checks the compact-input hashes, reparses the saved decoded answers,
reconstructs every target's guards and outcomes, regenerates both subject-level
bootstrap intervals, checks the strict-parser sensitivity and compares complete
repeat cases. Outputs are `summary.json`, `target_counts.csv`, `case_1617.json`
and `CHECKS.json`. Its results must match [the retained scientific summary](../reproducibility/mqmh/RESULTS.json).

[Source hashes and model revisions](../reproducibility/mqmh/SOURCE_HASHES.json)
bind the source evidence archives and the saved measurements. The public package
contains only scientific projections, not the large source ZIPs or execution
records. Saved truncation and control-token-validity flags are retained; this
command does not re-decode native token IDs, load a tokenizer or perform new
inference. Fresh-process separation was verified against the saved journals;
process identifiers are not included in this release.

The source is MQuAKE-CF-3k-v2, with its original question phrasings preserved.
This is an explicit-record, in-context counterfactual evaluation, **not** standard
closed-book MQuAKE performance. It does not establish deployment prevalence,
transfer to parameter editing, a unique stale-memory mechanism, or cross-model
superiority. Residual pretraining exposure is unknown. See the preserved
[dataset and model notices](../DATA_AND_LICENSES.md).

MQMH is the main public-data generalization result in Section 5. The earlier
RippleEdits-derived result remains in Appendix F.2 as secondary supporting
evidence; its saved outputs and historical accounting are preserved.
