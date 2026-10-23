# Calibration and consequence recovery

Repository topic A04. See manuscript Sections 3.1–3.2 and Appendices B.1–B.5.

## Inputs and fixed comparisons

The synthetic corpus contains 32 protocol-selection worlds, 64 direction-fit worlds
and 96 final worlds. Seven binary factors control relation direction, holder slot,
target slot, sentence order, distractor sign, target type and template family.
Each factor is balanced within each split. The two context wordings use different
lexical families and sentence orders. Only the focal relation changes between the
base and counterfactual states.

The [retained corpus](../../reproducibility/representation/counterfactual_fidelity/corpus.jsonl.gz)
and its generator agree exactly. The corpus seed is 20260904172. The earlier
study's eight demonstrations are retained as a required prompt dependency. Their
scientific content matches the frozen input identity after JSON-whitespace
normalization. Fresh demonstrations are separate from all study worlds.

| Query | Function | Included in recovery? |
|---|---|---|
| Q1 | Focal statement that becomes true in the counterfactual; calibrates strength | No |
| Q2 | Complement of the focal statement | Yes, focal group |
| Q3 | Paraphrase of the focal support statement | Yes, focal group |
| Q4 | Paraphrase of the focal opposition statement | Yes, focal group |
| Q5 | Other-holder relation control | Yes, invariance group |
| Q6 | Other-target relation control | Yes, invariance group |

Exact experimental strings remain in the corpus and
[contract](../../reproducibility/representation/counterfactual_fidelity/contract.json).
The primary answer format uses semantic entailment labels. The alternative maps
them to symbols. Each prompt includes eight demonstrations.

The model checkpoints are `google/gemma-2-9b-it` at revision
`11c9b309abf73637e4b6f9a3fa1e92e615547819` and
`meta-llama/Meta-Llama-3.1-8B-Instruct` at
`0e9e39f249a16976918f6564b8830bc894c89659`. The intervention uses the final
prompt-position residual after zero-based block 27 for Gemma and block 19 for
Llama. Exact BF16 weights and the right-padded batch-of-two candidate-sequence
scoring path stay fixed.

## Direct-answer calibration

Let the direct-answer margin be the semantic counterfactual-label log probability
minus the other label's log probability. For target fraction tau, the target is
the base Q1 margin plus tau times its natural counterfactual change. Fractions
are 0.25, 0.50, 0.75 and 1.00; the primary fraction is 0.75.

The implemented rule scans both signed rays of the direction and takes the reached
root with the smallest displacement. A tie selects the natural-projection ray.
Along each ray, it linearly interpolates the first grid crossing. Direction
displacement is measured in units of the Q1 natural whole-state difference.
The direction grid runs from 0.05 to 4.00 in steps of 0.05, then 4.25 to 8.00
in steps of 0.25. The interpolation grid ends at 4.00.

This rule was amended on fit worlds after the first five calibration checkpoints.
The earlier contract described scanning only the natural-projection ray.
The source records the two-ray amendment as fixed before final evaluation.
Both versions and that chronology remain explicit in
[`stcrf_v2_common.py`](../../reproducibility/representation/counterfactual_fidelity/source/stcrf_v2_common.py).

Achieved Q1 must be within `max(0.5 nat, 0.10 * abs(natural Q1 change))` of the
target. The chosen magnitude then stays fixed for Q2 through Q6 and the other
wording. Those outcomes never select or adjust it. A separate cached grid-scan
path is checked against the frozen measured scoring path at tolerance 0.25 nat.
Applied rows are remeasured on the frozen path.

The complete dose rule additionally requires the counterfactual hard answer,
signed margin at least one nat, finite scores and no saturation. A crossing jump
over five nats or selecting the grid maximum is saturation. At least 90% of worlds
must pass. The norm ratio's median must be at most two and its 90th percentile
at most four. Grid-unreachable, invalid, saturated and applied statuses remain
distinct in the saved records.

Both relation directions pass the norm budget and calibration tolerance. Only
80 of 96 final worlds pass the entire dose rule. Natural whole-state interpolation
at target fraction 0.75 also passes the norm budget and calibration tolerance but
only 81 of 96 worlds pass the entire dose rule; it is likewise not dose-valid. The source's dose-valid subset
and all-applied subset are reported separately, including their different counts.

## Recovery score and uncertainty

For each world, N is the natural counterfactual margin-change vector over Q2 through
Q6. I is the intervention's margin-change vector. The recovery score is
`1 - D(I,N)^2 / D(0,N)^2`. It is not clipped to zero or one.

The squared distance is standardized coordinate by coordinate. Each focal query
has weight 1/3; each invariance query has weight 1/2. The two groups therefore
have equal total weight. Scales are 1.4826 times the fit-world median absolute
deviation, floored at one quarter of the median focal scale. They are frozen
before final analysis. The decomposition satisfies
`recovery = focal recovery + invariance recovery - 1`.

Intervals use 10,000 world-level bootstrap draws with seed 202609041720.
Paired comparisons preserve whole-world membership and share the relevant draws.
They use worlds where both arms were applied. Holm correction uses alpha 0.05.
No world is discarded because of its Q2-through-Q6 outcome. Arm-specific applied
counts and dose-valid counts remain available in
[the complete analysis](../../reproducibility/representation/counterfactual_fidelity/gemma_analysis.json.gz).

Controls include matched final-logit forcing, sentiment, answer-token and factual
directions, the answer-residualized relation direction, natural interpolation,
full replacement and no intervention. The saved panels retain each target fraction,
cross-wording comparison and mapped-format comparison.

## Full-precision plotted comparison

Figure 3 uses the following applied-world summaries for all 96 final Gemma
contexts. The two calibrated arms use target fraction 0.75. These values retain
the precision in the [plotting input](../../figures/source/plot_data.json);
the plot displays the same means and intervals without the full numeric text.

| Intervention | Mean recovery | 95% context-bootstrap interval |
|---|---:|---|
| Original relation steering | -0.8508023069772639 | [-1.0047594830011988, -0.7066918757910918] |
| Whole-state interpolation | 0.9135822378114593 | [0.8994605138779845, 0.9259728090863992] |
| Full replacement | 0.981272301693043 | [0.979915374061593, 0.9825537048516965] |

These are the full applied-world summaries, not estimates restricted to the
80 (steering) or 81 (interpolation) contexts meeting the per-context dose rule.
Because neither tuned arm meets the 90% dose requirement, the manuscript reports
this comparison descriptively; the qualification below remains part of its
interpretation.

## Qualification and failed checks

Natural-reference qualification requires semantic balanced accuracy at least 0.90,
mapped-format agreement at least 0.95, and correctly directed focal changes in
at least 95% of comparisons. The control-to-focal change ratio must be at most
0.15. Cross-wording recovery must have mean at least 0.60 and lower interval bound
at least 0.40. Full replacement must recover at least 0.90.

Llama chooses among three predeclared protocols using only the 32 selection worlds.
It ranks the number of natural gates passed first, then cross-wording consistency,
then balanced accuracy. None passes all five natural-only selection gates.
The [failed protocol rows](../../reproducibility/representation/counterfactual_fidelity/llama_reference_qualification.json)
remain available. No unqualified actor is treated as a scientific null or averaged
into a cross-model conclusion.

The refitted Gemma direction uses only the 64 fit worlds. Eight-fold world-grouped
qualification requires relation-sign balanced accuracy at least 0.85,
cross-wording accuracy at least 0.80 and mapping agreement at least 0.95.
The observed values are 0.785156, 0.755859 and 0.935547, respectively. All fail.
The original frozen direction is a distinct secondary transport arm.

The later-layer witness is secondary to behavioral recovery. Its fitting projects
out the two relation directions and uses only unsteered fit-world activations.
Its criteria and saved outcomes remain in the supplied analysis and
[witness metadata](../../reproducibility/representation/counterfactual_fidelity/witness_metadata.json).

## Reproduction boundary

The command on the [Section 3 page](../03_changing_direct_answer.md#reproduce)
reconstructs the retained numerical panels from saved margins. Original synthetic
tests cover calibration, weighting, grouped bootstrap, classification precedence
and direction qualification. Corpus generation and current split membership are
checked against the retained rows.

Both original relation vectors are supplied in the
[probe parameter files](../../artifacts/probe_weights/README.md). Their numerical
array hashes match the frozen counterfactual input record exactly. One original
prior corpus is also supplied. The full two-corpus overlap test still has
an explicit missing-input skip; its supplied findings remain source-record evidence.
The [reproduction matrix](../../REPRODUCIBILITY.md#reproduction-consequences)
lists the remaining parameters and corpus dependency. Saved-score replay and
current-corpus checks remain executable.
