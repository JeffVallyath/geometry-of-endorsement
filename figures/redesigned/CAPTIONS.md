# Figure captions

Insertion size: 5.5 inches = 396 PDF points. Asset names do not dictate figure numbering.

## matched_history (main)

Different starting histories receive identical update commands and reach the same intended current facts. A is a conceptual two-lane illustration, not measured equal hidden states or equal probability vectors. B and C retain all eight prospective model/method cells. Each cell contains 64 cases, eight histories and 18 fixed coded targets per case: 1,152 opportunities per learned seed or textual condition. A target qualifies only when required facts are valid and correct in every history, native-final and same-method no-op references are correct, and valid downstream answers disagree. Ineligible targets remain in the denominator. Intervals are the saved 99.375% case-bootstrap intervals; learned seeds are averaged inside cases. These are target rates, not case prevalence.

## construct_validity (main)

Selected-layer scorer and specificity evidence is combined without dropping controls. Model answer margin, relation direction, logistic probe, held-out wording, MiniLM, deterministic random item scores and all three separate-input controls remain distinct. These are development evaluation boards, separate from fitting and layer selection, not the strict test. Intervals are 95% normal dyadic-robust intervals; open diamonds have no retained transfer interval. Direction-minus-text differences are 1.363 [1.089,1.637] for Llama and 2.038 [1.661,2.415] for Gemma. Algebraic cancellation is not neural evidence, and random item scores are not random activation directions. The factual control gives Llama primary 1.924 [1.893,1.956], transfer 1.840 [1.807,1.874], and Gemma 1.944 without a retained interval; its full mapping-reversal sweep and limitations are retained in the truth-control supplement.

## consequence_qualification (main-qualified)

This panel supports the draft introduction's direct-answer/consequence contrast only with explicit qualification limits. Both tuned interventions hit the requested relative margin in all 96 Gemma contexts at target fraction 0.75; this is neither perfect hard-answer accuracy nor a full strong-target pass. Recovery is -0.850802 [-1.004759,-0.706692] for frozen-direction steering, 0.913582 [0.899461,0.925973] for whole-state interpolation, and 0.981272 [0.979915,0.982554] for untuned full replacement. Intervals are saved 95% context bootstraps. The last interval is narrower than its plotting marker, so its exact bounds are printed. Zero denotes the unchanged state and one the natural change, not chance or accuracy. Strong-target gates failed; the matched direction was unqualified and Llama's natural reference failed. This is not promoted as a confirmatory intervention success.

## fresh_confirmation_utility (main)

Qualification requires valid, correct direct facts and same-pair/other-pair logical controls across histories, correct native-final and no-op references, and differing answers on a history pair matched for changed-fact count and prefix-token length. These rules were fixed before fresh outcomes. Fresh confirmation and utility are distinct endpoints of the same new 64-case study. The mean qualifying-target rate is 0.954861% with 95% paired-case interval [0.520833,1.432292]%; 22 target/seed observations arise in 14/64 contributing cases. Each seed retains all 1,152 scheduled coded targets, including ineligible targets. A uses six validation checks at one history; B uses three checks at each of two histories. Literal-format common-case counts are 59 and 57. At 50% retention the named error contrast B minus A is -1.74 percentage points, interval [-3.53,+0.65]. Lower error favors B, but superiority is not established. A/B-format sensitivity is -0.86 points [-2.68,+1.23]; all retention/seed/format values remain in the full-precision table. Equal query counts do not establish equal GPU cost.

## selected_crossover (main)

This selected Gemma case tests whether Dion and Orla both support Bridge; the correct answer is Yes. Each hybrid combines the first 16 layers from the early source and the remaining 26 from the later source before the question, with model weights fixed. H1/H2 identify histories; color and hatching track the later donor, not correctness. The direct-fact table retains all eight exact controls. Across six selected cases the main format gives five clear later-source effects and one mixed result; the alternate format gives one clear effect, two mixed results and three with insufficient original separation. All six-case rows are retained. These counts are not prevalence, and the manipulation does not identify a unique circuit or the origin of the historical information.

## repair_benefit_damage (main)

A repair requires the target, all 12 direct questions, two same-pair functions and six other-pair functions to be valid and correct in all eight histories in that answer format. Wrong consensus, invalidity or collateral loss counts as no repair. All four endpoints retain their original estimands and 98.75% paired-case intervals from 10,000 draws over all 256 cases. A is unchanged computation, B refreshes values 16-23, C refreshes values 24-31. Efficacy B minus C is measured over all 9,216 target opportunities, not only the baseline qualifying cohort. Harm denominators retain all 49,152 direct and 73,728 joint responses per arm, across both seeds, both formats and eight histories. Baseline W contains 83 target/format/seed rows in 63 cases; B repairs 36 rows in 28 cases. The prespecified 50% repair requirement fails. Joint errors fall from 3,490 to 2,893, but the full reference has 2,000 errors in the same 73,728 responses. B corrects 944 joint errors and introduces 347; it corrects 84 direct errors and introduces 76. Both partial refreshes require preparing the full reference; no efficiency advantage is established.

## update_reliability (appendix)

All 34 questions must be correct under all four histories for complete-case success. Dots and saved 98.75% paired-case intervals average the two learned seeds within each of 64 cases; downward/upward triangles show seeds 0/1. Both Qwen gains exclude zero; the Gemma constrained interval touches zero and the consistency interval crosses it. Absolute bars include the executed baseline and both learned procedures and span the full 0-100% scale. The baseline used fallback in 192/256 contexts per model. Neither learned procedure reaches 50% mean complete-case success. This supporting comparison is not a replacement for qualification-based matched-history testing.

## layer_development (appendix)

Every original layer and both original metrics are retained, without smoothing or added bands. The marked layers are Llama 19 and Gemma 27, selected using mean_mirrored_pairwise on a separate 300-row/75-board selection split. Evaluation uses 500 rows/125 boards. The evaluation maxima at Llama 21 and Gemma 28 did not choose the layers. These are development curves, not strict-test or confirmatory evidence; zero interaction and chance AUROC 0.5 remain visible.

## permutation_nulls (appendix)

The complete four-cell permutation evidence is retained with the exact original null draws and histogram bin grid, without smoothing or Gaussian substitution. Random training-situation flips preserve within-situation label relations; probes and select normalization were refitted in the saved analysis. Llama has B=10,000: direction k=131, p=0.01319868; logistic k=0, p=0.00009999. Gemma has B=200: direction k=2, p=0.01492537; logistic k=0, p=0.00497512. The one-sided rule is (k+1)/(B+1), never p=0. Llama direction two-sided sensitivity is 0.0245. The earlier Llama B=200 test concerned improvement over the text comparator, not this identical estimand; its historical record is preserved in frozen data.

## truth_control (appendix)

The development sweep and held-out factual tests have separate roles. Neutral symbols reverse the True/False mapping. Llama layer 14 uses the prespecified maximin selection criterion and lower-layer tie break. Held-out primary and transfer intervals are the original 95% group-bootstrap intervals from 2,000 replicates. Gemma layer 25 has only a retained point estimate and is deliberately shown without an interval. The earlier literal-label control failed (reversed-instruction AUROC 0.0003 despite reversed-literal AUROC 0.9997); later mapping divergence remains visible. A passing selected layer does not validate every layer.

## crossover_extension (appendix)

One retrospectively selected pair is tested in two formats, not two independent replications. Ada and Gita both oppose Theater, so the correct either-supports answer is No while the bars show probability of Yes. Both prefixes contain 517 tokens and both histories require two fact changes. First 16/remaining 26 layer activations are combined before the question, with weights fixed. Coded answers are mapped separately to semantic Yes/No. All eight strict direct checks pass: minimum correct probability 0.999230 and maximum donor-relative change 0.00008612. State order is explicitly aligned with the other crossover panel: AA, BB, AB, BA; no state is removed. Saved evidence also retains 108 baseline replays and 24 exact identity controls.

## editing_coverage (appendix)

Both editors train on single edits; the broader condition trains on more consequences. The same held-out situations are evaluated after two and three edits, with outcomes averaged within situation and seeds averaged before paired whole-situation resampling. All four contrasts, both seed markers and original 98.75% intervals are retained. Gemma has 64 situations and Qwen 32; reversed orders are not extra cases. The Gemma free-overwrite interval crosses zero. These relative gains do not establish absolute joint-control qualification or broad external generalization.

## fixed_readers (appendix)

Paraphrase and explicit-rule readers operate on the same saved edited states after three updates. The explicit rule adds information to the interface and is not a cosmetic rename. Each endpoint is same-side accuracy under the original fixed early ordering, with 64 Gemma or 32 Qwen situations. All eight conditions, two seed-specific effects and original 99.375% paired whole-situation intervals remain visible. None establishes an improvement, and zero-crossing intervals do not establish equivalence. This is distinct from full source-independence testing.

## archival_source_history (appendix)

The two old S3 exports are terminology/layout revisions of the same V6 analysis, not separate experiments. This canonical display retains all eight cells and the original four-start design: Quin, Kira and Juno about Meadow (FINAL-RSI6-0000), with initial values 000,100,011,111 and common requested 000. The diagram is a design, not model answers. Atomic perfection requires all 16 direct/opposes questions correct in all four histories; disagreement is on any of 18 joint questions. Bars use all 64 Gemma or 32 Qwen roots; right counts use fact-correct roots. Stable joint answers can be consistently wrong. Both original PDFs remain archived unchanged, and this descriptive census is not pooled with prospective V10.

## removal_screen (appendix)

This fixed exposed historical screen is not a prevalence sample or a validated population intervention. Removal corrects three distinct cases, all under seed 1, versus two for the matched control. Admission required at least three corrections, at least one per seed, at least two more than control, lower mean joint error and at most 2% new direct errors. Four requirements fail. Joint errors rise from 75 to 129/1,152 (control 72); removal introduces 67 new joint errors and corrects 13. New direct errors are 38/768, or 4.95%, above the fixed 2% cap. Both seeds, both selected histories and all scheduled family responses remain in denominators. The branch is closed; no fresh final cases were run.

## fixed_worked_example (appendix)

The original first hash-selected case and first No/Yes joint query are retained, not replaced by the later Esme/Vera success illustration. All 64 displayed target cells are valid and correct; the control counts can still differ. Values are the original normalized probabilities, displayed to four decimals with full-precision exports retained. The full-current-record comparator D reuses one seed-specific context across histories, making its invariance automatic rather than independent empirical success. Both partial refreshes require preparing D. B refreshes values 16-23 and C values 24-31; keys, other values, positions, prompts and model weights remain unchanged. The common suffix has 20 tokens. The original archive, all 506 declared members, capsule and selected score hashes were verified before the figure projection.
