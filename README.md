# Geometry of Endorsement

**Geometry of Endorsement studies when a model that correctly reports updated information actually uses that information as the state governing its later decisions. It combines measurements of a readable support/opposition relation, learned activation updates and causal interventions. The central finding is that these notions can end up coming apart: different histories can be updated to the same intended current facts, the model can report those facts correctly, yet later decisions can still depend on information that was supposedly replaced.**

For the short logical progression, see [Project Strategy](PROJECT_STRATEGY.md). The quantitative record is in [Results and Claims](docs/RESULTS_AND_CLAIMS.md). Human-review procedures are described in [Human Review](docs/HUMAN_REVIEW.md), and reproduction instructions are in [Reproducibility](REPRODUCIBILITY.md).

## Paper-first navigation

The [short terminology lookup](docs/TERMINOLOGY.md) connects scientific descriptions to exact saved-record names.

Correctly reporting an updated fact does not establish that later behavior depends
only on the current facts. Prior work already shows that local edits can fail to
propagate to downstream answers. Our central matched-history test asks a narrower
question: after different pasts reach the same verified current facts, does the
old path still predict the same later answer?

| Paper role | Evidence and reader route |
|---|---|
| Main matched-history result | [Prospective result and Figure 7](PAPER_GUIDE.md#matched-history-sufficiency) |
| Causal intervention | [Six-case panel and Figure 8](PAPER_GUIDE.md#causal-cache-intervention) |
| Matched causal extension | [One selected case and Figure 9](PAPER_GUIDE.md#matched-one-case-extension) |
| Public-data extension | [Qwen/Ripple saved-response replay](PAPER_GUIDE.md#public-data-extension) |
| Update-method comparison | [Matched-history comparison and Figure 10](PAPER_GUIDE.md#update-method-comparison) |
| Representation and decoding background | [Earlier figures and saved measurements](PAPER_GUIDE.md#representation-and-decoding) |

For a CPU-only start, use Python 3.11 or newer in a virtual environment:

```sh
python -m pip install -r reproducibility/requirements-replay.lock -e .
python -m repro.state_sufficiency --output reproduced/main
python -m repro.in_context_updates --output reproduced/qwen
```

These commands recalculate saved evidence. They do not load pretrained models.
The [paper guide](PAPER_GUIDE.md) gives expected outputs and the stronger
per-example main-result check. The [project-status notebook](notebooks/00_project_status.ipynb)
is historical representation/rephrasing material, not the current paper quickstart.
Its original snapshot pins are preserved.

## Is the relation represented?

The representation tests use moral and evaluative relations: whether considerations such as fairness, autonomy, privacy, or harm count for or against an action. This gave us a controlled setting for a broader question: when a model reports that one of these relations has changed, is that new relation actually what ends up governing its later judgments?

For example, if an update makes fairness count against an action and the model reports that correctly, later moral judgments should no longer depend on whether fairness used to count for it.

The main starting dataset is ValuePrism. Each example gives a situation, an action, and a consideration such as autonomy, fairness, privacy, or harm. The dataset records whether that consideration **Supports** or **Opposes** the action in that situation.

The useful feature is that the same named consideration can point in different directions across different cases.

For example, autonomy might support respecting a patient’s refusal of treatment. In another situation, an appeal to personal freedom might support letting a toddler run into a dangerous street. The surrounding situation determines how the consideration applies.

We organize examples into four-part comparisons where two considerations reverse direction across two situations. We call this a checkerboard.

| **Situation** | **Reason A** | **Reason B** |
| --- | --- | --- |
| Situation 1 | Supports | Opposes |
| Situation 2 | Opposes | Supports |

A fixed preference for one situation cancels across the checkerboard. A fixed preference for one consideration cancels too. A model score that still separates Supports from Opposes therefore has to respond to the relation between the consideration and the particular situation.

The evaluation also includes several controls. Familiar situation and consideration identities are held out from the strict test. Text-only comparisons check how much of the label is already available from wording. Answer labels are swapped to expose output-token preferences. A factual-truth control checks that the activation pipeline can recover a semantic distinction already known to be linearly represented. The main relation result was then reproduced in Gemma after first being developed in Llama.

### Why the labels need human review

<a id="where-human-review-came-into-this"></a>

Checkerboards control several statistical shortcuts, but their semantic interpretation still depends on the examples making sense to a human reader.

A separate human-review process checked whether the support/opposition labels and rewritten cases preserved the intended meaning. The public repository includes the review procedure and calibration material. The purpose of this review is semantic quality control: reviewers judge whether the relation and rewrite are coherent, without deciding which moral position is universally correct.

### Does the score measure resistance to rephrasing?

<a id="what-changed-after-the-first-result"></a>

The original second hypothesis treated distance from the Supports/Opposes boundary as a possible measure of commitment. Cases near the boundary seemed like they might be easier to flip under meaning-preserving rewrites.

That prediction did not hold reliably. The internal score separated Supports from Opposes, while its magnitude added little beyond the original text and the model’s own answer confidence when predicting rephrasing flips.

This narrowed what the direction could mean. It tracks which side of the relation the model occupies much better than how firmly that judgment will survive a rewrite.

Around the same time, category and cross-dataset tests broadened the interpretation of the signal. Directions learned from ValuePrism transferred to ordinary argument support/attack datasets and back again. The signal also survived measured removal of answer-token, sentiment, and factual-truth components. The most useful interpretation became a general **counts-for / counts-against** relation embedded in a broader evaluative representation.

## Can information be updated?

<a id="from-representation-to-control"></a>

Once the relation direction was readable, the next question was whether changing it would change behavior.

Steering the model along the direction moved Supports/Opposes answers in the expected direction. That established causal access to the answer-producing computation.

The behavior of other semantic directions complicated the picture. Truth and sentiment directions could often move the same Supports/Opposes endpoint, including under matched intervention sizes. Several distinct internal signals could therefore reach overlapping downstream behavior.

This motivated a broader measurement of what an intervention does across the model.

### Do intervention effects stay stable across contexts?

<a id="what-does-a-direction-actually-do"></a>

A **causal fingerprint** records how a small intervention changes a fixed panel of downstream measurements.

The first fingerprint design failed its own validation because random directions could look reliable under the original diagnostic. The corrected analysis separated two questions: whether a local intervention effect can be predicted at one prompt, and whether the same semantic direction keeps a similar causal role across different prompts.

Local gradient predictions were highly accurate in both models. Cross-context stability differed sharply. Llama’s fingerprints were relatively stable, while Gemma’s were much less so. The planned cross-model claim about one stable compressed control geometry therefore remained unsupported.

That result changed the role of the one-dimensional direction in the project. It remained a useful readout and intervention axis, but the broader internal state became increasingly important.

### Does the signal follow the relation being asked about?

<a id="what-exactly-is-the-direction-tracking"></a>

Controlled examples with several people and targets let us ask whether the direction follows the specific relation being queried.

It does. When the question switches from one person to another, or from one target to another, the frozen direction follows the addressed relation. This gives a sharper interpretation than generic positive or negative tone.

The same experiments also show why a scalar relation score cannot describe the entire relational structure of the context.

### What does the full activation reveal beyond the answer?

<a id="the-direction-the-broader-state-and-the-explicit-answer"></a>

On ordinary ValuePrism disagreements, the one-dimensional score usually follows the model’s own answer.

A controlled experiment initially appeared to show a deeper gap: the internal direction tracked the intended relation while zero-shot answers were poor. Fixed demonstrations of the expected answer format largely removed that discrepancy. The model could often report the relation once the task format was made clear.

The broader activation state still carried additional relation information. Readouts from the full activation improved prediction after accounting for the text and the model’s explicit answer. A frozen-plan replication reproduced this incremental information in both models on separate pre-existing populations.

The emerging picture was a relational state with several useful views: a scalar support/opposition coordinate, an explicit answer, and additional information distributed through the broader activation.

### Does changing an answer reproduce the consequences of changing a fact?

<a id="testing-whether-an-intervention-recreates-the-semantic-change"></a>

This gave us a stronger way to evaluate steering.

Controlled context pairs differ in exactly one relation. For example, the original context might say that Alice supports a proposal and the changed context says that Alice opposes it. That change has consequences across several later questions. Questions about Alice’s support should flip, complementary questions should move consistently, paraphrases should agree, and unrelated relations should remain stable.

In Gemma, an edit calibrated on the direct question could move that answer while still failing to reproduce the broader consequences of actually changing the relation. We first thought the edit might be failing because it was too simple, but making it higher-dimensional did not fix the problem. What ended up helping much more was tailoring the edit to the specific question being asked, which suggested the harder problem was making one reusable change that still works across future questions.

This changed the interpretation of the earlier result. The harder problem was carrying one semantic change across future questions. Since the question-matched intervention already uses information about the question being asked, the next step was to construct an edit before those future questions were known.

Llama could not support the same comparison because its natural changed-context reference was unstable across equivalent wordings. That reference failure limits the comparison to Gemma rather than counting as an editing failure.

### Can one update support questions it has not seen?

<a id="editing-a-shared-state-before-future-questions"></a>

The next stage moved the intervention earlier.

Instead of editing a representation after the downstream question was already present, the editor changes an addressed relation in the shared context first. The edited context is then reused for questions that were unknown when the edit was made.

This turns the problem into a state-update task. A successful edit has to support fresh questions about the changed relation while preserving unrelated information.

The first shared-state editors produced strong direct changes, although their reliability under repeated and reversed use was limited. Later constructions improved single-edit performance and made repeated assignment and restoration much more stable.

### Can updates be repeated, undone and combined?

<a id="repeated-restored-and-combined-updates"></a>

The later studies treat an update more like an assignment to a reusable state.

Repeating the same requested value should keep the state stable. Restoring an earlier value should recover the corresponding answers. Several updates should combine without damaging relations that were never touched.

Later experiments made this result substantially stronger. Editors trained only on individual assignments learned to repeat, restore, and combine two or three updates across both Gemma and Qwen while still being able to mostly preserve relations that were never changed.

The stronger recipes met bounded criteria for changed answers and preservation of unchanged relations in both models and training seeds. They did not make every joint question correct or establish general reliability.

However, some questions combining several relations could still change depending on the model’s earlier state, even when the individual edited relations were all reported correctly. That led to the next question: are the reported current facts actually enough to explain the model’s later decisions?

## Do current facts determine later behavior?

<a id="are-the-reported-current-facts-sufficient-for-later-decisions"></a>

Different starting histories were updated to the same intended current facts. We then asked the exact same follow-up question after each history.
We call this requirement **source-history sufficiency**: once the current facts are the same, the answer should no longer depend on what was true before the update.

Formally, if the current semantic state \(S\) is sufficient for answering question \(Q\), then the old history \(H\) should provide no additional information about the answer \(Y\):

$$
P(Y \mid H, S, Q) = P(Y \mid S, Q)
$$

Once the current facts and the question are known, knowing what used to be true should not change the expected answer. The comparison holds the question and response interface fixed; it does not require physically identical hidden states or erasure of history.

The pattern appears in a retrospective census (`V9`), a fixed-criterion check of archived data (`V6`), and a prospective study on fresh generated cases (`V10`). The prospective study confirmed it across Gemma and Qwen under learned and textual updates, including questions whose relevant direct facts and reference answers were correct. These are controlled relational tasks, not public factual benchmarks.

One repeat-checked example from the prospective study makes the distinction concrete. Wren ends up supporting Garden and Orla opposing it. The model correctly reports those individual relations in every starting history. Asked whether they take the same side, it answers No after some histories and Yes after others. This selected example illustrates the effect; the complete prospective results establish its recurrence.

## Does the effect appear on public data?

Qwen showed the same pattern in a completed [public-data study](docs/RESULTS_AND_CLAIMS.md#public-data-in-context-extension). It correctly reported the current records yet answered a question combining them using an obsolete value. This extends the internal finding to public RippleEdits-derived questions. Updates were supplied in the prompt while model weights stayed fixed. The old records remained earlier in the prompt.

## Can the dependence be changed by intervening on stored activations?

A separate Gemma experiment swapped stored activations from two histories before asking the same question. In five of six selected examples under the main answer format, the answer followed the history supplying the later layers' activations, while direct fact answers stayed correct. The result was less consistent under an alternate answer format. A completed [one-case extension](docs/RESULTS_AND_CLAIMS.md#one-case-extension-with-matched-histories) now combines stronger behavioral controls with the causal result. Its histories required the same number of fact changes and had the same prompt length. Under both tested answer formats, the downstream answer followed the later layers' history while direct fact answers remained essentially unchanged. This was one retrospectively selected example, not a prevalence estimate. It shows a causal role for the later stored activations; a complete internal mechanism remains unresolved.

See the [six-case crossover result](figures/fig08_cache_crossover.png) and the [matched one-case extension](figures/fig09_cache_crossover_extension.png), with [saved-score replay and provenance](reproducibility/cache_crossover/README.md).

## Do alternative update procedures improve downstream coherence?

A completed [update-method comparison](reproducibility/update_method_comparison/README.md)
shows why downstream coherence deserves separate evaluation. Learned updates
clearly improve Qwen's rate of answering every tested question correctly from
every starting history, while Gemma's comparisons are
mixed. Absolute reliability remains limited. This supports the importance of the matched-history test without replacing it.

The result is a separation between what the model can correctly report as its current facts and the information that still affects its later decisions.

## Conclusion

<a id="what-the-project-is-arguing-for-now"></a>

The project began by asking whether a support/opposition relation could be identified inside a model. Following that relation through readout, intervention, and learned editing led to a broader question:

**When an AI says it has updated a fact, has the information governing its later decisions actually changed too?**

The experiments show that reported current facts can be insufficient to account for later behavior. Distinct histories reach the same intended final records, the relevant facts can be read correctly, and a fixed downstream answer can still depend on what used to be true.

We use readable state and operative state as shorthand for this distinction, not as names for two physically separate state objects. Correct updated facts combined with a history-sensitive reader remain compatible with the evidence.

For moral reasoning, the consequence is straightforward. A model might correctly report that fairness now counts against an action while later moral judgments still depend on whether fairness used to count for it. More generally, this matters whenever we want to know whether an AI has actually incorporated an update into the state driving its behavior.

Conventional weight-editing validation and tests in broader settings remain open. The activation swaps provide coarse causal evidence; a complete internal mechanism and the deployment failure rate remain unresolved.

**Correctly reporting the current facts does not guarantee that those facts alone account for later answers.**
