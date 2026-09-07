# Results and Claims

**Checkerboard: a two-situation, two-consideration design where each consideration flips from Supports to Opposes across situations. This cancels fixed situation and fixed consideration preferences.**

**Support/opposition direction: a linear direction in activation space fitted to separate Supports from Opposes examples.**

**Causal fingerprint: the vector of local changes that a direction induces in a fixed panel of downstream semantic answer margins. The formal definition appears where the result is introduced.**

**More concepts than control knobs: my informal hypothesis that representations can distinguish semantic variables more finely than the downstream mechanisms through which interventions change behavior.**

**I left the raw formulas in mostly for transparency. You definitely do not need to follow every step of the math to understand the results. In my opinion, the more important part is what each metric is actually trying to measure. The formulas are there so the exact definitions can stay transparent, and so we can come back later and catch anything dumb or questionable if needed.**

---

## Part I. Establishing the relation signal

First establish that the task, measurement pipeline, and cross-model relation signal are real.

### 1. Does ValuePrism contain a real context-dependent relation task?

Yes. A ValuePrism row is a short natural-language situation or action paired with one named consideration, such as Compassion, Autonomy, or a Right, plus a label saying whether that consideration **Supports** or **Opposes** the action. After removing the third `Either` label, the dataset contains **183,023 Supports/Opposes rows**. Among them, **3,437 exact consideration strings** receive both labels in different situations, yielding **13,923 possible checkerboards** across **6,073 consideration pairs**.

The strict split starts from the deduplicated binary data and contains **116,000 training rows** and **7,394 test rows**. A test row must be held out on **both** axes: its situation is unseen in training and its consideration cluster is unseen in training. A training row must cross neither held-out boundary. The remaining **59,208 mixed-boundary rows** cross only one of those two boundaries, so they are excluded from both partitions rather than partially exposing a strict-test situation or consideration family.

This grouping matters. Deliberately exposing held-out consideration wording improved a text-only predictor by **7.29 percentage points [4.90, 9.69]**. Exposing familiar situations changed performance by only **0.53 points [-0.81, 1.86]**. Familiar consideration wording is therefore a meaningful shortcut even without activations.

A slightly stricter semantic filter retained **1,865 within-situation comparisons**, while the next automatic filter collapsed to only **23**. Strong grouping was therefore necessary without filtering away nearly all of the relational structure.

### 2. Can the activation pipeline recover a known semantic distinction?

Yes. The same extraction and direction-fitting machinery recovered factual True/False structure in both models after the answer format was redesigned to avoid a token-semantic confound.

The repaired control used neutral `A/B` symbols whose meanings were reversed across examples, then tested transfer to a held-out `1/2` format. For each of eight training partitions $k$, the training rows define a unit truth direction $w_k$. Projection scores on held-out rows are centered and scaled using **training-only** statistics:

```math
z_{ik}=\frac{h_i^\top w_k-m_k}{\sigma_k},
```

where $m_k$ and $\sigma_k$ are the mean and standard deviation of training projections for partition $k$. The held-out effect is

```math
\Delta_k=\mathbb{E}[z_{ik}\mid \mathrm{True}]
-\mathbb{E}[z_{ik}\mid \mathrm{False}],
\qquad
T=\frac{1}{8}\sum_{k=1}^{8}\Delta_k.
```

The averaged effect is the **standardized separation**, written $T$. It is a True-minus-False mean difference measured in training-projection standard deviations, averaged across eight train-oriented directions. A value near 1.92 therefore means that held-out True examples lie about 1.92 training-projection standard deviations farther along the train-defined truth direction than held-out False examples do.

| Model | Selected layer | Standardized separation $T$ | 95% interval | Transfer check |
|---|---:|---:|---:|---|
| Llama-3.1-8B-Instruct | 14 | **1.9245** | [1.8930, 1.9556] | **1.8402** on held-out `1/2` |
| Gemma-2-9B-it | 25 | **1.9444** | [1.9207, 1.9666] | same neutral-mapping design |

For Llama, none of **1,000** group-preserving randomized labelings matched the observed separation. The pipeline can therefore recover a clear semantic distinction while separating semantic labels from the physical token used to report them.

### 3. Do Llama and Gemma contain a context-sensitive support/opposition representation?

Yes. In both models, activation scores distinguish Supports from Opposes in a way that depends on the particular situation-consideration pairing.

A concrete checkerboard from the development data:

| Situation | Compassion | Right to truthful information |
|---|---|---|
| Telling a white lie to your friend. | Supports | Opposes |
| Telling a friend the truth if her dress looks ugly | Opposes | Supports |

The model sees **one cell at a time**: one short situation/action plus one named consideration. The checkerboard is only the evaluation structure.

**For each example, the Supports versus Opposes score is put on a common normalized scale. We then combine the four scores in a checkerboard so that any fixed preference for a situation or moral reason cancels to zero. The resulting checkerboard score therefore measures how strongly the model’s score changes with the particular situation-and-reason pairing. Zero means no such interaction under this measure. Larger positive values indicate a stronger context-sensitive pattern.**

Formally, each scorer is standardized using mean and standard deviation frozen on the selection split:

```math
\tilde f(s,c)=\frac{f(s,c)-\mu_{\mathrm{select}}}{\sigma_{\mathrm{select}}},
\qquad
I_b=
[\tilde f(s_1,c_1)-\tilde f(s_1,c_2)]
+[\tilde f(s_2,c_2)-\tilde f(s_2,c_1)].
```

The four-cell combination is the **checkerboard interaction**, written $I_b$. Equivalently, $I_b=\tilde S_{11}-\tilde S_{12}-\tilde S_{21}+\tilde S_{22}$.

Any additive score $f(s,c)=a(s)+b(c)$ gives $I_b=0$ exactly. Values such as **1.6089** or **2.3221** are therefore standardized difference-in-differences, expressed in selection-split standard deviations.

The raw score $f$ depends on the model being evaluated. The model-answer row uses the semantic Supports-minus-Opposes candidate log-probability margin; the direction uses $h^\top d$; the logistic probe uses its linear decision function. The frozen MiniLM comparator is a linear classifier over MiniLM situation and consideration embeddings plus their elementwise product and absolute difference.

Llama used 1,500 fit rows and 300 layer-selection rows. It was then evaluated on 500 rows across 125 checkerboards, at the selected layer 19.

| Llama measurement | AUROC | $I_b$ |
|---|---:|---:|
| Model answer margin | 0.721 | 1.6089 |
| Support/opposition direction | 0.732 | **1.6470** |
| Logistic activation probe | **0.780** | **2.0836** |
| Frozen MiniLM comparator | n/a | 0.2842 |

The direction exceeded the text comparator by **1.3628 [1.0885, 1.6370]** and transferred to a held-out answer format at **2.1569**.

Gemma independently selected layer 27:

| Gemma measurement | Within-situation accuracy | Within-consideration accuracy | $I_b$ |
|---|---:|---:|---:|
| Model answer margin | 0.8920 | 0.8597 | 2.4068 |
| Support/opposition direction | 0.8720 | 0.8776 | **2.3221** |
| Logistic activation probe | 0.8680 | 0.8425 | 2.1499 |
| Frozen MiniLM comparator | 0.5560 | 0.5885 | 0.2842 |

The Gemma direction exceeded the text comparator by **2.0378 [1.6608, 2.4149]** and transferred at **2.2617**.

## Part II. Identifying what the direction means

Next ask what the direction means: whether its magnitude measures commitment and whether its semantics are specifically moral.

### 4. Does distance along the direction measure how firmly the model holds the judgment?

Not reliably. Rephrasings sometimes changed the model's Supports/Opposes judgment, but distance from the direction's decision boundary did not add useful prediction beyond the original text and native answer confidence.

On the representative set, **23 of 170 rephrasings changed sign**, or **13.5%**. Those flips were spread across 13 of 48 originals. Exact repeats and trivial restatements were stable in **11/11** cases each. Requiring the semantic verdict to agree under both swapped answer mappings reduced the robust event count to **5 flips across 4 originals**.

For Llama's 49 representative originals, baseline log loss was **0.440353**. Adding the direction score moved it to **0.444223**, an incremental gain of **-0.003870 [-0.024731, 0.010818]**. Gemma's prospective estimate was **+0.002753 [-0.000507, 0.007011]**. A later 67-original test was also adverse, with normalized-RMSE improvement **-0.06625 [-0.13450, -0.00870]**.

The scalar identifies relation polarity, but its magnitude is not a reliable ordinal measure of resistance to rephrasing.

### 5. Is the direction specifically moral?

No. Within ValuePrism, separately fitted Value, Right, and Duty directions were almost parallel and performed similarly to one global direction.

| Model | Within-type AUROC | Cross-type AUROC | Global AUROC | Minimum cosine |
|---|---:|---:|---:|---:|
| Llama | 0.737 | 0.737 | 0.739 | **0.966** |
| Gemma | 0.838 | 0.833 | 0.835 | **0.965** |

The ValuePrism direction also transferred without refitting to three argument-relation corpora. **AMPERE++** covers support and attack relations in ICLR referee reports. **AbstRCT** covers argument relations in biomedical randomized-trial abstracts. **US2016** covers argumentative relations in 2016 U.S. presidential debates and linked Reddit reactions. An AMPERE++ direction also transferred back to ValuePrism.

| Transfer | Llama AUROC | Gemma AUROC |
|---|---:|---:|
| ValuePrism → AMPERE++ | **0.762 [0.682, 0.847]** | **0.801 [0.731, 0.882]** |
| AMPERE++ → ValuePrism | **0.785 [0.754, 0.817]** | **0.835 [0.804, 0.866]** |
| ValuePrism → AbstRCT | 0.867 | 0.877 |
| ValuePrism → US2016 | 0.709 | 0.785 |

Relation decoding and transfer remained after removing measured answer-token, sentiment, and factual-truth components. The raw geometry was still entangled, including relation/sentiment cosine **0.680** in Llama and **0.792** in Gemma.

The direction is therefore better described as a **general context-sensitive support/opposition component** embedded in broader evaluative geometry.

## Part III. From semantic readout to causal control

Now move from readout to intervention: causal effect, specificity, and cross-context stability.

### 6. Does steering the direction change the model's answer?

Yes. The intervention is

```math
h' = h + \alpha d_{\mathrm{relation}}.
```

The frozen headline statistic is a **dimensionless composite contrast** over paired signed intervention responses. Zero means no consistent intended-direction effect across the panel; positive values mean the answer moves systematically with steering. Its magnitude sits on the contrast's own dimensionless scale.

| Model | Composite contrast | 95% interval |
|---|---:|---:|
| Llama | **0.390** | [0.363, 0.417] |
| Gemma | **1.280** | [1.171, 1.388] |

Both had permutation **p = 0.0002**.

The underlying response curve uses the semantic candidate margin $\log p(\mathrm{Supports})-\log p(\mathrm{Opposes})$. Regressing this margin on signed intervention coefficient $\alpha$ gave selected-layer slopes **0.770** in Llama and **1.772** in Gemma, in semantic log-probability-margin units per frozen $\alpha$ unit. Nearby-layer slopes were **0.798** and **1.004**, so the effect was not sharply localized.

### 7. Is that causal effect specific to the relation direction?

Mostly no on the tested answer endpoint. Truth and sentiment directions often moved the same Supports/Opposes margin as much as, or more than, the relation direction.

Each number below is **relation response slope minus comparator response slope** after matching intervention scale. Positive favors relation steering; negative favors the comparator.

| Comparison | Llama VP | Llama AMPERE++ | Gemma VP | Gemma AMPERE++ |
|---|---:|---:|---:|---:|
| Relation minus factual | -0.015 | -0.125 | -0.247 | -0.681 |
| Relation minus sentiment | -0.542 | -0.392 | **+0.602** | -0.113 |

Only **1 of 8** headline comparisons favored relation steering, and equal-L2 normalization did not remove the pattern.

A refusal-domain reconstruction gave a related result: 11 directions had mean pairwise cosine **0.457**, while geometric similarity predicted refusal-effect strength with **r ≈ 0.89**. This motivated the **more concepts than control knobs** hypothesis.

### 8. Does a semantic direction have a stable causal role across contexts?

Not automatically. A direction can have locally predictable effects while its pattern across downstream endpoints changes with context.

For direction $d_i$, activation $h$, and semantic answer margin $m_j$,

```math
c_{ij}(h)=\nabla_h m_j(h)^\top d_i,
\qquad
f_i(h)=\big(c_{i1}(h),\ldots,c_{iJ}(h)\big).
```

The vector $f_i(h)$ is the **causal fingerprint**. Its coordinates are local changes in semantic output margins, not residual-stream similarities. Gradients were validated against centered finite interventions before fingerprint stability was evaluated.

The original random-direction reliability check was invalid. It effectively compared $\bar g_{\mathrm{left}}^\top r_k$ with $\bar g_{\mathrm{right}}^\top r_k$. If the two halves have similar mean gradient fields, fixed random directions look reproducible even without semantic structure. A semantics-free synthetic fixture scored about **0.99995**, so that statistic measured repeatable projection geometry rather than semantic specificity.

The final validation used **32 independent items per model**, eight each for language, relation, sentiment, and truth.

| Metric | Llama | Gemma |
|---|---:|---:|
| Gradient / finite-intervention correlation | **0.9926** | **0.9984** |
| Sign agreement | 0.9863 | 0.9814 |
| Median relative error | 0.0663 | 0.0467 |
| Median same-direction fingerprint similarity | **0.8435** | **0.4594** |
| Relation-family similarity | 0.7500 | 0.2580 |

Local numerical fidelity was excellent in both models, while cross-context fingerprint stability was much stronger in Llama.

## Part IV. Recovering the richer relational state

The next stage treats the scalar as one coordinate and asks what information exists in the broader state.

### 9. Does the model track who supports what, or only polarity?

The broader activation contains substantially more than a generic positive/negative score. Controlled contexts contain multiple people and targets while the **query** changes.

- **Dual-query reversal:** keep the context fixed and ask two relation questions whose correct signs are opposite.
- **Holder swap:** keep context and target fixed but switch the queried holder, in contexts where the two holders have opposite stances.
- **Target swap:** keep context and holder fixed but switch the queried target, in contexts where the correct relation sign reverses.

Using the frozen one-dimensional direction with no refitting:

| Measurement | Llama | Gemma |
|---|---:|---:|
| Scalar sign balanced accuracy | **0.857** | **0.827** |
| Dual-query reversal | 1.000 | 0.896 |
| Holder swap | 1.000 | 0.917 |
| Target swap | 1.000 | 0.979 |
| Four-way full residual activation | **0.940** | **0.940** |
| Four-way scalar alone | 0.531 | 0.534 |

The four-way task jointly identifies **which holder** is queried and **whether that holder supports or opposes the target**.

For the residual readout, fit-context activations are linearly residualized against the scalar score, native answer probability, prompt length, context-sentence count, answer-mapping one-hots, and lexical-family one-hots; the fitted transformation is then applied unchanged to held-out contexts. Relation-presence decoding remained **0.939** in Llama and **0.950** in Gemma.

An additive holder-plus-sign baseline reached only **0.604** joint balanced accuracy in Llama and **0.513** in Gemma, versus about **0.95** from the full activation. The richer state therefore contains holder-target-relation interaction information missing from the scalar.

### 10. Is the support/opposition direction different from the model's own answer?

Usually not by much. On ValuePrism cells where the model's report disagreed with the task label, the one-dimensional direction overwhelmingly followed the report.

| Model | Disagreement cells | Probability scalar sides with task label |
|---|---:|---:|
| Llama | 158 across 110 boards | **0.082 [0.039, 0.130]** |
| Gemma | 102 across 78 boards | **0.049 [0.010, 0.092]** |

A controlled zero-shot task initially looked like a representation/report dissociation, but fixed eight-shot elicitation largely removed it. Native answers then agreed with the frozen direction on more than **99%** of signed Llama rows and about **98%** of Gemma rows.

The broader activation still adds information beyond text, mapping, and the native answer. These replication populations came from pre-existing M1 development partitions. The 1,500-row fit split formed **1,454 situation units**. The 300-row selection split formed **75 checkerboards**. Neither is being presented as the 7,394-row strict test set.

| Population | Llama log-loss improvement | Gemma improvement |
|---|---:|---:|
| 1,454-situation population | **0.064 [0.044, 0.083]** | **0.038 [0.025, 0.051]** |
| 75-checkerboard population | **0.073 [0.028, 0.117]** | **0.031 [0.004, 0.059]** |

The scalar is strongly report-aligned, while the broader activation contains additional relation information.

### 11. Is the support/opposition distinction created by instruction tuning?

No. On an explicit stance task, both base checkpoints performed above chance, with balanced accuracy **0.771** for Llama and **0.792** for Gemma. At the frozen layers, relation AUROC was **1.000** for base→base, base→instruct, instruct→base, and instruct→instruct decoding.

The exact geometry still rotated. Base/instruct direction cosine was **0.627** in Llama and **0.603** in Gemma. Chat-template comparisons fell further, to **0.207** and **0.290**, despite high AUROC.

The distinction is therefore linearly separable before instruction tuning even though post-training and prompting alter its detailed geometry.

## Part V. Does steering recreate the semantic counterfactual?

Finally ask whether strong steering reproduces the broader consequences of actually changing the relation.

### 12. If steering changes the answer, does it recreate the natural counterfactual?

In Gemma, no. Rank-1 relation steering could convert the direct answer at counterfactual-like strength while the other consequences moved away from the pattern produced by actually changing the relation.

The test uses deterministic synthetic contexts with two holders, two targets, and explicit relation facts. Base and counterfactual versions are identical except that **one focal holder-target relation is programmatically flipped SUPPORT↔OPPOSE**; all other facts stay fixed. A second independently written realization changes wording and sentence order while preserving those facts.

$Q_1$ asks about the focal relation and calibrates intervention strength. $Q_2$ asks its complement, $Q_3,Q_4$ paraphrase the focal facts, and $Q_5,Q_6$ query unchanged holder/target relations.

For question $q$, the behavioral endpoint is a semantic margin in natural-log probability units:

```math
m_q(x)=\log p(\mathrm{ENTAILED}\mid x,q)-\log p(\mathrm{NOT\_ENTAILED}\mid x,q).
```

The natural signature is

```math
N_w=\big(m_q(x_{cf})-m_q(x_{base})\big)_{q=2}^{6},
```

and $I_w$ is the corresponding intervention-induced change.

The distance uses development-frozen robust scales:

```math
s_j=\max\left(1.4826\,\mathrm{MAD}(N_{\cdot,j}),\;
0.25\times\mathrm{median}_{k\in\{2,3,4\}}
1.4826\,\mathrm{MAD}(N_{\cdot,k})\right),
```

```math
D(u,v)^2=\sum_{j=2}^{6}a_j
\left(\frac{u_j-v_j}{s_j}\right)^2,
```

with $a_2=a_3=a_4=\frac13$ and $a_5=a_6=\frac12$. Counterfactual Signature Recovery is

```math
\mathrm{CSR}_w(I)=1-\frac{D(I_w,N_w)^2}{D(0,N_w)^2}.
```

CSR = **1** is exact recovery, **0** is no movement from base, and negative values mean the intervention moved farther from the natural counterfactual.

At the main target, rank-1 steering hit the requested $Q_1$ margin on **100%** of 96 final Gemma contexts. It flipped the hard answer on **99%** of them. CSR was nevertheless **-0.851 [-1.005, -0.707]**.

Recovery worsened as the steering strength rose, measured as a fraction of the natural $Q_1$ change:

| Fraction of the natural $Q_1$ change | 25% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|
| CSR | -0.144 | -0.394 | -0.851 | -1.681 |

| Held-out consequence | Natural | Full-state interpolation | Rank-1 steering |
|---|---:|---:|---:|
| Complement $Q_2$ | +2.61 | +1.91 | **-0.28** |
| Focal paraphrases $Q_3,Q_4$, mean | +0.66 | +0.49 | +0.18 |
| Stability controls $Q_5,Q_6$, mean | +0.06 | +0.04 | **+1.15** |

The same direct-answer target was reachable through the full state. A complete counterfactual-state patch gave **CSR 0.981 [0.980, 0.983]**; interpolating the whole activation only far enough to reach the same $Q_1$ target gave **0.914 [0.899, 0.926]**.

A later-layer witness corroborated the behavioral result. It is an **L2-regularized logistic probe at layer 32**, five layers after steering at layer 27, fitted on 64 separate contexts. Before fitting, each layer-32 residual-stream vector is orthogonally projected out of the span of the two tested rank-1 relation directions:

```math
X_\perp=X-(XQ)Q^\top.
```

The probe is trained on these residualized activations; its weight vector is not projected after fitting. On unsteered final states it reached **1.000** balanced accuracy and **0.979** cross-template accuracy.

| Intervention | Later-layer witness recovery |
|---|---:|
| Full-state patch | **0.993** |
| Full-state interpolation | **0.779 [0.727, 0.828]** |
| Rank-1 relation steering | **0.198 [0.123, 0.269]** |

Behavior and later-layer state agree: whole-state movement approaches the natural counterfactual, while rank-1 steering converts the answer without reproducing the broader consequence pattern. Llama's natural counterfactual reference was not stable enough across equivalent prompt realizations for the same fidelity comparison.

### 13. Overall evidence picture

The support/opposition direction survives checkerboard controls, replicates across Llama and Gemma, transfers to ordinary argument relations, and remains informative after measured answer-token, sentiment, and truth components are removed. Its magnitude is not a reliable commitment meter, semantically different directions can move the same answer endpoint, and controlled multi-person tests show that the broader activation carries relational structure missing from the scalar.

The counterfactual experiment sharpens the distinction. In Gemma, rank-1 steering produces the counterfactual answer while moving the held-out consequence pattern away from the natural semantic change. A higher-dimensional state change reaches the same answer target while preserving that pattern.

> **A linear semantic direction can be readable, transferable, and causally powerful while still being only one coordinate of a richer representation. Controlling that coordinate can reproduce the target answer without reproducing the broader computation associated with changing the underlying semantic fact.**
