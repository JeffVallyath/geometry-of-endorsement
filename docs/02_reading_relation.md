# Section 2: Reading the relation

<a id="sec:reading-relation"></a>

Linear readouts distinguish whether a consideration supports or opposes an action in context. The checkerboard comparison isolates an interaction between situation and consideration. This is development evidence, including a held-out development evaluation. An audited confirmatory result is not supplied.

## Method

Each checkerboard crosses two situations with two considerations. The stored relation reverses across both dimensions. A situation-only, consideration-only or additive scorer therefore has zero interaction contrast.

The model receives the situation, consideration and a requested answer mapping. Activations are extracted at the final pre-answer token after each decoder block. A normalized difference-in-means direction and an L2 logistic probe read the joint state. Layer selection uses a separate selection split. A frozen MiniLM text comparator tests what can be predicted directly from the input strings.

The activation probes use 1,500 training items, 300 selection items and 500
evaluation items. Selection and evaluation contain 75 and 125 checkerboards.
[Construction and statistics](appendix/A01_relation_construction.md) specify
grouping, leakage controls, fitting and normalization. The [text-feature guide](appendix/A03_text_features.md)
separately documents the comparator's features and training data.

## Result

The Llama readouts have standardized interaction means of 1.647 for the difference-in-means direction and 2.084 for the logistic probe. The frozen text comparator has a mean of 0.284. Both models' complete interaction, control and layer tables are in [figure_data.json](../artifacts/figures/figure_data.json).

[Figure 2: construct validity](../figures/main/construct_validity.pdf) ([PNG](../figures/main/construct_validity.png)). The figure compares joint-activation readouts with isolated-input, additive and text-only controls. The interaction signal is strongest in the joint representation, within the retained development evaluation.

The [permutation and factual controls](appendix/A02_controls.md) preserve the original smaller nulls, the larger Llama null and the failed first factual-control version. Reading a relation is distinct from showing that changing this readout changes downstream reasoning.

## Reproduce

```sh
python -m repro.relation_readout --output reproduced/relation
```

The CPU command verifies delivered inputs, reconciles aggregate tables, recounts
stored permutation tails and checks the available Llama factual-control gates.
It does not refit probes or extract activations. Source, recorded model revisions,
fitted parameters and Gemma membership identifiers are inspectable.
The [reproduction matrix](../REPRODUCIBILITY.md#reproduction-relation) lists the
remaining inputs needed for fresh execution.
