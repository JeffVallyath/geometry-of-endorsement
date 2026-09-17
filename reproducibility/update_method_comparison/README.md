# Update procedures and downstream coherence

Internal comparison package: `UPDATE_METHOD_RESULTS`; completed panel: `ORIGIN`.

Different ways of updating the same facts produce different downstream answers.
This completed matched-history comparison measures whether a model answers all
34 questions correctly from each of four histories ending at the same facts.
There are 64 benchmark cases per model. A benchmark case succeeds only if all 136 answers
are correct. The learned methods use both saved seeds, averaged within each benchmark case.
The models are `google/gemma-2-9b-it` and `Qwen/Qwen3-8B`, with exact revisions
in the retained preflight records. This is separate from the Qwen2.5/Ripple study.

## Reconstruct from saved scores

```sh
python -m repro.update_methods --output reproduced/update-methods
```

This CPU-only command checks 208,896 score rows against the frozen questions and
semantic labels, reconstructs the four primary contrasts, and verifies every
cell of the supplied headline, per-case, per-scene and timing/memory tables.
It produces `results.json` and those four CSVs without model inference.

The primary comparison uses a factual-update baseline: an attempted internal
fact update combined with the latest textual correction
(`FIELD_PLUS_LATEST_ERRATUM`). Effects are percentage-point changes in the rate
of getting every answer in a case correct, with 98.75% paired case-bootstrap
intervals and 10,000 draws.

| Model | Update procedure | Change (percentage points) | Interval |
| --- | --- | ---: | ---: |
| Gemma | Constrained learned update (`INV_PAIR_NLL`) | +15.625 | [0, 30.469] |
| Gemma | Consistency-trained update (`FREE_PAIR_CONSISTENCY`) | +13.281 | [-0.781, 27.344] |
| Qwen | Constrained learned update | +31.250 | [20.312, 43.750] |
| Qwen | Consistency-trained update | +39.844 | [28.906, 50.781] |

Both Qwen intervals exclude zero. Gemma is mixed: one interval touches zero and
the other crosses it. The constrained Gemma method also varies substantially
between seeds (30/64 versus 12/64 successful benchmark cases).

Absolute reliability remains limited. The baseline succeeds on 11/64 Gemma benchmark cases
and 1/64 Qwen benchmark cases. The learned methods' seed-mean success rates are 32.8125%
and 30.46875% for Gemma, and 32.8125% and 41.40625% for Qwen. These gains do not
establish a universal consistency cure. The baseline used its recorded fallback
in 192/256 contexts for each model; its label alone should not be interpreted as
successful internal fact refresh in every context.

This is supporting evidence for separating local update quality from downstream
coherence. The paper's centerpiece remains the matched-history sufficiency test:
whether the old path predicts a later answer after current facts are verified.

[Figure 10](../../figures/fig10_update_methods.png) shows both relative effects
and absolute success. [Technical provenance](PROVENANCE.md) records archive
verification, code provenance, and the boundaries of the retained evidence.
