# Section 6: Causal follow-ups

<a id="sec:causal-followups"></a>

Transferring stored activations changed selected downstream answers while current-fact checks stayed correct. A separate population study found partial repair after replacing a fixed band of cached values. These interventions establish causal influence under the tested manipulations.

## Crossover

The selected-case experiment exchanges earlier or later layers of the stored key/value state between two histories. The downstream question is introduced after the exchange. Both transfer directions and both answer formats are measured.

Five of six selected cases show a clear later-layer pattern in the main format. The alternate format is less consistent. A separate, structurally matched one-case extension passes the stronger checks in both formats. See [all cases, boundaries and controls](appendix/A09_crossover.md).

[Figure 5: selected crossover](../figures/main/selected_crossover.pdf) ([PNG](../figures/main/selected_crossover.png)). The figure shows how answers change after exchanging parts of the stored state. Its selected examples support causal influence without locating a unique storage site. See manuscript Appendix E.1.

## Question-blind refresh

The refresh experiment starts from learned-updated contexts and a full-current-record reference. It replaces values at a fixed layer band before opening the questions. Keys, remaining values, weights and the recipient prompt stay fixed. Each question starts from an independent state copy.

Targeted refresh repairs 36 of 83 qualifying failures across 28 designs. The comparison band repairs two. The prespecified 50% strong-repair threshold fails. Across all scheduled joint responses, targeted refresh corrects 944 errors and introduces 347 errors. The full-current-record reference has fewer total errors than either the original context or targeted refresh.

[Appendix Figure 7: repair benefit and damage](../figures/main/repair_benefit_damage.pdf) ([PNG](../figures/main/repair_benefit_damage.png)). The figure compares targeted refresh, comparison refresh and the full reference. Partial repair must be read alongside introduced errors and the failed strong-repair threshold. See manuscript Appendix E.2.

See [population, uncertainty, reverse interventions and repeats](appendix/A10_refresh_accounting.md). Later successful-case repeats reproduce selected repairs and add no independent population observations.

## Reproduce

From the installed repository, use new output paths:

```sh
python -m repro.cache_crossover --output reproduced/crossover
python -m repro.cache_crossover_extension --output reproduced/crossover-extension
python -m repro.question_blind_repair --output reproduced/refresh
```

The crossover commands reconstruct results from saved journals. The refresh command reconstructs primary gates, harms and selected-repeat statistics from saved scores. It retains the original intervals without recalculating them. See the [reproduction matrix](../REPRODUCIBILITY.md#reproduction-levels) for fresh-execution limits. [Additional carrier and mediation evidence](appendix/A12_mechanism.md) records the bounded follow-ups and their failures.
