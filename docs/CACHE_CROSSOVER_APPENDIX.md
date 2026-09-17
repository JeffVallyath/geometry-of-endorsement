# Saved outcomes from swapping activations between histories

Exact study: `V10_CACHE_CROSSOVER_RESULTS`. Each benchmark case is a final fact
configuration with its starting histories; the saved records call it a `root`.

All 24 original outcomes are retained below. Main and alternate refer to the
original study's chosen answer formats, not always literal and coded answers.
Values are normalized probability of semantic Yes, not accuracy.

The compact state codes in this table identify which history supplies each part:
AA is original history A, BB is original history B, AB combines the first 16
layers from A with the remaining 26 from B, and BA reverses that combination.
`LATER-FOLLOWING` means the answer follows the later layers' history under the
frozen criterion. `MIXED/INCONCLUSIVE` means that criterion is not met.
`LOW-DISCRIMINATION` means the original answers are too similar for a clear test.
The `witness` rows are the selected positive examples; `control` rows are their
matched comparisons. The table preserves the original technical labels.

| Matched pair | Case ID | Role | Answer format | Question suffix | AA | AB | BA | BB | Recorded classification | Fact checks pass |
|---|---|---|---|---|---:|---:|---:|---:|---|---|
| 00 | SSC1-FINAL-0034 | witness | primary | d0-07 | 0.197267 | 0.680723 | 0.130336 | 0.699027 | LATER-FOLLOWING | True |
| 00 | SSC1-FINAL-0034 | witness | alternate | d1-07 | 0.074499 | 0.119936 | 0.051842 | 0.123626 | LOW-DISCRIMINATION | True |
| 00 | SSC1-FINAL-0011 | control | primary | d0-10 | 0.009682 | 0.160234 | 0.005522 | 0.078357 | LOW-DISCRIMINATION | True |
| 00 | SSC1-FINAL-0011 | control | alternate | d1-10 | 0.016606 | 0.338814 | 0.009709 | 0.187466 | MIXED/INCONCLUSIVE | True |
| 02 | SSC1-FINAL-0026 | witness | primary | d1-extra2 | 0.965777 | 0.024701 | 0.977336 | 0.017052 | LATER-FOLLOWING | True |
| 02 | SSC1-FINAL-0026 | witness | alternate | d0-extra2 | 0.987266 | 0.087946 | 0.991868 | 0.050182 | LATER-FOLLOWING | True |
| 02 | SSC1-FINAL-0031 | control | primary | d1-09 | 0.996532 | 0.994799 | 0.996976 | 0.995170 | LOW-DISCRIMINATION | True |
| 02 | SSC1-FINAL-0031 | control | alternate | d0-09 | 0.998501 | 0.997017 | 0.998718 | 0.997193 | LOW-DISCRIMINATION | True |
| 04 | SSC1-FINAL-0033 | witness | primary | d0-07 | 0.004402 | 0.885073 | 0.003108 | 0.527056 | MIXED/INCONCLUSIVE | True |
| 04 | SSC1-FINAL-0033 | witness | alternate | d1-07 | 0.002160 | 0.347510 | 0.001737 | 0.067420 | LOW-DISCRIMINATION | True |
| 04 | SSC1-FINAL-0007 | control | primary | d0-extra3 | 0.000572 | 0.002357 | 0.000804 | 0.002622 | LOW-DISCRIMINATION | True |
| 04 | SSC1-FINAL-0007 | control | alternate | d1-extra3 | 0.001765 | 0.004745 | 0.002018 | 0.005549 | LOW-DISCRIMINATION | True |
| 05 | SSC1-FINAL-0004 | witness | primary | d0-07 | 0.997720 | 0.012876 | 0.998276 | 0.022445 | LATER-FOLLOWING | True |
| 05 | SSC1-FINAL-0004 | witness | alternate | d1-07 | 0.997069 | 0.927196 | 0.997082 | 0.930988 | LOW-DISCRIMINATION | True |
| 05 | SSC1-FINAL-0023 | control | primary | d0-07 | 0.999790 | 0.996873 | 0.999794 | 0.995535 | LOW-DISCRIMINATION | True |
| 05 | SSC1-FINAL-0023 | control | alternate | d1-07 | 0.999173 | 0.996959 | 0.999149 | 0.996153 | LOW-DISCRIMINATION | True |
| 06 | SSC1-FINAL-0057 | witness | primary | d1-08 | 0.014909 | 0.938181 | 0.044362 | 0.967555 | LATER-FOLLOWING | True |
| 06 | SSC1-FINAL-0057 | witness | alternate | d0-08 | 0.067339 | 0.836544 | 0.183286 | 0.912848 | MIXED/INCONCLUSIVE | True |
| 06 | SSC1-FINAL-0022 | control | primary | d1-02 | 0.001111 | 0.000683 | 0.001827 | 0.000601 | LOW-DISCRIMINATION | True |
| 06 | SSC1-FINAL-0022 | control | alternate | d0-02 | 0.000854 | 0.000482 | 0.001525 | 0.000269 | LOW-DISCRIMINATION | True |
| 07 | SSC1-FINAL-0054 | witness | primary | d0-08 | 0.997556 | 0.146965 | 0.998115 | 0.049775 | LATER-FOLLOWING | True |
| 07 | SSC1-FINAL-0054 | witness | alternate | d1-08 | 0.996634 | 0.632231 | 0.997075 | 0.155832 | MIXED/INCONCLUSIVE | True |
| 07 | SSC1-FINAL-0029 | control | primary | d0-08 | 0.998574 | 0.996365 | 0.998003 | 0.996088 | LOW-DISCRIMINATION | True |
| 07 | SSC1-FINAL-0029 | control | alternate | d1-08 | 0.992360 | 0.974276 | 0.991255 | 0.974257 | LOW-DISCRIMINATION | True |

Main-format preservation requires correct direct answers, correct-answer
probability at least 0.95, candidate mass at least 0.90, and change at most 0.02
from the original supplying the later layers. Alternate-format preservation
tests correctness only where both original direct answers are correct. These
original rules and outcomes are unchanged; the extension's strict rule applies
to both of its formats separately.

Full-precision values, errors and source selectors are in
[original_panel.csv](../artifacts/cache_crossover/original_panel.csv). All direct
checks and probability/log-odds effects are in the existing
[saved summaries](../reproducibility/cache_crossover/saved_summaries.json), which
are byte-identical to the overlay's copy. The source is
`V10_CACHE_CROSSOVER_RESULTS.zip::combined/summaries.json`, checked against
`combined/raw.jsonl`. This panel and the selected extension are not pooled.
