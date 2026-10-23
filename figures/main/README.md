# Paper figures

The main text contains five figures: a matched-history design schematic, matched-history result rates, relation-readout controls, consequence recovery, and activation crossover. Appendix figures report fresh confirmation/utility and refresh benefits/damage. PDFs are vector outputs with searchable text; SVG and PNG versions provide editable sources and previews.

## Reproduce

The [plotting functions](../../src/repro/main_figure_renderers.py) draw the compositions from retained measurements. Run from the repository root after installing the documented CPU dependencies:

```sh
python -m repro.main_figures --output reproduced/main-figures
```

This writes seven figures as PDF, SVG, and PNG. Input hashes are checked first. Plotting performs no fitting, model execution, bootstrap, or new statistical analysis.

## Main-text figures

- Figure 1, [matched-history design](matched_history_design.pdf): two starting histories (H₁, H₂) receive the same update commands and reach the same intended current record before the same question is asked; the caption specifies which question-required facts are verified under every history. The caption defines the correctness checks.
- Figure 2, [relation readout](construct_validity.pdf): context-sensitive checkerboard interactions and controls in Llama and Gemma. These are development measurements.
- Figure 3, [consequence recovery](consequence_qualification.pdf): direct relation steering, whole-state interpolation, and full replacement. Both tuned arms miss the prespecified 90% full-dose rule (80/96 and 81/96 contexts), so the comparison is descriptive and qualification-limited.
- Figure 4, [prospective matched-history rates](matched_history.pdf): Gemma and Qwen results across the four update procedures. Denominators and intervals are defined in the manuscript caption.
- Figure 5, [causal crossover](selected_crossover.pdf): answer probabilities after exchanging early and later activations in a selected Gemma example.

## Appendix figures

- Figure 6, [fresh confirmation and validation utility](fresh_confirmation_utility.pdf): recurrence under prespecified stronger controls and a separate inconclusive utility comparison.
- Figure 7, [refresh benefits and damage](repair_benefit_damage.pdf): repair gain over the later-layer comparison, net joint-error change, and newly introduced direct and joint errors for the targeted refresh.
