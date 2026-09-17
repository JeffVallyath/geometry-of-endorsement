# Paper guide

The main question is whether the same verified current facts are sufficient for
the same later answer, regardless of the path to those facts. Downstream
propagation failures have prior art. The matched-history test is the centerpiece;
method comparisons and the public-data extension provide supporting evidence.

Read by scientific question: [representation](#representation-and-decoding),
[updating information](docs/RESULTS_AND_CLAIMS.md#7-can-we-change-a-relation-before-knowing-which-question-will-be-asked),
[current facts and later behavior](#matched-history-sufficiency),
[public data](#public-data-extension), [causal intervention](#causal-cache-intervention),
and [alternative update procedures](#update-method-comparison).
The [short terminology lookup](docs/TERMINOLOGY.md) maps exact record names to these ideas.

Use the [locked CPU environment](REPRODUCIBILITY.md#run-the-saved-evidence-replay).
Commands below run from the checkout root and require fresh output directories.
They do not contact a model service, require a cloud account, or run inference.

## Representation and decoding

The background establishes a readable support/opposition relation and the limits
of treating a successful direct intervention as a reusable semantic update.
[Figures 1 through 6 and supplementary plots](figures/CAPTIONS.md) link to
[representation evidence](reproducibility/representation/) and
[earlier editing evidence](reproducibility/relational_editing/).

```sh
python -m repro tables --output reproduced/all-tables
python -m repro figures --output reproduced/all-figures
```

Expected: all supported historical tables and ten main plus three supplementary
figures, each in PNG and PDF. Some background findings are aggregate-only;
the [study index](REPRODUCIBILITY.md#experiment-index) identifies them separately.

## Matched-history sufficiency

[Figure 7](figures/fig07_source_history.png) presents the prospective main result.
The [independent verification package](reproducibility/state_sufficiency/independent_verification/README.md)
contains saved scores, complete inputs, semantic mappings, references and repeats.

```sh
python -m repro.state_sufficiency --output reproduced/main
python reproducibility/state_sufficiency/independent_verification/reconstruct.py analyze --output reproduced/independent
python reproducibility/state_sufficiency/independent_verification/reconstruct.py compare --output reproduced/independent
```

Expected: eight primary model/method cells reconstructed, all adjusted intervals
above zero; independent comparison matches eight cells, 896 benchmark case rows and 16,128
joint-query rows. A benchmark case is a target fact configuration
with several starting histories and questions. Joint questions combine facts.
The first command starts from case-level counts; the independent command checks
which questions meet the factual/reference requirements and still receive
different answers across histories.
Neither regenerates neural outputs or retrains editors.

## Causal cache intervention

[Figure 8](figures/fig08_cache_crossover.png) illustrates the
[six-case experiment](reproducibility/cache_crossover/README.md).

```sh
python -m repro.cache_crossover --output reproduced/causal-panel
```

Expected: 24 classifications and 288 identity controls. Under the main format,
five of six selected positive examples follow the history supplying the later-layer
activations; one is mixed. Under the alternate format, one follows that history,
two are mixed and three lack adequate original separation. Selection prevents
interpreting these counts as prevalence.

## Matched one-case extension

[Figure 9](figures/fig09_cache_crossover_extension.png) presents
[the matched extension](reproducibility/cache_crossover/README.md#one-case-extension).

```sh
python scripts/export_cache_crossover.py --output reproduced/causal-extension
```

Expected: the history supplying the later-layer activations controls the answer in both formats;
eight strict direct checks pass, with 108 baseline replays and 24 exact controls.
This is one retrospective case, separate from the six-case panel and all
prevalence denominators. Both commands normalize and classify saved raw scores.

## Public-data extension

The [Qwen/Ripple package](reproducibility/in_context_updates/README.md) retains
both frozen parsers and the same recorded responses. Its
[generated table](reproducibility/in_context_updates/public_data_table.md)
keeps the denominator and parser sensitivity visible without another figure.

```sh
python -m repro.in_context_updates --output reproduced/qwen
```

Expected: primary 3/98 qualifying history-dependent questions across three benchmark cases, alternative 2/98,
65/98 primary eligible questions (42/98 under the alternative), and accuracy on unrelated facts
170/272 without history versus 94/272 with history. Both repeated benchmark cases have
zero qualifying questions; all 86 repeated responses agree with the final pass.
Qualifying questions have different answers after different histories despite
correct current-fact and clean-reference checks.
These are in-context updates with fixed weights, not parameter editing.

## Update-method comparison

The [completed matched-history comparison](reproducibility/update_method_comparison/README.md)
and [Figure 10](figures/fig10_update_methods.png) show how update procedures change
downstream coherence, including uncertainty, seed variation and absolute rates.

```sh
python -m repro.update_methods --output reproduced/update-methods
```

Expected: 208,896 score rows and 6,144 telemetry rows reconstruct four primary
contrasts and the headline, per-case, per-scene and timing/memory CSVs. Both Qwen
improvements have intervals above zero; Gemma's intervals touch or cross zero.
No method establishes generally reliable performance on all questions. This is
supporting evidence, not the central matched-history sufficiency result.

## Reproduction levels and release boundaries

Inspecting figures, reconstructing saved measurements, regenerating neural
outputs and retraining are distinct levels. The first two are the supported
paper interfaces here. See [fresh-inference status](reproducibility/fresh_inference/README.md)
for the recovered portable entry point, exact implementation/checkpoints and
passed CPU smoke. Its optional pretrained path has not been run or certified.
The original historical notebook pins do not certify the current main result.

The author approved the separate anonymous review copy's contents. No new license
is selected or added. Existing upstream attribution and terms stay intact;
approval of the copy is not authorization to publish it.
