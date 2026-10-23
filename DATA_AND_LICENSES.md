# Data and licenses

This anonymous review copy contains scientific code, saved measurements, authorized
benchmark-derived inputs, synthetic examples and small fitted probes/editors.
It contains no pretrained backbone weights, tokenizer caches, private activation
banks, full key/value caches, gated ValuePrism text or completed human judgments.

No project-wide software license is specified. The review-copy permission does
not grant broader redistribution rights or replace upstream terms. The upstream
licenses below apply to their respective materials, not to the entire project.

## Models and fitted parameters

Obtain any separately needed backbone directly from its publisher, using the
recorded immutable revision and accepting its terms. The default CPU commands
need none of those downloads.

| Family | Preserved terms | Where revisions are recorded |
|---|---|---|
| Llama 3.1 | [Community License](third_party/licenses/Llama-3.1.txt) and its incorporated acceptable-use policy | [Model specifications](src/relation_study/m1/model_specs.py), [breadth roster](reproducibility/text_scope/saved/runtime/MODEL_ROSTER_FROZEN.json) |
| Gemma 2 | [Gemma Terms of Use](third_party/licenses/Gemma-terms.txt) and [prohibited-use policy](https://ai.google.dev/gemma/prohibited_use_policy) | [Editor settings](reproducibility/update_method_comparison/inputs/models.json), [figure data](artifacts/figures/figure_data.json) |
| Qwen3 and Qwen2.5 | [Supplied Apache 2.0 notice](third_party/licenses/Apache-2.0.txt) | Study configurations and [public-data methods](docs/appendix/A11_ripple_accounting.md) |

Built with Llama. Llama 3.1 is licensed under the Llama 3.1 Community License,
Copyright © Meta Platforms, Inc. All Rights Reserved.

Gemma is provided under and subject to the Gemma Terms of Use found at
ai.google.dev/gemma/terms. Gemma-dependent parameters remain subject to those
terms and their incorporated use restrictions. Those restrictions apply to use
and further distribution; this repository does not waive them. A separate
[Notice.txt](third_party/Notice.txt) preserves the required notices.

Small learned probes and editors are separately fitted research parameters.
Their public manifests identify the underlying models and actual parameter-file
hashes. Container cleanup removes custody metadata without changing retained
numerical arrays. [Probe notes](artifacts/probe_weights/README.md) and
[editor parameter notes](docs/appendix/A05_learned_editors.md) describe those
modifications. Public source manifests identify portability/privacy adaptations.

## Dataset boundaries

ValuePrism is distributed by AllenAI under the AI2 ImpACT License for Medium Risk
Artifacts. Acquire licensed rows from `allenai/ValuePrism`, configuration
`valence`, revision `d439ca90825e5b4e5ef97798d9b5950e16ba7065`, through the
publisher's access procedure after accepting its terms. This release retains
aggregate measurements, construction code and the recovered Gemma activation-probe
membership identifiers. It withholds protected row text, reference labels and
cell-role labels. Downloading the dataset does not recover the still-missing
MiniLM fitted-row list or establish the current Llama comparator's provenance.

The factual control names immutable `cities` and `neg_cities` sources and integrity
digests in [its configuration](configs/truth_control_v2.yaml). The retained
split identifiers and aggregate nulls support the documented CPU control checks.

RippleEdits is by Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson and Mor Geva,
*Evaluating the Ripple Effects of Knowledge Editing in Language Models*, TACL
2024. Its [upstream repository](https://github.com/edenbiran/RippleEdits) supplies
the preserved [MIT license](third_party/licenses/RippleEdits-MIT.txt).
The retained prompts, aliases, counterfactual records and generated responses
support the selected offline analysis. Benchmark assignments are experimental
targets, not assertions about real people.


MQuAKE-CF-3k-v2 supplies the source subjects and original question phrasings for
the saved-response MQMH evaluation. The study's prompts are explicit-record,
in-context counterfactual evaluations; benchmark stipulations are not real-world
biographical claims. The upstream MQuAKE repository is distributed under the
preserved [MIT license](third_party/licenses/MQuAKE-MIT.txt). Source dataset
revision and hashes are recorded in `reproducibility/mqmh/SOURCE_HASHES.json`.

The locality diagnostic uses eight prepared CounterFact items. Obtain the full
benchmark from the [original ROME project](https://github.com/kmeng01/rome).
The supplied compact evidence does not establish an immutable upstream dataset
checkout for these derivatives. Unknown provenance remains unknown; model
licensing does not establish broader rights in benchmark content.

Synthetic meeting records, explicit relation contexts and saved evaluation scores
remain with the studies that use them. Human-audit documentation describes a
procedure, not completed protected judgments or a passed confirmatory outcome.
The [reproduction table](REPRODUCIBILITY.md#reproduction-levels) distinguishes
available measurements from unavailable row-level or neural reconstruction.
