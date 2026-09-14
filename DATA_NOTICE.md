# Data notice

The demo package contains aggregate ValuePrism measurements. Full reconstruction retrieves the licensed rows directly from AllenAI after the dataset terms have been accepted. The exact dataset version and integrity digests remain in the machine-readable configuration.

ValuePrism uses the AI2 ImpACT License for Medium Risk Artifacts. Full reconstruction requires Hugging Face access linked to that acceptance. Generated row-level files remain inside the active reconstruction runtime and stay outside the public repository.

The factual control retrieves the public cities and negated-cities sources named in the Truth configuration. Retrieval verifies both sources before activation extraction begins.

## Models and learned parameters

No pretrained model weights, tokenizers, activation banks or full key/value caches
are distributed. Obtain backbones from their original model repositories, using
the immutable revisions in the experiment configurations and accepting their terms.

| Model family | Upstream terms |
|---|---|
| Llama 3.1 | [Llama 3.1 Community License](reproducibility/licenses/Llama-3.1.txt) and [acceptable-use policy](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/USE_POLICY.md) |
| Gemma 2 | [Gemma Terms of Use](reproducibility/licenses/Gemma-terms.txt) and [prohibited-use policy](https://ai.google.dev/gemma/prohibited_use_policy) |
| Qwen3-8B | [Apache License 2.0](reproducibility/licenses/Apache-2.0.txt), as declared by the [model publisher](https://huggingface.co/Qwen/Qwen3-8B) |

Built with Llama. Llama 3.1 is licensed under the Llama 3.1 Community License,
Copyright © Meta Platforms, Inc. All Rights Reserved.

Gemma is provided under and subject to the Gemma Terms of Use found at
ai.google.dev/gemma/terms. Gemma-dependent learned parameters are subject to
those terms and their incorporated use restrictions. These restrictions remain
applicable to use and further distribution; this repository does not waive them.

Small learned probes and editors are separately fitted research parameters, not
copies of the pretrained backbones. Their manifests identify the underlying
model, scientific configuration and parameter hashes. They do not confer access
to gated datasets or models. Portability/privacy modifications to copied source
are identified in the public source manifests.

## Data and review boundary

The public tree retains saved numeric measurements, anonymous split identifiers,
synthetic relation-construction code and empty review instruments. The instruments
describe how review is conducted; they are not completed judgments or evidence of
a confirmatory outcome. Individual completed judgments are not distributed.

Third-party corpus text is not automatically redistributable because measurements
on it are public. Use the original sources and their terms for corpus acquisition.
Model/data availability, aggregate replay and full experiment reproduction are
distinct, as documented in [Reproducibility](REPRODUCIBILITY.md).

No project-wide software license is specified. Reuse rights beyond the explicitly
included upstream licenses require clarification from the project authors before
redistribution of project code or learned parameters.
