# When Apparent Updates Still Depend on History

Correctly reporting an updated fact does not establish that a model uses it in
its later answers. We investigate this with matched histories: different starting
sets of facts receive the same updates, reach the same final facts, and are asked
the same downstream question. The model must report the required facts correctly
under each history. We also check that it can answer when given the final facts
from the start. Even then, its answers can depend on the earlier facts.

This repository contains the experiments behind the [paper](papers/final-paper.pdf),
from the matched-history tests to cache interventions and external-state repair.
The [slides](papers/synthetic-method-slides.pdf) introduce the synthetic setup.

## The result in one case

The [Ada/Bryn example](examples/synthetic/README.md) asks whether two people take
the same side on a proposal. After the updates, Ada supports it and Bryn opposes
it, so the answer is No. Gemma reports both facts correctly under both histories.
Yet it answers No under H5 and Yes under H6. The clean and no-op references both
answer No, and the additional checks pass.

The walkthrough gives the exact inputs, updates, answers and controls. H0-H7 are
eight histories within one synthetic case; the study contains many such cases.
The [public examples](examples/public/README.md) apply the test to MQuAKE questions,
starting with a two-hop question about Funky Drummer and then the paper's Levada
case. These use factual chains rather than synthetic stances.

## Run the saved-example check

Use Python 3.13 and an isolated environment. From this repository's root:

```sh
python -m venv ../review-env
```

Activate it with `source ../review-env/bin/activate` on Linux/macOS, or
`..\review-env\Scripts\Activate.ps1` in Windows PowerShell. Then run:

```sh
python -m pip install -r requirements-onboarding.lock
python -B -X utf8 tools/review_examples.py --all --output "../review examples"
```

Open `../review examples/REPORT.md`. It should show **160/160 synthetic checks**
and **94/384 Qwen and 101/384 Gemma history-dependent public questions**.
`views/` contains readable inputs and checks; `results.json` has the detailed results.

The command runs the scoring code on saved model outputs, without a model or GPU.
Installation needs package-index access; replay does not. Choose a new output
directory for each run. Platform details are in the [replay guide](history_updates/docs/reproduction.md).

## Follow the paper

| Paper section | Question | Results |
|---|---|---|
| **2 and 3.1** | Can answers differ after identical updates and passed checks? | [Synthetic rates](history_updates/results/tables/table_03.md); [stricter controls and fresh cases](docs/experiment_overview.md) |
| **3.2 and 3.3** | Does the effect persist on public questions and larger models? | [Public counts](history_updates/results/tables/table_01.md); [larger-model results](docs/experiment_overview.md) |
| **4** | Which cache changes transfer or repair the effect? | [Repair and harm counts](history_updates/results/tables/table_05.md); [head-intervention effects](history_updates/results/tables/cache_reader_effects.md) |
| **5.1-5.3** | What changes when the facts are stored outside the model? | [Synthetic repair](history_updates/results/tables/table_06.md); [public repair](history_updates/results/tables/table_07.md) |
| **5.4 and Appendix G** | What if the interpreter sees history or answers differently? | [History exposure](history_updates/results/tables/table_10.md); [QA, PROGRAM and MeLLo](history_updates/results/tables/table_11.md) |
| **Appendix G, Table 13** | How does the repair compare with corrected GMeLLo? | [GMeLLo results](history_updates/results/tables/gmello_corrected.md); [paired comparisons](history_updates/results/tables/gmello_contrasts.md) |

The [complete paper map](history_updates/docs/paper_map.md) connects every figure,
table and numerical passage to its sources. Use its links rather than matching
table numbers to filenames.

## Find the implementation

[Code navigation](CONTRIBUTING.md#code-navigation) links directly to case generation,
update rendering, activation hooks, scoring, training and dictionary execution.
For an agent, [AGENTS.md](AGENTS.md) gives the short file map and search commands.

The current paper's code and results are under `history_updates/`.
The examples and result tables can be read without setting up an environment.

## Working on the repository

Full-paper replay uses the separately distributed `software.zip` and `data.zip`.
Follow the [setup instructions](history_updates/docs/reproduction.md#archive-handoff-for-full-replay)
and keep that extraction outside the Git checkout. The small example above needs
neither archive. `python reproduce.py --help` explains the five full-replay routes.

New model experiments also need weights, adapters and their runtime dependencies.
The [asset inventory](HANDOFF.md#available-assets-and-gaps) identifies what is
available and what still needs to be supplied.

For framing, begin with the examples and [review questions](docs/review_and_extensions.md).
For implementation, follow the code links and the [contribution workflow](CONTRIBUTING.md).
Work on a branch, run the relevant checks, and open a pull request describing the
change. New experiments should have their own study ID and outputs so the
recorded results remain available for comparison. The [handoff](HANDOFF.md) covers
the commands and assets needed to pick up that work.

Built with Llama. [Data and model notices](DATA_AND_LICENSES.md) and
[rights and disclosure](history_updates/docs/rights.md).
