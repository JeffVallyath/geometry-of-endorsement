from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_REPOSITORY = "https://github.com/JeffVallyath/geometry-of-truth.git"
PUBLIC_REF = "v1.0.0"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def colab_badge(filename: str) -> str:
    target = f"https://colab.research.google.com/github/JeffVallyath/geometry-of-truth/blob/{PUBLIC_REF}/notebooks/{filename}"
    image = "https://colab.research.google.com/assets/colab-badge.svg"
    return f"[![Open in Colab]({image})]({target})"


def setup_cell(run_mode: str) -> str:
    return f"""
from pathlib import Path
import os
import subprocess
import sys

PUBLIC_REPOSITORY = {PUBLIC_REPOSITORY!r}
PUBLIC_REF = {PUBLIC_REF!r}
RUN_MODE = {run_mode!r}

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    REPO_ROOT = Path("/content/geometry-of-truth")
    if (REPO_ROOT / ".git").is_dir():
        subprocess.run(["git", "-C", str(REPO_ROOT), "fetch", "--depth", "1", "origin", "tag", PUBLIC_REF], check=True)
        subprocess.run(["git", "-C", str(REPO_ROOT), "checkout", "--detach", PUBLIC_REF], check=True)
    elif not (REPO_ROOT / "pyproject.toml").is_file():
        if REPO_ROOT.exists() and any(REPO_ROOT.iterdir()):
            raise RuntimeError("The Colab repository directory exists but is not a usable checkout")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", PUBLIC_REF, PUBLIC_REPOSITORY, str(REPO_ROOT)],
            check=True,
        )
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_ROOT)], check=True)
else:
    candidates = [Path.cwd(), *Path.cwd().parents]
    REPO_ROOT = next(path for path in candidates if (path / "pyproject.toml").is_file())

SOURCE_ROOT = str(REPO_ROOT / "src")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

print({{"mode": RUN_MODE}})
"""


def secret_cell(extra: str, call: str) -> str:
    return f"""
if RUN_MODE == "FULL":
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", f"{{REPO_ROOT}}[{extra}]"],
        check=True,
    )
    if IN_COLAB:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("Add HF_TOKEN to Colab secrets and enable notebook access")
    {call}
else:
    print("Set RUN_MODE to FULL and rerun from the first cell for raw reconstruction")
"""


def truth_notebook():
    cells = [
        markdown(f"""
# Geometry of Truth

{colab_badge("geometry_of_truth.ipynb")}

## Why this experiment exists

The larger study asks whether a language model represents the relation between a situation and a consideration, such as whether privacy supports one action but opposes another. Before interpreting that relation experiment, the activation pipeline needs to recover a known semantic distinction when surface answer symbols change. This control comes first.

A probe could appear successful by tracking the token used for an answer. The experiment breaks that shortcut by reversing the meanings of A and B during training and testing, then transferring the fitted procedure to unseen 1 and 2 answers. A stable truth direction under both changes shows that the pipeline reads factual class information from the hidden state rather than one preferred answer symbol.

The confirmatory result selects layer 14, finds a signed separation of 1.924455 training standard deviations, measures directional consensus of 0.998695 on a zero to one scale, obtains a permutation p value of 0.000999, and retains a separation of 1.840165 after the answer symbols change to 1 and 2.
"""),
        markdown("""
## Start here

From GitHub, click the Open in Colab badge above. In Colab, choose Runtime and Run all. Demo mode verifies the public result files and renders the analysis immediately. Full reconstruction downloads the model and factual datasets, extracts the activations again, and requires an accepted model license, an HF_TOKEN Colab secret, and a CUDA GPU with at least 23,000 MiB of free memory. The memory threshold is an execution requirement rather than an experimental result.
"""),
        code(setup_cell("DEMO")),
        code("""
from IPython.display import display

from geometry_of_truth.truth.contracts import load_bundle, number_lineage
from geometry_of_truth.truth.plots import layer_selection, permutation_null
from geometry_of_truth.truth.results import (
    consensus,
    design_counts,
    direction_method,
    mapping_checks,
    partition_effects,
    permutation_summary,
    prompt_examples,
    transfer,
    v1_diagnostics,
)

bundle = load_bundle(REPO_ROOT)
v1 = bundle["v1"]
v2 = bundle["v2"]
print("Artifact integrity verified")
"""),
        markdown("""
## Experimental setup

The model is Meta Llama 3.1 8B Instruct. For each prompt, the extractor records every transformer layer at the final question token before the model produces its answer. The probe therefore receives an internal state from the same point in the response process for every example.

The factual sources contribute 1,496 affirmative city records and 1,496 matched negated records to each answer scheme. The experiment forms 748 proposition groups and keeps every linked version in one split, preventing an affirmative form from training a probe that later sees its negated partner in testing. Groups stay intact.

Each answer scheme contains 2,992 prompt records. Within each scheme, 1,796 records train the directions, 600 select the layer, and 596 remain untouched for the confirmatory test. The held-out 1 and 2 scheme mirrors the A and B design, producing 5,984 cached records across both schemes.

Standard prompts map true to A and false to B. Reversed prompts map true to B and false to A. The transfer prompts replace those symbols with 1 and 2. Every semantic class therefore appears under both answer assignments.
"""),
        code("""
display(prompt_examples())
display(design_counts(bundle["split"]))
display(mapping_checks(bundle["split"], v2))
"""),
        markdown("""
## Frozen first attempt

The first design used the literal words True and False while instructing the model to reverse their meanings. AUROC measures ranking quality from zero to one, with 0.5 representing chance. Under the reversed instruction, semantic scoring reached 0.000349 while literal True minus False scoring reached 0.999651. The model followed the familiar words almost perfectly and ignored their assigned meanings almost perfectly.

That version remains a formal failure. The neutral-symbol design was specified afterward and keeps the failed result visible instead of rewriting the original rule.
"""),
        code("""
print(v1["frozen_v1_disposition"])
v1_table = v1_diagnostics(v1)
decisive_v1_rows = [
    "reversed instruction mapped macro AUROC",
    "reversed literal True minus False macro AUROC",
]
display(v1_table[v1_table["quantity"].isin(decisive_v1_rows)].reset_index(drop=True))
"""),
        markdown("""
## Activation measure

The code divides the training records into eight partitions. For each held-out partition and each layer, it subtracts the mean hidden state for false training rows from the mean for true training rows, then normalizes that vector to length one. The held-out partition never contributes to its own direction.

Each test activation is projected onto its training direction, centered with training class means, and divided by the training projection standard deviation. One partition effect is the mean true score minus the mean false score, with the two factual sources weighted equally. The signed statistic T averages the eight partition effects. T has no fixed upper bound, and a positive value means the training orientation transfers to held-out data.

Directional consensus C is the length of the average of the eight unit directions. It ranges from zero to one. A value near one means that separate training partitions recover nearly the same direction.
"""),
        code("""
display(direction_method())
display(consensus(v2))
"""),
        markdown("""
## Layer selection

The development sweep evaluates layers 8 through 24, with the horizontal axis showing the transformer layer and the vertical axis showing signed development separation for each answer mapping. A candidate layer must separate truth correctly under both standard and reversed A and B mappings. The selection score uses the weaker of those two effects, so one easy mapping cannot determine the winner. Layer 14 maximizes that rule. Test data remains untouched until this choice is fixed.
"""),
        code("""
display(layer_selection(v2).figure)
"""),
        markdown("""
## Confirmatory result

At layer 14, all eight held-out partition effects are positive and range from 1.876134 to 1.955230. Their average is T equal to 1.924455, meaning that true and false test projections differ by about 1.92 training standard deviations after equal weighting of the two factual sources.

The eight unit directions produce C equal to 0.998695 and an implied mean pairwise cosine of 0.997019. Both quantities show that the direction changes very little across training partitions.

The null keeps groups intact. The permutation test flips labels for complete proposition groups, preserving every linked affirmative and negated set. Zero of 1,000 shuffled results reach T equal to 1.924455. The prespecified add-one calculation gives 1 divided by 1,001, or p equal to 0.000999. This p value measures incompatibility with the group-preserving null, while T measures the size of the separation.
"""),
        code("""
primary_effects = partition_effects(v2)
primary_effects = primary_effects[
    (primary_effects["verbalizer"] == "A/B") & (primary_effects["mapping"] == "overall")
][["partition", "signed effect"]]
display(primary_effects.reset_index(drop=True))
permutation_table = permutation_summary(v2)
shown_fields = ["permutations", "observed T", "null values at least observed", "add one p"]
display(permutation_table[permutation_table["field"].isin(shown_fields)].reset_index(drop=True))
display(permutation_null(v2).figure)
"""),
        markdown("""
## Symbol transfer

The selected layer and fitted procedure next score prompts that use 1 and 2 instead of A and B. The table reports standard mapping, reversed mapping, and their equal-weight average for both symbol schemes. Signed T retains the training orientation. Macro AUROC reports ranking quality, where 0.5 is chance and 1.0 is perfect ranking within each factual source.

For A and B, standard mapping gives T equal to 2.033345 and reversed mapping gives 1.815513, producing the 1.924455 average. For held-out 1 and 2, the corresponding values are 1.987243 and 1.694375, producing the 1.840165 average. Macro AUROC equals 1.0 in every row. The decrease preserves strong separation while removing the original answer tokens.
"""),
        code("""
display(transfer(v2))
"""),
        markdown("""
## Where every headline number comes from

Every displayed result comes from the public result bundle loaded at the start of the notebook. Integrity verification runs before any result table. The final two columns below name the data file and exact field or calculation for each headline quantity, allowing a reader to trace the prose back to stored values.
"""),
        code("""
display(number_lineage(bundle))
"""),
        markdown("""
## Reproduction modes

DEMO verifies and presents the retained result files. ANALYSIS recomputes every statistic from a retained activation cache after checking every cache part. FULL downloads the fixed inputs, extracts activations, runs the analysis, and compares the reproduced measurements with the public reference.

Analysis mode reads the cache location from TRUTH_CACHE_ROOT. Full mode writes generated files inside the active Colab runtime.
"""),
        code("""
if RUN_MODE == "ANALYSIS":
    from geometry_of_truth.truth.reproduce import reproduce_analysis

    cache_root = os.environ.get("TRUTH_CACHE_ROOT")
    if not cache_root:
        raise RuntimeError("Set TRUTH_CACHE_ROOT to the retained activation cache")
    analysis_run = reproduce_analysis(cache_root, "/content/truth-analysis")
    display(analysis_run["comparison"])
else:
    print("Analysis reconstruction skipped")
"""),
        code(secret_cell(
            "truth-full",
            'from geometry_of_truth.truth.reproduce import reproduce_full; full_run = reproduce_full("/content/truth-reproduction"); display(full_run["checks"])',
        )),
        markdown("""
## Interpretation

The control establishes that this extraction and cross-fit procedure can recover a known semantic distinction after two answer-mapping changes. That result makes a later positive endorsement finding more credible and makes a null endorsement result harder to attribute to broken activation extraction, while the next stage asks whether ValuePrism can support the same measurement under unseen consideration identities and reciprocal checkerboards.
"""),
    ]
    return notebook(cells)

def leakage_notebook():
    cells = [
        markdown(f"""
# ValuePrism leakage controls

{colab_badge("valueprism_leakage.ipynb")}

## Why this experiment exists

The larger study asks whether a model represents how a consideration bears on an action in a particular situation, and ValuePrism supplies situations, considerations, and Supports or Opposes labels that can also reward repeated moral phrases instead of that relation. A phrase such as respect for privacy often carries a stable prior even when the relevant situation changes.

This notebook tests the dataset design before the activation experiment. The strict split places recognized consideration identities and situations on one side of the training and test boundary. The checkerboard endpoint uses the same two considerations in two situations where their labels reverse, causing any fixed preference for one consideration to cancel. These protections are complementary.

The main result of this review is methodological. Restoring held-out consideration identities raises a text-only paired-accuracy baseline by 7.294793 percentage points across five fixed draws, while restoring situations raises it by 0.526937 points. Phrase identity is therefore a material shortcut for this baseline.
"""),
        markdown("""
## Start here

From GitHub, click the Open in Colab badge above. In Colab, choose Runtime and Run all. Demo mode verifies and presents public aggregate measurements. Full reconstruction retrieves ValuePrism after the dataset license has been accepted, reads HF_TOKEN from Colab Secrets, rebuilds the public measurements on CPU, and keeps licensed row text inside the active runtime.
"""),
        code(setup_cell("DEMO")),
        code("""
from IPython.display import display

from geometry_of_truth.leakage.contracts import load_bundle, number_lineage
from geometry_of_truth.leakage.plots import stress_test
from geometry_of_truth.leakage.results import (
    audit,
    candidate_supply,
    normalization,
    protections,
    split_lineage,
    stress_draws,
    sensitivity,
    uncertainty,
)

bundle = load_bundle(REPO_ROOT)
results = bundle["results"]
print("Artifact integrity verified")
"""),
        markdown("""
## Candidate structure

A ValuePrism row pairs one situation with one consideration and labels that consideration Supports or Opposes. Removing the third Either label leaves 183,023 eligible rows. Among them, 20,032 situations contain at least one row of each valence, and 3,437 exact consideration phrases appear with both valences somewhere in the dataset.

Combining those reversals produces 13,923 possible checkerboards across 6,073 distinct consideration pairs. A checkerboard contains two situations and two considerations. The first consideration supports in one situation and opposes in the other, while the second consideration follows the opposite pattern. The 13,923 figure counts possible boards before semantic review, and the 6,073 figure counts unique pairs even when one pair supports several boards.

For scores f, the checkerboard interaction is

I = [f(s1,c1) - f(s1,c2)] + [f(s2,c2) - f(s2,c1)]

Any score formed by adding a situation term to a fixed consideration term gives I equal to zero. A nonzero reciprocal effect therefore requires sensitivity to the situation and consideration together.
"""),
        code("""
display(audit(results))
display(protections(results))
"""),
        markdown("""
## Algorithmic identity grouping

Exact string matching misses spelling and wording variants. The grouping pipeline starts with 17,678 distinct raw forms. Case, punctuation, spacing, leading articles, and light plural normalization merge 1,971 variants and leave 15,707 forms. Removing standard Value, Right, and Duty prefixes merges another 1,106 and leaves 14,601 forms.

The final automatic stage represents each phrase with character fragments of length 3 through 5 and groups forms whose similarity reaches 0.85, with at most 25 forms in one cluster. It merges 4,410 additional forms and leaves 10,191 algorithmic identity clusters. The table column named collapsed from prior reports forms merged at that stage rather than dataset rows removed.

L3 is an automatic near-duplicate grouping. Human review decides whether borderline phrases express the same consideration in meaning and stakeholder role.
"""),
        code("""
display(normalization(results))
"""),
        markdown("""
## Dataset lineage

Two branches begin from the same 183,023 binary rows and answer different questions. The audit branch keeps 164,279 rows attached to consideration identities that reverse somewhere, then keeps 110,656 rows in situations containing both labels. Those rows cover 19,068 situations and measure whether enough reciprocal structure exists for review.

The frozen split branch removes 421 identical rows from the original binary pool and leaves 182,602. It holds out 25 percent of the L3 clusters and 30 percent of situations. Their joint assignment produces 116,000 strict training rows and 7,394 strict test rows, while the remaining 59,208 rows touch the held-out side for only one of the two required identities and stay outside both strict partitions.

This branching explains why 116,000 training rows plus later categories can exceed the 110,656-row audit pool. The figures come from separate filters applied to the shared source rather than successive steps in one shrinking table.
"""),
        code("""
display(split_lineage(results))
"""),
        markdown("""
## Deliberate shortcut restoration

The stress test fits the same logistic text classifier five times using seeds 0 through 4, with word unigrams and bigrams from the situation and consideration together as its features. Within-situation paired accuracy asks whether a Supports row receives a higher score than an Opposes row from the same situation, with ties worth one half. A score of 0.5 is chance and 1.0 is perfect ordering.

Each draw first uses the strict split. One intervention restores 30 percent of held-out consideration identities to training while the other restores situations, with test membership, classifier, features, and metric fixed within each draw. In the table, strict score is the baseline accuracy and each overlap score is the accuracy after its named restoration. The difference columns multiply that subtraction by 100, so they report percentage points rather than percent change.

Restoring consideration identities raises paired accuracy by 7.294793 points on average. Across the five draws, the sample standard deviation is 1.929464 points, the standard error is 0.862883, and the Student t 95 percent interval runs from 4.899047 to 9.690539. Restoring situations raises the mean by 0.526937 points. Its sample standard deviation is 1.072724, its standard error is 0.479737, and its interval runs from negative 0.805026 to 1.858899. The consideration interval stays above zero, while the situation interval includes zero.

The plotted lines connect strict and restored scores for the same seed. This result belongs to the text-only pair_text baseline and within-situation paired accuracy. Activation probes and checkerboard interactions use separate endpoints.
"""),
        code("""
display(stress_draws(results))
display(uncertainty(results))
display(stress_test(results))
"""),
        markdown("""
## Stricter sensitivity sets

U0 is the 7,394-row strict test set. It contains 4,372 situations, 2,009 within-situation comparisons, and 378,506 cross-situation row pairs that share a consideration cluster. U1 removes the clearest remaining near-duplicate risks and retains 7,081 rows, 4,237 situations, 1,865 within-situation comparisons, and 374,160 within-consideration pairs in total.

U2 also removes every automatically ambiguous high-risk cluster. The set collapses to 587 rows in 547 situations, with 23 within-situation comparisons and 477 within-consideration pairs. The large within-consideration counts in U0 and U1 arise because every eligible row under one consideration can pair with rows from many other situations.

The U2 collapse shows where automatic exclusion loses the comparison structure needed for the relation test. Human adjudication is the next filter for borderline identity pairs.
"""),
        code("""
sensitivity_table = sensitivity(results)
display(sensitivity_table[sensitivity_table["set"].isin(["U0", "U1", "U2"])].reset_index(drop=True))
"""),
        markdown("""
## Two human audits

The leakage audit asks whether exposure to a training phrase substantially defeats the intended holdout of a test phrase because both express the same consideration across splits. Reviewers see phrase pairs without labels, activation results, probe scores, or baseline performance.

The checkerboard audit examines a different unit. Reviewers decide whether each consideration keeps the same meaning and stakeholder role across both situations, then check whether all four labels support the reciprocal pattern. Separate records preserve the distinction between train-test identity control and checkerboard construct validity.
"""),
        markdown("""
## Checkerboard supply

The ranked pool contains exactly 1,090 candidate checkerboards, and the planned endpoint needs 800 human-confirmed boards. Dividing 800 by 1,090 gives a required acceptance fraction of 0.733945, or 73.3945 percent.

The 1,090 figure is a census of the ranked pool, so sampling uncertainty does not apply to that count. Human acceptance remains unknown. Two independent reviewers first estimate it on a sample, and the resulting interval determines whether the pool can plausibly reach 800 accepted boards before the full audit proceeds.
"""),
        code("""
display(candidate_supply(results))
"""),
        markdown("""
## Where every headline number comes from

Every displayed result comes from results.json after its integrity check passes. The final two columns below identify the exact field or calculation for each headline quantity. The fixed seeds, per-draw scores, split counts, sensitivity coverage, and candidate supply all remain available in that same public aggregate.
"""),
        code("""
display(number_lineage(bundle))
"""),
        markdown("""
## Full reconstruction

Full mode installs the ValuePrism dependencies, reads HF_TOKEN from Colab Secrets, reconstructs the public aggregate on CPU, and compares all 13 headline quantities with the stored reference. This rerun is optional. Generated row-level files remain inside the active Colab runtime under the reconstruction directory.
"""),
        code(secret_cell(
            "valueprism-full",
            'from geometry_of_truth.leakage.reproduce import reproduce; full_run = reproduce("/content/valueprism-reproduction"); display(full_run["comparison"])',
        )),
        markdown("""
## Place in the larger experiment

The strict split measures generalization to unseen consideration identities and unseen situations. The reciprocal checkerboard measures whether a score changes with the relation after fixed consideration preferences cancel. Passing both controls would support a context-sensitive endorsement signal rather than phrase recognition alone.

Human review now determines the semantic quality of the retained identity boundaries and checkerboards. Its acceptance estimates set the usable sample size for the activation experiment and therefore determine whether the final relation endpoint has enough audited comparisons to justify interpretation. That audit comes next.
"""),
    ]
    return notebook(cells)

def notebook(cells):
    for cell in cells:
        material = f"{cell.cell_type}\0{cell.source}".encode("utf-8")
        cell["id"] = hashlib.sha256(material).hexdigest()[:16]
    document = nbf.v4.new_notebook(cells=cells)
    document["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    document["metadata"]["language_info"] = {"name": "python", "version": "3.10"}
    return document


def main():
    target = ROOT / "notebooks"
    target.mkdir(exist_ok=True)
    nbf.write(truth_notebook(), target / "geometry_of_truth.ipynb")
    nbf.write(leakage_notebook(), target / "valueprism_leakage.ipynb")


if __name__ == "__main__":
    main()
