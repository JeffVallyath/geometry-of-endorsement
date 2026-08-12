from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_REPOSITORY = "https://github.com/JeffVallyath/geometry-of-endorsement.git"
PUBLIC_REF = "main"
PUBLIC_COMMIT = "cf605a169eef6cbe24ead242e0a5a39097df4f0d"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def colab_badge(filename: str) -> str:
    target = f"https://colab.research.google.com/github/JeffVallyath/geometry-of-endorsement/blob/{PUBLIC_REF}/notebooks/{filename}"
    image = "https://colab.research.google.com/assets/colab-badge.svg"
    return f"[![Open in Colab]({image})]({target})"


def setup_cell(
    run_mode: str,
    valid_modes: tuple[str, ...],
    *,
    truth_dependency_locks: bool = False,
) -> str:
    valid_mode_literal = "{" + ", ".join(repr(mode) for mode in sorted(valid_modes)) + "}"
    if truth_dependency_locks:
        dependency_setup = """
if RUN_MODE == "ANALYSIS":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_ROOT / "requirements-truth-analysis.txt")], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", str(REPO_ROOT), "--no-deps"], check=True)
elif RUN_MODE == "FULL":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_ROOT / "requirements-truth-reproduction.txt")], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", str(REPO_ROOT), "--no-deps"], check=True)
elif IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", str(REPO_ROOT)], check=True)
""".strip()
    else:
        dependency_setup = """
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", str(REPO_ROOT)], check=True)
""".strip()
    return f"""
from pathlib import Path
import os
import subprocess
import sys

PUBLIC_REPOSITORY = {PUBLIC_REPOSITORY!r}
PUBLIC_REF = {PUBLIC_REF!r}
PUBLIC_COMMIT = {PUBLIC_COMMIT!r}
RUN_MODE = {run_mode!r}
VALID_MODES = {valid_mode_literal}

if RUN_MODE not in VALID_MODES:
    raise ValueError(f"RUN_MODE must be one of {{sorted(VALID_MODES)}}, got {{RUN_MODE!r}}")

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    REPO_ROOT = Path("/content/geometry-of-endorsement")
    if (REPO_ROOT / ".git").is_dir():
        origin = subprocess.run(["git", "-C", str(REPO_ROOT), "remote", "get-url", "origin"], check=True, text=True, capture_output=True).stdout.strip()
        if origin.rstrip("/").removesuffix(".git") != PUBLIC_REPOSITORY.rstrip("/").removesuffix(".git"):
            raise RuntimeError("The existing checkout has an unexpected origin")
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], check=True, text=True, capture_output=True).stdout
        if dirty:
            raise RuntimeError("The existing checkout contains modified or untracked files")
        subprocess.run(["git", "-C", str(REPO_ROOT), "fetch", "--depth", "1", "origin", PUBLIC_COMMIT], check=True)
        subprocess.run(["git", "-C", str(REPO_ROOT), "checkout", "--detach", PUBLIC_COMMIT], check=True)
    elif not (REPO_ROOT / "pyproject.toml").is_file():
        if REPO_ROOT.exists() and any(REPO_ROOT.iterdir()):
            raise RuntimeError("The Colab repository directory exists but is not a usable checkout")
        REPO_ROOT.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(REPO_ROOT), "init"], check=True)
        subprocess.run(["git", "-C", str(REPO_ROOT), "remote", "add", "origin", PUBLIC_REPOSITORY], check=True)
        subprocess.run(["git", "-C", str(REPO_ROOT), "fetch", "--depth", "1", "origin", PUBLIC_COMMIT], check=True)
        subprocess.run(["git", "-C", str(REPO_ROOT), "checkout", "--detach", "FETCH_HEAD"], check=True)
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
    if head != PUBLIC_COMMIT:
        raise RuntimeError(f"Expected public commit {{PUBLIC_COMMIT}}, found {{head}}")
else:
    candidates = [Path.cwd(), *Path.cwd().parents]
    REPO_ROOT = next(path for path in candidates if (path / "pyproject.toml").is_file())

{dependency_setup}

OUTPUT_ROOT = Path(os.environ.get("GEOMETRY_OUTPUT_ROOT", "/content/geometry-results" if IN_COLAB else str(REPO_ROOT / "geometry-results")))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
SOURCE_ROOT = str(REPO_ROOT / "src")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

print({{"mode": RUN_MODE, "output_root": str(OUTPUT_ROOT)}})
"""


def secret_cell(extra: str, call: str) -> str:
    if extra == "truth-full":
        install = ""
    else:
        install = f"""
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"{{REPO_ROOT}}[{extra}]"], check=True)
"""
    return f"""
if RUN_MODE == "FULL":
{install.rstrip()}
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
# Geometry-of-Truth-style factual positive control

{colab_badge("geometry_of_truth.ipynb")}

## Scope and current status

The project tests whether a language model encodes how a consideration bears on an action in a particular situation. This notebook covers the factual positive control for the activation method. The later Llama moral-relation development test has now passed, while the human-audited confirmatory test and rephrasing-flip prediction remain open.

The control asks whether the extraction and probing pipeline can recover factual truth when the answer mapping changes. Training and testing both include standard and reversed A/B mappings. The held-out transfer changes both the output symbols and the surrounding user phrasing. It tests whether the signal follows the factual class across a second answer vocabulary and a modest prompt-template change.

Layer 14 gives T=1.92 with a 95 percent bootstrap interval from 1.89 to 1.96. Directional consensus is C=0.999. None of 1,000 group-preserving Monte Carlo null runs match the observed statistic, giving p=1/1001 with the prespecified add-one correction. The 1/2 transfer gives T=1.84 with a 95 percent interval from 1.81 to 1.87.

## Construct boundary

This control matches the answer-mapping and activation-extraction mechanics used in the moral experiment. Factual truth is easier and represents a different target from moral endorsement. A pass establishes instrument sensitivity, while the checkerboard experiments supply the construct-specific evidence.

## Contents

1. Experimental population and split
2. Preserved first failure
3. Eight-partition direction ensemble
4. All-layer development selection
5. Confirmatory A/B result and 1/2 transfer
6. Source lineage and reproduction modes
"""),
        markdown("""
## Start here

From GitHub, click the Open in Colab badge above. In Colab, choose Runtime and Run all. DEMO verifies the retained result files on CPU. ANALYSIS recomputes statistics from the hash-bound activation cache and is maintainer-only until that cache is published. FULL downloads the pinned model and datasets, extracts activations again, and requires an accepted model license, an HF_TOKEN Colab secret, and a CUDA GPU with at least 23,000 MiB of free memory.

Colab is the public reproduction path and enforces the repository origin, commit, and cleanliness checks. Local execution is a maintainer convenience that trusts the enclosing checkout.

| Mode | Public input | Typical resource | Output |
| --- | --- | --- | --- |
| DEMO | Aggregate result bundle | Colab CPU, minutes | Verified tables and figures |
| ANALYSIS | Retained activation cache and version-pinned analysis requirements | CPU plus cache storage | Recomputed statistics |
| FULL | Model and factual datasets | CUDA GPU, long run | New activations and statistics |
"""),
        code(setup_cell(
            "DEMO",
            ("DEMO", "ANALYSIS", "FULL"),
            truth_dependency_locks=True,
        )),
        code("""
from IPython.display import display

from geometry_of_truth.truth.contracts import load_bundle, number_lineage, retained_environment
from geometry_of_truth.truth.plots import layer_selection, permutation_null
from geometry_of_truth.truth.results import (
    consensus,
    bootstrap_intervals,
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

The model is Meta Llama 3.1 8B Instruct. For each prompt, the extractor records every transformer layer at the last nonpadding token of the rendered chat prompt, immediately before the answer candidate begins. The prompt includes the mapping instruction and generation prompt, so final question token would be too narrow.

The factual sources contribute 1,496 affirmative city records and 1,496 matched negated records to each answer scheme. The experiment forms 748 proposition groups and keeps every linked version in one split, preventing an affirmative form from training a probe that later sees its negated partner in testing. Groups stay intact.

Each answer scheme contains 2,992 prompt records. Within each scheme, 1,796 records train the directions, 600 select the layer, and 596 remain untouched for the confirmatory test. The held-out 1 and 2 scheme uses the same factual records and grouped split with a second user template, producing 5,984 cached records across both schemes.

Standard prompts map true to A and false to B. Reversed prompts map true to B and false to A. The transfer prompts use 1 and 2 and also replace the question wording. Every semantic class therefore appears under both answer assignments. This condition tests joint transfer across answer vocabulary and prompt template, rather than isolating answer symbols alone.
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

The code divides the training groups into eight disjoint subsets. At each layer, it independently fits one direction on the rows inside each subset by subtracting the mean false activation from the mean true activation and normalizing the result. All eight fitted directions then score the same held-out development or test examples.

Each held-out activation is projected onto a subset direction, centered using that subset's training class means, and divided by that subset's training projection standard deviation. One subset effect is the mean true score minus the mean false score, with the two factual sources weighted equally. T averages the eight held-out effects. T has no fixed upper bound, and a positive value means the training orientation transfers to held-out data.

Directional consensus C is the length of the average of the eight unit directions. It ranges from zero to one. A value near one means that disjoint training subsets recover nearly the same direction. The notebook calls this an eight-partition signed separation rather than conventional cross-fitting because each direction uses its own partition, not the other seven.
"""),
        code("""
display(direction_method())
display(consensus(v2))
"""),
        markdown("""
## Layer selection

The development sweep evaluates all 32 transformer layers. The horizontal axis shows layer index and the vertical axis shows signed development separation for each answer mapping. A candidate layer must separate truth correctly under both standard and reversed A/B mappings. The selection score uses the weaker effect, so one easy mapping cannot determine the winner. Layer 14 maximizes that rule. Test data remains untouched until this choice is fixed.
"""),
        code("""
display(layer_selection(v2).figure)
"""),
        markdown("""
## Confirmatory result

At layer 14, all eight direction-specific effects on the held-out test set are positive and range from 1.876 to 1.955. Their average is T=1.924, meaning that true and false test projections differ by about 1.92 training-standardized projection units after equal weighting of the two factual sources. A 2,000-group bootstrap gives a 95 percent interval from 1.893 to 1.956.

The eight unit directions produce C=0.9987 and an implied mean pairwise cosine of 0.9970. Both quantities show that the direction changes very little across training subsets.

The null keeps groups intact and flips labels for complete proposition groups, preserving every linked affirmative and negated set. Every null run repeats the full 32-layer development selection before evaluating the test statistic. None of 1,000 Monte Carlo runs reach T=1.924. The prespecified add-one calculation gives p=1/1001. This p value measures incompatibility with the group-preserving null, while T measures separation size.
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
display(bootstrap_intervals(v2))
"""),
        markdown("""
## Answer-vocabulary and prompt-template transfer

The selected layer and fitted procedure next score prompts that use 1 and 2 instead of A and B and phrase the factual question differently. The table reports standard mapping, reversed mapping, and their equal-weight average for both prompt schemes. Signed T retains the training orientation. Macro AUROC reports ranking quality, where 0.5 is chance and 1.0 is perfect ranking within each factual source.

For A/B, standard mapping gives T=2.03 and reversed mapping gives T=1.82, producing the T=1.92 average. For held-out 1/2 with the second prompt template, the corresponding values are T=1.99 and T=1.69, producing T=1.84. Its 2,000-group bootstrap interval runs from 1.807 to 1.874. Macro AUROC equals 1.0 in every row. The result establishes transfer across the combined answer-vocabulary and prompt-template change. A pure verbalizer control with identical user wording remains a separate experiment.
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
## Inherited method and project modifications

The experiment inherits the factual truth-geometry motivation, cities data family, and mean-difference direction from Marks and colleagues, *The Geometry of Truth* [arXiv 2310.06824](https://arxiv.org/abs/2310.06824) and its [source repository](https://github.com/saprmarks/geometry-of-truth).

| Component | This implementation |
| --- | --- |
| Answer surface | Neutral A/B mappings with a held-out 1/2 transfer |
| Grouping | Affirmative, negated, mapping-linked forms remain together |
| Estimator | Eight directions fitted on eight disjoint training subsets |
| Selection | All 32 layers selected on development data |
| Null | Group-sign Monte Carlo null repeats layer selection |
| Confirmation | Held-out test plus 2,000-group bootstrap intervals |
"""),
        markdown("""
## Reproduction modes

DEMO verifies and presents the retained result files. ANALYSIS recomputes every statistic from a retained activation cache after checking every cache part. The cache is not publicly distributed yet, so this middle tier remains maintainer-only. FULL requests the model at the exact frozen revision, extracts activations, runs the analysis, and compares the reproduced measurements with the public reference.

ANALYSIS installs `requirements-truth-analysis.txt` before importing the recorded numerical stack, then reads the cache location from TRUTH_CACHE_ROOT. FULL installs the separate GPU reproduction requirements at the same point. Platform and BLAS differences remain visible because the result comparison fails closed on any changed value. This is an exact replay contract for the retained environment. Independent cross-platform numerical equivalence remains a separate contract requiring prespecified tolerances. All generated files use OUTPUT_ROOT. Set GEOMETRY_OUTPUT_ROOT to a mounted Google Drive directory before the setup cell when a run must survive a Colab reset. The following table records the software versions used for the retained run.
"""),
        code("""
display(retained_environment(bundle))
"""),
        code("""
if RUN_MODE == "ANALYSIS":
    from geometry_of_truth.truth.reproduce import reproduce_analysis

    cache_root = os.environ.get("TRUTH_CACHE_ROOT")
    if not cache_root:
        raise RuntimeError("Set TRUTH_CACHE_ROOT to the retained activation cache")
    analysis_run = reproduce_analysis(cache_root, OUTPUT_ROOT / "truth-analysis")
    display(analysis_run["comparison"])
else:
    print("Analysis reconstruction skipped")
"""),
        code(secret_cell(
            "truth-full",
            'from geometry_of_truth.truth.reproduce import reproduce_full; full_run = reproduce_full(OUTPUT_ROOT / "truth-reproduction"); display(full_run["checks"])',
        )),
        markdown("""
## Interpretation

The control establishes that the extraction and eight-partition direction ensemble can recover a known semantic distinction across two answer vocabularies. The completed moral-relation development result therefore rests on a pipeline that already passed a factual positive control. The open tests now ask whether the relation survives human-audited checkerboards and whether the original geometry predicts rephrasing-induced answer flips beyond text and native confidence.
"""),
    ]
    return notebook(cells)

def leakage_notebook():
    cells = [
        markdown(f"""
# ValuePrism leakage controls

{colab_badge("valueprism_leakage.ipynb")}

## Scope and current status

The project tests whether a model encodes how a consideration bears on an action in a particular situation. ValuePrism supplies situations, considerations, and Supports or Opposes labels, but repeated or nearly repeated moral phrases can create a shortcut. This notebook covers the split and checkerboard controls for that problem.

The Llama moral-relation development test has now passed. The human-audited confirmatory test remains sealed, and rephrasing-flip prediction has not yet run. These leakage controls determine what the development result can support and what the confirmatory stage must still establish.

Across five strict-style split draws, a 30 percent controlled injection of held-out consideration exposure raises text-only within-situation paired accuracy by 7.29 percentage points. The 95 percent Student-t interval for the mean intervention effect across those five draws runs from 4.90 to 9.69 points. Restoring situation exposure raises the mean by 0.53 points.

## Contents

1. Candidate checkerboards and reciprocal endpoint
2. Algorithmic identity grouping and split lineage
3. Cross-split overlap audit
4. Controlled shortcut restoration
5. Nested sensitivity sets and human audits
6. Exact reconstruction checks
"""),
        markdown("""
## Start here

From GitHub, click the Open in Colab badge above. In Colab, choose Runtime and Run all. DEMO verifies and presents the public aggregate measurements on CPU. FULL retrieves ValuePrism after the dataset license has been accepted, reads HF_TOKEN from Colab Secrets, rebuilds the measurements on CPU, and keeps licensed row text inside the active runtime.

Colab is the public reproduction path and enforces the repository origin, commit, and cleanliness checks. Local execution is a maintainer convenience that trusts the enclosing checkout.

| Mode | Public input | Typical resource | Output |
| --- | --- | --- | --- |
| DEMO | Aggregate result bundle | Colab CPU, minutes | Verified tables and figures |
| FULL | Licensed ValuePrism source | Colab CPU, long run | Rebuilt manifests and comparisons |

Set GEOMETRY_OUTPUT_ROOT to a mounted Google Drive directory before the setup cell when a long reconstruction must survive a Colab reset.
"""),
        code(setup_cell("DEMO", ("DEMO", "FULL"))),
        code("""
from IPython.display import display

from geometry_of_truth.leakage.contracts import load_bundle, number_lineage
from geometry_of_truth.leakage.plots import stress_test
from geometry_of_truth.leakage.results import (
    audit,
    candidate_supply,
    normalization,
    overlap_checks,
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

A synthetic example makes the reciprocal structure concrete.

| Synthetic unit | Academic prize | Emergency housing |
| --- | --- | --- |
| Action | Award the highest-scoring applicant | Prioritize the lowest-income applicant |
| Rewarding demonstrated merit | Supports | Opposes |
| Prioritizing urgent need | Opposes | Supports |

The labels reverse because the action and allocation purpose change, while each consideration keeps the same meaning and stakeholder role. Human review applies that same requirement to real candidates.

For scores f, the checkerboard interaction is

I = [f(s1,c1) - f(s1,c2)] + [f(s2,c2) - f(s2,c1)]

Any score formed by adding a situation term to a fixed consideration term gives I equal to zero. A nonzero reciprocal effect therefore requires sensitivity to the situation and consideration together. Nonlinear text interactions can still produce a signal, and the statistic alone cannot establish moral understanding. Text baselines and human semantic review remain necessary.
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
## Cross-split overlap audit

The strict manifests have zero row, exact-consideration, L2 prefix-stripped, L3 cluster, situation overlap. They contain two L1-normalized collisions. L1 collapses case, punctuation, spacing, articles, and light plural variants, so those two cases require a visible disposition even though the L3 cluster count is zero.

Both L1 collisions are conservatively removed from U1. The public aggregate reports their count and disposition without redistributing licensed phrase text. The accurate split claim is therefore zero L3-cluster and situation overlap, plus two disclosed L1 collisions removed from the stricter sensitivity set.
"""),
        code("""
display(overlap_checks(results))
"""),
        markdown("""
## Deliberate shortcut restoration

The stress test fits the same logistic text classifier five times using seeds 0 through 4, with word unigrams and bigrams from the situation and consideration together as its features. Within-situation paired accuracy asks whether a Supports row receives a higher score than an Opposes row from the same situation, with ties worth one half. A score of 0.5 is chance and 1.0 is perfect ordering.

Each draw uses a strict-style double holdout with 25 percent of consideration clusters and 35 percent of situations held out. The final frozen manifest uses the same cluster fraction and a 30 percent situation holdout. Within each stress draw, an intervention replaces 30 percent of the capped training sample with rows exposing held-out consideration identities or held-out situations. Training size, test membership, classifier, features, and metric remain fixed.

The 30 percent consideration injection raises paired accuracy by 7.29 points on average. The sample standard deviation is 1.93 points, the standard error is 0.86, and the 95 percent Student-t interval for the mean intervention effect across these five seeds runs from 4.90 to 9.69. The situation injection raises the mean by 0.53 points, with an interval from -0.81 to 1.86. This interval covers variation across five chosen split draws. Residual duplicate exposure, label noise, human-audit uncertainty, other models, and untested ValuePrism splits remain outside its scope.

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

U2 also removes every automatically ambiguous high-risk cluster. The set collapses to 587 rows in 547 situations, with 23 within-situation comparisons and 477 within-consideration pairs. U3 is currently identical to U2 because zero human adjudications were available to construct a distinct human-confirmed layer. The terminal disposition is ULTRACLEAN_INCONCLUSIVE.

The U2 collapse shows where automatic exclusion loses the comparison structure needed for the relation test. The large within-consideration counts in U0 and U1 arise because every eligible row under one consideration can pair with rows from many situations. Human adjudication is the next filter for borderline identity pairs.
"""),
        code("""
sensitivity_table = sensitivity(results)
display(sensitivity_table.reset_index(drop=True))
"""),
        markdown("""
## Two human audits

The leakage audit asks whether exposure to a training phrase substantially defeats the intended holdout of a test phrase because both express the same consideration across splits. Reviewers see phrase pairs without labels, activation results, probe scores, or baseline performance.

The checkerboard audit examines a different unit. Reviewers decide whether each consideration keeps the same meaning and stakeholder role across both situations, then check whether all four labels support the reciprocal pattern. Separate records preserve the distinction between train-test identity control and checkerboard construct validity.
"""),
        markdown("""
## Checkerboard supply

The ranked pool contains exactly 1,090 candidate checkerboards, and the planned endpoint needs 800 human-confirmed boards. Dividing 800 by 1,090 gives a required acceptance fraction of 0.733945, or 73.3945 percent.

The 1,090 figure is an exact census of the ranked pool. Sampling uncertainty enters through the unknown human acceptance rate. Two independent reviewers first estimate that rate on a sample, and the resulting interval determines whether the pool can plausibly reach 800 accepted boards before the full audit proceeds.
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

FULL installs the ValuePrism dependencies, reads HF_TOKEN from Colab Secrets, and reconstructs the public aggregate on CPU. The visible comparison checks aggregate counts, all five seed-specific strict scores and intervention effects, every overlap count including L1, strict and common-training row hashes, frozen input file hashes, confirmatory row membership across all four board cells, and the U1 exclusion hash. The normalization code and counts are reproduced. The exact frozen SHA-256 for `manifest_confirmatory.csv` requires the same board grouping, rank, primary or reserve assignment, and review order. The separate row-membership hash verifies the underlying unique row set after sorting. A public hash for the complete form-to-cluster mapping remains unavailable. ValuePrism dependency ranges also remain broader than the version-pinned Truth requirements, so a fresh FULL validation matrix is still required before claiming environment-complete reproduction.
"""),
        code(secret_cell(
            "valueprism-full",
            'from geometry_of_truth.leakage.reproduce import reproduce; full_run = reproduce(OUTPUT_ROOT / "valueprism-reproduction"); display(full_run["comparison"])',
        )),
        markdown("""
## Place in the larger experiment

The strict split measures generalization to unseen consideration identities and unseen situations. The reciprocal checkerboard measures whether a score changes with the relation after fixed consideration preferences cancel. The completed Llama development result passed its relation controls against situation-only, consideration-only, additive, and matched-text alternatives.

Human review now determines whether that development signal survives a semantically audited confirmatory set. A separate rephrasing stage will test whether the original activation geometry predicts answer flips beyond matched text and native confidence. Those open stages set the current claim boundary.
"""),
    ]
    return notebook(cells)

def project_status_notebook():
    cells = [
        markdown(f"""
# Project status

{colab_badge("00_project_status.ipynb")}

## Research question

The project asks whether a language model encodes the relation between a situation and a moral consideration, and whether that internal relation predicts how the model's answer changes under meaning-preserving rephrasing. The first claim concerns relation geometry. The second concerns predictive value beyond text and the model's own answer margin.

Three pieces are complete. The ValuePrism split and checkerboard controls define an evaluation that resists recognized consideration shortcuts. A factual positive control shows that the activation method recovers truth across answer mappings. A Llama 3.1 8B Instruct development test then finds a moral-relation signal at layer 19.

Two pieces remain open. Human reviewers have not completed the semantic calibration needed for the confirmatory checkerboards, and the sealed confirmatory model result remains unopened. The rephrasing-flip experiment has not run. Current claims stop at development evidence.

## Contents

1. Stage table
2. Moral-relation development design
3. Main interaction and control results
4. Native-confidence comparison
5. Claim boundary and source lineage
"""),
        markdown("""
## Start here

Click the Open in Colab badge, then choose Runtime and Run all. This notebook verifies and presents public aggregate artifacts on CPU in minutes. It downloads neither Llama nor licensed ValuePrism rows.
"""),
        code(setup_cell("DEMO", ("DEMO",))),
        code("""
from IPython.display import display

from geometry_of_truth.project.contracts import load_status
from geometry_of_truth.project.results import (
    development_design,
    development_intervals,
    development_results,
    source_lineage,
    stage_table,
)

bundle = load_status(REPO_ROOT)
status = bundle["status"]
print({"artifact_integrity": "verified", "as_of": status["as_of"]})
"""),
        markdown("""
## Full experimental arc

Each stage protects a different inference. The split limits memorization across training and testing. Checkerboards cancel fixed additive situation and consideration preferences. The factual control checks the extraction and probing procedure against a known semantic distinction. The moral development slice tests the target relation. Human audit and rephrasing then determine whether the result survives semantic review and predicts behavior under new wording.
"""),
        code("""
display(stage_table(status))
"""),
        markdown("""
## Moral-relation development design

The model sees a situation and consideration together. A difference-in-means direction and a logistic activation probe score whether the consideration Supports or Opposes the action. The primary board statistic I_b sums the two within-situation differences on a reciprocal checkerboard. Fixed additive situation-only or consideration-only scores give I_b=0.

The frozen development slice uses 1,500 training rows, 300 selection rows across 75 boards, and 500 evaluation rows across 125 boards. Development selection chooses layer 19. The evaluation rows determine the measurements below, so this is a completed development result rather than the human-audited confirmation.
"""),
        code("""
display(development_design(status))
display(development_results(status))
"""),
        markdown("""
## Interaction result and controls

The difference-in-means direction gives I_b=1.65, while the logistic activation probe gives I_b=2.08. The frozen SBERT matched-text baseline gives I_b=0.28. Situation-only and consideration-only controls give I_b=0, and the separate-encoding additive control is numerically zero.

For both activation methods, none of 200 group-preserving permutation values match the observed improvement over SBERT. The add-one value is p=1/201. The 95 percent interval for the difference-in-means advantage over SBERT runs from 1.09 to 1.64. The logistic advantage runs from 1.50 to 2.10.
"""),
        code("""
display(development_intervals(status))
"""),
        markdown("""
## Native-confidence comparison

AUROC measures relation decoding on individual development rows. Native answer margin reaches 0.721. The difference-in-means activation direction reaches 0.732, and the logistic activation probe reaches 0.780. The simple direction only slightly exceeds native confidence, while the learned activation probe has a larger gap.

These AUROCs do not measure rephrasing-flip prediction. That experiment will ask whether the original hidden state predicts later answer changes after controlling for text and native confidence. Treating the current 0.780 result as a flip-prediction result would cross the project claim boundary.
"""),
        markdown("""
## Provenance and next decision point

The tables come from a hash-verified public status artifact. The source hashes below identify the retained development result and its independent audit archive without exposing local paths or row text.
"""),
        code("""
display(source_lineage(status))
print(status["claim_boundary"])
"""),
        markdown("""
The next material decision follows the blind human calibration. Its result determines whether the checkerboard pool can support the confirmatory endpoint and whether the sealed relation test can be opened under the frozen protocol. Rephrasing remains a separate experiment even if confirmation succeeds.
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
    nbf.write(project_status_notebook(), target / "00_project_status.ipynb")
    nbf.write(truth_notebook(), target / "geometry_of_truth.ipynb")
    nbf.write(leakage_notebook(), target / "valueprism_leakage.ipynb")


if __name__ == "__main__":
    main()
