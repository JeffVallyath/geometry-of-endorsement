# CPU audit: same-instance certification × SHS

## Result in brief

**GO for a small, prospectively frozen follow-up.** In this post-hoc saved-output
audit, same-pair logical-consistency violations (G2a+M) survive at 188 coded
opportunities across 53 shared designs; cross-pair propagation-qualified
violations (G2b+M) survive at 82 opportunities across 34 designs. Both gates
hold at 64 opportunities across 29 designs. Each category occurs in all eight
model/update groups, including learned updates. The respective same-matched-pair
strong counts are 29, 14 and 11, across 17, 9 and 7 designs. This weakens
"the updated relations failed to propagate into compositional reasoning at all"
while leaving target-specific propagation failure and context interference viable.
It does not establish identical latent states or passage of all conventional
editing benchmarks. All union counts pool a larger search across conditions and
are not new prospective prevalence estimates.

## Scope, sources and reconstruction

Only existing V10 scores were analyzed. No inference, tokenizer/model execution,
GPU, model download, cloud allocation, paid API, retraining, editor change or
paper change occurred. The original V10 primary result is unchanged.

Sources are the supplied snapshot
`[local path omitted]`
(V10 sections of `docs/RESULTS_AND_CLAIMS.md`, state-sufficiency README,
independent-verification README/protocol/reconstruction and committed outputs),
plus the two original confirmation archives in `D:/Downloads`:
`STATE_SUFFICIENCY_CONFIRMATION_RESULTS_AND_REPRODUCTION.zip` and
`STATE_SUFFICIENCY_CONFIRMATION_RAW_DATA_01.zip`.
Their SHA-256 identities are respectively
`96cdb665083c1c41e7cea49868efbb44207bca89f01c9bfa08779488f93ec455` and
`fd5718998929dd09465b5f6eb1b4933c519f6731458f6dec99093eadf1c73766`.
All 1,383 declared original payloads and 418 frozen scientific files verified.
There is no `PAPER_GUIDE.md` in the supplied snapshot.

The unchanged independent reconstruction is at
`source/upstream/reconstruct.py` in this audit's local working folder. It
reconstructs gold from `inputs/designs/*.json` fields `root.commands`,
`root.sources[].values`, `root.terminal.values`, and `questions[].spec/labels/text`.
Scores come from `inputs/responses/{model}/{root}/core.json.gz`: `physical[].logps`,
`tokens`, validity cross-checks, `bindings` and `aliases`. Exact prefix lengths
are `len(contexts[context].prefix_ids)`. `expected/` is accessed only after the
original independent calculation, never to construct new certification gates.

**V10 reproduction PASS:** 16,128/16,128 query classifications/hard answers
(including SOURCE), 896/896 root-condition rows, and 8/8 primary cells agree.
All classification/count fields agree exactly. Recomputed probabilities differ
in a few low bits (maximum 2.22e-16), within the original 1e-12 numerical
tolerance. The eight rates and original root-bootstrap intervals match. The
64 designs, eight histories, three addressed relations, nine semantic questions,
two code draws and 18 fixed coded opportunities per root are independently
validated. Coverage is 230,656 physical / 248,064 logical responses and 17,408
aliases; 8,448 saved repeat rows and 12,672 repeat comparisons have zero score
difference. Textual conditions lack saved independent repeat runs.

Original scientific source commit: `d549f9222e70626db8282e338e04f1023435badc`.
Backbones: `google/gemma-2-9b-it` revision
`11c9b309abf73637e4b6f9a3fa1e92e615547819`, and `Qwen/Qwen3-8B` revision
`b968826d9c46dd6066d109eabc6255188de91218`. INV denotes `INV_PAIR_NLL`, FREE
denotes `FREE_PAIR_CONSISTENCY`; both retain seeds 0 and 1. Correction and Latest
denote `EXISTING_CORRECTION` and `LATEST_SAME_WORDING`. Original learned rank 16,
sites 15/13 and the saved model/method/seed weight identities are unchanged.

Existing nuisance and answer-code results were **reconstructed, not rediscovered**.
`V10_CPU_FOLLOWUP_RESULTS.zip` (SHA-256
`c04f228f6e9475e3d775dfd055d3bea28e47a1fcb7e1212a5140b724427454f2`) passed
its existing verifier: 670 payloads, including 529 unchanged original payloads.
The actual archived `robustness.py` functions `ladder` and `paired_codes` are
reused verbatim. Every original flag and count/length/jointly-matched/strong pair
list agrees for **13,824/13,824** updating opportunities; every common-pair,
same-direction cross-code list agrees for **6,912/6,912** semantic opportunities.
G1 also agrees exactly with existing `all_direct_eligible/all_direct_witness`.

## Frozen gates and anti-leakage boundary

This report follows the **revised** user attachment
`9ec9e44c-fc83-479c-8f84-c37b5fd25d16/pasted-text-1.txt`.
An earlier requested audit had already exposed both-code same-pair and global
battery results before the live goal replacement was noticed. Thus this is not
a claim of fresh outcome blinding. The revised G2a/G2b mapping follows the new
user specification and Boolean structure exactly; it is independent of survivor
IDs, margins, models, methods and seeds. No thresholds or mappings were tuned.

`GATE_SPEC.json` was frozen at 2026-09-18 04:48:38Z with SHA-256
`8ad59d7d31952f5cbdc9bd2c6eb62a0f9b87c886c306cc814302e4f82b23c765`.
All 13,824 certification rows were constructed and sealed before the revised
outcome join. A per-target guard allows required native/no-op reference answers
but rejects either target-code rendering in other histories. Gate inputs are
direct and non-target correctness booleans, not target differences or W/D bits.
The serious-review/freeze workflow produced this explicit separation and the
independent counterfactual leakage regression.

- **G0 witness:** unchanged original A AND C AND D. `G0_pass` separately means
  A AND C, the pre-disagreement eligibility requirement.
- **G1:** exactly the existing all-measured-direct sensitivity: all twelve
  measured direct renderings correct/valid in every history, plus the target's
  native-final/no-op references. The set includes addressed reads and retention
  controls; no new addressed-only variant was substituted.
- **G2a:** G1 plus the other two SAME/BOTH/EITHER functions on the target pair,
  correct/valid in every history **under the target draw**. Any two functions
  determine the third; this is intentionally a logical-consistency test.
- **G2b:** G1 plus all three functions on each of the other two addressed
  actor pairs, correct/valid in every history under the target draw: six semantic
  checks. Neither target pair nor its alternate code is accidentally included.
- **Both:** G2a AND G2b; this is reported separately rather than merging the
  distinct objections addressed by the two gates.

The machine-readable specification contains every actor-pair/function/draw
mapping. Cross-code certification requires the corresponding gate under both
draws, including both native/no-op references. Cross-code witnesses require the
**same matched pair and same semantic direction** under both codes. They use
nine semantic opportunities per root, not an 18-opportunity doubled denominator.

## Matching ceilings and the existing G1 baseline

Source-to-final changed-fact counts are derived as
`sum(source.values[a][p] != command.value)` over the three addressed facts and
verify as `[0,1,1,2,1,2,2,3]` in every design. Edit count alone leaves six of 28
pairs. Adding exact saved prefix-token length leaves 2 or 6 pairs: respectively
10,368 and 3,456 coded opportunities. **Pair availability is 1152/1152 in every
individual condition** (2304/2304 in a two-seed group), so no target lacks an
available matched comparison. The response-dependent disagreement ceiling is
smaller and must not be confused with availability.

All rows below have fixed N=1152. G0+M is the disagreement-retention ceiling
before stronger certification. G1 is existing V10 robustness, not a new finding.

| Model / condition | G0 W | G0+M ceiling | G1 pass | G1 W | G1+M |
|---|---|---|---|---|---|
| gemma / Correction | 69 | 44 | 628 | 43 | 27 |
| gemma / FREE_s0 | 66 | 34 | 794 | 58 | 27 |
| gemma / FREE_s1 | 52 | 35 | 778 | 45 | 30 |
| gemma / INV_s0 | 30 | 16 | 904 | 25 | 13 |
| gemma / INV_s1 | 116 | 68 | 769 | 105 | 63 |
| gemma / Latest | 64 | 43 | 602 | 37 | 24 |
| qwen / Correction | 71 | 56 | 130 | 15 | 13 |
| qwen / FREE_s0 | 21 | 16 | 563 | 13 | 11 |
| qwen / FREE_s1 | 34 | 27 | 602 | 30 | 25 |
| qwen / INV_s0 | 30 | 15 | 450 | 22 | 12 |
| qwen / INV_s1 | 27 | 20 | 512 | 21 | 14 |
| qwen / Latest | 73 | 56 | 174 | 22 | 16 |

Across all conditions, G0 has 653 witnesses, of which 430 retain disagreement
under M; G1 has 6906 eligible opportunities, 436 witnesses, and 275 matched
witnesses. Among those 275 G1+M cases: **124 pass G2a only, 18 G2b only, 64 both,
and 69 neither**. This is a decomposition, not independent samples.

Before applying M, G2a/G2b/their conjunction retain respectively 275/152/98
original SHS opportunities, from 5240/3584/3058 certified opportunities.
Their matched totals are 188/82/64. Both the certification attrition and the
reduced pair-search opportunities therefore matter; neither loss is assigned
a causal explanation by these counts alone.

Matching count and total length does not match lexical content, clause positions,
which relation was stale, activation geometry, update quality, or every
computational nuisance. Reducing 28 pairs to 2 or 6 also reduces chances to see
disagreement. Lost witnesses are **not** a fraction explained by a confound.

## Primary same-draw results, per condition/seed

Each entry is **matched witnesses (distinct roots); strong matched witnesses
(distinct strong roots)**. Each row has 1152 coded opportunities. A strong witness
must use a qualifying matched pair that spans the original >=0.95/<=0.05
probability thresholds; an extreme pair elsewhere is insufficient.

| Model / condition | G2a M (roots); strong (roots) | G2b M (roots); strong (roots) | Both M (roots); strong (roots) |
|---|---|---|---|
| gemma / Correction | 22 (10); 3 (1) | 2 (2); 1 (1) | 1 (1); 1 (1) |
| gemma / FREE_s0 | 16 (11); 0 (0) | 11 (7); 0 (0) | 9 (6); 0 (0) |
| gemma / FREE_s1 | 22 (15); 1 (1) | 12 (9); 1 (1) | 9 (7); 1 (1) |
| gemma / INV_s0 | 12 (9); 2 (1) | 9 (7); 2 (1) | 8 (6); 2 (1) |
| gemma / INV_s1 | 41 (22); 9 (6) | 20 (15); 3 (2) | 11 (8); 2 (1) |
| gemma / Latest | 18 (8); 3 (1) | 2 (2); 2 (2) | 1 (1); 1 (1) |
| qwen / Correction | 2 (1); 1 (1) | 1 (1); 0 (0) | 1 (1); 0 (0) |
| qwen / FREE_s0 | 6 (5); 0 (0) | 4 (3); 0 (0) | 4 (3); 0 (0) |
| qwen / FREE_s1 | 23 (15); 0 (0) | 10 (8); 0 (0) | 10 (8); 0 (0) |
| qwen / INV_s0 | 8 (5); 6 (5) | 6 (5); 5 (5) | 5 (4); 4 (4) |
| qwen / INV_s1 | 12 (10); 1 (1) | 4 (4); 0 (0) | 4 (4); 0 (0) |
| qwen / Latest | 6 (3); 3 (3) | 1 (1); 0 (0) | 1 (1); 0 (0) |

## Original-convention group summaries

Learned group rates average seeds within each root. Because denominators are
fixed, this equals summed counts / 2304; textual N=1152. Group root counts would
be two-seed unions with a larger search budget, never equal-budget comparisons
to textual conditions. No new intervals or multiplicity framework are used.
Fixed rate = matched count / original N; conditional rate = matched count /
gate-pass count and is descriptive. Retention columns divide matched count by
original G0 witnesses and by its M disagreement ceiling respectively.

### G2a primary-draw rates

| Model / group | Gate pass | Matched / strong | Fixed rate | Conditional rate | Retention: G0 / M ceiling |
|---|---|---|---|---|---|
| gemma / INV | 1350 | 53 / 11 | 2.30% | 3.93% | 36.30% / 63.10% |
| gemma / FREE | 1256 | 38 / 1 | 1.65% | 3.03% | 32.20% / 55.07% |
| gemma / Correction | 387 | 22 / 3 | 1.91% | 5.68% | 31.88% / 50.00% |
| gemma / Latest | 376 | 18 / 3 | 1.56% | 4.79% | 28.12% / 41.86% |
| qwen / INV | 777 | 20 / 7 | 0.87% | 2.57% | 35.09% / 57.14% |
| qwen / FREE | 957 | 29 / 0 | 1.26% | 3.03% | 52.73% / 67.44% |
| qwen / Correction | 56 | 2 / 1 | 0.17% | 3.57% | 2.82% / 3.57% |
| qwen / Latest | 81 | 6 / 3 | 0.52% | 7.41% | 8.22% / 10.71% |

### G2b primary-draw rates

| Model / group | Gate pass | Matched / strong | Fixed rate | Conditional rate | Retention: G0 / M ceiling |
|---|---|---|---|---|---|
| gemma / INV | 918 | 29 / 5 | 1.26% | 3.16% | 19.86% / 34.52% |
| gemma / FREE | 868 | 23 / 1 | 1.00% | 2.65% | 19.49% / 33.33% |
| gemma / Correction | 213 | 2 / 1 | 0.17% | 0.94% | 2.90% / 4.55% |
| gemma / Latest | 228 | 2 / 2 | 0.17% | 0.88% | 3.12% / 4.65% |
| qwen / INV | 542 | 10 / 5 | 0.43% | 1.85% | 17.54% / 28.57% |
| qwen / FREE | 752 | 14 / 0 | 0.61% | 1.86% | 25.45% / 32.56% |
| qwen / Correction | 23 | 1 / 0 | 0.09% | 4.35% | 1.41% / 1.79% |
| qwen / Latest | 40 | 1 / 0 | 0.09% | 2.50% | 1.37% / 1.79% |

### G2ab primary-draw rates

| Model / group | Gate pass | Matched / strong | Fixed rate | Conditional rate | Retention: G0 / M ceiling |
|---|---|---|---|---|---|
| gemma / INV | 770 | 19 / 4 | 0.82% | 2.47% | 13.01% / 22.62% |
| gemma / FREE | 726 | 18 / 1 | 0.78% | 2.48% | 15.25% / 26.09% |
| gemma / Correction | 177 | 1 / 1 | 0.09% | 0.56% | 1.45% / 2.27% |
| gemma / Latest | 187 | 1 / 1 | 0.09% | 0.53% | 1.56% / 2.33% |
| qwen / INV | 467 | 9 / 4 | 0.39% | 1.93% | 15.79% / 25.71% |
| qwen / FREE | 679 | 14 / 0 | 0.61% | 2.06% | 25.45% / 32.56% |
| qwen / Correction | 19 | 1 / 0 | 0.09% | 5.26% | 1.41% / 1.79% |
| qwen / Latest | 33 | 1 / 0 | 0.09% | 3.03% | 1.37% / 1.79% |


## Cross-code sensitivity: same pair, same semantic direction

Only `semantic_representative=True` (draw 0) rows are counted in this analysis.
Fixed N=576 per condition and 1152 per pooled learned group. A second code is
not a second independent semantic fact. Every entry below is
**gate-pass count / matched semantic witnesses / strong semantic witnesses**.

| Model / group | Semantic N | M ceiling | G2a: pass/M/strong | G2b: pass/M/strong | Both: pass/M/strong |
|---|---|---|---|---|---|
| gemma / INV | 1152 | 23 | 610 / 12 / 2 | 384 / 7 / 2 | 309 / 5 / 2 |
| gemma / FREE | 1152 | 18 | 577 / 8 / 0 | 378 / 5 / 0 | 307 / 4 / 0 |
| gemma / Correction | 576 | 9 | 175 / 4 / 1 | 98 / 0 / 0 | 84 / 0 / 0 |
| gemma / Latest | 576 | 6 | 174 / 2 / 1 | 111 / 0 / 0 | 93 / 0 / 0 |
| qwen / INV | 1152 | 6 | 322 / 4 / 1 | 206 / 1 / 0 | 172 / 1 / 0 |
| qwen / FREE | 1152 | 11 | 415 / 7 / 0 | 316 / 3 / 0 | 279 / 3 / 0 |
| qwen / Correction | 576 | 6 | 11 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| qwen / Latest | 576 | 7 | 23 / 0 / 0 | 9 / 0 / 0 | 9 / 0 / 0 |

The original cross-code matched-disagreement ceiling is 86 semantic
opportunities across the whole search. G2a retains 37 (25 designs), G2b 16
(15 designs), and both 13 (12 designs). Their strong counts are 5 (3 designs),
2 (2 designs), and 2 (2 designs). The two strongest conjunction examples are
Gemma INV seed 0/root 0032 and seed 1/root 0035.

Cross-code seed/condition detail uses the same **witnesses (roots); strong
(strong roots)** notation, with N=576 for each row:

| Model / condition | G2a M (roots); strong (roots) | G2b M (roots); strong (roots) | Both M (roots); strong (roots) |
|---|---|---|---|
| gemma / Correction | 4 (4); 1 (1) | 0 (0); 0 (0) | 0 (0); 0 (0) |
| gemma / FREE_s0 | 4 (4); 0 (0) | 3 (3); 0 (0) | 2 (2); 0 (0) |
| gemma / FREE_s1 | 4 (4); 0 (0) | 2 (2); 0 (0) | 2 (2); 0 (0) |
| gemma / INV_s0 | 2 (2); 1 (1) | 2 (2); 1 (1) | 2 (2); 1 (1) |
| gemma / INV_s1 | 10 (10); 1 (1) | 5 (5); 1 (1) | 3 (3); 1 (1) |
| gemma / Latest | 2 (2); 1 (1) | 0 (0); 0 (0) | 0 (0); 0 (0) |
| qwen / Correction | 0 (0); 0 (0) | 0 (0); 0 (0) | 0 (0); 0 (0) |
| qwen / FREE_s0 | 1 (1); 0 (0) | 1 (1); 0 (0) | 1 (1); 0 (0) |
| qwen / FREE_s1 | 6 (6); 0 (0) | 2 (2); 0 (0) | 2 (2); 0 (0) |
| qwen / INV_s0 | 3 (3); 1 (1) | 1 (1); 0 (0) | 1 (1); 0 (0) |
| qwen / INV_s1 | 1 (1); 0 (0) | 0 (0); 0 (0) | 0 (0); 0 (0) |
| qwen / Latest | 0 (0); 0 (0) | 0 (0); 0 (0) | 0 (0); 0 (0) |

Zeros must be read against the ceiling: textual G2b cross-code survival is
0/9 and 0/6 for Gemma correction/latest, and 0/6 and 0/7 for Qwen. Qwen
correction has **zero G2b cross-code eligible opportunities**, so its conditional
rate is undefined, not an observed zero-violation certified population. These
small ceilings limit negative conclusions. By contrast, primary G2b has nonzero
survival in all eight model/update groups. Per-cell rates for any row are exactly
reconstructible from RESULTS.csv using the formulas above; do not double-count
the cross-code fields repeated on both coded rows.

## Strong definition and semantic audit

The unchanged V10 strong rule is original W; all target joint responses valid;
relevant same-code direct correct-label probabilities >=0.95; candidate mass
>=0.90 for operands, all target histories and native reference; and target
correct-label probabilities reaching >=0.95 and <=0.05. Added M-strong requires
these extrema in the **same matched pair**. Cross-code strong requires that
same pair to satisfy both draws and the same semantic direction. Strong does
not newly require every G2a/G2b check or every reference to exceed 0.95.

Manual review covered **all 56 distinct roots** in G2a-or-G2b+M. With 206 coded
surviving opportunities, the deterministic review set was all 32 strong
opportunities plus the first lexicographic `(root,model,condition,query)` survivor
per root: **85 opportunities**. The other 121 ordinary opportunities have full
records and automated checks but are not represented as separately manually
reviewed. Every selected record was checked for named actor pair/function,
intended/source facts, code meaning, direct reads, both same-pair and all six
cross-pair checks, references, matching metadata and target margins.
`SURVIVORS.json` records the exact selection and verdicts. Reviewed root suffixes:

0000, 0001, 0002, 0003, 0004, 0005, 0006, 0007, 0008, 0009, 0010, 0012, 0013, 0014, 0015, 0016, 0017, 0018, 0019, 0020, 0022, 0023, 0024, 0025, 0026, 0028, 0032, 0033, 0034, 0035, 0036, 0037, 0038, 0039, 0040, 0041, 0042, 0043, 0044, 0045, 0046, 0047, 0048, 0049, 0050, 0051, 0052, 0053, 0054, 0055, 0056, 0057, 0059, 0060, 0061, 0063

No invalid case was found or removed. G2a checks were not rejected for logically
determining the target. Some cases pass only one gate: for example Qwen INV s0
root 0024 `d0-extra1` is strong G2b-only, while many G2a cases fail cross-pair
checks. A displayed pair may have correct checks even when another history
breaks an all-history gate; the independent verifier checks the entire battery.
Near-tie flips such as Gemma textual root 0038, p(correct)=0.496279308, remain
ordinary rather than strong. Codes, aliases, repeated roots and methods are not
counted as independent semantic facts.

The most stringent illustrative case is Gemma INV seed 0/root 0032, target
BOTH(Ravi,Faye) on Meadow. All final addressed values are 0. Origins 1 and 4
each require one changed fact and have 413 prefix tokens. Both gates pass
across all histories and both codes. Target p(correct) is
0.999595554 versus 0.037241859 in the literal code and 0.999606055 versus
0.030245450 in the alternate code, on the **same pair**. Both source target
truths are also false, so simply returning the old target truth cannot explain
this contrast. Root 0035 supplies a second strong cross-code conjunction case
in the other learned seed. These are existence examples, not a universal model
of the failure mechanism.

Independent verification recomputed scalar predictions, semantic gold, question
mappings and all flags from saved log probabilities: **13,824 opportunities,
608,256 Boolean checks, 13,824 mapping checks, and 200 summary rows PASS**.
The unchanged upstream suite passed 26 tests in the earlier reconstruction;
four revised regressions passed (mapping/draw separation, distinct gates,
forbidden target access and counterfactual invariance, common-pair code direction).
One earlier wrapper failure demanded bitwise probability equality; it was fixed
to the original numerical tolerance, preserving the completed reconstruction
and every classification. It occurred before the earlier certification join;
the revised scientific run completed on its first attempt.

## Limitations and scientific delta

This is post hoc, with a user-directed revised specification after earlier
results were known. The mapping is fixed by Boolean structure, but it is not
a preregistered confirmation or an outcome-naive study. Certification covers a
small generated relational battery, not all possible propagation or editing
benchmarks. Logical dependence among functions is intentional in G2a and the
checks are not statistically independent. Hard correctness on added probes may
have modest margins. Sparse textual/cross-code ceilings constrain negative
interpretations. Learned root unions have larger search budgets than textual
conditions. Saved-score integrity cannot independently attest the original
forward passes, tokenizer, runtime chronology or cache tensors.

G2a shows that the target pair's composition is not uniformly unreadable: two
logically sufficient functions are correct while a third varies with history.
G2b shows correct compositional use elsewhere in the same relational system,
weakening a **global** failure-to-propagate explanation. Neither excludes
target-specific incomplete propagation, interference, positional effects,
lexical effects or another context-dependent computation mechanism. The
conjunction is the strongest behavioral claim; it is not proof of identical
internal states or universal state failure.

## Exact execution commands and artifact interpretation

The four delivered files are this report, GATE_SPEC.json, RESULTS.csv and
SURVIVORS.json. No separate checksum or commands file is supplied. Working code,
bound source data and receipts remain under
`[local path omitted]` and the pre-existing verified reconstruction
under `[local path omitted]`.

Actual commands (PowerShell; no installation or downloads):

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONDONTWRITEBYTECODE='1'
$env:UV_OFFLINE='1'
$env:UV_PYTHON_DOWNLOADS='never'
$env:UV_PYTHON='[local path omitted]
Set-Location [local path omitted]
python -m unittest -v test_audit
python legacy/VERIFY_FOLLOWUP.py [local path omitted]
$rr='[local path omitted]
& $rr capsule build --spec ../CAPSULE_SPEC.json --output-root ../capsules
& $rr run --capsule ../capsules/1416d6acf2a743f0db95778f46b1c449ca21e902a3436138076b323d3964ee8a
```

Actual result work directory:
`D:/research-agent-platform/runs/cpu-revised-001-b3be3d89/scratch/source/work`.
It was copied without altering its outputs to the local `work/` directory.

```powershell
Set-Location [local path omitted]
python check_audit.py --source source --work work
python review_audit.py --work work --inventory --start 0 --count 85
python finish_report.py
```

The original gate freeze was `python freeze_audit.py` in `source/`; do not rerun
it over its immutable FROZEN directory. The original V10 calculation used the
unmodified independent `reconstruct.py analyze`, followed by `compare`; its
hash-bound completed output is reused, not recomputed to seek a favorable result.
For a fresh deterministic local recomputation within the current authorized
cutoff, `python audit.py --output ../replay` from source reproduces all gates,
pair lists and CSV fields in a new directory; run check_audit with that --work.
These commands refuse output overwrites. The original absolute cutoff is
2026-09-18 08:05:28Z; later work requires an explicitly fresh execution window,
not an extension of this completed attempt.

RESULTS.csv has one row per coded opportunity. `G0_witness` is original W,
`G*_pass` is certification without D, `G*_M_witness` is the certified matched
intersection, and `G*_strong_M` requires an extreme matched pair.
`crosscode_*` fields require both code certifications/common pair/direction;
count them only on `semantic_representative=True`. `original_both_code_W`
means both original coded witnesses before imposing a common pair; it is not
itself a cross-code SHS witness. `matched_pairs` and pair availability are
outcome-free; `matched_disagreeing_pairs` includes original eligibility. Arrays
retain changed-fact counts, exact lengths, target predictions/probabilities and
qualifying pair IDs. Group/root counts and all rates can be recomputed from
these fields. SURVIVORS.json shares complete design and response panels across
cases to avoid duplicated histories; its encoding legend, query mappings and
manual_review records make all checks traceable.

Run `cpu-revised-001-b3be3d89` is SUCCEEDED under the immutable LOCAL capsule
`1416d6acf2a743f0db95778f46b1c449ca21e902a3436138076b323d3964ee8a`.
The code used one math thread; independent checking used one CPU process.
No GPU/cloud resource was allocated; the CPU lease was released. The earlier
superseded audit and its technical-failure evidence were preserved, not deleted.

## Abstract-safe sentence

In a post-hoc reanalysis of V10 saved outputs, history-dependent answers persisted
on 29 shared designs despite correct measured direct reads, correct non-target
compositions on both the target and other actor pairs, and exact matching of
changed-fact count and prefix-token length.

## Final scientific verdict

1. **V10 RECONSTRUCTION — PASS.** Exact agreement on 16,128/16,128
   classifications/hard answers and 896/896 root rows; all 8/8 original primary
   cells and reported intervals match at the unchanged numerical tolerance.

2. **G2a RESULT.** 188 same-draw matched survivors across 53 designs and all
   eight model/update groups, including 29 strong survivors across 17 designs.
   Cross-code common-pair sensitivity retains 37 semantic opportunities across
   25 designs (5 strong across 3 designs).

3. **G2b RESULT.** 82 same-draw matched survivors across 34 designs and all
   eight groups, including 14 strong across 9 designs. Cross-code sensitivity
   retains 16 semantic opportunities across 15 designs (2 strong across 2).
   Both gates hold for 64 coded opportunities across 29 designs; 13 semantic
   opportunities across 12 designs also survive the common-pair cross-code test.

4. **SCIENTIFIC DELTA.** The explanation that no compositional use of the
   updated relations generalized beyond direct readout is weakened. The
   narrower explanation of target-specific propagation failure or context
   interference remains viable; identical latent state is not established.

5. **NEXT STEP — GO.** The smallest focused follow-up is one backbone/method:
   Gemma with the existing INV editor, both frozen learned seeds, and 64 fresh
   roots under the existing eight-history/two-code battery and native/no-op
   controls. Before any new scores, freeze G1/G2a/G2b, M, the conjunction,
   same-pair strong/cross-code criteria, root-unit analysis, fixed denominators,
   resource cap and a no-rescue stopping rule. Use the G2a+G2b+M conjunction as
   the single prospective primary behavioral question and report both seeds
   separately. This is a bounded validation recommendation, not authorization
   to allocate a GPU or a promise of power for a broad prevalence claim.
