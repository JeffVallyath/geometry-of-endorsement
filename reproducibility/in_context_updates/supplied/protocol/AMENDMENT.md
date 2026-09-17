# ICMH1-RIPPLE-v2: proposed execution amendment

Status: PROPOSED, NOT RUN. Copying the companion launcher is the proposed user approval. This handoff is a protocol and source-metadata audit, NOT an implemented or tested neural runner. Preserve the original ICMH1_FOLLOWUP_SPEC.zip unchanged. This amendment and SCOPE.json supersede its conflicting choices; resource authority comes only from the user's actual launcher.

## 1. Decision and scientific scope

Run one nonparametric, public-data matched-history study. Fix Qwen/Qwen2.5-7B-Instruct before outputs; do not compare it with Llama or search for a model that fails supersession. Use only the original five strong RippleEdits development roots, six strong qualification roots (16 questions), and 34 strong final roots (98 questions). Keep the two preselected full-root repeats: popular:670 and random:618. Exact prefixed IDs, question IDs, current-record order, splits and source hashes are in SCOPE.json.

Do not run ANY MQuAKE inference in this task: not the exposed eight, the proposed 12 qualification roots, the 37 secondary finals, or the sealed nine. Saved exposed MQuAKE records may explain prior failures, but need not be regenerated. This removes the unrelated MQuAKE gate/secondary block and preserves the nine and other unused pools. No parametric RippleEdits fallback is authorized after any outcome.

The question is whether a changed obsolete value in an explicitly retained prompt history changes correctness on the same downstream question after identical current records are supplied and individually recovered. A positive is public-data/nonparametric evidence. It is NOT a successful MEMIT/AlphaEdit replication, erased-memory failure, proof of a unique internal mechanism, or representative prevalence estimate. It does not establish novelty by itself. Instruction tuning does not guarantee qualification.

## 2. Source and input construction

Read the original SPEC.md/SPEC.json and this amendment. Reuse the v7 source archive's prepared data, not its parametric worker, environment lock, covariance files, weight resets, or whole-model hashing loop. Check generated manifests against the nested PACKAGE_MANIFEST; SOURCE_LOCK covers upstream source files and must not be assumed to bind generated manifests. Use the selected hashes in SCOPE.json as additional checks. No need to rerun the old MEMIT fidelity suite.

Construct one fixed current table S per root, before serving any particular downstream question. Include every final edited record AND all supporting records needed by the selected strong questions. Deduplicate by subject/relation/temporal address and require consistent targets. Use the root-wide topological order in SCOPE.json, derived from the source chain paths with deterministic address tie-breaking. The model must not receive question-specific gold answers, scoring aliases, B/D labels, eligibility flags, or a computed answer from a symbolic solver.

Validate every source chain and every required support before model execution. B, D and C must be different entity IDs at the SAME edited address. The B and D streams have equal numbers of assignments and identical templates, current tables, downstream questions, decoding and post-history state policy. Only the obsolete record value differs. Record token lengths; equal record counts do not imply equal token counts or rule out all lexical explanations. Do not add outcome-driven padding.

Render the source's plain record sentences without bracketed metadata. Use the five original context types: none, given_facts, canonical, chronological_B and chronological_D. Given_facts and canonical contain the SAME S in the SAME order; only their declared headers differ. Chronological streams contain the original supersession header, then the B or D assignment, then the COMPLETE S, then the question. Derive canonical current contents through a deterministic last-write ledger and assert both histories yield the same table. Do not blindly reuse a textual branch that contains only edited facts but omits unedited support. All selected records are scoped authoritative benchmark assignments, not assertions about actual geography, family history, etc.

Preserve public question wording and the strong-chain restriction. For Compositionality II the edited address may be hop 2, not hop 1. Read the actual edited address for prior/current-value diagnostics. The original 'prior answer differs from B and D' quantity is a diagnostic, NOT structural proof that B and D differ, and is no longer an execution gate.

## 3. Runtime, rendering and scoring

Pin an immutable Qwen model/tokenizer revision and record license, files, chat template and full software versions BEFORE first inference. Use a compatible installed/prebuilt x86-64 CUDA runtime; do not reuse Transformers 4.23.1, which does not support Qwen2.5. Official model documentation requires at least Transformers 4.37.0; this minimum alone does not certify the whole environment.

Use bf16 weights, model.eval(), inference_mode/no gradients, greedy generation, num_beams=1, max_new_tokens=16, model chat template and the fixed system instruction 'Answer with only the entity name.' No quantization, training, parameter edits, model comparison, CoT search or LLM judge. Record generation settings including EOS, padding, repetition penalties and backend precision. A KV cache may be used WITHIN an individual generation; start it fresh for every independent prompt. No parameter snapshots or full-weight hashes per query.

Primary C and D use the original ICMH1 strict scorer: first generated line, outer whitespace removed, at most one final punctuation character from . , ; : ! ? removed, then case-sensitive equality with a public answer/alias processed by the same rule. Do not skip lines to rescue an answer. Retain original raw strings and strict scores.

Primary A uses a fixed full-string entity-ID extractor for atomic reads: apply the same line/punctuation rule, then NFKC, casefold and whitespace collapse; exact-match against the full source entity alias registry restricted by the record's source object_type. Exactly one ID must match; otherwise abstain. No gold-dependent candidate restriction, prefix matching, substring rescue, or extra prose boundaries. The same normalization applies to registry aliases. Include labels/aliases from the source records deterministically, not from model outputs. Retain strict atomic scores alongside this extraction. This explicitly replaces the original underspecified longest-prefix diagnostic parser. Tests must cover ambiguous same-name IDs, typed aliases, wrong answers, extra prose, empty output, truncation and punctuation.

Use development for implementation/format/throughput checks, not adaptive scientific changes. Fix the above prompt/parser rules now; any substantive reader redesign after seeing outputs is outside this launch. Freeze the actual implementation, batch size/order, model, prompts and scorers before qualification. Technical repairs must preserve these semantics and be logged; no retuning after qualification.

## 4. Flags, gates, and what the gate may inspect

A(root): every record in the declared S is extracted as the correct entity under BOTH chronological contexts. C(q): strict downstream correctness under BOTH canonical and given_facts. D(q): both chronological downstream answers map unambiguously to DISTINCT entity IDs under the question's final-hop range type, and exactly one is strictly correct. Use the same fixed full-string typed parser for this validity/distinctness check, without consulting the gold answer. W(q)=A(root)&C(q)&D(q). Two different wrong answers, unmatched/ambiguous answers and alias/casing differences for the same entity are diagnostics, not W. Keep failed strict scores visible even where entity IDs agree. This explicitly prevents a formatting-only difference from becoming a semantic witness.

Development: five original Ripple development roots, 14 strong questions. Run the fixed readouts, retain records and measure realistic end-to-end throughput. Materialize all final input metadata and exact job counts without FINAL model inference. Check runtime/format, then freeze for qualification.

Qualification: all six predeclared roots and all 16 strong questions. Generate atomic reads under chronological_B/D plus canonical and given_facts downstream references and prior diagnostics. Chronological DOWNSTREAM generation is not needed for this gate and is excluded from qualification. This avoids exposing the target effect during a readiness decision.

All gates are fixed before qualification:
- G1: at least 13/16 correct under given_facts AND 13/16 under canonical.
- G2: at least 18/22 distinct current records correct under BOTH chronological contexts.
- G4: at most 4/44 chronological atomic reads abstain or fail to map uniquely. Prior/no-context reads do not enter this denominator.
- G5: A&C coverage at least 10/16 questions, spanning at least 3/6 roots.
- Technical: all records complete, no unresolved model/runtime/scoring fault.
- Resource: measured complete final+repeat job forecast fits the approved remainder with at least 25% work-time allowance and 30 minutes for export/shutdown.

G5 and canonical G1 are NEW prospective safeguards: the original qualification tested supplied-facts success without requiring the clean comparison used by W to be viable. These are operational readiness choices, not statistical significance requirements or retrospective changes to failed Phase 1. Thresholds must not be lowered to proceed.

If any completed qualification gate fails, no FINAL. Keep failed qualification evidence and stop this study after export. Do not switch to Llama, change the prompt, draw replacement qualification roots, use the sealed nine, or automatically start a parametric study. A technical/resource inability is reported separately from a completed scientific gate failure.

## 5. Final execution and repeats

On full gate pass, write PHASE_GATE and FINAL_FREEZE binding source/model/runtime, exact 34 root IDs and 98 question IDs, two repeat IDs, all prompts and record maps, parser/scorer, batch/order/cache policy, code, denominators and resource limits. Then run all 34 in the supplied fixed order without effect-based stopping.

For each final root obtain downstream responses in all five contexts, current-record atomic reads under B/D/none, and prior/current edited-address diagnostics. Preserve the eight prepared CounterFact locality items under none and chronological_B as a DIAGNOSTIC, not a MEMIT fidelity claim or a success gate. Compile the same locality panel before outcomes. Cache identical none-context diagnostics only with explicit logical aliases; never reuse outputs for a repeat. Preserve raw source-native references alongside any rendering adaptation.

Repeat the ENTIRE schedules for popular:670 and random:618 in two fresh processes after the main block, same seeds, prompts, settings and batch layout. Compare raw text, extracted entities, A/C/D/W flags and diagnostic outcomes. Do not count repeats as extra independent roots or select replacements after seeing which roots produced witnesses. Agreement on two W=0 roots does not establish repeatability of un-repeated positive roots. Greedy decoding is not a guarantee of hardware-independent reproducibility.

A technical interruption produces missing rows, not W=0. Resume only exact unfinished work with clear provenance; no scientific retries or best-run selection. If final cannot finish within the cap, export an explicitly incomplete dataset rather than shrinking/replacing its denominator.

## 6. Analysis and claim limits

Always report every A/C/D/W count, A&C coverage, W/98 and roots with W/34, per-category counts and root-level distributions. Also show witness rate among eligible opportunities as secondary, never replacing full denominators. Report all repeats including disagreements. No p-values and no 3/98 'rule of three' population bound: queries share roots and this restricted sample is not a demonstrated random population draw.

Remove the original positive/null/mixed classifier. It left some counts unclassified and described W with failed A/C, which is impossible under its own definition. Use transparent results instead: qualification failed (FINAL unopened); completed with zero witnesses; completed with one or more witnesses; coverage-limited (<60% final A&C); or technically incomplete. Repeat stability is a separate axis. One witness is not a null; zero observed is not evidence of universal invariance. Do not suppress unstable or zero rows as a 'stable subset.' Retain original proposal labels only as superseded history.

## 7. Execution and custody

One lightweight runner and analyzer, not the old parametric pipeline. Build and dry-test on CPU before renting. Implement only data rendering, generation, journaling, scoring, gates, repeats and analysis. Do not make another generic automation framework or promise the unimplemented spec already executes.

The companion launcher proposes a new, bounded 3-hour/USD10 study, with one A100 40GB SXM4 at <=USD1.99/h and a 30-minute closeout reserve. Respect its exact resource/accounting rules. Monitor actual measured throughput, not the memo's unmeasured 15-25 minute estimate. Unit tests and metadata audits do not count as GPU qualification.

Use supported existing budget supervision with exact instance binding. Failed diagnostics stop the workload, not the machine during authorized useful recovery. Once a completed qualification NO-GO or completed result leaves no authorized work, verify durable export and close out promptly; do not idle to the cap. Keep hard-deadline protection. Live provider/SSH/export calls remain allowed after model/data staging; 'offline inference' does not disable supervision.

Save status at actual milestones. Do not wait for user approval at routine phase boundaries when the preauthorized gates pass. Never assume human monitoring is available. No push, other method/model, extra instance, paid API, local pretrained fallback, new persistent filesystem, or automatic cap increase.

Deliver RESULTS.md, all raw responses and logical-to-physical identities, exact prompt/record manifests, gates, freezes, environment/config/commands, parser/analysis and tests, repeat comparison, cost/timing/memory, and verified export/provider status. Describe what was actually tested and what remains unmeasured. No new result is promised by this protocol.
