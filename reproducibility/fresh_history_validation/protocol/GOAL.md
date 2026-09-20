# Goal

Test the strongest version of the current paper claim on fresh cases, then ask
whether checking more than one history helps predict failure on withheld questions.

Read PROTOCOL.md, SCOPE.json and SOURCE_BINDINGS.json. Reuse the supplied repo's
scientific inference functions and learned checkpoints. The protocol is the
execution specification; the launcher supplies resource authority.

Run one Gemma-2-9B-it study with the existing INV_PAIR_NLL update and both stored
seeds. Generate 8 technical cases and 64 fresh final cases using the existing task
family, eight histories, two answer formats and 34-question battery.

Before inference, freeze the full same-pair and cross-pair controls, count/length
matching, all data, exact questions, response scoring and analysis. The primary
question is whether history-dependent target answers recur while all those
controls hold together.

On the same scores, compare six ordinary consequence checks at one history with
three checks repeated across two histories. Reserve three other questions per
case for testing the validation decisions. Keep the read budget fixed, compare
acceptance coverage and downstream error separately, and use equal-retention
comparisons to expose an advantage caused only by rejecting more cases.

This package supplies a protocol and model-free planning helpers. It is not a
finished fresh-case model runner. The repo's saved-score replay is working
evidence; its optional new-inference entry point still needs a real replay check.

Complete only the thin new-case adapter, restricted method schedule, analyzer and
decision-input boundary. Verify two fixed old cases, run the technical cases,
measure memory/throughput, and admit the full final block only if it fits with the
required margin. Do not select cases, parameters or questions from new outcomes.

Complete both learned seeds. After final execution, repeat up to four cases using
the predeclared reproducibility rule. Preserve the original counts and all replay
discrepancies. Keep current repository and manuscript artifacts unchanged.

Use one isolated worker and the existing independent exact-instance deadline
guard. Finish CPU preparation before renting. The proposed ceiling is ten
billable hours/$50 all-in on one H100 80GB, with a 30-minute closeout reserve.
Resolve startup/replay within the first 90 billable minutes or export and close.
The original scientific protocol and all old budgets stay separate.

Return the prospective result, the held-out validation comparison, transparent
CPU reconstruction, simple figures, actual resource accounting and
STORY_DELTA.md. Distinguish a replicated counterexample from evidence that the
new validation policy predicts future errors better. Finish with verified durable
export and exact-instance shutdown. No automatic additional experiment.
