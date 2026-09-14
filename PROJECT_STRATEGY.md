# Project strategy

Geometry of Endorsement asks how much we can infer from a model intervention that appears to change a semantic judgment. The work started with a readable support/opposition direction and gradually shifted toward a stronger goal: changing a relation in a way that later questions can reliably use.

Measurements are in [Results and Claims](docs/RESULTS_AND_CLAIMS.md); reproduction details are in [Reproducibility](REPRODUCIBILITY.md).

## Start with a relation whose meaning can be tested

ValuePrism gives a controlled target: a particular consideration can support an action in one situation and oppose it in another. Checkerboard evaluations make fixed preferences for the situation or consideration unhelpful, while text-only and factual-truth controls test simpler explanations.

Both Llama and Gemma contain a linear signal for this contextual relation. Its behavior across moral categories and ordinary argument datasets suggests a broader counts-for / counts-against representation.

The first important limit appeared when we asked what distance along that direction meant. It separated the relation labels, yet added little to predictions of rephrasing flips after accounting for text and answer confidence. Its sign carried a useful semantic interpretation; its magnitude did not behave like a measure of commitment.

## Follow the direction into the model’s computation

Steering the direction changed the model’s answer, which showed that the coordinate could influence downstream behavior. Truth and sentiment directions could often reach the same endpoint. That made the answer itself an incomplete description of what had changed internally.

Causal-fingerprint experiments pushed on the same issue from another angle. Small intervention effects could be predicted very accurately at a particular prompt, while the same semantic direction did not always keep a stable causal role across contexts. Gemma made this separation especially clear.

Controlled multi-person contexts showed that the direction follows the addressed relation: who supports what matters. Broader activations also retained relation information beyond the scalar score and explicit answer.

These results shifted the project toward the state in which the relation is represented and used.

## Compare intervention effects with a genuine change in the relation

A natural counterfactual gives a concrete reference. Change one relation in the text, then ask what else should follow from that change.

This test exposed a large gap. In Gemma, rank-one steering could hit the requested direct answer almost perfectly while the rest of the answer pattern failed to resemble the genuinely changed context. Moving the broader activation toward the natural counterfactual recovered much more of the expected consequences.

That result supplied the central evaluation idea for the rest of the project: later behavior should reflect the edited fact across several consequences, not only at the answer used to tune the intervention.

## Learn updates that happen before the future question exists

The next step moved editing into the shared context. An addressed relation is updated first, the context is sealed, and later questions are supplied afterward.

This makes reuse observable. The same edit can be queried in several ways, repeated, undone, or combined with other edits.

Successive shared-state editors improved this behavior substantially. Single edits became strong, repeated assignments became almost idempotent in the best Gemma settings, and restoration recovered earlier answers at high rates. Unrestricted overwrite also worked, so the capability did not depend on one specially constrained parameterization.

Training on a wider set of consequences from individual edits improved several unseen multi-update programs, especially in Qwen. Derived joint questions remained the main weakness.

## Ask whether the final state still carries its history

The source-independence study gives the current project its strongest unresolved test.

Several different starting contexts receive updates that lead to exactly the same requested final relations. Later questions then probe the resulting state. A reliable assignment-like update should make those answers depend on the final relations.

The current editors preserve useful single-edit, repetition, and restoration behavior across both tested backbones. Yet some joint answers still change with the overwritten starting values. This can happen even when the individual relations needed for the joint question are read correctly in every starting context. Fixed attempts to clarify the reader did not remove the effect.

That result points to a concrete research target: the edited information should govern consequences that were never shown to the updater.

## Current position

The project now has a constructive result and a clear boundary around it. Addressed, question-independent updates can behave like reusable state changes under single edits, repetition, restoration, and parts of multi-update evaluation. Source history still leaks into some joint consequences.

The next useful work should explain or remove that history dependence while keeping the existing capabilities intact. The stronger principle is simple: once a fact is updated, later computation should follow the new state regardless of the value that was overwritten.
