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

## Ask whether current facts are sufficient for later use

Useful editors made a stronger question worth asking. Distinct starting contexts receive the same commands and reach the same intended final records. For a fixed downstream question, should knowing the overwritten values still help predict the answer? We call the requirement that it should not source-history sufficiency.

This differs from ordinary answer accuracy. A reader can be consistently wrong without depending on history. Conversely, it can correctly report the relevant current facts while giving different joint answers after different histories. Readable state and operative state are shorthand for these behavioral roles, not separate physical objects we have identified.

The later constructive tests retained useful updates, repetition, restoration and bounded composition across Gemma and Qwen. The failure was therefore not confined to an intervention that never worked. Improving editor accuracy alone could not resolve whether the declared facts accounted for downstream behavior.

V9 supplied discovery and a full retrospective census of the history effect. Applying the same certificate to V6 supplied archival corroboration on nonoverlapping exact inputs, although the studies shared task and software ancestry and a learned checkpoint. These were reasons to seek prospective confirmation, not substitutes for it.

V10 then confirmed the central prediction on fresh generated cases with a fixed primary analysis. Both backbones showed the pattern under learned and textual updates, after protecting the relevant direct reads and reference answers. This supports the evaluation criterion without requiring the learned editor to outperform every simpler method. The ordinary-reader coherence control remains adverse to a different claim: editing did not uniquely create logical incoherence.

## Current position

The constructive result and the source-history result now belong together. Useful question-independent updates exist, yet correctly reported current facts can be insufficient to account for measured downstream responses. Correct facts plus a history-sensitive reader remain possible; the experiments do not show that the facts are absent internally.

The next step is to test this criterion on public factual-editing methods and benchmarks, and sharpen its relationship to prior work on consequence testing, repeated edits and state abstraction. External validation remains pending. The first external execution attempt produced no scientific result. Neither external generality nor priority over that prior work is established here.
