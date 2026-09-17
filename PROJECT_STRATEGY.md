# Project strategy

Geometry of Endorsement asks how much we can infer from a model intervention that appears to change a semantic judgment. It connects a readable support/opposition direction to a stronger goal: changing a relation in a way that later questions can reliably use.

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

The update is made to the shared context before any later question is supplied. The updated context is then kept fixed and reused for those questions.

This makes reuse observable. The same edit can be queried in several ways, repeated, undone, or combined with other edits.

Successive shared-state editors improved this behavior substantially. Single edits became strong, repeated assignments became almost idempotent in the best Gemma settings, and restoration recovered earlier answers at high rates. Unrestricted overwrite also worked, so the capability did not depend on one specially constrained parameterization.

Training on a wider set of consequences from individual edits improved several unseen sequences of updates, especially in Qwen. Questions combining several facts remained the main weakness.

## Ask whether current facts are sufficient for later use

Prior work already establishes that local edit success can coexist with failed
downstream propagation. Comparing update procedures can show how much downstream
coherence depends on the procedure; it cannot establish a universal cure. The
completed [matched-history comparison](reproducibility/update_method_comparison/README.md)
shows clear improvements for both learned procedures in Qwen and mixed results
in Gemma. Reliably answering every question remains difficult. Local update quality and
downstream coherence therefore remain distinct evaluation objectives.

Useful editors made a stronger question worth asking. Distinct starting contexts receive the same commands and reach the same intended final records. For a fixed downstream question, should knowing the overwritten values still help predict the answer? We call the requirement that it should not source-history sufficiency.

This differs from ordinary answer accuracy. A reader can be consistently wrong without depending on history. Conversely, it can correctly report the relevant current facts while giving different joint answers after different histories. Readable state and operative state are shorthand for these behavioral roles, not separate physical objects we have identified.

The later constructive tests retained useful updates, repetition, restoration and bounded composition across Gemma and Qwen. The failure was therefore not confined to an intervention that never worked. Improving editor accuracy alone could not resolve whether the declared facts accounted for downstream behavior.

The retrospective census (`V9`) measured how often the history effect occurred in the retained cases. Applying the same discrepancy criterion to earlier shared-state editing data (`V6`) supplied archival corroboration on nonoverlapping exact inputs. The studies nevertheless shared task and software ancestry and a learned checkpoint. These checks motivated a study planned before its outcomes; they did not substitute for one.

The prospective matched-history study (`V10`) confirmed the central prediction on fresh generated cases with a fixed primary analysis. Both models showed the pattern under learned and textual updates, after the relevant individual facts and reference answers passed their checks. This supports the evaluation criterion without requiring the learned update to outperform every simpler method. A separate logical-coherence control limits a different claim: editing did not uniquely create logical incoherence.

## Current position

The constructive result and the source-history result now belong together. Useful question-independent updates exist, yet correctly reported current facts can be insufficient to account for measured downstream responses. Correct facts plus a history-sensitive reader remain possible; the experiments do not show that the facts are absent internally.

We now also have coarse causal evidence from Gemma. Swapping stored activations from before the question changed which history the later answer followed, while direct fact answers stayed correct. In five of six selected examples under the main answer format, the answer followed the history supplying the later layers' activations. This shows that those activations can carry enough information to sustain the history-sensitive answer under the swap. A complete mechanism, including where the information first arose, remains unresolved.

The one-case robustness follow-up is now completed. One retrospectively selected Gemma history pair combines strong original disagreement under both tested answer formats with equal changed-fact counts and equal prompt lengths. Swapping the later stored activations changed which history controlled the downstream answer under both formats, while separately measured direct facts remained correct. This example now joins the strong behavioral controls and the causal result. The formats are two tests of the same example, not independent cases or a prevalence estimate. Full mechanism identification remains unresolved. The mechanism branch is closed for the current paper.

A completed Qwen study found source-history dependence on public RippleEdits-derived compositional questions. Qwen correctly reported the current records, yet old information still affected some later answers. This extends the internal finding beyond generated relational tasks. Updates were supplied in the prompt while model weights stayed fixed. The obsolete records remained earlier in the prompt.

The [offline replay](reproducibility/in_context_updates/README.md) now derives
this result from the saved responses under both frozen parsing rules. It retains
the drop in accuracy on unrelated facts and the lack of qualifying history-dependent questions in the repeated
benchmark cases. The [paper guide](PAPER_GUIDE.md) connects each current claim to its evidence.

Conventional parameter-editing validation remains open. The early AlphaEdit route stopped before pretrained inference and produced no scientific result. The later MEMIT/GPT-J MQuAKE Phase 1 ran pretrained inference but failed prospective reference qualification before FINAL. The run stopped at qualification, so it provides no final evidence about source-history dependence. Broader external generality remains open. The closest-prior-work comparison is still being finalized; priority over that work remains unresolved.
