from __future__ import annotations


DISPOSITIONS = frozenset({"GENERAL_CONTEXTUAL_RELATION_SUPPORTED",
    "VALUEPRISM_BOUNDED_SEMANTIC_FEATURE", "OUTPUT_CHANNEL_DOMINATES",
    "VALUEPRISM_TEMPLATE_DEPENDENCE", "MIXED_OR_UNDERDETERMINED"})


def decide(evidence: dict[str, bool]) -> str:
    if evidence.get("models_materially_disagree") or not evidence.get("native_external_competence", True) \
            or evidence.get("sentiment_alternative_unresolved") or evidence.get("broad_intervals"):
        return "MIXED_OR_UNDERDETERMINED"
    if (evidence.get("physical_token_alignment_dominates") and
            not evidence.get("residual_checkerboard_signal")):
        return "OUTPUT_CHANNEL_DOMINATES"
    if (not evidence.get("semantic_mapping_invariance") or
            evidence.get("strong_template_dependence")):
        return "VALUEPRISM_TEMPLATE_DEPENDENCE"
    general = all(evidence.get(key, False) for key in (
        "semantic_mapping_invariance", "residual_checkerboard_signal",
        "zero_shot_amperepp_both_models", "secondary_directionally_consistent",
        "external_within_task_valid", "reverse_transfer_both_models",
        "cosines_exceed_grouped_null", "sentiment_not_complete_explanation"))
    if general: return "GENERAL_CONTEXTUAL_RELATION_SUPPORTED"
    if evidence.get("direct_token_explanation_weakened") and evidence.get("residual_checkerboard_signal"):
        return "VALUEPRISM_BOUNDED_SEMANTIC_FEATURE"
    return "MIXED_OR_UNDERDETERMINED"


PERMITTED_CLAIMS = {
    "GENERAL_CONTEXTUAL_RELATION_SUPPORTED": "A shared linear feature tracks contextual support versus opposition across moral and argumentative relation tasks.",
    "VALUEPRISM_BOUNDED_SEMANTIC_FEATURE": "The feature is not reducible to the direct reporting channel, but its generality beyond the tested ValuePrism relation task is not established.",
    "OUTPUT_CHANNEL_DOMINATES": "The tested feature is dominated by the measured reporting channel under the frozen diagnostics.",
    "VALUEPRISM_TEMPLATE_DEPENDENCE": "The tested feature materially depends on the frozen ValuePrism task or prompt family.",
    "MIXED_OR_UNDERDETERMINED": "The frozen experiment does not distinguish the competing explanations.",
}
