"""Tests for the prose-validation pipeline.

The important cases are the deliberately bad rewrites — a dropped caveat, an
altered number, terminology drift, an unsupported causal claim, an unreadably
overloaded sentence — and the pending-claim semantics: an unverified number must
stay explicitly tracked, never decay into a generic warning, and must fail under
submission-strict when it carries a headline claim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prose_check as pc  # noqa: E402

CAPTIONS = ROOT / "figures" / "CAPTIONS.md"
RESULTS = ROOT / "docs" / "RESULTS_AND_CLAIMS.md"


def run_captions(text: str, tmp_path: Path) -> pc.Report:
    path = tmp_path / "CAPTIONS.md"
    path.write_text(text, encoding="utf-8")
    report, _ = pc.run("captions", override_path=path)
    return report


def rules(report: pc.Report, gate: str | None = None) -> set[str]:
    return {f.rule for f in report.findings if gate is None or f.gate == gate}


@pytest.fixture(scope="module")
def captions() -> str:
    return CAPTIONS.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# shipped prose
# --------------------------------------------------------------------------

def test_all_documents_pass_the_hard_gates():
    report, _ = pc.run()
    assert report.ok(), "\n".join(str(f) for f in report.errors)


def test_no_readability_warnings_anywhere():
    report, _ = pc.run()
    assert not report.warnings, "\n".join(str(f) for f in report.warnings)


def test_every_contract_binds_to_a_unit():
    report, _ = pc.run()
    assert "unit-missing" not in rules(report)


def test_every_comprehension_spec_binds_to_a_unit():
    _, units = pc.run()
    for doc in pc.documents():
        for path in sorted((ROOT / doc["comprehension"]).glob("*.json")):
            spec = pc.load_json(path)
            assert pc.find_unit(units[doc["id"]], spec["heading_prefix"]) is not None, spec["unit_id"]
            for question in spec["questions"]:
                assert question["accept_patterns"] and question["gold"], question["id"]


# --------------------------------------------------------------------------
# pending claims stay visible
# --------------------------------------------------------------------------

def test_pending_claims_are_tracked_not_warned():
    """An unverified number is its own category, never a readability warning."""
    report, _ = pc.run("results")
    assert report.pendings, "Results should still carry pending claims"
    for finding in report.pendings:
        assert finding.gate == "FIDELITY"
        assert finding.rule == "artifact-pending"
    assert not any(f.rule == "artifact-pending" for f in report.warnings)


def test_pending_claims_name_their_family_and_reason():
    report, _ = pc.run("results")
    for finding in report.pendings:
        assert "family:" in finding.message
        assert finding.message.rstrip().endswith(("resolved", "json", "artifact", "sections"))


def test_submission_strict_fails_on_pending_headline_claims():
    writing, _ = pc.run("results", strict=False)
    strict, _ = pc.run("results", strict=True)
    assert writing.ok(), "writing mode must not block on pending claims"
    assert not strict.ok(), "submission-strict must fail while headline claims are unverified"
    assert all(f.rule == "artifact-pending" for f in strict.errors)


def test_pending_value_absent_from_prose_is_an_error():
    """Declaring a claim pending does not excuse it from appearing."""
    report = pc.Report()
    contract = {
        "numbers": [{"value": "0.4242", "status": "pending",
                     "claimed_source_family": "made_up", "reason": "test"}],
        "literals": [], "required_facts": [],
    }
    pc.check_fidelity("d", "u", "prose with no such value", contract, {}, {}, report, False)
    assert "number-missing" in {f.rule for f in report.errors}


def test_verified_binding_without_a_path_is_malformed():
    report = pc.Report()
    contract = {"numbers": [{"text": "1.23", "status": "verified"}], "literals": [],
                "required_facts": []}
    pc.check_fidelity("d", "u", "1.23", contract, {}, {}, report, False)
    assert "binding-malformed" in {f.rule for f in report.errors}


def test_coverage_counts_bound_and_pending():
    report, _ = pc.run()
    bound = sum(r["bound"] for r in report.coverage)
    pending = sum(r["pending"] for r in report.coverage)
    assert bound > 0 and pending > 0
    assert all(r["total"] == r["bound"] + r["pending"] for r in report.coverage)
    headline = sum(r["headline_pending"] for r in report.coverage)
    assert headline > 0, "headline pending claims should be counted separately"


# --------------------------------------------------------------------------
# deliberately bad rewrites
# --------------------------------------------------------------------------

def test_dropped_caveat_is_caught(captions, tmp_path):
    bad = captions.replace(
        "Gemma was not re-run because its layer-27 activations\nwere not retained.", "")
    assert bad != captions
    report = run_captions(bad, tmp_path)
    assert not report.ok()
    assert any("gemma-not-rerun-reason" in f.message for f in report.errors)


def test_dropped_sidedness_qualification_is_caught(captions, tmp_path):
    bad = captions.replace("a one-sided empirical\npermutation p-value", "a permutation p-value")
    assert bad != captions
    report = run_captions(bad, tmp_path)
    assert any("p-is-one-sided" in f.message for f in report.errors)


def test_altered_number_is_caught(captions, tmp_path):
    bad = captions.replace("0.0132, with 131 draws", "0.0072, with 131 draws")
    assert bad != captions
    report = run_captions(bad, tmp_path)
    assert "number-missing" in rules(report, "FIDELITY")
    assert "number-undeclared" in rules(report, "FIDELITY")


def test_silently_improved_number_is_caught(captions, tmp_path):
    bad = captions.replace("It reaches a checkerboard interaction of 0.28.",
                           "It reaches a checkerboard interaction of 0.3.")
    assert bad != captions
    report = run_captions(bad, tmp_path)
    assert "number-missing" in rules(report, "FIDELITY")


def test_terminology_drift_is_caught(captions, tmp_path):
    bad = captions.replace("frozen\nMiniLM comparator", "frozen text baseline")
    assert bad != captions
    report = run_captions(bad, tmp_path)
    assert "alias-used" in rules(report, "TERMINOLOGY")


def test_direction_named_by_its_fitting_method_is_caught(captions, tmp_path):
    bad = captions.replace("The support/opposition direction gives\n0.0132",
                           "The fitted direction gives\n0.0132")
    assert bad != captions
    report = run_captions(bad, tmp_path)
    assert "alias-used" in rules(report, "TERMINOLOGY")


def test_unsupported_causal_claim_is_caught(captions, tmp_path):
    bad = captions.replace(
        "This experiment does not isolate why the\ntwo nulls differ;",
        "The two nulls differ because regularization stabilises the logistic probe;")
    assert bad != captions
    report = run_captions(bad, tmp_path)
    assert "forbidden-claim" in rules(report, "FIDELITY")


def test_random_direction_mislabel_is_caught(captions, tmp_path):
    bad = captions.replace(
        "Its implementation hashes each\nitem identifier to a scalar in the range 0 to 1",
        "It is a random direction of the same norm")
    assert bad != captions
    report = run_captions(bad, tmp_path)
    assert not report.ok()
    assert {"forbidden-claim", "alias-used"} & rules(report)


def test_overloaded_sentence_is_warned_not_failed(captions, tmp_path):
    stuffed = (
        "The support/opposition direction gives 0.0132, with 131 draws at or above it, "
        "while the logistic activation probe holds at 0.0001 across 10,000 draws, and "
        "the Gemma panels remain at 200 draws with 0.0149 for the direction, all of "
        "which together indicate that the corrected values should now be preferred "
        "over the superseded ones in every downstream table and summary sentence."
    )
    bad = captions.replace(
        "The support/opposition direction gives\n0.0132, with 131 draws at or above it.", stuffed)
    assert bad != captions
    report = run_captions(bad, tmp_path)
    assert {"long-sentence", "numeric-overload"} & rules(report, "READABILITY")
    assert not [f for f in report.findings if f.gate == "READABILITY" and f.level == pc.ERROR]


def test_statistic_defined_by_negation_is_warned(captions, tmp_path):
    """Defining a statistic by listing what it is not is flagged, not forbidden."""
    bad = captions.replace(
        "It is a standardized difference-in-differences,\nexpressed in selection-split standard deviations.",
        "It is a standardized difference-in-differences, and it is not an\naccuracy, a logit, or Cohen's $d$.")
    assert bad != captions
    report = run_captions(bad, tmp_path)
    assert "negation-list" in rules(report, "READABILITY")
    assert not [f for f in report.findings if f.gate == "READABILITY" and f.level == pc.ERROR]


def test_saying_what_a_control_rules_out_stays_legal():
    """The rule must not fire on evidence statements, only on statistic definitions."""
    report = pc.Report()
    body = ("So the figure establishes that the effect is not wording, not answer "
            "format, and not an artefact of the board algebra.")
    pc.check_readability("d", "u", body, {}, report)
    assert "negation-list" not in {f.rule for f in report.findings}


def test_ambiguous_reference_is_warned(captions, tmp_path):
    bad = captions.replace("That is what selection on a separate\nsplit should look like.",
                           "The former is what this quantity should look like.")
    assert bad != captions
    report = run_captions(bad, tmp_path)
    assert "ambiguous-reference" in rules(report, "READABILITY")


def test_provenance_in_a_caption_is_warned(captions, tmp_path):
    bad = captions.replace(
        "The dashed red line marks the frozen layer",
        "Rebuilt from artifacts/figures/figure_data.json. The dashed red line marks the frozen layer")
    assert bad != captions
    report = run_captions(bad, tmp_path)
    assert "provenance-in-prose" in rules(report, "READABILITY")


def test_moved_fact_is_required_at_its_destination():
    """Relocating a fact does not drop it: it must appear where it was sent."""
    contract = {
        "numbers": [], "literals": [],
        "required_facts": [{"id": "moved", "description": "detail lives elsewhere",
                            "moved_to": "Appendix", "any_of": ["the detail"]}],
    }
    present = pc.Report()
    pc.check_fidelity("d", "u", "body", contract, {},
                      {"Appendix": "here is the detail"}, present, False)
    assert not present.errors

    absent = pc.Report()
    pc.check_fidelity("d", "u", "body", contract, {},
                      {"Appendix": "nothing relevant"}, absent, False)
    assert "fact-missing" in {f.rule for f in absent.errors}

    missing_destination = pc.Report()
    pc.check_fidelity("d", "u", "body", contract, {}, {}, missing_destination, False)
    assert "fact-destination-missing" in {f.rule for f in missing_destination.errors}


# --------------------------------------------------------------------------
# terminology plumbing and comprehension
# --------------------------------------------------------------------------

def test_terminology_is_its_own_source_of_truth():
    terminology = pc.load_json(pc.TERMINOLOGY)
    assert terminology["source_of_truth"] == "this file"
    assert terminology["seeded_from"] == "docs/RESULTS_AND_CLAIMS.md"


def test_results_is_audited_like_any_other_document():
    ids = {d["id"] for d in pc.documents()}
    assert {"results", "captions"} <= ids


def test_strip_canonical_protects_longer_names():
    terminology = pc.load_json(pc.TERMINOLOGY)
    assert "test set" not in pc.strip_canonical("the 7,394-row strict test set", terminology)


def test_symbol_must_be_introduced_with_its_name():
    terminology = pc.load_json(pc.TERMINOLOGY)
    report = pc.Report()
    pc.check_symbol_order("d", "A value of $I_b$ appeared with no name.", terminology, report)
    assert "symbol-undefined" in {f.rule for f in report.findings}


def test_subscripts_are_not_reported_values():
    assert pc.number_tokens("$I_b=\\tilde S_{11}-\\tilde S_{12}$") == []
    assert "0.0132" in pc.number_tokens("p = 0.0132")


def test_blind_prompt_excludes_gold_answers():
    spec = pc.load_json(ROOT / "paper" / "comprehension" / "results" / "12_section.json")
    bundle = pc.build_prompt("12.", "passage text", spec["questions"])
    serialized = json.dumps(bundle)
    assert "gold" not in serialized and "accept_patterns" not in serialized
    assert len(bundle["questions"]) == len(spec["questions"])


def test_grade_requires_every_critical_question():
    questions = [
        {"id": "a", "critical": True, "accept_patterns": ["one-sided"]},
        {"id": "b", "critical": False, "accept_patterns": ["125"]},
    ]
    assert pc.grade(questions, {"a": "it is one-sided", "b": "125 boards"})["critical_pass"]
    assert not pc.grade(questions, {"a": "not stated", "b": "125"})["critical_pass"]
    partial = pc.grade(questions, {"a": "one-sided", "b": "NOT STATED"})
    assert partial["critical_pass"] and partial["score"] == 0.5


def test_unavailable_reader_refuses_rather_than_faking():
    with pytest.raises(RuntimeError):
        pc.UnavailableReader().answer("passage", [])


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def test_shipped_documents_render():
    report, _ = pc.run()
    rendering = [f for f in report.findings if f.gate == "RENDERING"]
    assert not rendering, "; ".join(str(f) for f in rendering)


def test_blocked_math_macro_is_an_error():
    """A macro the renderer rejects leaves an error box where a formula should be."""
    report = pc.Report()
    pc.check_rendering("d", r"the scale is $$s=\operatorname{median}(x)$$ here", report)
    assert "blocked-macro" in {f.rule for f in report.errors}


def test_inline_math_across_a_line_break_is_an_error():
    report = pc.Report()
    wrapped = "a formula $I_b = a -\nb$ wrapped over two lines"
    pc.check_rendering("d", wrapped, report)
    assert "unbalanced-math" in {f.rule for f in report.errors}


def test_fenced_math_with_allowed_macros_is_clean():
    document = "\n".join([
        "text $x$",
        "",
        "```math",
        r"y=\mathrm{median}(x)",
        "```",
        "",
    ])
    report = pc.Report()
    pc.check_rendering("d", document, report)
    assert not report.errors
