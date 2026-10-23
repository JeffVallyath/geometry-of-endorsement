# Saved-evidence CPU reproduction

> **Historical version.** This page records an earlier paper and its acceptance
> checks. It is not the status of the current onboarding branch. Use the
> [collaborator README](README.md), [current handoff](HANDOFF.md#objective-and-current-state),
> and [current paper map](history_updates/docs/paper_map.md). Historical records follow.

This release supplies the retained measurements, selected parameters, study
records and CPU analysis interfaces. The documented workflow reconstructs 16
paper analyses, checks the editor interfaces, regenerates all seven paper figures
and verifies the shipped-file inventory before and after the test workflow.

[The paper map](PAPER_MAP.md) follows the manuscript's five main-text figures and
two appendix figures. It connects repository topics A01–A13 to manuscript
Appendices A–H and identifies additional repository-only analyses. Start with the
[measured matched-history example](reproducibility/fresh_history_validation/examples/CASE_0017.md).

The MQuAKE-derived public multi-hop result is integrated as Section 5 and Appendix F.1. Its saved-response reproduction command and test are documented in `REPRODUCIBILITY.md`; it adds no new pretrained inference. RippleEdits is retained as Appendix F.2 supporting evidence.

The supplied Gemma MiniLM record establishes a 107,368-row fitted population and
an encoder revision. Its exact fitted-row membership remains unavailable. See
[training provenance](docs/appendix/A03_text_features.md#training-provenance).

[Reproduction limits](REPRODUCIBILITY.md#reproduction-levels) distinguish saved
scores, aggregate checks and unavailable fresh execution. No new pretrained
inference or complete retraining was performed. No hosting surface or publication
is verified. Anonymizing local identifiers does not guarantee that distinctive
content cannot be matched to another copy.

## Historical validation

These checks apply to the earlier paper, not the current paper's acceptance status.
They used Windows x86-64, Python 3.13.12, the CPU dependency lock and CPU PyTorch 2.8.0.

| Check | Result |
|---|---|
| `repro.paper_results --output reproduced/paper-review` | All 16 steps passed |
| `repro.paper_results --test --output reproduced/tests-review` | 263 passed, 1 skipped |
| `repro.mqmh --output reproduced/mqmh-review` | Both models reconstructed; Qwen 94/384 and Gemma 101/384 qualified targets |
| `repro.main_figures --output reproduced/main-figures-review` | All seven figures generated; all plotted values unchanged |
| Manuscript and clean-reader builds | Both passed with `latexmk -pdf`, pdfTeX 1.40.29 / TeX Live 2026; both 19 pages |
| PDF text and build logs | Rebuilt review PDF's extracted text matches the supplied PDF; no unresolved references, missing characters, font substitutions or overfull boxes in final logs |
| Reader-facing navigation | 330 local file links across 24 Markdown documents resolved; generated paper map matched its source |

The skipped overlap test requires two original prior-study corpora; one is absent,
as already documented. This is not a passed overlap audit. No fresh pretrained
inference, GPU allocation or retraining was performed. The reproduction limits
above remain unchanged.

The runner requires at least 5 GiB free on the output volume before starting.
This is a preflight check, not a disk reservation; concurrent workflows need
additional headroom. The frozen analyses and scientific criteria are unchanged.
