# A repeated 32B example, limited to the Yes/No interface

## Result

Qwen3-32B correctly reports that Mina and Wren oppose Gallery. It also reports that they take the same side and that neither supports it. Nevertheless, after one starting history it says both support it. The target difference reproduces exactly in the already-completed full-case repeat.

This example was selected for exposition after the outcomes. The underlying single-format control analysis and full-case repeat-selection rule were fixed beforehand. It is not a new observation or a full cross-format success.

## Exact binding

- Archive: `QWEN3_32B_FULL_FORWARD_RESULTS.zip`.
- SHA256: `5777db17c80cb0145207c447a0303d395f309e045f17f4b2e6d84157a294fca3`.
- Model: `Qwen/Qwen3-32B`, revision `9216db5781bf21249d130ec9da846c4624c16137`, `enable_thinking=False`.
- Design: `SSC1-FINAL-0003`.
- Procedure: `LATEST_SAME_WORDING`.
- Target: `SSC1-FINAL-0003-d0-09`.
- Compared origins: 1 and 2. Each changes one designated fact; each prefix has 834 tokens.
- Each read processes the complete prompt independently, with no supplied or returned cache.

## Saved answer comparison

| Independently asked question | Correct answer | Origin 1 after correction | Origin 2 after correction |
|---|---|---|---|
| Does Wren favor the Gallery proposal? | No | No | No |
| Does Mina favor the Gallery proposal? | No | No | No |
| Do Mina and Wren take the same side on Gallery? | Yes | Yes | Yes |
| Does at least one of Mina and Wren support Gallery? | No | No | No |
| Is the Gallery proposal supported by both Mina and Wren? | No | No | Yes |

The target's normalized correct-answer score is 0.9983893724 under origin 1 and 0.04365346482 under origin 2. Candidate masses are 0.9999828285 and 0.9999966971. These are answer-interface scores, not calibrated beliefs.

All 12 direct checks are correct across all eight histories, including both answer formats. The eight other semantic joint questions are correct across all histories in the Yes/No format. Native and same-procedure no-op target references are correct in that format.

The second answer format does not qualify: its no-op reference already answers this target incorrectly. Its selected origins also both answer Yes. The full cross-format logical-control result therefore remains zero.

Origin 1 originally has Mina supporting Gallery; origin 2 originally has Ivo supporting it. Wren opposes it in both. The old truth of the displayed target ('both Mina and Wren support Gallery') is No in both origins. Thus this example establishes dependence on history, not that the model literally returns the old target truth.

## Repetition and reproduction

The case was already among the four full-case repeats. Every score/token binding in its 578-read block matches the main run in the independent comparison. Do not rent compute to repeat it again.

Use `independent_check.py --source <extracted-original-archive> --output <fresh-check-directory>` for the saved-output verification. Original raw records are in `runtime/results/qwen32_t4/final/SSC1-FINAL-0003/RAW_SCORES.jsonl`, with the repeat under `repeat_full/`. Questions and terminal facts are in `runtime/cpu/designs/SSC1-FINAL-0003.json`. The inherited coded analysis is in `runtime/analysis/cases/SSC1-FINAL-0003.json`.
