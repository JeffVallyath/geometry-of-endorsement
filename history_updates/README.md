# When Apparent Updates Still Depend on History

For the Git checkout, start with the [README](../README.md),
[examples](../examples/README.md) and [code map](../CONTRIBUTING.md#code-navigation).
The [replay guide](docs/reproduction.md) covers the separate software/data archives
and the commands to run their analyses from the Git checkout.

Can answers still depend on earlier facts after a model reports every updated
fact correctly? This package reconstructs the paper's matched-history tests,
cache interventions, external-state repair and interpreter comparisons from
the recorded observations.

Full replay with `--all` includes both interpreter studies, the synthetic repair
bridge and corrected GMeLLo. Use `--section NAME` for one part:

| Question | Paper location | Section |
|---|---|---|
| Do equal final facts yield history-dependent answers? | Sections 2-3; Appendices A-C | matched_histories |
| Which cache changes transfer or repair the effect? | Section 4; Appendix D | cache_interventions |
| Does an isolated current dictionary support correct answers? | Sections 5.1-5.3; Appendices E-F | external_state |
| What changes when the reader or its history access changes? | Section 5.4; Appendix G | interpreter_checks |
| How does the corrected local-graph comparison perform? | Appendix G; Table 13 | gmello_comparison |

Find individual figures, tables and numerical results in the
[paper map](docs/paper_map.md). Read the [methods](docs/methods.md) for inputs
and metrics, and the [design notes](docs/design_notes.md) for selection and scope.

The paper map follows the final paper with 13 tables. Result filenames are
stable identifiers, not the current paper's table numbers. For example, paper
Table 13 combines table_11 with gmello_corrected; gmello_contrasts contains its
two primary comparisons. The superseded GMeLLo observations are retained only
in legacy_gmello and are not the paper's current GMeLLo results.

This is saved-output reproduction, not a complete fresh-model rerun. Base-model
weights and the complete trained-adapter set are not bundled; see
[reproduction.md](docs/reproduction.md) for the scope and environment setup.
