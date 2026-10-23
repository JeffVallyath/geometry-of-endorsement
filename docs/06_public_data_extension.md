# Appendix F.2: RippleEdits-derived secondary extension

<a id="sec:public-data"></a>

Qwen sometimes returns an obsolete value on a composed question after correctly reporting the current facts. The study uses RippleEdits-derived questions and fixed model weights. Earlier records remain explicitly present in the prompt.

## Method

Each case has a shared current record table and two earlier histories. The histories differ only in the obsolete value assigned to one address. They receive identical current records and downstream questions. Clean-reference prompts contain the current table without the earlier obsolete record.

A question qualifies when all current facts are recovered under both histories, both clean references answer correctly, and the history-conditioned answers identify distinct entities with exactly one correct. All 98 scheduled questions across 34 cases remain in the main denominator.

## Result

| Saved-response measure | Primary label-first parser | Alternative union parser |
|---|---:|---:|
| Qualifying history-dependent questions | 3/98 | 2/98 |
| Questions passing factual and clean-reference checks | 65/98 | 42/98 |

The primary witnesses occur in three cases. Every incorrect qualifying answer is the obsolete value associated with that history. Parser sensitivity re-scores the same responses; it adds no inference.

Unrelated-fact accuracy is 170/272 without the history and 94/272 with history B. These are eight prepared questions repeated across 34 contexts. Both preselected repeat cases reproduce their recorded responses exactly, and neither contains a qualifying witness. Positive cases were not independently repeated.

See [construction, parsers, gates, witnesses and repeats](appendix/A11_ripple_accounting.md). The findings concern this restricted benchmark sample, not deployment prevalence.

## Reproduce

```sh
python -m repro.in_context_updates --output reproduced/ripple
```

This offline command checks the public manifests and journal chains. It reconstructs every retained phase under both frozen parsers and compares every scored unit with the original saved analysis. Outputs are `results.json`, `per_root_results.csv` and `public_data_table.md`.

The retained model is `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`. Fresh inference and reconstruction of the broader upstream preparation pool were not verified. See [data and licensing](../DATA_AND_LICENSES.md).
