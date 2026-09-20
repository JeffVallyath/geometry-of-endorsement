# CPU portability and provenance notes

The source archives, manifests, raw files and supplied checkers were left
unchanged. The adapter adds integrity checks, fresh output directories,
path redirection, an explicit worked-example check and readable plot labels.
It does not change the experimental definitions or choose new cases.

## Derived-output serialization

The original repeat checker writes regenerated summaries inside its runtime.
When run against restored records, its immutable writer rejected four derived
candidate-mass values differing by one floating-point step, at most about
`1.1e-16`. Those values are recalculated from identical original log probabilities.

The adapter now redirects only its derived repeat outputs to the fresh output
directory. Original raw bytes remain hash-checked. All classifications, counts,
identities and the repeat disposition require exact equality. Finite derived
summaries retain the repository's `1e-12` absolute, zero-relative comparison bound.
The complete primary summary and intervals reproduced exactly in this environment.

## Timing metadata is not a score

An initial example check compared the whole score record, including elapsed
execution time. The recorded repeats naturally have different durations. The
corrected check excludes only `seconds` and requires every other score field and
all token bindings to match exactly. Candidate log-probability differences are zero.

The example checker resolves actor and project indices through the frozen design's
name lists. The figure adapter converts NumPy bar widths to ordinary JSON numbers;
this changes no plotted value. The regression suite covers these boundaries and
rejects changed classifications, missing repeats, modified source bytes and
pre-existing output directories.

These were local integration checks, not failures of the measured study and not
scientific retries. No inference, training, paid service or live outcome access
was used for recovery.
