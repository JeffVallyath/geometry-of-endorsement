# Figure presentation update, 2026-09-20

The [current gallery](../../figures/redesigned/README.md) publishes 16 reviewed
redesigns covering 20 earlier exports. Numbered PDF/PNG paths are compatibility
copies of these assets, with merged figures made explicit in the
[mapping](../../figures/redesigned/old_to_new.csv). Original source SVGs inside
experiment archives and all scientific evidence remain unchanged.
The separately guarded original supplementary source-history rendering is
preserved in the [archive](../../figures/archive/20260919/README.md). Its original
hard-coded checksum tests now point there; both current supplementary paths
display the redesign.

The combined review, contact sheet, captions, editable SVGs, high-resolution
previews and grayscale previews are included. The viewing ZIP exporter validates
the manifest before packaging and refuses to overwrite an existing ZIP.
Historical CPU plotting remains a separate, explicitly documented interface;
this publication change does not rerun scientific analysis or claim that those
historical renderers generate the new layout.

## Presentation incident and regression guard

The initial staged whitespace check rejected Matplotlib's trailing spaces inside
generated SVG path data. Reformatting these already-reviewed exports would break
their exact source identities. A narrowly scoped `figures/.gitattributes` rule
therefore permits that formatting only for `redesigned/*.svg`; it does not
disable whitespace checking for source or documentation.

`tests/test_redesigned_figures.py` verifies all presentation checksums, every
compatibility copy, gallery targets and ZIP contents, including refusal to
overwrite an existing archive. The staged whitespace check remains a separate
acceptance gate. No scientific numbers or frozen source hashes were changed to
resolve the presentation check.

The claim checker also rejected the changed archival color label and two new
figure-number references outside their number contracts. The references were
rewritten as labels. Only the presentation color phrase in the caption registry
changed; numerical bindings, denominators and scientific requirements did not.
A caption corruption test now rejects assigning the orange segments to failed
fact checks, preserving the original interpretation guard under the new palette.
