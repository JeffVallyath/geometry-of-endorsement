# Proposed fix only

Change only the regenerated-SVG comparison to compare CRLF-normalized bytes. Keep all archive and manifest integrity checks byte-exact. Do not modify the frozen package, source capsule, figures or manifests.

The original generator writes SVG using platform-default text newline handling. The replay then compares the regenerated raw digest with the archived raw digest. The frozen SVGs contain CRLF. Native Windows regeneration matches them. LF serialization changes their hashes without changing their content. [Original replay](inputs/0d968769fb62335991d365086ed50902bd2a47dfe5bb6881e87652d36d8047a8/tools/replay_bundle.py), [generator](inputs/0d968769fb62335991d365086ed50902bd2a47dfe5bb6881e87652d36d8047a8/tools/screen_figures.py).

For a future, separately versioned replay helper, replace the existing figure loop with:

```python
for name in regenerated["figures"]:
    expected = (bundle / "figures" / name).read_bytes()
    actual = (work / "figures" / name).read_bytes()
    if expected.replace(b"\r\n", b"\n") != actual.replace(b"\r\n", b"\n"):
        raise ValueError("Figure regeneration differs")
```

Rename its receipt field to `figures_canonical_lf_identical`, so it does not claim raw byte identity. The package's earlier raw hash and length checks stay unchanged. Normalization applies only to the known SVG regeneration test, not data, tensor, capsule or source verification. No XML whitespace stripping, rounding or numeric tolerance is needed.

The retained regression check establishes that CRLF/LF variants compare equal, both original raw hashes remain intact, and a changed plotted number compares unequal. The independent Linux failure log is preserved. This audit ran the counterfactual serialization check on Windows, not a new Linux process. [Portability receipt](checks/SVG_PORTABILITY.json), [Linux traceback](inputs/cb45eed6be974dadc3badf9e174319cff88c59ab6afb552c0b4cfb5c7962b26e/checks/supplied_replay.log).

Making a future generator explicitly write LF would improve deterministic output, but alone would still disagree with the existing CRLF-frozen figures. It is not required for the minimal comparator repair. No proposed change has been applied.
