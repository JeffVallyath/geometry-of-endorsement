"""Scientific retained-report integrity helper; billing and scheduling omitted."""
from __future__ import annotations
from pathlib import Path, PurePosixPath
import rsi6_common as S

SCHEMA = "RSI6_C_ONLY_CONTINUATION_V1"

def retained_reports(root, expected_manifest_sha256):
    """Read only a hash-bound, fully verified A/B report set, never model again."""
    root = Path(root)
    if S.C.sha(root / 'MANIFEST.json') != expected_manifest_sha256:
        raise ValueError('Prior reports manifest changed')
    manifest = S.C.load(root / 'MANIFEST.json')
    required = {'CLAIMS_AND_PROGRESS.json', 'INDEPENDENT_ARTIFACT_CHECKS.json',
                'INPUT_PROVENANCE.json'}
    if not required.issubset(manifest['files']):
        raise ValueError('Retained claims and provenance must be declared in the manifest')
    for name, digest in manifest['files'].items():
        relative = PurePosixPath(name)
        path = root / name
        if (relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name
                or relative.as_posix() != name or path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve()) or S.C.sha(path) != digest):
            raise ValueError('Retained report member path/hash mismatch')
    claims = S.C.load(root / 'CLAIMS_AND_PROGRESS.json')
    checks = S.C.load(root / 'INDEPENDENT_ARTIFACT_CHECKS.json')
    if (claims.get('completed_panels') != ['A', 'B']
            or set(claims.get('missing_panels', {})) != {'C'} or set(checks) != {'A', 'B'}
            or any(checks[p].get('status') != 'EXACT_RAW_PHYSICAL_CUSTODY_AND_COVERAGE_PASS'
                   for p in ('A', 'B'))):
        raise ValueError('Only complete, verified A/B may be retained for C')
    return {'root': root, 'manifest': manifest, 'manifest_sha256': expected_manifest_sha256,
            'claims': claims, 'checks': checks,
            'provenance': S.C.load(root / 'INPUT_PROVENANCE.json')}
