from pathlib import Path

import pytest

from repro.common import ROOT, new_output, safe_path


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a\\b", "a/../b", "x:y", "a//b"])
def test_artifact_paths_reject_escape_and_noncanonical_forms(name):
    with pytest.raises(ValueError):
        safe_path(name)


def test_replay_cannot_overwrite_evidence():
    for folder in ("artifacts", "reproducibility", "figures", "src", "tests", "docs"):
        with pytest.raises(ValueError):
            new_output(ROOT / folder / "replacement")


def test_normal_artifact_path():
    assert safe_path("artifacts/figures/figure_data.json").is_file()


def test_v6_replays_original_statistics_and_negative_conjunctions():
    from repro.relational import replay_v6
    result = replay_v6()
    assert len(result['primary_contrasts']) == 12
    assert not any(row['lower'] > 0 for row in result['primary_contrasts'])
    assert len(result['source_independence']) == 54
    assert not any(row['pass'] for row in result['source_independence'])
    assert len(result['program_tables']) == 1014
    assert not any(row['two_seed_joint'] for row in result['functional'])
    constrained = [r for r in result['functional'] if r['architecture'] == 'INV_COMPLETE_SINGLE' and r['reader'] == 'R0']
    assert {r['actor'] for r in constrained} == {'gemma', 'qwen'}
    assert all(r['two_seed_functional'] for r in constrained)


def test_v6_fails_closed_on_missing_seed(monkeypatch):
    import repro.relational as r
    original = r.read
    def altered(name):
        data = original(name)
        if name == 'functional_decisions.json.gz':
            del data['functional_progress'][0]['seeds']['0']
        return data
    monkeypatch.setattr(r, 'read', altered)
    with pytest.raises(ValueError, match='Both original training seeds'):
        r.replay_v6()


def test_v6_rejects_changed_canonical_number():
    from repro.relational import scene_summary, read
    row = read('program_scene_statistics.json.gz')[0]
    row['changed'] += .01
    with pytest.raises(ValueError, match='replay differs'):
        scene_summary(row)


def test_selected_checkpoint_metadata_matches_study_selection():
    import re
    from repro.common import load_json, sha256
    for version in ('v1', 'v2', 'v3', 'v4'):
        records = load_json(ROOT / f'reproducibility/relational_editing/{version}/checkpoints.json')['checkpoints']
        assert records
        for row in records:
            path = safe_path(row['path'])
            assert path.stat().st_size == row['bytes']
            assert sha256(path) == row['sha256']
            epoch = re.search(r'/epoch(\d+)/', row['selection_identity'])
            if epoch and 'epoch' in row['metadata']:
                assert row['metadata']['epoch'] == int(epoch[1]), row['selection_identity']


def test_frozen_qualifications_are_required_by_evidence_registry():
    from repro.common import load_json
    expected = {'specificity': {'equal-distance-mixed'},
                'full_state': {'retrospective-confound'},
                'counterfactual': {'frozen-qualification', 'witness-boundary'}}
    registry = load_json(ROOT / 'reproducibility/evidence.json')
    document = next(d for d in registry['documents'] if d['id'] == 'results')
    for name, identities in expected.items():
        unit = next(u for u in document['units'] if u['id'] == name)
        assert identities <= {r['id'] for r in unit['facts']}


@pytest.mark.parametrize('mutation', ['content', 'missing', 'reference', 'model', 'source'])
def test_manifest_verifier_fails_closed(tmp_path, mutation):
    import hashlib
    import json
    from repro.verify import verify
    folder = tmp_path / 'reproducibility'
    folder.mkdir()
    data = tmp_path / 'score.json'
    data.write_text('{}', encoding='utf8')
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    source = tmp_path / 'source.json'
    source.write_text(json.dumps({'files': [{'path': 'score.json', 'sha256': digest}]}), encoding='utf8')
    manifest = {'schema_version': 1,
                'files': [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in (data, source)],
                'experiments': [{'id': 'fixture', 'artifact_paths': ['score.json'], 'models': [{'id': 'publisher/model', 'revision': 'a' * 40}]}],
                'source_manifests': ['source.json']}
    (folder / 'manifest.json').write_text(json.dumps(manifest), encoding='utf8')
    assert verify(tmp_path)['files'] == 2
    if mutation == 'content':
        data.write_text('{"changed": true}', encoding='utf8')
    elif mutation == 'missing':
        data.unlink()
    elif mutation == 'reference':
        manifest['experiments'][0]['artifact_paths'] = ['absent.json']
    elif mutation == 'model':
        manifest['experiments'][0]['models'][0]['revision'] = 'main'
    else:
        manifest['source_manifests'] = ['absent.json']
    (folder / 'manifest.json').write_text(json.dumps(manifest), encoding='utf8')
    with pytest.raises(ValueError):
        verify(tmp_path)
