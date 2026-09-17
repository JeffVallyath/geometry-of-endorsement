"""Saved-response regression guards, including absence/corruption and sensitivity."""
import copy
import gzip
import importlib.util
import json
import sys

import pytest

from repro import in_context_updates as q


@pytest.fixture(scope='module')
def result():
    return q.replay()


def test_all_headlines_and_adverse_results(result):
    assert result['primary']['denominators'] == {'roots': 34, 'questions': 98}
    assert result['primary']['counts']['W_questions'] == 3
    assert result['primary']['counts']['A_and_C_questions'] == 65
    assert result['alternative']['counts']['W_questions'] == 2
    assert result['alternative']['counts']['A_and_C_questions'] == 42
    assert result['primary']['counts']['roots_with_any_W'] == 3
    assert result['all_wrong_witness_answers_are_obsolete']
    assert result['primary']['locality_panel_diagnostic'] == {
        'none': {'measured': 272, 'correct': 170},
        'chronological_B': {'measured': 272, 'correct': 94}}
    assert result['locality_distinct_questions'] == 8
    assert result['parser_changed_questions'] == 24
    assert result['repeat_roots'] == 2 and result['repeat_units'] == 86
    assert result['repeat_witnesses'] == 0 and result['all_repeat_answers_identical']
    assert all(x['planned'] == x['measured'] for x in result['phase_coverage'].values())
    assert result['qualification']['primary']['G4']['pass']
    assert not result['qualification']['alternative']['G4']['pass']
    assert not result['qualification']['alternative']['G5']['pass']


def test_generated_projection_is_exact(result):
    assert result == json.loads((q.BASE / 'results.json').read_text(encoding='utf8'))


def test_frozen_parser_collision_and_strict_rules():
    analyzer = q.frozen_modules(q.BASE, 'source')
    parser = sys.modules[analyzer.__package__ + '.parse']
    entities = {'a': {'types': ['place'], 'labels': ['X'], 'aliases': []},
                'b': {'types': ['place'], 'labels': ['Y'], 'aliases': ['X']}}
    primary, union = parser.build_registry(entities), parser.build_registry_union(entities)
    assert parser.extract('X.', primary, 'place')['entity_id'] == 'a'
    assert parser.extract('X.', union, 'place')['status'] == 'ambiguous'
    assert parser.extract('The answer is X', primary, 'place')['status'] == 'unmatched'
    assert not parser.strict_correct('x', [{'value': 'X'}])


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'chain'])
def test_journal_corruption_fails_closed(tmp_path, mutation):
    analyzer = q.frozen_modules(q.BASE, 'source')
    phase = 'repeat_670'
    original = gzip.decompress((q.BASE / 'supplied/journals' / phase / 'journal.jsonl.gz').read_bytes())
    path = tmp_path / 'journal.jsonl'
    if mutation == 'chain':
        path.write_bytes(original.replace(b'"event":"generation"', b'"event":"corruption"', 1))
    else:
        # Re-chain so the coverage guard, not the byte/chain guard, must reject it.
        records = [json.loads(line)['payload'] for line in original.splitlines()]
        generation = next(r for r in records if r['event'] == 'generation')
        if mutation == 'missing':
            records.remove(generation)
        else:
            extra = copy.deepcopy(generation)
            extra['unit_id'] = 'undeclared-unit'
            records.insert(-1, extra)
        journal = analyzer.Journal(path)
        for record in records:
            journal.append(record)
    registry = q.load_json(q.BASE / 'supplied/inputs/built/REGISTRY.json')['registry']
    with pytest.raises(Exception, match='hash mismatch|Missing or extra'):
        q.reconstruct_phase(q.BASE, phase, analyzer, registry, path)


def test_output_never_overwrites(tmp_path):
    q.export(tmp_path / 'out')
    with pytest.raises(FileExistsError):
        q.export(tmp_path / 'out')
