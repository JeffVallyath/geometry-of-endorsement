"""Replay source-independence results from original scene/root sufficient statistics.

This checks the published analyses, not model inference or physical-state identity.
Equations and thresholds follow the retained executed rfr5/rsi6 metrics sources.
Origins, questions and training seeds are never treated as independent scenes.
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict

import numpy as np

from .common import ROOT

DATA = ROOT / 'reproducibility/relational_editing/v6/scores'


def read(name):
    with gzip.open(DATA / name, 'rt', encoding='utf8') as stream:
        return json.load(stream)


def equal(actual, expected, name):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f'Undefined denominator changed: {name}')
    elif isinstance(actual, (bool, np.bool_)) or isinstance(expected, bool):
        if bool(actual) != bool(expected):
            raise ValueError(f'Scientific disposition changed: {name}')
    elif not np.isclose(actual, expected, rtol=0, atol=1e-12):
        raise ValueError(f'Saved-statistic replay differs: {name}: {actual} != {expected}')


def average(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def interval(values, level, seed, draws=10000):
    x = np.asarray(list(values), dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite whole-scene/root statistics required')
    rng = np.random.default_rng(seed)
    means = x[rng.integers(len(x), size=(draws, len(x)))].mean(axis=1)
    alpha = (1 - level) / 2
    return {'effect': float(x.mean()), 'lower': float(np.quantile(means, alpha)),
            'upper': float(np.quantile(means, 1 - alpha))}


def scene_summary(record):
    scenes = record['per_scene']
    if not scenes or len(scenes) != record['n_scenes']:
        raise ValueError('Incomplete scene population')
    result = {}
    for key in ('changed', 'harm', 'conditional_harm', 'all24', 'accuracy',
                'touched_accuracy', 'other_address_error', 'preservation',
                'nonaddressed_change', 'all_program_questions', 'accuracy_all_program_questions'):
        if key in record:
            result[key] = average(v[key] for v in scenes.values())
            equal(result[key], record[key], key)
    directions = {str(d): average(v['changed'] for v in scenes.values() if v['direction'] == d)
                  for d in sorted({v['direction'] for v in scenes.values()})}
    for direction, value in record['changed_by_direction'].items():
        equal(directions[direction], value, 'changed_by_direction')
    result['changed_by_direction'] = directions
    return result


def f1(metrics, native):
    directions = list(metrics['changed_by_direction'].values())
    return (metrics['changed'] is not None and metrics['changed'] >= .85
            and metrics['harm'] <= .05 and len(directions) == 2
            and all(v is not None and v >= .8 for v in directions)
            and metrics['all24'] >= .7 * native['all24'])


def replay_v6():
    a = read('reader_scene_statistics.json.gz')
    c = read('program_scene_statistics.json.gz')
    expected_programs = {'single', 'noop', 'repeat2', 'repeat4', 'repeat8', 'restore',
                         'restore_after4', 'overwrite', 'AB', 'BA', 'ABC', 'CBA', 'AB_cross'}
    if len(c) != 702 or {r['program'] for r in c} != expected_programs:
        raise ValueError('Full program/control inventory missing')
    family_tables = []
    for panel, records in (('A', a), ('C', c)):
        for r in records:
            summary = scene_summary(r)
            family_tables.append({**{k: r[k] for k in ('actor', 'condition', 'reader', 'program', 'order')},
                                  'panel': panel, **summary})

    primaries = []
    for r in read('reader_paired_contrasts.json.gz'):
        result = interval(r['per_scene'].values(), r['level'], r['bootstrap_seed'], r['draws'])
        for key, value in result.items():
            equal(value, r[key], 'reader ' + key)
        primaries.append({**{k: r[k] for k in ('actor', 'architecture', 'reader', 'level')},
                          'panel': 'A', **result})

    source = read('source_root_statistics.json.gz')
    grouped = defaultdict(list)
    for row in source['per_root']:
        grouped[(row['actor'], row['condition'], row['reader'])].append(row)
    source_tables = []
    for record in source['absolute']:
        key = tuple(record[k] for k in ('actor', 'condition', 'reader'))
        rows = grouped[key]
        if len(rows) != record['roots'] or len({r['root_id'] for r in rows}) != len(rows):
            raise ValueError('Incomplete or duplicate root population')
        output = dict(zip(('actor', 'condition', 'reader'), key))
        for bundle in ('all34', 'original24'):
            output[bundle] = {}
            for metric, expected in record[bundle].items():
                result = interval([r[bundle][metric] for r in sorted(rows, key=lambda r: r['root_id'])], .95, 260912602)
                for name, value in result.items():
                    equal(value, expected[name], 'source ' + metric + ' ' + name)
                output[bundle][metric] = result['effect']
        stats = output['all34']
        passed = (stats['all_origin_question_correct'] >= .9 and stats['origin_disagreement'] <= .05
                  and stats['all_origin_bundle_correct'] >= .7)
        equal(passed, record['prospective_operating_point'], 'source independence')
        output['pass'] = passed
        source_tables.append(output)
    for r in source['primary']:
        result = interval(r['per_root'].values(), r['level'], 260912602, r['draws'])
        for name, value in result.items():
            equal(value, r[name], 'source reader contrast ' + name)
        primaries.append({**{k: r[k] for k in ('actor', 'architecture', 'reader', 'level')},
                          'panel': 'B', **result})

    claims = read('functional_decisions.json.gz')
    if claims['completed_panels'] != ['A', 'B', 'C'] or claims['missing_panels']:
        raise ValueError('Not a terminal completed scientific package')
    if claims['historical_V5'] != 'PREPARED_NO_EFFICACY; zero historical fresh outcomes':
        raise ValueError('Prepared predecessor must not acquire efficacy outcomes')
    functional = []
    for record in claims['functional_progress']:
        seeds = {}
        if set(record['seeds']) != {'0', '1'}:
            raise ValueError('Both original training seeds are required')
        for seed, r in record['seeds'].items():
            main = scene_summary(r['main_metrics'])
            main_native = scene_summary(r['main_native_metrics'])
            single = scene_summary(r['program_SINGLE_metrics'])
            native = scene_summary(r['program_SINGLE_native_metrics'])
            main_pass, single_pass = f1(main, main_native), f1(single, native)
            equal(main_pass, r['main_F1'], 'main single edit')
            equal(single_pass, r['program_SINGLE_F1_quality'], 'subset single edit')
            repeat_passes, restore_passes = [], []
            repeats, restorations = [], []
            for category, metric in (('repeat', 'disagreement'), ('restoration', 'preservation')):
                for program, endpoint in r[category].items():
                    vals = [v[metric] for v in endpoint['per_scene'] if v[metric] is not None]
                    result = interval(vals, .95, 260912502)
                    for name, value in result.items():
                        equal(value, endpoint['interval'][name], program + ' ' + name)
                    passed = result['effect'] <= .05 if category == 'repeat' else result['effect'] >= .95
                    equal(passed, endpoint['pass'], program)
                    (repeat_passes if category == 'repeat' else restore_passes).append(passed)
                    (repeats if category == 'repeat' else restorations).append(result['effect'])
            joint_passes = []
            joint = {}
            for program, endpoint in r['joint'].items():
                stat = scene_summary(endpoint['metrics'])
                passed = stat['changed'] >= .8 and stat['harm'] <= .05
                equal(passed, endpoint['pass'], program)
                joint_passes.append(passed)
                joint[program] = stat
            success = bool(main_pass and single_pass and all(repeat_passes) and all(restore_passes))
            full = bool(success and all(joint_passes))
            equal(success, r['functional_F1_repeat_restoration'], 'functional conjunction')
            equal(full, r['G2_full_functional_conjunction'], 'full joint conjunction')
            seeds[seed] = {'single_changed': single['changed'], 'single_harm': single['harm'],
                           'weaker_direction': min(single['changed_by_direction'].values()),
                           'max_repeat_disagreement': max(repeats), 'min_restoration': min(restorations),
                           'ABC_changed': joint['ABC']['changed'], 'ABC_harm': joint['ABC']['harm'],
                           'functional_pass': success, 'joint_pass': full}
        two_functional = all(v['functional_pass'] for v in seeds.values())
        two_joint = all(v['joint_pass'] for v in seeds.values())
        equal(two_functional, record['two_seed_functional_replication'], 'both-seed functional')
        equal(two_joint, record['two_seed_G2'], 'both-seed joint')
        functional.append({**{k: record[k] for k in ('actor', 'architecture', 'reader')},
                           'seeds': seeds, 'two_seed_functional': two_functional, 'two_seed_joint': two_joint})
    return {'primary_contrasts': primaries, 'source_independence': source_tables,
            'functional': functional, 'program_tables': family_tables,
            'reproduction_level': 'replayed from committed scene/root sufficient statistics'}
