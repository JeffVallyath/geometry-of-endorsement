"""Evaluator-only joining, independent truth, exact grouping and fixed contrasts."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from canonical.origins import truth, paired_interval
from canonical.metric_core import origin_rates, root_statistics, program_retention
from canonical.program_metrics import summarize
from canonical.operand_metrics import recomposition
from inputs import RECIPES

BASELINE='FIELD_PLUS_LATEST_ERRATUM'

def label_answer(answer, evaluation_record, source_answer):
    """Join only after serving; SOURCE remains the original shared interface."""
    if answer['question']!=evaluation_record['text'] or source_answer['question']!=evaluation_record['text']:
        raise ValueError('Question strings changed across methods or SOURCE')
    return dict(evaluation_record,**{k:answer[k] for k in ('prediction','logps','valid','answer_mass','argmax_in_labels','tie')},
                suffix_ids=answer.get('suffix_ids'),label_token_ids=answer.get('label_token_ids'),
                source_prediction=source_answer['prediction'],source_valid=source_answer['valid'],
                source_logps=source_answer['logps'],correct=bool(answer['valid'] and answer['prediction']==evaluation_record['gold']))

def score_case(context, evaluation_records, source_answers, *, backbone, seed=None):
    """Turn one fixed case into metric-ready rows, only AFTER compilation.

    source_answers is a query_id -> scored SOURCE-answer dictionary produced by
    the original interface, never passed to the updater. For SOURCE itself pass
    None. Compilation must already be complete before this function is called.
    """
    from serve_questions import serve_questions
    if not evaluation_records or len({r['input_id'] for r in evaluation_records})!=1:
        raise ValueError('Exactly one supplied fixed case required')
    if len({r['query_id'] for r in evaluation_records})!=len(evaluation_records):
        raise ValueError('Duplicate supplied question')
    if {r['actor'] for r in evaluation_records}!={backbone.cfg['key']}:
        raise ValueError('Evaluation model differs from compiled backbone')
    public=[dict(text=r['text'],labels=r['labels']) for r in evaluation_records]
    answers=serve_questions(context,public,backbone=backbone)
    if context.method=='SOURCE':
        source_answers={r['query_id']:a for r,a in zip(evaluation_records,answers,strict=True)}
    if source_answers is None or set(source_answers)!={r['query_id'] for r in evaluation_records}:
        raise ValueError('Original same-interface SOURCE answers required for every question')
    learned=context.method in RECIPES or context.method=='CANONICAL_REPLAY'
    if learned and seed not in (0,1) or not learned and seed is not None:
        raise ValueError('Original writer seeds only; deterministic baselines have seed=None')
    if learned:
        from inputs import inventory
        inv=inventory();actor=backbone.cfg['key']
        recipe=inv['canonical_replay'][actor][str(seed)] if context.method=='CANONICAL_REPLAY' else context.method
        expected=next(r['setter_sha256'] for r in inv['selected'] if (r['actor'],r['recipe'],r['seed'])==(actor,recipe,seed))
        if expected!=context.selected_weight_sha256:
            raise ValueError('Scoring seed/recipe differs from compiled editor')
    condition=context.method+(f'_s{seed}' if learned else '')
    result=[]
    for r,a in zip(evaluation_records,answers,strict=True):
        joined=label_answer(a,r,source_answers[r['query_id']])
        joined.update(condition=condition,seed=seed,reader='R0',
            input_panel=r['panel'],panel={'MAIN':'A','ORIGIN':'B','PROGRAM':'C','DEV':'DEV'}[r['panel']],
            artifact_sha256=context.artifact_sha256,master_unchanged=a['master_unchanged'],
            physical_batch=a['physical_batch'],physical_slot=a['physical_slot'],
            batch_size=a['batch_size'],batch_seconds=a['batch_seconds'])
        result.append(joined)
    return result

def paired_primary(vectors, expected_roots):
    """vectors[(actor,recipe,seed)] -> {root: complete-all-origin indicator}.

    Baseline has seed=None and is subtracted ONCE after within-root seed mean.
    Missing roots, duplicate-baseline seeds or incomplete contrasts fail closed.
    """
    expected={(a,r,s) for a in ('gemma','qwen') for r in RECIPES for s in (0,1)}
    expected|={(a,BASELINE,None) for a in ('gemma','qwen')}
    if set(vectors)!=expected:
        raise ValueError('Exactly four contrasts, both seeds, and one baseline per model required')
    out=[]
    for actor in ('gemma','qwen'):
        roots=set(expected_roots[actor])
        if len(roots)!=64:
            raise ValueError('64 terminal roots per model required')
        for k,v in vectors.items():
            if k[0]==actor and (set(v)!=roots or any(x not in (0,1) for x in v.values())):
                raise ValueError('Missing/extra roots or non-binary whole-bundle endpoint')
        baseline=vectors[(actor,BASELINE,None)]
        for recipe in RECIPES:
            seed_d={s:{r:vectors[(actor,recipe,s)][r]-baseline[r] for r in sorted(roots)} for s in (0,1)}
            differences={r:(seed_d[0][r]+seed_d[1][r])/2 for r in sorted(roots)}
            out.append(dict(actor=actor,recipe=recipe,baseline=BASELINE,
                endpoint='complete 34-question bundle correct from all four origins',
                unit='terminal root; seed mean inside each root; baseline once',
                bootstrap_seed=260913902,per_root=differences,
                per_seed={str(s):paired_interval(list(seed_d[s].values()),level=.9875,draws=10000,seed=260913902) for s in (0,1)},
                **paired_interval(list(differences.values()),level=.9875,draws=10000,seed=260913902)))
    return out

def scene_summary(rows, scenes):
    """Canonical all24/coverage, changed accuracy and invariant-harm denominators."""
    view=[]
    for r in rows:
        s=scenes[r['scene_id']]; a,p=s['target_actor'],s['target_project']; spec=r['spec']
        view.append(dict(r,prediction=r['prediction'] if r['valid'] else -1,
            source_prediction=r['source_prediction'] if r['source_valid'] else -1,
            direction=1-s['values'][a][p],cell=[len(s['actors']),len(s['projects']),s['values'][a][p]],
            touches_target=spec['a']==a and spec['p']==p,
            mentions_target=(spec['a']==a or spec.get('b')==a) and spec['p']==p))
    grouped=defaultdict(list)
    for r in view:
        grouped[(r['actor'],r['condition'],r['program'])].append(r)
    return {k:summarize(v) for k,v in grouped.items()}

def fixed_criteria(main, native_main, program, native_program, retention):
    """Inherited point criteria, not a new comparative F2 or deployment claim."""
    def f1(m,n):
        return (m['changed'] is not None and m['changed']>=.85 and m['harm'] is not None and m['harm']<=.05
                and len(m['changed_by_direction'])==2
                and all(v is not None and v>=.8 for v in m['changed_by_direction'].values())
                and m['all24']>=.7*n['all24'])
    names={'repeat2','repeat4','repeat8','restore','restore_after4'}
    if set(retention)!=names or not {'single','AB','ABC'}<=set(program):
        raise ValueError('Incomplete fixed program criteria')
    repeats={k:retention[k]['disagreement'] is not None and retention[k]['disagreement']<=.05 for k in names if k.startswith('repeat')}
    restores={k:retention[k]['preservation'] is not None and retention[k]['preservation']>=.95 for k in names if k.startswith('restore')}
    joint={k:program[k]['changed'] is not None and program[k]['changed']>=.8 and program[k]['harm'] is not None and program[k]['harm']<=.05 for k in ('AB','ABC')}
    return dict(main_F1=f1(main,native_main),program_single_F1=f1(program['single'],native_program['single']),repeat=repeats,restoration=restores,joint=joint)
