"""Fresh fixed input populations using the unchanged V7 candidate generator.

This is model-free orchestration of the supplied scientific generators, not a
replacement benchmark. Labels and terminal worlds are emitted separately.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from canonical import candidate, origins, readers, tasks
from canonical.scenes import scene_from_dict
from inputs import AddressedUpdate, digest

ROOT = Path(__file__).resolve().parent
NAMESPACE = 'UPDATE_COMPARISON'
SEED = 260913901

def stable(x):
    return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, x):
    with path.open('x',encoding='utf8',newline='\n') as f:
        f.write(json.dumps(x,sort_keys=True,indent=2)+'\n')

def write_rows(path,rows):
    with path.open('x',encoding='utf8',newline='\n') as f:
        for row in rows:
            f.write(stable(row)+'\n')

def cell(scene):
    a,p=scene.target_actor,scene.target_project
    bits=sum(scene.values[(a+j)%len(scene.actors)][p]<<j for j in range(3))
    return len(scene.actors),len(scene.projects),bits

@contextmanager
def main_question_seed():
    previous=tasks.OLD.SEED
    tasks.OLD.SEED=260912501  # Exact inherited V5/V7 R0 answer-mapping procedure
    try:
        yield
    finally:
        tasks.OLD.SEED=previous

def queries(scene, terminal, program, origin_panel=False):
    if origin_panel:
        return [q for draw in (0,1) for q in origins.questions(readers,terminal,scene,'R0',draw)]
    with main_question_seed():
        result=[]
        for draw in (0,1):
            for q in tasks.program_queries(scene,program,draw):
                q.update(draw=draw,reader='R0')
                result.append(q)
        return result

def build(prior_path):
    prior=json.loads(Path(prior_path).read_text())
    used=set(prior['sha256']); rejected=[]; populations={}
    for split,n in [('MAIN',64),('ORIGIN',64),('DEV',8)]:
        rows=[]
        for i in range(n):
            for attempt in range(32):
                # +1024 preserves the original size and three-bit cell on collision.
                s=candidate.candidate(split,i+1024*attempt)
                s=replace(s,scene_id=f'{NAMESPACE}-{split}-{i:04d}',split='FINAL')
                branches=origins.origins(tasks.OLD,s) if split=='ORIGIN' else [s]
                hashes=[candidate.semantic_hash(x) for x in branches]
                if len(set(hashes))!=len(hashes):
                    raise ValueError('Duplicate branches')
                if set(hashes)&used:
                    rejected.append({'split':split,'index':i,'attempt':attempt})
                    continue
                used.update(hashes)
                if split=='ORIGIN':
                    rows.append(dict(root_id=s.scene_id,terminal=asdict(s),
                                     sources=[asdict(x) for x in branches],
                                     commands=[asdict(e) for e in origins.commands(tasks.OLD,s)],
                                     origin_masks=list(map(list,origins.ORIGIN_MASKS)),source_hashes=hashes))
                else:
                    rows.append(asdict(s))
                break
            else:
                raise ValueError('Fixed collision-replacement supply exhausted')
        populations[split.lower()]=rows
    main=[scene_from_dict(s) for s in populations['main']]
    counts=Counter(cell(s) for s in main)
    roots=Counter(cell(scene_from_dict(r['terminal'])) for r in populations['origin'])
    if len(counts)!=32 or set(counts.values())!={2} or roots!=counts:
        raise ValueError('64-scene/root balanced truth-cell requirement failed')
    # Lowest source SHA256 within each size x projects x three-bit truth cell.
    groups={}
    for s in main:
        groups.setdefault(cell(s),[]).append(s)
    subset=[min(ss,key=candidate.semantic_hash) for _,ss in sorted(groups.items())]
    populations['program']=[asdict(s) for s in subset]
    dev=Counter((len(s['actors']),len(s['projects']),s['values'][s['target_actor']][s['target_project']]) for s in populations['dev'])
    if len(dev)!=8 or set(dev.values())!={1}:
        raise ValueError('Development size/direction balance failed')
    return populations,dict(namespace=NAMESPACE,seed=SEED,
        historical_unique_sources=len(prior['sha256']),fresh_unique_sources=len(used)-len(prior['sha256']),
        rejected_collisions=rejected,remaining_collisions=0,prior_index_sha256=sha(prior_path),
        main_per_model=64,terminal_roots_per_model=64,origins_per_root=4,program_scenes_per_model=32,
        development_scenes_per_model=8,models_share_scene_designs=True,writer_seeds_share_cases=True,
        independent_unit='terminal root (primary); scene (main/program); not origin/query/model/writer seed',
        program_selection='lowest source-text SHA256 within each of 32 size/project/three-bit cells',
        development_selection='8 disjoint candidate scenes covering each size/project/direction cell once',
        question_mapping_seeds=dict(main=260912501,origin=260912601),
        outcomes_used=False,model_evaluations=0)

def export(destination, prior_path=ROOT/'cases/PRIOR_SOURCE_HASHES.json'):
    destination=Path(destination)
    if destination.exists():
        raise FileExistsError('Immutable case export: choose a new directory')
    pops,audit=build(prior_path)
    destination.mkdir(parents=True)
    for name,rows in pops.items():
        write_rows(destination/(name+'.jsonl'),rows)
    requests=[]; scores=[]; plans=[]
    def add(scene,terminal,commands,program,panel,order,root_id,origin):
        qs=queries(scene,terminal,program,panel=='ORIGIN')
        if len([q for q in qs if not q.get('additional',False)])!=24:
            raise ValueError('Original question membership changed')
        if panel=='ORIGIN' and len(qs)!=34:
            raise ValueError('Origin question count changed')
        for actor,procedure in [('gemma','C1'),('qwen','T4')]:
            text,spans=tasks.prefix_text(scene,procedure,order)
            cmds=[asdict(AddressedUpdate(*spans[(e.actor,e.project)],e.value)) for e in commands]
            identifier=digest(stable([actor,panel,scene.scene_id,program,order]))
            requests.append(dict(input_id=identifier,actor=actor,source_text=text,commands=cmds))
            plans.append(dict(input_id=identifier,actor=actor,panel=panel,scene_id=scene.scene_id,
                              root_id=root_id,origin=origin,program=program,order=order,
                              source_sha256=digest(text),question_count=len(qs),writer_seeds=[0,1]))
            for q in qs:
                spec=q['spec']
                if q['gold']!=origins.truth(terminal.values,spec) or q['source_gold']!=origins.truth(scene.values,spec):
                    raise ValueError('Independent label disagreement')
                scores.append(dict(q,input_id=identifier,actor=actor,panel=panel,scene_id=scene.scene_id,
                                   root_id=root_id,origin=origin,program=program,order=order))
    for panel in ('MAIN','PROGRAM','DEV'):
        for rec in pops[panel.lower()]:
            s=scene_from_dict(rec)
            programs=tasks.PROGRAMS if panel=='PROGRAM' else ('single',)
            orders=('early','late') if panel=='MAIN' else ('early',)
            for program in programs:
                for order in orders:
                    cmds=tasks.programs(s)[program]
                    add(s,tasks.OLD.apply_program(s,cmds),cmds,program,panel,order,s.scene_id,None)
    for r in pops['origin']:
        terminal=scene_from_dict(r['terminal']); cmds=tuple(tasks.OLD.Edit(**e) for e in r['commands'])
        for origin,rec in enumerate(r['sources']):
            s=scene_from_dict(rec)
            add(s,terminal,cmds,'SET_TERMINAL_ABC','ORIGIN','early',r['root_id'],origin)
    write_rows(destination/'inputs.jsonl',requests)
    write_rows(destination/'evaluation.jsonl',scores)
    write_rows(destination/'plan.jsonl',plans)
    example=next(r for r in requests if any(p['input_id']==r['input_id'] and p['panel']=='DEV' for p in plans))
    write_json(destination/'example_input.json',example)
    write_json(destination/'example_question.json',next({'text':q['text'],'labels':q['labels']} for q in scores if q['input_id']==example['input_id']))
    write_json(destination/'COLLISION_CHECK.json',audit)
    files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(destination.iterdir())}
    audit.update(files=files,input_requests=len(requests),scoring_rows=len(scores),
                 generator_sha256=sha(Path(__file__)),
                 canonical_generator_hashes={name:sha(ROOT/'canonical'/name) for name in ('candidate.py','v2_protocol.py','v3_protocol.py','origins.py','tasks.py')},
                 prompts_sha256=sha(ROOT/'prompts.json'))
    write_json(destination/'MANIFEST.json',audit)
    return audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--prior',default=ROOT/'cases/PRIOR_SOURCE_HASHES.json')
    a=p.parse_args(); result=export(a.output,a.prior)
    print(json.dumps({k:result[k] for k in ('input_requests','scoring_rows','remaining_collisions','fresh_unique_sources')}))
