"""Deterministic, query-separated semantic-control tasks.

The gold evaluator is intentionally executable. This is an experiment on model
interventions, NOT a proposal to replace the evaluator with a neural solver.
Production compiler inputs are limited to render_prefix(scene) and edit_request(scene).
No Query, answer mapping, option list, or gold label belongs in that interface.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import argparse, hashlib, json, random

SEED = 260910196
NAMES = ('Ari','Bela','Cato','Dara','Eli','Faye','Gale','Hana','Ivo','Juno','Kira','Luca',
         'Mina','Nico','Orla','Pia','Quin','Ravi','Sora','Tavi','Uma','Vera','Wren','Zara')
TARGETS = ('Harbor','Orchard','Meadow','Bridge','Gallery','Garden','Museum','Library')
CRITERIA = ('accessibility','variety')

def stable_rng(*parts):
    return random.Random('|'.join(map(str,(SEED,)+parts)))
def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

@dataclass(frozen=True)
class Scene:
    scene_id: str
    split: str
    task: str
    actors: tuple[str,...]
    targets: tuple[str,...]
    values: tuple[tuple[int,...],...]
    priorities: tuple[int,...]
    target_actor: int
    target_project: int
    layout: tuple[int,...]
    setting: str

def edit_request(s: Scene) -> dict:
    old = s.values[s.target_actor][s.target_project] if s.task=='relation' else s.priorities[s.target_actor]
    return {'task':s.task,'actor':s.actors[s.target_actor],
            'project':s.targets[s.target_project] if s.task=='relation' else None,
            'operation':'set','value':1-old}

def apply_edit(s: Scene, actor=None, project=None, value=None) -> Scene:
    a=s.target_actor if actor is None else actor
    p=s.target_project if project is None else project
    if s.task=='relation':
        v=[list(row) for row in s.values]; v[a][p]=1-v[a][p] if value is None else value
        return replace(s,values=tuple(tuple(row) for row in v))
    v=list(s.priorities);v[a]=1-v[a] if value is None else value
    return replace(s,priorities=tuple(v))

def render_prefix(s: Scene, variant=0) -> tuple[str,tuple[int,int]]:
    """Return source text and the addressed clause's CHARACTER interval.
    Address information is available from the requested semantic edit, not an oracle
    about which later question changes. Token mapping is the adapter's responsibility.
    """
    lines=[f'This is a fictional {s.setting}. Use only the records below.',
           'Each named member has independent views. Updating one record leaves the other records unchanged.']
    hit=None
    for a in s.layout:
        if s.task=='relation':
            for p,t in enumerate(s.targets):
                if variant==0: line=f'{s.actors[a]} is '+('in favor of ' if s.values[a][p] else 'opposed to ')+f'the {t} proposal.'
                else: line=f'On {t}, {s.actors[a]} '+('supports' if s.values[a][p] else 'opposes')+' the proposal.'
                begin=len('\n'.join(lines))+1; lines.append(line)
                if a==s.target_actor and p==s.target_project: hit=(begin,begin+len(line))
        else:
            c=CRITERIA[s.priorities[a]]
            line=(f'{s.actors[a]} selects the option with the highest {c} score. '
                  'If scores tie, use the first option in the displayed list.')
            begin=len('\n'.join(lines))+1;lines.append(line)
            if a==s.target_actor: hit=(begin,begin+len(line))
    lines.append('End of records.\n')
    assert hit is not None
    return '\n'.join(lines),hit

@dataclass(frozen=True)
class Query:
    query_id: str
    family: str
    text: str
    labels: tuple[str,...]
    gold_index: int
    source_gold_index: int
    affected: bool
    case: dict

def query_gold(s: Scene, d: dict) -> int:
    typ=d['type'];a=d['a'];p=d.get('p',0)
    if typ=='direct': ans=bool(s.values[a][p])
    elif typ=='opposes': ans=not bool(s.values[a][p])
    elif typ=='same': ans=s.values[a][p]==s.values[d['b']][p]
    elif typ=='both': ans=bool(s.values[a][p] and s.values[d['b']][p])
    elif typ=='either': ans=bool(s.values[a][p] or s.values[d['b']][p])
    elif typ=='recommend':
        idx=max(range(len(d['scores'])),key=lambda k:(d['scores'][k][s.priorities[a]],-k))
        return idx
    elif typ=='score_sign': ans=d['scores'][d['option']][d['criterion']] > 0
    elif typ=='rule': ans=s.priorities[a]==d['criterion']
    else: raise ValueError(typ)
    return int(ans)

def questions(s: Scene, target: Scene|None=None, *, challenge=False, draw=0) -> list[Query]:
    """Generate query-side choices only AFTER final edited-prefix artifacts are sealed.
    In FIT/CAL the same function provides legitimate supervised suffixes.
    'affected' and reference indices are evaluator-only and never sent to compiler.
    """
    target=apply_edit(s) if target is None else target
    r=stable_rng('queries',s.scene_id,draw,challenge);a=s.target_actor;p=s.target_project
    b=(a+1)%len(s.actors);otherp=(p+1)%len(s.targets)
    rows=[]
    if s.task=='relation':
        configs=[('direct',a,p),('opposes',a,p),('same',a,p),
                 ('direct',b,p),('opposes',b,p),('direct',a,otherp)]
        if challenge:
            c=(a+2)%len(s.actors)
            # Conjunction/disjunction are never training or editor-calibration queries.
            configs += [('both',a,p),('either',a,p),('same',c,p),('both',c,p),('either',c,p),('direct',c,otherp)]
        for typ,aa,pp in configs:
            bb=b if aa==a else a
            d={'type':typ,'a':aa,'b':bb,'p':pp}
            an,bn,proj=s.actors[aa],s.actors[bb],s.targets[pp]
            texts={'direct':f'Does {an} favor the {proj} proposal?',
                   'opposes':f'Does {an} oppose the {proj} proposal?',
                   'same':f'Do {an} and {bn} take the same side on {proj}?',
                   'both':f'Is the {proj} proposal supported by both {an} and {bn}?',
                   'either':f'Does at least one of {an} and {bn} support {proj}?'}
            rows.append((typ,texts[typ],d))
    else:
        counts=(2,3,4,6) if challenge else (2,3)
        for k in counts:
            for mode in ('conflict','agreement','other_actor'):
                aa=b if mode=='other_actor' else a
                scores=[(r.randrange(-3,10),r.randrange(-3,10)) for _ in range(k)]
                if mode=='agreement':
                    win=r.randrange(k);scores[win]=(12,12)
                else:
                    x,y=r.sample(range(k),2);scores[x]=(12,-2);scores[y]=(-2,12)
                perm=list(range(k));r.shuffle(perm);scores=[scores[j] for j in perm]
                d={'type':'recommend','a':aa,'scores':scores,'mode':mode}
                opts='\n'.join(f'Option {j+1}: accessibility {v[0]}, variety {v[1]}.' for j,v in enumerate(scores))
                text=f'For this new case, which option should {s.actors[aa]} select?\n'+opts
                rows.append(('recommend_'+mode,text,d))
        scores=[(r.choice((-2,2)),r.choice((-3,3))) for _ in range(2)]
        for c in range(2):
            d={'type':'score_sign','a':a,'scores':scores,'option':0,'criterion':c}
            text=f'An option has accessibility {scores[0][0]} and variety {scores[0][1]}. Is its {CRITERIA[c]} score positive?'
            rows.append(('score_sign',text,d))
        # Important: do not use an explicit "is the policy X" query as the
        # main consequence; recommendations constitute the primary task.
    out=[]
    for j,(fam,text,d) in enumerate(rows):
        g0=query_gold(s,d);g1=query_gold(target,d)
        if d['type']=='recommend':
            labs=list('ABCDEF'[:len(d['scores'])]);r.shuffle(labs)
            text+='\nAnswer mapping: '+ '; '.join(f'option {i+1} = {code}' for i,code in enumerate(labs))+'.'
        else:
            labs=list(('No','Yes'))
            # Alternate verbalizers without changing semantic index.
            if draw%2:
                labs=list(('B','A')) if r.random()<.5 else list(('A','B'))
                text+=f'\nUse {labs[1]} for yes and {labs[0]} for no.'
        text+='\nAnswer with exactly one of '+', '.join(labs)+'.'
        out.append(Query(f'{s.scene_id}-d{draw}-{j:02d}',fam,text,tuple(labs),g1,g0,g1!=g0,d))
    return out

def generate(split,n_per_task):
    out=[]
    for task in ('relation','priority'):
        for i in range(n_per_task):
            rng=stable_rng(split,task,i);n=4 if split!='FINAL' or i%2==0 else 6
            actors=tuple(rng.sample(NAMES,n));targets=tuple(rng.sample(TARGETS,2 if n==4 else 3))
            values=tuple(tuple(rng.randrange(2) for _ in targets) for _ in actors)
            priorities=tuple(rng.randrange(2) for _ in actors)
            layout=list(range(n));rng.shuffle(layout)
            s=Scene(f'{split}-{task}-{i:03d}',split,task,actors,targets,values,priorities,
                    rng.randrange(n),rng.randrange(len(targets)),tuple(layout),
                    rng.choice(('community planning meeting','arts committee','volunteer group','library advisory meeting')))
            out.append(s)
    return out

def from_dict(d):
    d=dict(d)
    for k in ('actors','targets','priorities','layout'): d[k]=tuple(d[k])
    d['values']=tuple(tuple(x) for x in d['values'])
    return Scene(**d)

def export(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    files={}
    for split,n in [('FIT',64),('CAL',24),('FINAL',64)]:
        rows=[asdict(s) for s in generate(split,n)]
        p=root/(split.lower()+'_scenes.jsonl');p.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows))
        files[p.name]={'rows':len(rows),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (root/'MANIFEST.json').write_text(json.dumps({'seed':SEED,'files':files},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);export(p.parse_args().output)
