from __future__ import annotations
from dataclasses import dataclass, asdict, replace
import hashlib, json, random
import numpy as np
SEED = 260911398
NAMES = ('Ari', 'Bela', 'Cato', 'Dara', 'Eli', 'Faye', 'Gale', 'Hana', 'Ivo', 'Juno', 'Kira', 'Luca', 'Mina', 'Nico', 'Orla', 'Pia', 'Quin', 'Ravi', 'Sora', 'Tavi', 'Uma', 'Vera', 'Wren', 'Zara', 'Ada', 'Bryn', 'Cleo', 'Dion', 'Esme', 'Finn', 'Gita', 'Hugo')
PROJECTS = ('Harbor', 'Orchard', 'Meadow', 'Bridge', 'Gallery', 'Garden', 'Museum', 'Library', 'Canal', 'Theater')

@dataclass(frozen=True)
class Scene:
    scene_id: str
    split: str
    actors: tuple[str,...]
    projects: tuple[str,...]
    values: tuple[tuple[int,...],...]
    target_actor: int
    target_project: int
    layout: tuple[int,...]
    setting: str = 'community planning meeting'


@dataclass(frozen=True)
class Edit:
    actor: int
    project: int
    value: int


def rng(*parts):
    return random.Random('|'.join(map(str, (SEED,) + parts)))


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def validate_scene(s):
    if len(set(s.actors)) != len(s.actors) or len(set(s.projects)) != len(s.projects):
        raise ValueError('Nonunique identifiers')
    if sorted(s.layout) != list(range(len(s.actors))): raise ValueError('Invalid layout')
    if len(s.values) != len(s.actors) or any(len(x)!=len(s.projects) for x in s.values):
        raise ValueError('Invalid state shape')
    if any(v not in (0,1) for row in s.values for v in row): raise ValueError('Not binary')
    if not 0<=s.target_actor<len(s.actors) or not 0<=s.target_project<len(s.projects): raise ValueError('Address')
    return s


def set_value(s, e):
    if e.value not in (0,1): raise ValueError('SET requires a bit')
    if not 0<=e.actor<len(s.actors) or not 0<=e.project<len(s.projects): raise ValueError('Address')
    a=[list(x) for x in s.values];a[e.actor][e.project]=e.value
    return replace(s, values=tuple(map(tuple,a)))


def apply_program(s, edits):
    for e in edits: s=set_value(s,e)
    return s


def primary_edit(s):
    return Edit(s.target_actor,s.target_project,1-s.values[s.target_actor][s.target_project])


def make_scenes(split,n):
    result=[]
    for i in range(n):
        r=rng(split,i)
        if split in ('FINAL','WORKFLOW'):
            count=(4,6)[(i//2)%2];project_count=(2,3)[i%2]
        else:count=4;project_count=2
        actors=tuple(r.sample(NAMES,count));projects=tuple(r.sample(PROJECTS,project_count))
        values=[ [r.randrange(2) for _ in projects] for _ in actors]
        a=r.randrange(count);p=r.randrange(project_count)
        values[a][p]=((i//4)%2 if split in ('FINAL','WORKFLOW') else i%2)
        # Balance switch direction within, not across, every size cell.
        if split=='WORKFLOW':
            b=(a+1)%count;c=(a+2)%count
            values[b][p]=(i//8)%2
            values[c][p]=(1-values[b][p]) if (i//16)%2 else values[b][p]
        layout=list(range(count));r.shuffle(layout)
        s=Scene(f'{split}-SRS2-{i:04}',split,actors,projects,tuple(map(tuple,values)),a,p,tuple(layout))
        result.append(validate_scene(s))
    return result


def render(s, order='original', variant=0):
    """Return source text and address -> character interval; never an answer or question."""
    validate_scene(s)
    pairs=[(a,p) for a in s.layout for p in range(len(s.projects))]
    hit=(s.target_actor,s.target_project)
    if order=='early':pairs=[hit]+[x for x in pairs if x!=hit]
    elif order=='late':pairs=[x for x in pairs if x!=hit]+[hit]
    elif order!='original': raise ValueError(order)
    lines=[f'This is a fictional {s.setting}. Use only the records below.',
           'Each named member has independent views. Updating one record leaves the other records unchanged.']
    spans={}
    for a,p in pairs:
        if variant==0:
            line=f'{s.actors[a]} is '+('in favor of ' if s.values[a][p] else 'opposed to ')+f'the {s.projects[p]} proposal.'
        elif variant==1:
            line=f'On {s.projects[p]}, {s.actors[a]} '+('supports' if s.values[a][p] else 'opposes')+' the proposal.'
        else: raise ValueError(variant)
        start=len('\n'.join(lines))+1;lines.append(line);spans[(a,p)]=(start,start+len(line))
    lines.append('End of records.\n')
    return '\n'.join(lines),spans


def evaluate(s, spec):
    a,p=spec['a'],spec['p'];x=s.values[a][p];kind=spec['kind']
    if kind=='direct':return int(x)
    if kind=='opposes':return int(not x)
    y=s.values[spec['b']][p]
    if kind=='same':return int(x==y)
    if kind=='both':return int(x and y)
    if kind=='either':return int(x or y)
    raise ValueError(kind)


def questions(s, edited=None, challenge=False, draw=0):
    """Evaluator-side only. Independent compiler must never receive these records."""
    edited=set_value(s,primary_edit(s)) if edited is None else edited
    a,p=s.target_actor,s.target_project;b=(a+1)%len(s.actors);c=(a+2)%len(s.actors);o=(p+1)%len(s.projects)
    specs=[('direct',a,p,b),('opposes',a,p,b),('same',a,p,b),('direct',b,p,a),('opposes',b,p,a),('direct',a,o,b)]
    if challenge:specs += [('both',a,p,b),('either',a,p,b),('same',c,p,a),('both',c,p,a),('either',c,p,a),('direct',c,o,a)]
    r=rng('queries',s.scene_id,draw,challenge);out=[]
    for j,(k,aa,pp,bb) in enumerate(specs):
        an,bn,pr=s.actors[aa],s.actors[bb],s.projects[pp]
        wording={'direct':f'Does {an} favor the {pr} proposal?',
                 'opposes':f'Does {an} oppose the {pr} proposal?',
                 'same':f'Do {an} and {bn} take the same side on {pr}?',
                 'both':f'Is the {pr} proposal supported by both {an} and {bn}?',
                 'either':f'Does at least one of {an} and {bn} support {pr}?'}[k]
        labels=['No','Yes'] if draw%2==0 else (['A','B'] if r.randrange(2) else ['B','A'])
        if draw%2:wording+=f'\nUse {labels[1]} for yes and {labels[0]} for no.'
        wording+='\nAnswer with exactly one of '+', '.join(labels)+'.'
        spec={'kind':k,'a':aa,'b':bb,'p':pp}
        g0=evaluate(s,spec);g1=evaluate(edited,spec)
        out.append({'query_id':f'{s.scene_id}-d{draw}-{j:02}', 'scene_id':s.scene_id,'family':k,
                    'text':wording,'labels':labels,'spec':spec,'source_gold':g0,'gold':g1,'changed':g0!=g1})
    return out


def token_mask(clause_positions,prefix_length,footprint):
    cp=tuple(clause_positions)
    if not cp or tuple(range(cp[0],cp[-1]+1))!=cp or cp[0]<0 or cp[-1]>=prefix_length:raise ValueError('Invalid clause positions')
    if footprint=='clause':return cp
    if footprint=='tail':return tuple(range(cp[0],prefix_length))
    raise ValueError(footprint)

