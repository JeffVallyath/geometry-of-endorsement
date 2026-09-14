"""Model-free contracts for SHARED_RELATION_STATE_V2.
No inference, claims of causal identity, or executable production adapter here.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from itertools import permutations
import hashlib
import json
from pathlib import Path
import random
import numpy as np

SEED = 260910217
NAMES = ('Ari','Bela','Cato','Dara','Eli','Faye','Gale','Hana','Ivo','Juno','Kira','Luca',
         'Mina','Nico','Orla','Pia','Quin','Ravi','Sora','Tavi','Uma','Vera','Wren','Zara',
         'Ada','Bryn','Cleo','Dion','Esme','Finn','Gita','Hugo')
PROJECTS = ('Harbor','Orchard','Meadow','Bridge','Gallery','Garden','Museum','Library','Canal','Theater')

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

def programs(s):
    a=primary_edit(s);b=Edit((a.actor+1)%len(s.actors),a.project,1-s.values[(a.actor+1)%len(s.actors)][a.project])
    c=Edit((a.actor+2)%len(s.actors),a.project,1-s.values[(a.actor+2)%len(s.actors)][a.project])
    restore=Edit(a.actor,a.project,s.values[a.actor][a.project])
    return {'single':[a], 'noop':[restore], 'repeat':[a,a], 'restore':[a,restore],
            'AB':[a,b], 'BA':[b,a], 'ABC':[a,b,c], 'CBA':[c,b,a]}

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

@dataclass(frozen=True)
class CompileRequest:
    prefix_ids: tuple[int,...]
    addresses: tuple[tuple[int,...],...]
    desired_values: tuple[int,...]
    checkpoint_sha: str
    def __post_init__(self):
        if len(self.addresses)!=len(self.desired_values):raise ValueError('Unmatched edits')
        for cp,v in zip(self.addresses,self.desired_values):
            token_mask(cp,len(self.prefix_ids),'clause')
            if v not in (0,1): raise ValueError('Bad desired value')
    def key(self):return digest(asdict(self))


def augmented_delta(h,z,q,C,b,V,center):
    """Shared and aware use the same map class; q is zero for shared.
    q is a centered source-question summary, never a teacher or gold answer.
    """
    h=np.asarray(h,dtype=np.float64);z=np.asarray(z,dtype=np.float64);q=np.asarray(q,dtype=np.float64)
    if h.ndim!=2 or z.shape!=(h.shape[1],) or q.shape!=z.shape:raise ValueError('Feature shape')
    x=np.concatenate((h-center,np.broadcast_to(z-center,h.shape),np.broadcast_to(q,h.shape)),axis=1)
    if C.shape!=(3*h.shape[1],V.shape[0]) or b.shape!=(V.shape[0],) or V.shape[1]!=h.shape[1]:raise ValueError('Factor shape')
    return (x@C+b)@V


def extend_old(C):
    if C.ndim!=2 or C.shape[0]%2: raise ValueError('Expected 2d x r input factor')
    return np.concatenate((C,np.zeros((C.shape[0]//2,C.shape[1]),dtype=C.dtype)),axis=0)

def clip_rows(delta,cap,strength=1.0):
    if cap<=0 or strength<0:raise ValueError('Invalid norm rule')
    d=np.asarray(delta,float);n=np.linalg.norm(d,axis=-1,keepdims=True)
    return d*np.minimum(1.,cap/np.maximum(n,1e-12))*strength

def common_frobenius(a,b):
    """Diagnostic only: reduce BOTH nonzero tensors to the lower observed norm.
    Cannot enlarge a per-token cap. Computed before queries for shared arms.
    """
    a=np.asarray(a,float);b=np.asarray(b,float);na=np.linalg.norm(a);nb=np.linalg.norm(b)
    if na==0 or nb==0:return None
    n=min(na,nb)
    return a*n/na,b*n/nb


def export(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);meta={}
    for split,n in [('FIT',96),('CAL',32),('FINAL',128),('WORKFLOW',128)]:
        scenes=make_scenes(split,n);p=root/f'{split.lower()}_scenes.jsonl'
        p.write_text(''.join(json.dumps(asdict(s),sort_keys=True)+'\n' for s in scenes),encoding='utf8')
        meta[p.name]={'scenes':n,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (root/'MANIFEST.json').write_text(json.dumps({'seed':SEED,'files':meta},indent=2)+'\n')
    return meta

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--export',required=True)
    export(p.parse_args().export)
