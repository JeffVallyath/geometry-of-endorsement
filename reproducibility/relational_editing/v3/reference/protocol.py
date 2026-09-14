"""Fresh scene draws and evaluator-side edit programs. No model inference.
Single-operation training only; program semantics are explicit SET assignments.
"""
from __future__ import annotations
from dataclasses import asdict, replace
from pathlib import Path
import json, hashlib
from . import v2_protocol as old

SEED=260911397
old.SEED=SEED
Scene=old.Scene
Edit=old.Edit
render=old.render
apply_semantics=old.apply_program

def scenes(split,n):
    return [replace(s,scene_id=s.scene_id.replace('SRS2','RSO3')) for s in old.make_scenes(split,n)]

def programs(s):
    a=old.primary_edit(s);b=Edit((a.actor+1)%len(s.actors),a.project,1-s.values[(a.actor+1)%len(s.actors)][a.project]);c=Edit((a.actor+2)%len(s.actors),a.project,1-s.values[(a.actor+2)%len(s.actors)][a.project]);undo=Edit(a.actor,a.project,1-a.value)
    return {'single':[a],'noop':[undo],'repeat2':[a]*2,'repeat4':[a]*4,'repeat8':[a]*8,
            'restore':[a,undo],'restore_after4':[a]*4+[undo],'overwrite':[a,undo,a],
            'AB':[a,b],'BA':[b,a],'ABC':[a,b,c],'CBA':[c,b,a]}

def program_queries(s,name,draw=0):
    """Only evaluator sees these. Additional direct questions cover each touched address."""
    prog=programs(s)[name];world=apply_semantics(s,prog)
    out=old.questions(s,world,True,draw)
    covered={(q['spec']['a'],q['spec']['p']) for q in out if q['family']=='direct'}
    for e in prog:
        if (e.actor,e.project) in covered:continue
        spec={'kind':'direct','a':e.actor,'b':(e.actor+1)%len(s.actors),'p':e.project}
        labels=['No','Yes'] if not draw%2 else ['B','A']
        text=f'Does {s.actors[e.actor]} favor the {s.projects[e.project]} proposal?'
        if draw%2:text+=f'\nUse {labels[1]} for yes and {labels[0]} for no.'
        text+='\nAnswer with exactly one of '+', '.join(labels)+'.'
        v0=old.evaluate(s,spec);v1=old.evaluate(world,spec)
        out.append({'query_id':f'{s.scene_id}-d{draw}-extra{len(out)}','scene_id':s.scene_id,'family':'direct','text':text,'labels':labels,'spec':spec,'source_gold':v0,'gold':v1,'changed':v0!=v1,'additional':True})
        covered.add((e.actor,e.project))
    for q in out:q['program_id']=name
    return out

def export(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);files={};seen=set()
    for split,n in [('FIT',96),('CAL',48),('FINAL',128),('WORKFLOW',128)]:
        data=scenes(split,n)
        for s in data:
            text=render(s)[0];sig=hashlib.sha256(text.encode()).hexdigest()
            if sig in seen:raise ValueError('Duplicate source; do not leak splits')
            seen.add(sig)
        p=root/f'{split.lower()}_scenes.jsonl';p.write_text(''.join(json.dumps(asdict(s),sort_keys=True)+'\n' for s in data))
        files[p.name]={'scenes':n,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (root/'MANIFEST.json').write_text(json.dumps({'seed':SEED,'files':files,'programs':list(programs(data[0]))},indent=2))
    return files

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--export',required=True);export(p.parse_args().export)
