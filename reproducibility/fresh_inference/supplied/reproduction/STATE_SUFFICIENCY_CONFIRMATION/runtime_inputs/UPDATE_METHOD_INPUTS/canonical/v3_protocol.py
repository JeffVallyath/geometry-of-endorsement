from . import v2_protocol as old
Scene = old.Scene
Edit = old.Edit
apply_semantics = old.apply_program

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

