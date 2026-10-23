from . import v2_protocol as OLD
from . import v3_protocol as P3
from .config import INSTRUCTION
from .prompts import demo_block
PROGRAMS = ('single', 'noop', 'repeat2', 'repeat4', 'repeat8', 'restore', 'restore_after4', 'overwrite', 'AB', 'BA', 'ABC', 'CBA', 'AB_cross')

def programs(s):
    """The twelve supplied programs plus one cross-project two-edit variant (B on the next project), reported separately."""
    out = dict(P3.programs(s)); a = out['single'][0]; b = (a.actor+1) % len(s.actors); o = (a.project+1) % len(s.projects)
    out['AB_cross'] = [a, P3.Edit(b, o, 1-s.values[b][o])]
    if tuple(out) != PROGRAMS: raise ValueError('program set drift')
    return out


def final_world(s, program): return OLD.apply_program(s, programs(s)[program])


def program_queries(s, name, draw=0):
    """FINAL/WORKFLOW question bundle of a program: the original 24-question bundle (12 per draw) of the single edit's specs evaluated in the
    program's true final world, plus one additional direct question per touched address not already covered (flag `additional`)."""
    if name in P3.programs(s): out = P3.program_queries(s, name, draw)
    else:
        prog = programs(s)[name]; world = final_world(s, name); out = OLD.questions(s, world, True, draw)
        covered = {(q['spec']['a'], q['spec']['p']) for q in out if q['family'] == 'direct'}
        for e in prog:
            if (e.actor, e.project) in covered: continue
            spec = {'kind': 'direct', 'a': e.actor, 'b': (e.actor+1) % len(s.actors), 'p': e.project}
            labels = ['No', 'Yes'] if not draw % 2 else ['B', 'A']
            text = f'Does {s.actors[e.actor]} favor the {s.projects[e.project]} proposal?'
            if draw % 2: text += f'\nUse {labels[1]} for yes and {labels[0]} for no.'
            text += '\nAnswer with exactly one of '+', '.join(labels)+'.'
            v0 = OLD.evaluate(s, spec); v1 = OLD.evaluate(world, spec)
            out.append({'query_id': f'{s.scene_id}-d{draw}-extra{len(out)}', 'scene_id': s.scene_id, 'family': 'direct', 'text': text, 'labels': labels, 'spec': spec, 'source_gold': v0, 'gold': v1, 'changed': v0 != v1, 'additional': True})
            covered.add((e.actor, e.project))
        for q in out: q['program_id'] = name
    for q in out: q.setdefault('additional', False)
    return out


def correction_line(s, e):
    v = int(e.value)
    return f"Correction: the record for {s.actors[e.actor]} on the {s.projects[e.project]} proposal is updated. {s.actors[e.actor]} is now "+('in favor of' if v else 'opposed to')+f' the {s.projects[e.project]} proposal.'


def prefix_text(s, procedure, order='original', variant=0, world=None, corrections=None):
    """(text, spans): V2 demonstrations + rendered records of `world` (default the source scene) in the given order/wording; optional
    correction lines (one per declared command, in command order) inserted before 'End of records.'.  Never contains a question or answer."""
    w = s if world is None else world
    body, spans = OLD.render(w, order, variant); demos = demo_block(procedure)
    if corrections:
        tail = 'End of records.\n'
        if not body.endswith(tail): raise ValueError('unexpected render tail')
        body = body[:-len(tail)]+''.join(correction_line(s, e)+'\n' for e in corrections)+tail
    off = len(demos)
    return demos+body, {k: (off+a, off+b) for k, (a, b) in spans.items()}


def suffix_text(text): return text+INSTRUCTION

