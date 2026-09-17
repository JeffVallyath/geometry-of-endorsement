import numpy as np
from . import scene_metrics as S

def summarize(rows):
    """V2 summary on the ORIGINAL question bundle (all24 = every original question of an artifact correct); plus `all_program_questions`
    (every question including coverage additions correct) and the additions' accuracy when additions exist."""
    orig = [r for r in rows if not r.get('additional')]; m = S.summarize(orig)
    if any(r.get('additional') for r in rows):
        full = S.summarize(rows); m['all_program_questions'] = full['all24']; m['accuracy_all_program_questions'] = full['accuracy']
        adds = [r for r in rows if r.get('additional')]; m['additional_accuracy'] = float(np.mean([r['prediction'] == r['gold'] for r in adds])); m['additional_rows'] = len(adds)
        for sid, v in m['per_scene'].items(): v['all_program_questions'] = full['per_scene'][sid]['all24']; v['accuracy_all_program_questions'] = full['per_scene'][sid]['accuracy']
    else:
        m['all_program_questions'] = m['all24']; m['accuracy_all_program_questions'] = m['accuracy']
        for v in m['per_scene'].values(): v['all_program_questions'] = v['all24']; v['accuracy_all_program_questions'] = v['accuracy']
    return m

