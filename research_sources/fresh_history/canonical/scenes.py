from . import v2_protocol as P

def scene_from_dict(d):
    d = dict(d); d['actors'] = tuple(d['actors']); d['projects'] = tuple(d['projects']); d['layout'] = tuple(d['layout']); d['values'] = tuple(tuple(x) for x in d['values'])
    return P.validate_scene(P.Scene(**d))

