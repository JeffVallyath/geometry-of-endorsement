from . import tasks as Q
READERS = ("R0", "R1", "R2")
TEMPLATES = {'R0': 'Do {A} and {B} take the same side on {project}?', 'R1': 'Do {A} and {B} hold matching positions regarding {project}?', 'R2': 'Do {A} and {B} hold matching positions regarding {project}? The answer is yes if both support it or both oppose it, and no if one supports it and the other opposes it.'}

def equality_text(reader, A, B, project):
    """The reader interface has names only: no scene, values, gold or program."""
    return TEMPLATES[reader].format(A=A, B=B, project=project)


def apply_reader(text, family, A, B, project, reader):
    if reader not in READERS:
        raise ValueError('Undeclared reader')
    if family != 'same':
        return text
    first, sep, tail = text.partition('\n')
    if first != equality_text('R0', A, B, project):
        raise ValueError('Original equality wording drift')
    return equality_text(reader, A, B, project) + sep + tail

