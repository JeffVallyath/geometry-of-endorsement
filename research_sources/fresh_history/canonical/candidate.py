import random, hashlib
from . import v2_protocol as OLD
SEED = 260913901

def semantic_hash(scene):
    return hashlib.sha256(OLD.render(scene, 'original')[0].encode()).hexdigest()


def candidate(split, i):
    """Same vocabulary/renderer; local RNG, not outcome-driven generation."""
    r = random.Random(f'{SEED}|{split}|{i}')
    if split in ('FIT', 'CAL'): n, m = 4, 2
    else: n, m = ((4,2),(4,3),(6,2),(6,3))[i % 4]
    names = tuple(r.sample(OLD.NAMES,n)); projects = tuple(r.sample(OLD.PROJECTS,m))
    values = [[r.randrange(2) for _ in range(m)] for _ in range(n)]
    a, p = r.randrange(n), r.randrange(m)
    # Balanced three-bit truth cells; FIT/CAL have 8 cells, FINAL 32 cells.
    code = i % 8 if split in ('FIT','CAL') else (i // 4) % 8
    for j in range(3): values[(a+j)%n][p] = (code >> j) & 1
    layout = list(range(n)); r.shuffle(layout)
    return OLD.validate_scene(OLD.Scene(f'{split}-RTC7-{i:04d}',split,names,projects,
            tuple(map(tuple,values)),a,p,tuple(layout)))

