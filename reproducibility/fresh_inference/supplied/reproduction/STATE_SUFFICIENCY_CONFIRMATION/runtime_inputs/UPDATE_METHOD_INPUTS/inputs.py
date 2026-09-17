"""Question-blind source/update boundary and immutable clone-served context.

No evaluator module is imported here. No labels, final-world dictionary, origin,
or question can be supplied to prepare(). Address intervals are half-open Unicode
character offsets in the supplied source text, not token positions.
"""
from dataclasses import dataclass, field
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
RECIPES = ('INV_PAIR_NLL', 'FREE_PAIR_CONSISTENCY')
TEXT_METHODS = ('SOURCE', 'REBUILD', 'EXISTING_CORRECTION', 'LATEST_SAME_WORDING')
CLAUSE = re.compile(r'(?P<actor>[A-Za-z]+) is (?P<stance>in favor of|opposed to) the (?P<project>[A-Za-z]+) proposal\.')

def digest(value):
    return hashlib.sha256(value.encode('utf8')).hexdigest()

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@dataclass(frozen=True)
class AddressedUpdate:
    start: int
    end: int
    value: int

    def __post_init__(self):
        if any(type(x) is not int for x in (self.start, self.end, self.value)):
            raise TypeError('Integer address bounds and bit required')
        if not 0 <= self.start < self.end or self.value not in (0, 1):
            raise ValueError('Nonempty half-open character span and binary value required')

def normalize(commands):
    """Canonical last-write normalization in source-record order; no truth lookup."""
    from canonical.setters import Command, normalize_commands
    canonical = normalize_commands([Command(tuple(range(c.start, c.end)), c.value) for c in commands])
    return tuple(AddressedUpdate(c.address[0], c.address[-1]+1, c.value) for c in canonical)

def validate(source_text, commands):
    if type(source_text) is not str or not source_text.endswith('End of records.\n'):
        raise ValueError('Expected supplied source text ending with End of records.')
    if type(commands) is not tuple or any(type(c) is not AddressedUpdate for c in commands):
        raise TypeError('Tuple of AddressedUpdate only; no evaluator records accepted')
    actual_start = source_text.rfind('Now the actual records:\n')
    for c in commands:
        if c.end > len(source_text) or c.start < actual_start:
            raise ValueError('Address outside actual source records')
        if source_text[c.start-1:c.start] != '\n' or source_text[c.end:c.end+1] != '\n':
            raise ValueError('Address must cover exactly one complete source record')
        if not CLAUSE.fullmatch(source_text[c.start:c.end]):
            raise ValueError('Address is not a canonical record clause')
    normalize(commands)  # rejects partial overlap without removing commands

def rewrite(source_text, commands, method):
    from canonical.tasks import correction_line
    from canonical.v2_protocol import Edit
    if method == 'SOURCE':
        return source_text
    if method == 'REBUILD':
        result = source_text
        for c in reversed(normalize(commands)):
            match = CLAUSE.fullmatch(source_text[c.start:c.end])
            a,b = match.span('stance')
            result = result[:c.start+a] + ('in favor of' if c.value else 'opposed to') + result[c.start+b:]
        return result
    if method not in ('EXISTING_CORRECTION', 'LATEST_SAME_WORDING'):
        raise ValueError('Unknown textual method')
    selected = commands if method == 'EXISTING_CORRECTION' else normalize(commands)
    lines = []
    for c in selected:
        match = CLAUSE.fullmatch(source_text[c.start:c.end])
        names = SimpleNamespace(actors=(match['actor'],), projects=(match['project'],))
        lines.append(correction_line(names, Edit(0, 0, c.value)))
    tail = 'End of records.\n'
    return source_text[:-len(tail)] + ''.join(line+'\n' for line in lines) + tail

def inventory(root=ROOT):
    path = Path(root)/'weights/SELECTED.json'
    if not path.is_file():
        raise FileNotFoundError('Missing required weights/SELECTED.json')
    return json.loads(path.read_text())

def load_editor(backbone, actor, recipe, seed, root=ROOT):
    """Load only hash-verified selected arrays; never select a new checkpoint."""
    from canonical.bank import SetterBank
    if actor not in ('gemma','qwen') or seed not in (0,1):
        raise ValueError('Fixed actor and original seed required')
    data = inventory(root)
    if recipe == 'CANONICAL_REPLAY':
        recipe = data['canonical_replay'][actor][str(seed)]
    elif recipe not in RECIPES:
        raise ValueError('Only fixed learned recipes or canonical replay are exposed')
    matches = [r for r in data['selected'] if (r['actor'],r['recipe'],r['seed']) == (actor,recipe,seed)]
    if len(matches) != 1:
        raise FileNotFoundError(f'Missing selected editor: {actor}/{recipe}/s{seed}')
    rec = matches[0]; path = Path(root)/rec['path']
    for filename, key in [('setter.npz','setter_sha256'),('META.json','meta_sha256')]:
        p = path/filename
        if not p.is_file():
            raise FileNotFoundError('Missing required artifact: '+rec['path']+'/'+filename)
        if file_hash(p) != rec[key]:
            raise ValueError('Artifact hash mismatch: '+rec['path']+'/'+filename)
    bank = SetterBank.load(backbone,path)
    from canonical.config import MODELS
    if (bank.actor,bank.seed,bank.site) != (actor,seed,MODELS[actor]['site']):
        raise ValueError('Editor model/seed/site mismatch')
    if bank.model.width != MODELS[actor]['hidden']:
        raise ValueError('Editor hidden width mismatch')
    bank.model.eval()
    for parameter in bank.parameters():
        parameter.requires_grad_(False)
    return bank, rec

@dataclass(frozen=True)
class UpdatedContext:
    method: str
    text: str
    source_sha256: str
    updated_text_sha256: str
    commands: tuple[AddressedUpdate,...]
    artifact_sha256: str | None
    preprocess_seconds: float
    compile_seconds: float | None
    selected_weight_sha256: str | None = None
    _artifact: object = field(default=None, repr=False, compare=False)

    def describe(self):
        return dict(schema='UPDATED_CONTEXT_V1', method=self.method,
                    source_sha256=self.source_sha256, updated_text_sha256=self.updated_text_sha256,
                    artifact_sha256=self.artifact_sha256, command_count=len(self.commands),
                    selected_weight_sha256=self.selected_weight_sha256,
                    preprocessing_seconds=self.preprocess_seconds, compile_seconds=self.compile_seconds,
                    backend='canonical_backbone' if self._artifact is not None else 'text_only_no_model',
                    model_checks='PENDING' if self._artifact is None else 'runtime_hash_checks_only')

def _sync(bb):
    if bb is not None and bb.device.type == 'cuda':
        bb.t.cuda.synchronize(bb.device)

def prepare(source_text, commands, *, method, backbone=None, actor=None, seed=None, root=ROOT):
    """source text + explicit commands -> context, before any question exists.

    Learned methods preserve the full supplied sequence. Only named normalized
    baselines and CANONICAL_REPLAY normalize; no-ops/restorations are never filtered.
    Current canonical learned/replay code recompiles the source prefix and charges
    that full work. It does not pretend to update an already cached prefix cheaply.
    """
    started = time.perf_counter()
    validate(source_text, commands)
    if method in ('LATEST_ERRATUM','FIELD_PLUS_LATEST_ERRATUM'):
        raise NotImplementedError('External Stage 1 adapter and stock-equivalence/token compatibility checks are required; no replacement implementation supplied')
    learned = method in RECIPES or method == 'CANONICAL_REPLAY'
    if method not in TEXT_METHODS and not learned:
        raise ValueError('Undeclared method')
    text = source_text if learned else rewrite(source_text, commands, method)
    active = normalize(commands) if method in ('LATEST_SAME_WORDING','CANONICAL_REPLAY') else commands
    pre_seconds = time.perf_counter()-started
    art, weight, seconds = None,None,None
    if learned and backbone is None:
        raise ValueError('Learned editing requires a supplied backbone; use REBUILD for the CPU example')
    if backbone is not None:
        if actor != backbone.cfg['key']:
            raise ValueError('Backbone actor mismatch')
        bank=None
        if learned:
            bank,record=load_editor(backbone,actor,method,seed,root)
            weight=record['setter_sha256']
        _sync(backbone); t0=time.perf_counter()
        ids=tuple(backbone.prefix_ids(text))
        if learned:
            from canonical.request import WriterInput
            from canonical.writer import writer_plan
            from canonical.bank import canonical_program
            positions=tuple(tuple(backbone.clause_positions(text,(c.start,c.end))) for c in commands)
            values=tuple(c.value for c in commands)
            if method == 'CANONICAL_REPLAY':
                positions,values=canonical_program(positions,values)
            request=WriterInput(ids,tuple(map(tuple,positions)),tuple(values))
            plan=writer_plan(bank,request)
            art=backbone.compile(ids,maps=plan['maps'],index=plan['index'],site=plan['site'],workspace='fp32',grad=False)
            if len(positions) and art['realized']['maps'] != len(positions):
                raise RuntimeError('An explicit assignment was skipped')
        else:
            art=backbone.compile(ids,grad=False)
        _sync(backbone); seconds=time.perf_counter()-t0
        if backbone.cache_hash(art['cache']) != art['hash']:
            raise RuntimeError('Compiled cache hash mismatch')
    return UpdatedContext(method,text,digest(source_text),digest(text),active,
                          art['hash'] if art else None,pre_seconds,seconds,weight,art)

def answer(context, question, answer_labels, *, backbone):
    """Separate serving boundary. Gold labels/world/origin are not accepted.

    answer_labels are the public answer symbols already stated in the question,
    in semantic [false,true] order, NOT a gold index. Scoring truth is separate.
    """
    from canonical.tasks import suffix_text
    if type(context) is not UpdatedContext or context._artifact is None:
        raise ValueError('Model-compiled UpdatedContext required; text-only demo cannot answer')
    if type(question) is not str or len(answer_labels)!=2 or len(set(answer_labels))!=2:
        raise ValueError('One question and two distinct answer symbols required')
    if digest(context.text)!=context.updated_text_sha256:
        raise ValueError('Updated text changed')
    art=context._artifact
    if backbone.cache_hash(art['cache'])!=context.artifact_sha256:
        raise ValueError('Master cache changed before serving')
    suffix=suffix_text(question)
    enc=backbone.encode(context.text,suffix)
    if enc['prefix_ids']!=art['prefix_ids']:
        raise ValueError('Question crossed immutable prefix token boundary')
    label_ids=backbone.label_ids(enc['full_text'],answer_labels)
    _sync(backbone); started=time.perf_counter()
    result=backbone.ask_batch(art,[enc['suffix_ids']],[label_ids],verify_master=True)[0]
    _sync(backbone); seconds=time.perf_counter()-started
    if not result.get('master_unchanged'):
        raise RuntimeError('Master cache changed during serving')
    valid=bool(result.get('finite',False) and not result['tie']
               and all(math.isfinite(v) for v in result['logps']))
    return dict(schema='SCORED_ANSWER_V1',question=question,question_sha256=digest(question),
                answer=answer_labels[result['prediction']],prediction=result['prediction'],
                logps=result['logps'],answer_mass=result['answer_mass'],
                argmax_in_labels=result['argmax_in_labels'],tie=result['tie'],valid=valid,
                master_unchanged=True,artifact_sha256=context.artifact_sha256,
                label_token_ids=label_ids,suffix_ids=enc['suffix_ids'],suffix_seconds=seconds)

def commands_from_json(rows):
    result=[]
    for row in rows:
        if set(row)!={'start','end','value'}:
            raise ValueError('Update command schema forbids evaluator fields')
        result.append(AddressedUpdate(**row))
    return tuple(result)
