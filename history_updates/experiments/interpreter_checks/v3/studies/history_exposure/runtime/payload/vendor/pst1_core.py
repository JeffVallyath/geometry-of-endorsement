"""PST1 candidate-only schema-parametric route. No evaluator or answer labels."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'governing/implementation'))
from public_graph import ContractError, VersionStore

REGISTRY = json.loads((ROOT/'governing/audit/RELATION_REGISTRY.json').read_bytes())
assert len(REGISTRY) == 37

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()

SCHEMA_DIGEST = digest(REGISTRY)
PROGRAM_SYSTEM = '''Interpret the original question against the complete relation registry below.
The records in this task may be counterfactual. Do not predict an answer and do not use real-world truth to override supplied records.
Output ONLY one JSON object: {"op":"FOLLOW","entity":"literal name from the question","path":["P..."],"snapshot":"CURRENT"}, or {"op":"UNSUPPORTED"}.
The entity must be an exact, case-sensitive, contiguous literal span in the question. Do not output an entity ID. No spelling changes, aliases, or name normalization.
The path is the ordered sequence of 1 to 4 forward relations expressly requested by the question, starting from that entity. Include all requested hops; relations may repeat. A type-valid path can still be semantically wrong.
CURRENT is the requested snapshot. Do not output a final answer, code, explanations, or additional fields.
Use the complete global registry; no relation is implied merely because it appears in supplied records.
COMPLETE GLOBAL RELATION REGISTRY:
''' + canonical(REGISTRY)

ANSWER_SYSTEM = '''Answer the original question using only this task's supplied records and the complete relation registry below.
Records may be counterfactual: never use external knowledge to negate them. If an atomic SET batch is supplied, apply every SET simultaneously to the preceding records before answering; other records remain unchanged.
Use the supplied entity labels. Output ONLY a single entity label, or UNSUPPORTED if the question cannot be answered from these records. Do not output an entity ID, a sentence, reasoning, or JSON.
COMPLETE GLOBAL RELATION REGISTRY:
''' + canonical(REGISTRY)

PROGRAM_SCHEMA = {
    'oneOf': [
        {'type': 'object', 'properties': {'op': {'const': 'UNSUPPORTED'}}, 'required': ['op'], 'additionalProperties': False},
        {'type': 'object', 'properties': {
            'op': {'const': 'FOLLOW'}, 'entity': {'type': 'string', 'minLength': 1},
            'path': {'type': 'array', 'items': {'type': 'string', 'enum': sorted(REGISTRY)}, 'minItems': 1, 'maxItems': 4},
            'snapshot': {'const': 'CURRENT'}},
         'required': ['op', 'entity', 'path', 'snapshot'], 'additionalProperties': False}
    ]
}

MODEL_ID = {
    'base': 'Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28',
    'seed0': '8d0167d7c057716aaad77974708c324fbf3fd78ab4b8748044754e00bc3e6078',
    'seed1': 'fba15b25793b925b1ac3d11a5dd0edc39772f2a958d49a7cc544472ec363f9ec'
}

class Store(VersionStore):
    def __init__(self, triples, registry=REGISTRY):
        super().__init__(triples, set(registry))
        self.schema_digest = digest(registry)
        self.cache = {}
        self.hits = 0

    def apply(self, edits):
        self._check(edits)
        keys=[(s,p) for s,p,_ in edits]
        if len(keys)!=len(set(keys)):
            raise ContractError('DUPLICATE_EDIT_KEY')
        return super().apply(edits)

    def query(self, start, path, snapshot=None):
        idx = len(self.snapshots)-1 if snapshot is None else snapshot
        # Failures, including missing snapshot and missing edge, are not cached.
        if type(idx) is not int or not 0 <= idx < len(self.snapshots):
            raise ContractError('MISSING_SNAPSHOT')
        if not isinstance(path, list) or not 1 <= len(path) <= 4:
            raise ContractError('PATH_DEPTH')
        if any(p not in self.relation_ids for p in path):
            raise ContractError('UNDECLARED_RELATION')
        key = (self.schema_digest, idx, start, tuple(path))
        if key in self.cache:
            self.hits += 1
            return dict(self.cache[key], cache_hit=True)
        cur, deps = start, []
        for pid in path:
            value = self.snapshots[idx].get((cur, pid))
            if value is None:
                raise ContractError('MISSING_EDGE')
            deps.append([cur, pid, value])
            cur = value
        out = dict(endpoint=cur, snapshot=idx, dependencies=deps, schema_digest=self.schema_digest, cache_hit=False)
        self.cache[key] = out
        return dict(out)

def clean_output(text):
    # Only terminal model control markers are stripped. No label/name normalization.
    text = text.strip()
    for token in ('<|im_end|>', '<|endoftext|>'):
        if text.endswith(token):
            text = text[:-len(token)].rstrip()
    return text

def parse_program(text):
    """Mechanical errors only; the returned error may be used in the one retry."""
    try:
        obj = json.loads(clean_output(text))
    except (ValueError, TypeError):
        return None, 'JSON_SYNTAX'
    if not isinstance(obj, dict):
        return None, 'OBJECT_TYPE'
    if obj == {'op': 'UNSUPPORTED'}:
        return obj, None
    if set(obj) != {'op','entity','path','snapshot'}:
        return None, 'OBJECT_FIELDS'
    if obj['op'] != 'FOLLOW':
        return None, 'OP_TYPE'
    if not isinstance(obj['entity'], str) or not obj['entity']:
        return None, 'ENTITY_STRING_TYPE'
    if obj['snapshot'] != 'CURRENT':
        return None, 'SNAPSHOT_TYPE'
    if not isinstance(obj['path'], list) or not 1 <= len(obj['path']) <= 4:
        return None, 'PATH_ARRAY_LENGTH'
    if any(not isinstance(p, str) or p not in REGISTRY for p in obj['path']):
        return None, 'UNREGISTERED_PROPERTY_TYPE'
    return obj, None

def execute(root, question, obj, store=None):
    if obj['op'] == 'UNSUPPORTED':
        return dict(error='UNSUPPORTED', answer=None)
    literal = obj['entity']
    if literal not in question:
        return dict(error='NAME_NOT_LITERAL_IN_QUESTION', answer=None)
    ids = [k for k, values in root['labels'].items() if literal in values]
    if len(ids) != 1:
        return dict(error='NAME_ABSENT' if not ids else 'NAME_AMBIGUOUS', answer=None)
    if store is None:
        store = Store(root['initial'])
        store.apply(root['edits'])
    try:
        result = store.query(ids[0], obj['path'])
    except ContractError as exc:
        return dict(error=str(exc), answer=None, start_id=ids[0])
    labels = root['labels'].get(result['endpoint'], [])
    if not labels:
        return dict(error='ENDPOINT_LABEL_ABSENT', answer=None, **result)
    # Deterministic public-label rendering independent of evaluator answer/aliases.
    return dict(error=None, answer=labels[0], source_labels=labels, start_id=ids[0], **result)

def view(root, condition):
    if condition not in ('H0','H1','C','C+'):
        raise ValueError('condition')
    if condition.startswith('H'):
        triples = list(root['initial'])
        if condition == 'H1':
            triples.reverse()
    else:
        state = Store(root['initial'])
        state.apply(root['edits'])
        triples = sorted([s,p,o] for (s,p),o in state.view().items())
    result = dict(records=triples, entity_labels=root['labels'])
    if condition != 'C':
        result['atomic_SET_batch'] = root['edits']
    return result

def make_request(root, question, arm, condition):
    program = 'PROGRAM' in arm or arm.startswith('SCA1_')
    isolated = arm == 'ISOLATED_PROGRAM' or arm.startswith('SCA1_')
    content = {'question': question}
    if not isolated:
        content['supplied_records'] = view(root, condition)
    req = dict(messages=[{'role':'system','content':PROGRAM_SYSTEM if program else ANSWER_SYSTEM},
                         {'role':'user','content':canonical(content)}], max_new=192 if program else 96,
               schema=PROGRAM_SCHEMA if program else None,
               schema_name='PST1_PROGRAM_'+SCHEMA_DIGEST if program else None,
               lora='seed0' if arm == 'SCA1_0_ISOLATED' else 'seed1' if arm == 'SCA1_1_ISOLATED' else None)
    return req

def retry_request(req, output, mechanical_error):
    return dict(req, messages=req['messages'] + [
        {'role':'assistant','content':output},
        {'role':'user','content':'Mechanical format/type error: '+mechanical_error+'. Output only a valid object under the same schema. No other feedback is provided.'}])

def request_key(req):
    # Full actual prompts, schema, token policy and immutable weight identities.
    return digest({'request':req, 'base':MODEL_ID['base'], 'adapter':MODEL_ID.get(req.get('lora')), 'greedy':True})

ARM_CONDITIONS = {
    'HISTORY_ANSWER': ('H0','H1','C+'), 'CURRENT_ANSWER': ('C',),
    'HISTORY_PROGRAM': ('H0','H1'), 'ISOLATED_PROGRAM': ('H0',),
    'SCA1_0_ISOLATED': ('H0',), 'SCA1_1_ISOLATED': ('H0',)
}

def tasks(root, regression=False):
    out = []
    for qidx, question in enumerate(root['questions']):
        for arm, conditions in ARM_CONDITIONS.items():
            if regression and 'PROGRAM' not in arm and not arm.startswith('SCA1_'):
                continue
            for cond in conditions:
                out.append(dict(kind='target', index=qidx, question=question, arm=arm, condition=cond))
    if not regression:
        for idx, question in enumerate(root['control_questions']):
            for cond in ('H0','H1','C','C+'):
                out.append(dict(kind='guard', index=idx, question=question, arm='HISTORY_ANSWER', condition=cond))
    return out
