"""ID-keyed source transport and reference state operations, NOT a neural experiment.

The caller must label use of orig.{new_,}triples as annotation-assisted source evidence.
No function here generates a question, selects its path, or reads a benchmark answer.
The evaluator must retain raw source bytes independently for complete round trips.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Any

class ContractError(ValueError):
    pass

Triple = tuple[str, str, str]

def _triple(value: Any) -> Triple:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ContractError('TRIPLE_SHAPE')
    if not all(isinstance(v, str) and v for v in value):
        raise ContractError('TRIPLE_IDENTIFIER')
    return tuple(value)

def unique_map(triples: list[Triple]) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for subject, relation, obj in triples:
        key = (subject, relation)
        if key in out and out[key] != obj:
            raise ContractError('MULTIVALUED_LOCAL_KEY')
        out[key] = obj
    return out

@dataclass(frozen=True)
class Transport:
    initial: tuple[Triple, ...]
    edits: tuple[Triple, ...]
    labels: dict[str, tuple[str, ...]]
    relations: dict[str, tuple[str, ...]]
    questions: tuple[str, ...]
    provenance: tuple[dict[str, Any], ...]

    def candidate(self) -> dict[str, Any]:
        # No gold path, answer, alias-answer list, single-hop answer, or terminal state.
        return {'initial': [list(t) for t in self.initial],
                'edits': [list(t) for t in self.edits],
                'labels': {k: list(v) for k, v in self.labels.items()},
                'questions': list(self.questions)}


def prepare_root(row: dict[str, Any]) -> Transport:
    original = row['orig']
    labels: dict[str, set[str]] = {}
    relations: dict[str, set[str]] = {}
    by_label: dict[str, set[str]] = {}
    provenance: list[dict[str, Any]] = []
    def add_name(identity: str, label: str) -> None:
        if not isinstance(identity, str) or not identity or not isinstance(label, str) or not label:
            raise ContractError('NAME_TYPE')
        labels.setdefault(identity, set()).add(label)
        by_label.setdefault(label, set()).add(identity)
    old = [_triple(t) for t in original['triples']]
    new = [_triple(t) for t in original['new_triples']]
    for kind, triples in [('triples', old), ('new_triples', new)]:
        glosses = original[kind + '_labeled']
        if len(triples) != len(glosses):
            raise ContractError('GLOSS_LENGTH')
        for i, ((s, p, o), gloss) in enumerate(zip(triples, glosses)):
            if not isinstance(gloss, list) or len(gloss) != 3:
                raise ContractError('GLOSS_SHAPE')
            add_name(s, gloss[0]); add_name(o, gloss[2])
            relations.setdefault(p, set()).add(gloss[1])
            provenance.append({'pointer': '/orig/' + kind + '/' + str(i), 'triple': [s, p, o]})
    initial = unique_map(old)
    edits: list[Triple] = []
    for i, edit in enumerate(row['requested_rewrite']):
        ids = by_label.get(edit['subject'], set())
        if len(ids) != 1:
            raise ContractError('EDIT_SUBJECT_AMBIGUOUS_OR_ABSENT')
        sid = next(iter(ids)); pid = edit['relation_id']
        before, after = edit['target_true'], edit['target_new']
        add_name(before['id'], before['str']); add_name(after['id'], after['str'])
        key = (sid, pid)
        if key in initial and initial[key] != before['id']:
            raise ContractError('BEFORE_VALUE_CONFLICT')
        initial[key] = before['id']
        edits.append((sid, pid, after['id']))
        provenance.append({'pointer': '/requested_rewrite/' + str(i),
                           'before': [sid, pid, before['id']], 'after': list(edits[-1])})
    keys = [(s, p) for s, p, _ in edits]
    if len(keys) != len(set(keys)):
        raise ContractError('DUPLICATE_EDIT_KEY')
    touched = set(keys)
    for s, p, o in new:
        if (s, p) not in touched:
            if (s, p) in initial and initial[(s, p)] != o:
                raise ContractError('UNCHANGED_SUPPORT_CONFLICT')
            initial[(s, p)] = o
    # Integrity check ONLY. No final values are exported as a terminal-world shortcut.
    endpoint = dict(initial)
    endpoint.update({(s, p): o for s, p, o in edits})
    if any(endpoint.get((s, p)) != o for s, p, o in new):
        raise ContractError('RELEASED_ENDPOINT_MISMATCH')
    questions = row['questions']
    if not all(isinstance(q, str) for q in questions):
        raise ContractError('QUESTION_TYPE')
    return Transport(tuple(sorted((s, p, o) for (s, p), o in initial.items())),
                     tuple(edits), {k: tuple(sorted(v)) for k, v in sorted(labels.items())},
                     {k: tuple(sorted(v)) for k, v in sorted(relations.items())},
                     tuple(questions), tuple(provenance))

class VersionStore:
    """Small independently testable reference implementation; not a patch to NSR."""
    def __init__(self, triples: list[Triple], relation_ids: set[str]):
        self.relation_ids = frozenset(relation_ids)
        self._check(triples)
        self.snapshots = [unique_map(triples)]
    def _check(self, triples: list[Triple]) -> None:
        for t in triples:
            _, p, _ = _triple(t)
            if p not in self.relation_ids:
                raise ContractError('UNDECLARED_RELATION')
    def apply(self, edits: list[Triple]) -> int:
        self._check(edits)
        writes = unique_map(edits)  # validates entire transaction before publication
        new = dict(self.snapshots[-1]); new.update(writes)
        self.snapshots.append(new)
        return len(self.snapshots) - 1
    def follow(self, start: str, path: list[str], snapshot: int | None = None) -> str:
        if not 1 <= len(path) <= 4:
            raise ContractError('PATH_DEPTH')
        idx = len(self.snapshots)-1 if snapshot is None else snapshot
        if type(idx) is not int or not 0 <= idx < len(self.snapshots):
            raise ContractError('MISSING_SNAPSHOT')
        current = start
        for p in path:
            if p not in self.relation_ids: raise ContractError('UNDECLARED_RELATION')
            key = (current, p)
            if key not in self.snapshots[idx]: raise ContractError('MISSING_EDGE')
            current = self.snapshots[idx][key]
        return current
    def view(self, snapshot: int | None = None):
        idx = len(self.snapshots)-1 if snapshot is None else snapshot
        if type(idx) is not int or not 0 <= idx < len(self.snapshots):
            raise ContractError('MISSING_SNAPSHOT')
        return copy.deepcopy(self.snapshots[idx])
