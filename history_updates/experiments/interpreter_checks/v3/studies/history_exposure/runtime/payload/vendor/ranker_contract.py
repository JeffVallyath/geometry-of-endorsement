"""Pure, untrained ranker interface. Does not infer meaning or edit state."""
from __future__ import annotations
import json, math
from typing import Any, Mapping, Sequence

SYSTEM = '''Decide whether the supplied candidate program expresses the complete original question.
All supplied current records are authoritative, including counterfactual records. The candidate is executable;
executability alone is not enough. Its starting entity, every relation, direction, and stopping point must match
what the question requests. Do not prefer the longest path, answer from real-world knowledge, or substitute a
final entity for the question's named start. Return Yes only for the complete requested program, otherwise No.
The judgment is about the program's meaning, not whether its endpoint is a true fact.'''

def key(candidate: Mapping[str,Any]) -> str:
    return json.dumps([candidate['start_id'], candidate['entity'], candidate['path']], ensure_ascii=False, separators=(',',':'))

def request(question: str, current_payload: Mapping[str,Any], candidate: Mapping[str,Any]) -> dict:
    if not isinstance(question,str) or not question: raise ValueError('EMPTY_QUESTION')
    if candidate['entity'] not in question: raise ValueError('NONLITERAL_ANCHOR')
    if not 1<=len(candidate['path'])<=4: raise ValueError('PATH_LENGTH')
    # No list position, option label, root identity, query-family flag, answer or gold program.
    program={'start_id':candidate['start_id'], 'entity':candidate['entity'], 'path':candidate['path'],
             'snapshot':'CURRENT'}
    registry=current_payload['relation_registry']
    if any(p not in registry for p in candidate['path']): raise ValueError('UNREGISTERED_RELATION')
    payload={'question':question,'relation_registry':registry,
             'current_records':current_payload['current_records'],'candidate_program':program}
    return {'messages':[{'role':'system','content':SYSTEM},
                        {'role':'user','content':json.dumps(payload,ensure_ascii=False,separators=(',',':'))}]}

def choose(candidates: Sequence[Mapping[str,Any]], margins: Sequence[float]) -> dict|None:
    if len(candidates)!=len(margins):raise ValueError('LENGTH_MISMATCH')
    if not candidates:return None
    if any(not math.isfinite(float(s)) for s in margins):raise ValueError('NONFINITE_SCORE')
    # Zero is a predeclared binary-decision boundary, not an outcome-fitted confidence threshold.
    # Stable semantic tie-breaking is independent of presentation order.
    ranked=sorted(zip(candidates,margins),key=lambda x:(-float(x[1]),key(x[0])))
    candidate,score=ranked[0]
    return None if score<=0 else dict(candidate)
