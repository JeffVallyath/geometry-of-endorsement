"""CSG1 reference primitives: current-state grounding, not a semantic oracle.

These functions receive only candidate-visible records, labels, and original text.
They neither read evaluation answers nor choose a query's intended path.
"""
from __future__ import annotations
import json
from collections import defaultdict
from typing import Any, Mapping

class GroundingError(ValueError):
    pass

def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def materialize(initial: list[list[str]], edits: list[list[str]]) -> dict[tuple[str,str], str]:
    state: dict[tuple[str,str], str] = {}
    for s,p,o in initial:
        if (s,p) in state and state[s,p] != o:
            raise GroundingError("CONFLICTING_INITIAL_KEY")
        state[s,p] = o
    seen = set()
    for s,p,o in edits:
        if (s,p) in seen:
            raise GroundingError("DUPLICATE_EDIT_KEY")
        seen.add((s,p))
    state.update({(s,p):o for s,p,o in edits})
    return state

def active_labels(state: Mapping[tuple[str,str], str],
                  labels: Mapping[str, list[str]]) -> dict[str,list[str]]:
    nodes = {s for s,p in state} | set(state.values())
    result = {}
    for node in sorted(nodes):
        vals = labels.get(node)
        if not vals or any(not isinstance(v,str) or not v for v in vals):
            raise GroundingError("MISSING_PUBLIC_LABEL")
        result[node] = sorted(set(vals))
    return result

def current_payload(initial: list[list[str]], edits: list[list[str]],
                    labels: Mapping[str,list[str]], registry: Mapping[str,list[str]]) -> dict:
    """No event log, path order, prior version number, or inactive-name catalog."""
    state = materialize(initial, edits)
    labs = active_labels(state, labels)
    edges = []
    for (s,p),o in sorted(state.items()):
        if p not in registry:
            raise GroundingError("UNKNOWN_RELATION")
        edges.append({"subject":{"id":s,"labels":labs[s]},
                      "relation":{"id":p,"labels":list(registry[p])},
                      "object":{"id":o,"labels":labs[o]}})
    return {"current_records":edges,"relation_registry":dict(sorted(registry.items()))}

def enumerate_paths(initial: list[list[str]], edits: list[list[str]],
                    labels: Mapping[str,list[str]], question: str,
                    max_depth: int=4, max_candidates: int=64) -> list[dict]:
    """All forward prefixes from every exact literal anchor, not only terminals.

    No case folding, fuzzy matching, gold start hint, longest-path selection, or
    answer-guided pruning. Repeated edges/nodes are allowed up to max_depth.
    A candidate's existence establishes executability, not requested meaning.
    """
    if not isinstance(question,str) or not 1 <= max_depth <= 4:
        raise GroundingError("INVALID_INPUT")
    state = materialize(initial, edits)
    labs = active_labels(state, labels)
    anchors = [(node,lab) for node,values in labs.items()
               for lab in values if lab in question]
    adj = defaultdict(list)
    for (s,p),o in sorted(state.items()):
        adj[s].append((p,o))
    paths = {}
    for start,literal in anchors:
        stack = [(start,(),())]
        while stack:
            node,rels,edges = stack.pop()
            if rels:
                key = (start,rels)
                record = {"start_id":start,"entity":literal,"path":list(rels),
                          "edges":[list(e) for e in edges],"endpoint":node}
                if key not in paths or literal < paths[key]["entity"]:
                    paths[key] = record
                if len(paths)>max_candidates:
                    raise GroundingError("CANDIDATE_BUDGET_EXCEEDED")
            if len(rels)<max_depth:
                for rel,obj in reversed(adj[node]):
                    stack.append((obj,rels+(rel,),edges+((node,rel,obj),)))
    return [paths[key] for key in sorted(paths)]

def render_exact_public_id(text: str | None, labels: Mapping[str,list[str]]) -> tuple[str | None,bool]:
    """Transport-only diagnostic/codec. Unknown text is retained, never guessed.

    Exact source-ID -> source label. One terminal period is handled as in PST1.
    No answer aliases, evaluator expected ID, casefolding, substring answer search,
    or prose cleanup. User code should retain both original and converted output.
    """
    if text is None:
        return None,False
    key = text.strip()
    if key.endswith("."):
        key = key[:-1]
    if key in labels:
        vals = labels[key]
        if not vals:
            raise GroundingError("EMPTY_PUBLIC_LABEL")
        return vals[0],True
    return text,False
