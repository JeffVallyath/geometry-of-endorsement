"""Model-free planning checks for the proposed fresh validation study.

This file contains no model runner or statistical claim. It builds a fixed
validation/withheld-question split from inherited question specifications.
"""
from __future__ import annotations
import hashlib
from itertools import combinations

FUNCTIONS = ("same", "both", "either")

def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def truth(kind: str, x: int, y: int) -> int:
    if type(x) is not int or type(y) is not int or x not in (0,1) or y not in (0,1):
        raise ValueError("Operands must be Boolean integers.")
    if kind == "same": return int(x == y)
    if kind == "both": return x & y
    if kind == "either": return x | y
    raise ValueError("Unknown relation function.")

def choose_pair(case_id: str, changed_counts: list[int], prefix_lengths: list[int]):
    if len(changed_counts) != 8 or len(prefix_lengths) != 8:
        raise ValueError("Eight histories required.")
    if any(type(x) is not int or x < 0 for x in changed_counts + prefix_lengths):
        raise ValueError("Invalid count or length.")
    pairs = [(a,b) for a,b in combinations(range(8),2)
             if changed_counts[a] == changed_counts[b]
             and prefix_lengths[a] == prefix_lengths[b]]
    if not pairs:
        return None
    a,b = min(pairs, key=lambda ab:digest(f"pair|{case_id}|{ab[0]}|{ab[1]}"))
    if int(digest(f"anchor|{case_id}|{a}|{b}"),16) % 2:
        a,b = b,a
    return a,b

def query_partition(case_id: str, case_index: int, questions: list[dict], draw: int):
    """Select six validation / three withheld semantic questions per code.

    Query IDs are kept only to identify immutable source rows. Outcomes and
    probabilities are not accepted by this function.
    """
    if draw not in (0,1) or case_index < 0:
        raise ValueError("Invalid draw or case index.")
    qs=[q for q in questions if f"-d{draw}-" in q["query_id"]]
    joint=[q for q in qs if q["spec"]["kind"] in FUNCTIONS]
    direct=[q["query_id"] for q in qs if q["spec"]["kind"]=="direct"]
    if len(joint)!=9 or len(direct)!=6:
        raise ValueError("Expected inherited 9 joint and 6 direct questions per draw.")
    by_pair={}
    for q in joint:
        s=q["spec"]
        pair=(s["p"], *sorted((s["a"],s["b"])))
        d=by_pair.setdefault(pair,{})
        if s["kind"] in d:
            raise ValueError("Duplicate semantic function.")
        d[s["kind"]]=q["query_id"]
    if len(by_pair)!=3 or any(set(v)!=set(FUNCTIONS) for v in by_pair.values()):
        raise ValueError("Three complete relation pairs required.")
    validation=[]; withheld=[]; across_histories=[]
    for j,(pair,qd) in enumerate(sorted(by_pair.items())):
        held=FUNCTIONS[(case_index+j)%3]
        remaining=[f for f in FUNCTIONS if f!=held]
        selected=int(digest(f"query|{case_id}|{pair}"),16)%2
        withheld.append(qd[held])
        validation.extend(qd[f] for f in remaining)
        across_histories.append(qd[remaining[selected]])
    if len(validation)!=6 or len(withheld)!=3 or set(validation)&set(withheld):
        raise AssertionError("Invalid validation/withheld partition.")
    if not set(across_histories)<=set(validation):
        raise AssertionError("History audit accessed a withheld question.")
    return {"direct":sorted(direct), "accuracy_queries":validation,
            "history_queries":across_histories,"withheld_queries":withheld,
            "accuracy_read_budget":len(validation),
            "history_read_budget":2*len(across_histories)}

def validate_decision_inputs(allowed_ids: set[str], received_ids: set[str]):
    if received_ids != allowed_ids:
        raise ValueError("Decision process input does not match permitted query IDs.")
