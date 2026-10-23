"""Source-faithful question/ontology GMeLLo interfaces, no repair imports.

Inputs are neutral source facts, actual correction sentences, the global ontology,
four fixed TRAIN demonstrations, and raw base-model output. No evaluator data.
"""
from dataclasses import asdict
import hashlib
import json
import re
import unicodedata
from chain_interface import Chain, InterfaceError, Linker, normalize, parse_chain, parse_update
from independent_graph import IndependentGraph, graph_priority

ENDPOINTS = ("G_CORE", "G_FULL", "G_TEXT")

def canonical(value): return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()

def request(system, user, cap):
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_new_tokens": cap, "do_sample": False, "stop_strings": []}

def ontology_text(ontology):
    return "\n".join(key + ": " + "; ".join(labels) for key, labels in sorted(ontology.items()))

def chain_request(question, ontology, examples, revision=0):
    """This signature deliberately cannot receive state, paths, labels or a query tag."""
    assert len(examples) == 4 and revision in (0, 1, 2)
    instruction = (
        "Extract the relation chain expressed by the question. Choose relations from the global ontology below. "
        "Begin with the entity named in the question, then alternate a relation and its next entity. "
        "Represent unknown intermediate entities and the unknown answer by distinct variables such as ?x, ?y, ?z, ?w. "
        "Do not guess factual names for unknown entities. Output a single line: Relation Chain: entity->relation->?x->relation->?y. "
        "Use the question's meaning, not outside factual knowledge. Do not answer the question or add explanation."
    )
    if revision >= 1:
        instruction += " Every relation must be followed by a node. The final unknown node is a variable. Keep explicitly named entities concrete and use variables for unknown entities."
    if revision >= 2:
        instruction += " Read nested phrases from the named entity outward, preserving every requested relationship in its original order. Emit only the arrow chain, on one line."
    samples = "\n\n".join("Question: " + ex["question"] + "\nRelation Chain: " + ex["chain"] for ex in examples)
    return request(instruction + "\n\nGlobal relation ontology:\n" + ontology_text(ontology) + "\n\n" + samples,
                   "Question: " + question + "\nRelation Chain:", 256)

def edit_request(sentence, ontology, examples):
    assert len(examples) == 4
    instruction = (
        "Extract the factual relation in the supplied correction sentence as one triple. "
        "Select a relation from the global ontology. Both entities are concrete names in the sentence. "
        "Output one line: Relation Chain: subject->relation->object. No variables or explanation."
    )
    samples = "\n\n".join("Sentence: " + ex["edit_sentence"] + "\nRelation Chain: " + ex["edit_chain"] for ex in examples)
    return request(instruction + "\n\nGlobal relation ontology:\n" + ontology_text(ontology) + "\n\n" + samples,
                   "Sentence: " + sentence + "\nRelation Chain:", 128)

def fallback_request(question, facts, examples):
    instruction = (
        "Answer the question using the supplied facts. Facts can be counterfactual; accept the supplied facts as authoritative. "
        "Give only one final entity name, with no explanation. If the answer cannot be supported, output UNSUPPORTED."
    )
    samples = "\n\n".join("Facts:\n" + "\n".join(ex["facts"]) + "\nQuestion: " + ex["question"] + "\nAnswer: " + ex["answer"] for ex in examples)
    return request(instruction + "\n\n" + samples, "Facts:\n" + "\n".join(facts) + "\nQuestion: " + question + "\nAnswer:", 128)

class RecordingLinker(Linker):
    def __init__(self, entities, relations):
        super().__init__(entities, relations)
        self.events = []

    def entity(self, text):
        event = {"role": "node", "generated": text}
        try:
            result = super().entity(text)
            event.update(normalized=result, method="source-id-or-exact-alias")
            return result
        except InterfaceError as exc:
            event["error"] = str(exc); raise
        finally: self.events.append(event)

    def relation(self, text, nearest=None):
        event = {"role": "relation", "generated": text}
        def logged_nearest(value):
            result, retrieval = nearest(value)
            event.update(method="contriever-nearest", retrieval=retrieval)
            return result
        try:
            result = super().relation(text, logged_nearest if nearest else None)
            event.setdefault("method", "ontology-id-or-exact-alias")
            event["normalized"] = result
            return result
        except InterfaceError as exc:
            event["error"] = str(exc); raise
        finally: self.events.append(event)

def relation_mapper(ontology, retrieve):
    options = [(key, label) for key, labels in sorted(ontology.items()) for label in labels]
    def nearest(text):
        result = retrieve(text, [label for _, label in options], 1)
        assert len(result) == 1
        return options[result[0]["index"]][0], result
    return nearest

def parse_generated(generation, entities, ontology, retrieve):
    linker = RecordingLinker(entities, ontology)
    trace = {"status": "INVALID_QUERY", "decisions": linker.events, "chain": None}
    if generation.get("cap_hit") or generation.get("invalid_control_tokens"):
        trace["error"] = "CAP_HIT" if generation.get("cap_hit") else "INVALID_CONTROL_TOKENS"
        return None, trace
    try:
        chain = parse_chain(generation["text"], linker, relation_mapper(ontology, retrieve))
        trace.update(status="VALID", chain=asdict(chain))
        return chain, trace
    except InterfaceError as exc:
        trace["error"] = str(exc)
        return None, trace

def query_generated(graph, chain):
    if chain is None: return {"status": "INVALID_QUERY", "entity": None}
    try: return graph.query(chain)
    except InterfaceError as exc: return {"status": str(exc), "entity": None}

def text_update(generation, entities, ontology, retrieve, person_detector):
    """Appendix A.5: only explicitly named in-ontology person-valued relations.

    'chief of department' is absent from this task ontology, so no invented alias
    expands that rule to CEO, chairperson or other relations.
    """
    linker = RecordingLinker(entities, ontology)
    trace = {"status": "UNAPPLIED", "decisions": linker.events, "swap": False, "triple": None}
    if generation.get("cap_hit") or generation.get("invalid_control_tokens"):
        trace["error"] = "CAP_HIT_OR_INVALID_CONTROL_TOKENS"; return trace
    try:
        triple = parse_update(generation["text"], linker, relation_mapper(ontology, retrieve))
        trace["before_direction_rule"] = list(triple)
        if triple[1] in ("P35", "P6"):
            object_name = entities[triple[2]][0]
            person, evidence = person_detector(object_name)
            trace["person_detection"] = {"text": object_name, "is_person": person, "entities": evidence}
            if not person:
                triple = (triple[2], triple[1], triple[0]); trace["swap"] = True
        trace.update(status="APPLIED", triple=list(triple))
    except InterfaceError as exc: trace["error"] = str(exc)
    return trace

def graph_for_history(history, entities, text_updates=None, phase="post"):
    labels = {key: values[0] for key, values in entities.items()}
    graph = IndependentGraph(history["initial"], labels)
    if phase == "post":
        edits = history["edits"] if text_updates is None else [u["triple"] for u in text_updates if u["status"] == "APPLIED"]
        graph.apply(edits)
    return graph

def current_facts(graph, ontology, templates):
    facts = []
    for s, p, o in graph.triples():
        template = templates.get(p, "{s} -- " + ontology[p][0] + " --> {o}.")
        facts.append(template.format(s=graph.labels[s], o=graph.labels[o]))
    return facts

def fallback_evidence(endpoint, question, graph, edit_sentences, ontology, templates, retrieve, phase="post"):
    if endpoint not in ENDPOINTS: raise ValueError("UNSUPPORTED_ENDPOINT")
    pool = edit_sentences if endpoint == "G_TEXT" and phase == "post" else [] if endpoint == "G_TEXT" else current_facts(graph, ontology, templates)
    if endpoint == "G_FULL":
        selected = [{"index": i, "fact": fact, "score": None} for i, fact in enumerate(pool)]
    else: selected = retrieve(question, pool, min(6, len(pool))) if pool else []
    return [pool[row["index"]] for row in selected], {"pool": pool, "selected": selected, "pool_sha256": digest(pool)}

def decode_answer(text, entities, cap_hit=False, invalid_control_tokens=False):
    # Same whole-response codec as the original comparison, including case,
    # punctuation, multiline and refusal rules; no looser comparator scoring.
    def norm(value): return " ".join(unicodedata.normalize("NFC", value).split())
    if cap_hit or invalid_control_tokens: return {"entity": None, "status": "INVALID_OUTPUT"}
    lines = text.strip().splitlines()
    if not lines: return {"entity": None, "status": "INVALID_OUTPUT"}
    if len(lines) > 1: return {"entity": None, "status": "MULTILINE_OUTPUT"}
    line = norm(lines[0])
    if line == "UNSUPPORTED": return {"entity": None, "status": "REFUSAL"}
    forms = {line}
    if line.startswith("Answer:"): forms.add(line[len("Answer:"):].strip())
    for value in list(forms): forms.add(value[:-1].rstrip() if value.endswith(".") else value)
    for value in list(forms):
        for left, right in [('"', '"'), ("'", "'"), ("\u201c", "\u201d"), ("\u2018", "\u2019")]:
            if len(value) >= 2 and value.startswith(left) and value.endswith(right):
                inner = value[1:-1].strip(); forms.add(inner)
                forms.add(inner[:-1].rstrip() if inner.endswith(".") else inner)
    aliases = {}
    for key, names in entities.items():
        for name in names: aliases.setdefault(norm(name), set()).add(key)
    unique = {name: next(iter(ids)) for name, ids in aliases.items() if len(ids) == 1}
    found = {unique[norm(value)] for value in forms if norm(value) in unique}
    found |= {value for value in forms if value in entities}
    if len(found) != 1: return {"entity": None, "status": "AMBIGUOUS_OUTPUT" if found else "UNRECOGNIZED_OUTPUT"}
    return {"entity": next(iter(found)), "status": "VALID"}

def combined_answer(graph_result, fallback_generation, entities):
    chosen = graph_priority(graph_result, fallback_generation["text"])
    if chosen["route"] == "graph":
        outcome = {"status": "VALID", "entity": chosen["entity"]}
    else:
        outcome = decode_answer(fallback_generation["text"], entities, fallback_generation.get("cap_hit", False), fallback_generation.get("invalid_control_tokens", False))
    return {**chosen, "outcome": outcome}

def legacy_sensitivity(text, graph, entities, ontology, retrieve):
    """Original literal A.4 parser on the SAME saved generation, never a selector.

    Its label-only start binding and nonvariable-as-relation behavior are retained.
    Relation scores are separately recorded; no new neural chain is requested.
    """
    trace = {"status": "INVALID_QUERY", "chain": None, "normalization": []}
    lines = [line.strip() for line in text.splitlines() if "->" in line]
    if len(lines) != 1: trace["error"] = "INVALID_CHAIN"; return trace
    parts = [v.strip() for v in re.sub(r"^Relation [Cc]hain:\s*", "", lines[0]).split("->")]
    rels = [v for v in parts[1:] if not v.startswith("?")]
    if not 1 <= len(rels) <= 4: trace["error"] = "INVALID_CHAIN_DEPTH"; return trace
    used = {node for s, _, o in graph.triples() for node in (s, o)}
    matches = [key for key, aliases in entities.items() if key in used and any(" ".join(parts[0].split()).casefold() == " ".join(alias.split()).casefold() for alias in aliases)]
    if len(matches) != 1: trace["error"] = "UNLINKED_OR_AMBIGUOUS_NAME"; return trace
    nearest = relation_mapper(ontology, retrieve); path = []
    for value in rels:
        exact = [key for key, aliases in ontology.items() if value == key or value in aliases]
        if len(exact) == 1: relation = exact[0]; how = "exact"; retrieval = None
        else: relation, retrieval = nearest(value); how = "contriever-nearest"
        path.append(relation); trace["normalization"].append({"generated": value, "normalized": relation, "method": how, "retrieval": retrieval})
    chain = Chain(tuple([matches[0]] + ["?legacy" + str(i) for i in range(len(path))]), tuple(path))
    trace.update(status="VALID", chain=asdict(chain), graph=query_generated(graph, chain))
    return trace
