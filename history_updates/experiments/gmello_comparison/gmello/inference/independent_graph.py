"""Independent RDF/SPARQL store scaffolding. No imports from the user's Store.

Only validated identifiers are inserted into a fixed query template. Source
triples/commands are runtime inputs. Expected paths and answers are not.
"""
from __future__ import annotations
from typing import Iterable, Mapping, Sequence
from rdflib import Graph, Namespace, URIRef
from chain_interface import Chain, InterfaceError
import re

ENT = Namespace('https://example.invalid/entity/')
REL = Namespace('https://example.invalid/relation/')

def entity(identifier: str) -> URIRef:
    if not re.fullmatch(r'Q[0-9]+', identifier):
        raise InterfaceError('BAD_ENTITY_ID')
    return ENT[identifier]

def relation(identifier: str) -> URIRef:
    if not re.fullmatch(r'P[0-9]+', identifier):
        raise InterfaceError('BAD_RELATION_ID')
    return REL[identifier]

class IndependentGraph:
    def __init__(self, triples: Iterable[Sequence[str]], labels: Mapping[str, str]):
        self.graph = Graph()
        self.labels = dict(labels)
        seen = {}
        for s,p,o in triples:
            if (s,p) in seen and seen[(s,p)] != o:
                raise InterfaceError('NONFUNCTIONAL_SOURCE_OUT_OF_SCOPE')
            seen[s,p] = o
            self.graph.add((entity(s), relation(p), entity(o)))

    def apply(self, edits: Iterable[Sequence[str]]) -> None:
        # Validate the whole supplied batch before any mutation.
        checked = [(entity(s), relation(p), entity(o)) for s,p,o in edits]
        for s,p,o in checked:
            self.graph.remove((s,p,None))
            self.graph.add((s,p,o))

    def triples(self) -> list[tuple[str,str,str]]:
        return sorted((str(s).removeprefix(str(ENT)),str(p).removeprefix(str(REL)),str(o).removeprefix(str(ENT))) for s,p,o in self.graph)

    def query(self, chain: Chain) -> dict:
        def term(node: str) -> str:
            return node if node.startswith('?') else '<'+str(entity(node))+'>'
        body = '\n'.join(f'{term(chain.nodes[i])} <{relation(p)}> {term(chain.nodes[i+1])} .' for i,p in enumerate(chain.relations))
        # Alias the final variable/constant rather than dropping constant claims.
        query = f'SELECT DISTINCT ({term(chain.nodes[-1])} AS ?answer) WHERE {{\n{body}\n}}'
        ids = sorted({str(r[0]).removeprefix(str(ENT)) for r in self.graph.query(query)})
        if len(ids) > 1:
            raise InterfaceError('NONFUNCTIONAL_QUERY_OUT_OF_SCOPE')
        if not ids:
            return {'status':'NO_RESULT','entity':None,'sparql':query}
        if ids[0] not in self.labels:
            raise InterfaceError('MISSING_SOURCE_LABEL')
        return {'status':'VALID','entity':ids[0],'answer':self.labels[ids[0]],'sparql':query}

def graph_priority(graph_result: dict, fallback_text: str) -> dict:
    """No expected answer is accepted here. A wrong graph result still wins."""
    if graph_result.get('status') == 'VALID' and graph_result.get('entity') is not None:
        return {'route':'graph','answer':graph_result['answer'],'entity':graph_result['entity']}
    return {'route':'fallback','answer':fallback_text,'entity':None}
