"""New CPU interface scaffolding, NOT the authors' GMeLLo implementation.

Parses the paper's alternating entity/relation arrow language. Literal entities
remain constraints; this code never chooses a path by whether it gives an answer.
Retrieval normalization is injected by the inference adapter and must be logged.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
import unicodedata
from typing import Callable, Mapping, Sequence

class InterfaceError(ValueError):
    pass

def normalize(text: str) -> str:
    text = unicodedata.normalize('NFC', text.strip())
    if len(text) >= 2 and (text[0], text[-1]) in [('"','"'),("'","'"),('`','`')]:
        text = text[1:-1].strip()
    return ' '.join(text.split()).casefold()

@dataclass(frozen=True)
class Chain:
    nodes: tuple[str, ...]
    relations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.relations) <= 4 or len(self.nodes) != len(self.relations) + 1:
            raise InterfaceError('UNSUPPORTED_CHAIN_SHAPE')
        if not re.fullmatch(r'Q[0-9]+', self.nodes[0]):
            raise InterfaceError('START_MUST_BE_LINKED_ENTITY')
        if any(not re.fullmatch(r'P[0-9]+', p) for p in self.relations):
            raise InterfaceError('BAD_RELATION_ID')
        if any(not (re.fullmatch(r'Q[0-9]+', x) or re.fullmatch(r'\?[A-Za-z_][A-Za-z_0-9]*', x)) for x in self.nodes[1:]):
            raise InterfaceError('BAD_NODE')

class Linker:
    def __init__(self, entities: Mapping[str, Sequence[str]], relations: Mapping[str, Sequence[str]]):
        self.entities = {k: tuple(v) for k,v in entities.items()}
        self.relations = {k: tuple(v) for k,v in relations.items()}
        if any(not re.fullmatch(r'Q[0-9]+', k) for k in self.entities):
            raise InterfaceError('BAD_ENTITY_REGISTRY')
        if any(not re.fullmatch(r'P[0-9]+', k) for k in self.relations):
            raise InterfaceError('BAD_RELATION_REGISTRY')

    @staticmethod
    def _exact(text: str, registry: Mapping[str, Sequence[str]]) -> str | None:
        stripped = text.strip().strip('`')
        if stripped in registry:
            return stripped
        matches = {k for k,aliases in registry.items() if any(normalize(text) == normalize(a) for a in aliases)}
        if len(matches) > 1:
            raise InterfaceError('AMBIGUOUS_ALIAS')
        return next(iter(matches)) if matches else None

    def entity(self, text: str) -> str:
        result = self._exact(text, self.entities)
        if result is None:
            raise InterfaceError('UNLINKED_ENTITY')
        return result

    def relation(self, text: str, nearest: Callable[[str], str] | None = None) -> str:
        result = self._exact(text, self.relations)
        if result is not None:
            return result
        if nearest is None:
            raise InterfaceError('UNLINKED_RELATION')
        # The adapter supplies frozen Contriever nearest-label matching, never
        # path availability, target type, expected answer, or a trained scorer.
        result = nearest(text)
        if result not in self.relations:
            raise InterfaceError('BAD_RETRIEVAL_RESULT')
        return result

def arrow_parts(text: str) -> list[str]:
    cleaned = text.replace('→', '->').strip()
    if cleaned.startswith('```'):
        lines = cleaned.splitlines()
        if len(lines) < 3 or lines[-1].strip() != '```':
            raise InterfaceError('UNBALANCED_FENCE')
        cleaned = '\n'.join(lines[1:-1])
    lines = [x.strip() for x in cleaned.splitlines() if '->' in x]
    if len(lines) != 1:
        raise InterfaceError('EXPECTED_ONE_CHAIN_LINE')
    line = re.sub(r'^Relation\s+Chain\s*:\s*', '', lines[0], flags=re.IGNORECASE)
    parts = [x.strip() for x in line.split('->')]
    if any(not x for x in parts):
        raise InterfaceError('EMPTY_CHAIN_SEGMENT')
    return parts

def parse_chain(text: str, linker: Linker, nearest: Callable[[str], str] | None = None) -> Chain:
    parts = arrow_parts(text)
    if len(parts) < 3 or len(parts) % 2 == 0:
        raise InterfaceError('EXPECTED_ALTERNATING_NODE_RELATION_NODE')
    if not 1 <= (len(parts)-1)//2 <= 4:
        raise InterfaceError('UNSUPPORTED_CHAIN_DEPTH')
    nodes = [linker.entity(parts[0])]
    rels = []
    for i in range(1, len(parts), 2):
        rels.append(linker.relation(parts[i], nearest))
        node = parts[i+1]
        if re.fullmatch(r'\?[A-Za-z_][A-Za-z_0-9]*', node):
            nodes.append(node)
        else:
            nodes.append(linker.entity(node))
    return Chain(tuple(nodes), tuple(rels))

def parse_update(text: str, linker: Linker, nearest: Callable[[str], str] | None = None) -> tuple[str,str,str]:
    parts = arrow_parts(text)
    if len(parts) != 3:
        raise InterfaceError('EXPECTED_ONE_EDIT_TRIPLE')
    return linker.entity(parts[0]), linker.relation(parts[1], nearest), linker.entity(parts[2])
