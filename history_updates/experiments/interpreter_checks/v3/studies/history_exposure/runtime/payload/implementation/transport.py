"""Exact adapter for the released MQMH statement renderer, not general English.

Only printed source lines, a global template dictionary, and source name/ID
bindings determine writes. Queries, labels for correct answers, and gold paths
are not read by this module. It rejects unknown or multiply interpreted lines.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Mapping

TEMPLATES = {
 'P108': '{s} is employed by {o}.',
 'P159': 'The headquarters of {s} is located in the city of {o}.',
 'P112': '{s} was founded by {o}.',
 'P140': '{s} is affiliated with the religion of {o}.',
 'P937': '{s} worked in the city of {o}.',
 'P136': 'The type of music that {s} plays is {o}.',
 'P495': '{s} was created in the country of {o}.',
 'P50': 'The author of {s} is {o}.',
 'P170': '{s} was created by {o}.',
 'P27': '{s} is a citizen of {o}.',
 'P19': '{s} was born in the city of {o}.',
 'P6': 'The name of the current head of the {s} government is {o}.',
 'P175': '{s} was performed by {o}.',
 'P176': 'The company that produced {s} is {o}.',
 'P26': '{s} is married to {o}.',
 'P69': 'The univeristy where {s} was educated is {o}.',
 'P37': 'The official language of {s} is {o}.',
 'P36': 'The capital of {s} is {o}.',
 'P800': '{s} is famous for {o}.',
 'P178': '{s} was developed by {o}.',
 'P286': 'The head coach of {s} is {o}.',
 'P1412': '{s} speaks the language of {o}.',
 'P641': '{s} is associated with the sport of {o}.',
 'P30': '{s} is located in the continent of {o}.',
 'P106': '{s} works in the field of {o}.',
 'P449': 'The origianl broadcaster of {s} is {o}.',
 'P407': '{s} was written in the language of {o}.',
 'P740': '{s} was founded in the city of {o}.',
 'P35': 'The name of the current head of state in {s} is {o}.',
 'P169': 'The chief executive officer of {s} is {o}.',
 'P488': 'The chairperson of {s} is {o}.',
 'P20': '{s} died in the city of {o}.',
}

class TransportError(ValueError): pass

def dictionary(labels: Mapping[str,str]):
    d=defaultdict(set)
    for p,t in TEMPLATES.items():
        for s,sl in labels.items():
            for o,ol in labels.items():
                d[t.format(s=sl,o=ol)].add((s,p,o))
    return dict(d)

def parse_context(text: str, labels: Mapping[str,str]):
    """Returns every source edge and SET, preserving line order and raw text."""
    d=dictionary(labels); section=None; initial=[]; edits=[]; receipts=[]
    for i,line in enumerate(text.splitlines()):
        if not line: continue
        if line=='Initial record:':
            if section is not None: raise TransportError('DUPLICATE_INITIAL_HEADER')
            section='initial';continue
        if line=='Corrections:':
            if section!='initial':raise TransportError('CORRECTION_HEADER')
            section='edits';continue
        if section is None:raise TransportError('MISSING_HEADER')
        if section=='edits':
            if not line.startswith('SET '):raise TransportError('UNSUPPORTED_OPERATION')
            statement=line[4:]
        else: statement=line
        choices=d.get(statement,set())
        if len(choices)!=1:raise TransportError(f'NOT_UNIQUE_SOURCE_LINE:{i}:{len(choices)}')
        triple=list(next(iter(choices)))
        assert render(triple,labels)==statement
        (initial if section=='initial' else edits).append(triple)
        receipts.append({'line_number':i+1,'section':section,'text':line,'triple':triple})
    if not initial:raise TransportError('EMPTY_INITIAL')
    return {'initial':initial,'edits':edits,'receipts':receipts,'source_text':text}

def render(triple,labels):
    s,p,o=triple
    return TEMPLATES[p].format(s=labels[s],o=labels[o])

def materialize(initial,edits):
    result={};changes={}
    for s,p,o in initial:
        if (s,p) in result and result[s,p]!=o:raise TransportError('CONFLICTING_INITIAL')
        result[s,p]=o
    for s,p,o in edits:
        if (s,p) in changes:raise TransportError('DUPLICATE_EDIT')
        changes[s,p]=o
    result.update(changes)
    return sorted([s,p,o] for (s,p),o in result.items())

def current_text(edges,labels):
    return 'Initial record:\n'+'\n'.join(render(e,labels) for e in sorted(edges))
