"""Thin snapshot wrapper around the unchanged PST1/QSA1 Store and query method."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'vendor'))
from pst1_core import Store

def build_store(case,history):
    source=case['histories'][history];s=Store(source['initial'])
    if source['edits']:s.apply(source['edits'])
    return s

def execute(case,history,phase,selected):
    store=build_store(case,history)
    if selected is None:return {'entity':None,'status':'REFUSAL'}
    result=store.query(selected['start_id'],selected['path'],0 if phase=='pre' else None)
    if result['endpoint']!=selected['endpoint'] or result['dependencies']!=selected['edges']:
        raise ValueError('SELECTED_PROGRAM_STATE_MISMATCH')
    return {'entity':result['endpoint'],'answer':case['labels'][result['endpoint']],'status':'VALID','trace':result}
