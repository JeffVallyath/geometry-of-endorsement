"""Frozen new answer wrappers and scheduling; historical prompt template is unknown.
This is a new controlled comparison on old source bytes, not exact rerun of 2026 MQMH.
"""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
from state_grounding import current_payload
from ranker_contract import request as rank_request
from transport import materialize,current_text
from prepare import candidates

ANSWER_SYSTEM='''Answer the question using only the supplied source records. The records may be counterfactual: do not override them with outside knowledge. Each SET correction replaces the corresponding earlier record, even if it repeats its value; all other records remain unchanged. A preparation note, when supplied, is not a new source of facts and cannot override the records. Reply with one entity name as written in the sources or one exact supplied entity ID. If the requested answer is unavailable, reply UNSUPPORTED. Do not give reasoning or a sentence.'''
REFLECTION_SYSTEM='''Read the source records and any corrections. Before seeing a question, summarize which facts are currently authoritative after those corrections and consider consequences that follow from the supplied relationships. Preserve unchanged facts. Do not import outside knowledge, invent changes, or guess any future question. If there are no corrections, keep the supplied snapshot unchanged. This note is assistance, not an authoritative update. Use at most 256 tokens.'''
ANSWER_CUE='Return only one entity name, one supplied entity ID, or UNSUPPORTED.'

def history_source(case,history,phase):
    data=case['histories'][history]
    return data['source_text'] if phase=='post' else data['source_text'].split('\n\nCorrections:',1)[0]

def record_view(case,history,phase):
    data=case['histories'][history];labels={k:[v] for k,v in case['labels'].items()}
    registry=json.loads((ROOT/'mapping/RELATION_REGISTRY.json').read_bytes())
    edits=data['edits'] if phase=='post' else []
    edges=materialize(data['initial'],edits)
    return edges,current_payload(data['initial'],edits,labels,registry)

def answer_request(case,history,phase,question,arm,note=None):
    if arm in ('HISTORY','REFLECT'):
        source=history_source(case,history,phase)
    elif arm=='CURRENT_TEXT':
        edges,_=record_view(case,history,phase);source=current_text(edges,case['labels'])
    elif arm=='CURRENT_JSON':
        _,payload=record_view(case,history,phase);source=json.dumps(payload,ensure_ascii=False,separators=(',',':'))
    else:raise ValueError('NOT_ANSWER_ARM')
    messages=[{'role':'system','content':ANSWER_SYSTEM}]
    if arm=='REFLECT':
        if not isinstance(note,str):raise ValueError('MISSING_SHARED_PREQUESTION_NOTE')
        messages.extend([{'role':'user','content':source},{'role':'assistant','content':note},{'role':'user','content':question+'\n'+ANSWER_CUE}])
    else:messages.append({'role':'user','content':source+'\n\n'+question+'\n'+ANSWER_CUE})
    return {'messages':messages,'max_new_tokens':64,'do_sample':False}

def reflection_request(case,history,phase):
    return {'messages':[{'role':'system','content':REFLECTION_SYSTEM},{'role':'user','content':history_source(case,history,phase)}],'max_new_tokens':256,'do_sample':False}

def score_requests(case,history,phase,question):
    edges,payload=record_view(case,history,phase)
    labels={k:[v] for k,v in case['labels'].items()}
    paths=candidates(edges,labels,question)
    return paths,[rank_request(question,payload,p) for p in paths]

from schedule_plan import make_schedule
