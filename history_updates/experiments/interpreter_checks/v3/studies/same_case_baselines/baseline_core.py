"""Pure supplied-current-record endpoints. No evaluator imports or runtime gold paths."""
import hashlib
import json
import re
import sqlite3
import unicodedata

ENDPOINTS=['REPAIR_0','REPAIR_1','QA_0','QA_1','PROGRAM_0','PROGRAM_1','MELLO_CURRENT','GMELLO_LOCAL']
HISTORIES=['H0','H1','S0','S1','C','C+']
QA_SYSTEM='Answer the question using only the supplied authoritative current records and relation registry. Records may be counterfactual; never override them with outside knowledge. Output only one entity name as written in the records, or UNSUPPORTED. Do not explain.'
PROGRAM_SYSTEM='Interpret the question using the supplied authoritative current records and complete relation registry. Records may be counterfactual. Output only a JSON object with exactly start_id and path: {"start_id":"Q...","path":["P...", "P..."]}. The start is the entity asked about, not the final answer. Follow every requested relation in order, from one to four forward hops. Use only supplied entity IDs and the global relation registry. If unsupported, output {"unsupported":true}. Do not output an answer, code or explanation.'

def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def digest(x): return hashlib.sha256(canon(x).encode()).hexdigest()
def norm(x): return ' '.join(unicodedata.normalize('NFC',x).split())
def neural_request(messages, cap, stop=None):
    return {'messages':messages,'max_new_tokens':cap,'do_sample':False,'stop_strings':stop or []}
def input_text(payload):
    # Restore the original scorer's field ordering after content-addressed JSON serialization.
    ordered={'question':payload['question'],'relation_registry':dict(sorted(payload['relation_registry'].items())),
             'current_records':[{part:{'id':e[part]['id'],'labels':e[part]['labels']} for part in ['subject','relation','object']} for e in payload['current_records']]}
    return json.dumps(ordered,ensure_ascii=False,separators=(',',':'))
def direct_request(payload, family):
    return neural_request([{'role':'system','content':QA_SYSTEM if family=='QA' else PROGRAM_SYSTEM},
                           {'role':'user','content':input_text(payload)}],64 if family=='QA' else 128)

def records(payload):
    triples=[]; labels={}
    for edge in payload['current_records']:
        s,p,o=edge['subject'],edge['relation'],edge['object']
        triples.append([s['id'],p['id'],o['id']])
        for v in (s,o):
            old=labels.setdefault(v['id'],set());old.update(v['labels'])
    return triples,{k:sorted(v) for k,v in labels.items()}
def facts(payload):
    # Common source record rendering; includes every current branch and nothing historical.
    return [f"{e['subject']['labels'][0]} -- {e['relation']['labels'][0]} --> {e['object']['labels'][0]}." for e in payload['current_records']]
def parse_program(text, payload):
    try:
        obj=json.loads(text.strip())
        if obj=={'unsupported':True}: return None,'REFUSAL'
        if not isinstance(obj,dict) or set(obj)!={'start_id','path'}:return None,'INVALID_PROGRAM'
        _,labels=records(payload)
        if not isinstance(obj['start_id'],str) or obj['start_id'] not in labels:return None,'INVALID_START'
        path=obj['path']
        if not isinstance(path,list) or not 1<=len(path)<=4 or any(not isinstance(p,str) or p not in payload['relation_registry'] for p in path):return None,'INVALID_PATH'
        return obj,'VALID'
    except (ValueError,TypeError): return None,'INVALID_PROGRAM'

def formal_query(program):
    # IDs are validated by parser against the fixed registry/source before templating.
    assert re.fullmatch(r'Q\d+',program['start_id'])
    assert all(re.fullmatch(r'P\d+',x) for x in program['path'])
    clauses=[]; prev='ent:'+program['start_id']
    for i,relation in enumerate(program['path']):
        nxt='?id' if i==len(program['path'])-1 else '?x'+str(i)
        clauses.append(f'{prev} rel:{relation} {nxt} .');prev=nxt
    return 'PREFIX ent: <http://www.kg/entity/>\nPREFIX rel: <http://www.kg/relation/>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\nSELECT DISTINCT ?id ?label WHERE {\n'+'\n'.join(clauses)+'\n?id rdfs:label ?label .\n} LIMIT 1'

def sql_chain(payload, program):
    """Separately parity-tested chain-language equivalent of the formal SPARQL query."""
    triples,labels=records(payload)
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE fact(s TEXT,p TEXT,o TEXT,PRIMARY KEY(s,p))')
        db.executemany('INSERT INTO fact VALUES (?,?,?)',triples)
        clauses=['t0.s=?'];params=[program['start_id']]
        for i,p in enumerate(program['path']):clauses.append(f't{i}.p=?');params.append(p)
        joins='fact t0'+''.join(f' JOIN fact t{i} ON t{i}.s=t{i-1}.o' for i in range(1,len(program['path'])))
        statement=f'SELECT t{len(program["path"])-1}.o FROM {joins} WHERE '+' AND '.join(clauses)
        result=db.execute(statement,params).fetchall()
    if not result:return {'status':'NO_RESULT','entity':None,'formal_query':formal_query(program)}
    assert len(result)==1
    entity=result[0][0]
    return {'status':'VALID','entity':entity,'answer':labels[entity][0],'formal_query':formal_query(program)}

def parse_chain(text,payload,retrieve):
    # Paper Appendix A.4: choose chain line, split arrows, omit variable segments.
    lines=[x.strip() for x in text.splitlines() if '->' in x]
    if len(lines)!=1:return None,{'status':'INVALID_CHAIN','lines':len(lines)}
    line=re.sub(r'^Relation [Cc]hain:\s*','',lines[0])
    parts=[x.strip() for x in line.split('->')]
    name=parts[0]
    rels=[x for x in parts[1:] if not x.startswith('?')]
    if not 1<=len(rels)<=4:return None,{'status':'INVALID_CHAIN_DEPTH'}
    _,labels=records(payload)
    candidates=[key for key,names in labels.items() if any(norm(name).casefold()==norm(n).casefold() for n in names)]
    if len(candidates)!=1:return None,{'status':'UNLINKED_OR_AMBIGUOUS_NAME','name':name}
    registry=payload['relation_registry']; path=[]; normalization=[]
    options=[(key,label) for key,values in sorted(registry.items()) for label in values]
    for rel in rels:
        exact=[p for p,ls in registry.items() if rel==p or rel in ls]
        if len(exact)==1:p=exact[0];how='exact'
        else:
            match=retrieve(rel,[label for _,label in options],1)
            assert len(match)==1
            p=options[match[0]['index']][0];how='contriever-nearest'
        path.append(p);normalization.append({'generated':rel,'normalized':p,'method':how})
    return {'start_id':candidates[0],'path':path},{'status':'VALID','generated_name':name,'normalization':normalization}

def chain_demo_text(demos):
    result=[]
    for d in demos:
        _,labs=records(d['current_payload']);p=d['correct_program']
        chain=labs[p['start_id']][0]
        for i,rel in enumerate(p['path']): chain+='->'+d['current_payload']['relation_registry'][rel][0]+'->?x'+str(i)
        result.append('Question: '+d['current_payload']['question']+'\nRelation Chain: '+chain)
    return '\n\n'.join(result)

def gmello(payload,demos,retrieve):
    instruction='Extract the relation chain requested by the question, starting with the named entity and using relation labels from the complete supplied registry. Use the demonstrated arrow format; output one Relation Chain line only. The current records may be counterfactual. Do not output the answer.\n\n'+chain_demo_text(demos)
    generated=yield neural_request([{'role':'system','content':instruction},{'role':'user','content':input_text(payload)+'\nRelation Chain:'}],256)
    program,parsing=parse_chain(generated['text'],payload,retrieve)
    graph={'status':'INVALID_QUERY','entity':None} if program is None or generated['cap_hit'] else sql_chain(payload,program)
    trace={'chain_generation':generated,'parse':parsing,'program':program,'graph_only':graph,'fallback':None}
    if graph['entity'] is not None:
        # Graph priority is unconditional, including scientifically wrong but valid answers.
        return {'answer_text':graph['answer'],'entity':graph['entity'],'status':'VALID','program':program,'route':'graph','trace':trace}
    memory=facts(payload); selected=retrieve(payload['question'],memory,min(6,len(memory)))
    qa_examples='\n\n'.join('Facts: '+' '.join(facts(d['current_payload']))+'\nQuestion: '+d['current_payload']['question']+'\nAnswer: '+d['answer_label'] for d in demos)
    prompt=input_text(payload)+'\nRetrieved current facts:\n'+'\n'.join(memory[x['index']] for x in selected)+'\nAnswer:'
    fallback=yield neural_request([{'role':'system','content':QA_SYSTEM+'\n\n'+qa_examples},{'role':'user','content':prompt}],64)
    trace['fallback']={'retrieval':selected,'generation':fallback}
    return {'answer_text':fallback['text'],'status':'CAP_HIT' if fallback['cap_hit'] else 'GENERATED','program':program,'route':'fallback','trace':trace}

def mello(payload,demo_text,retrieve):
    instruction='Continue the demonstrated split-answer-check transcript. Generate a Subquestion and a Generated answer, then stop before Retrieved fact:. After a retrieved fact is supplied, explicitly check whether it contradicts your generated answer and give the intermediate answer before continuing. When ready, output Final answer: followed by one entity name. All supplied current records are authoritative and may be counterfactual.\n\n'+demo_text
    transcript=input_text(payload)+'\n\nQuestion: '+payload['question']+'\n'
    memory=facts(payload);trace=[];any_cap=False
    for i in range(4):
        request=neural_request([{'role':'system','content':instruction},{'role':'user','content':transcript}],256,['Retrieved fact:'])
        generated=yield request
        any_cap=any_cap or generated['cap_hit']
        text=generated['text'];lines=[x.strip() for x in text.strip().splitlines() if x.strip()]
        trace.append({'round':i+1,'generation':generated})
        if lines and lines[-1].startswith('Final answer: '):
            return {'answer_text':lines[-1][len('Final answer: '):],'status':'CAP_HIT' if any_cap else 'GENERATED','trace':trace,'route':'mello'}
        if len(lines)<2 or not lines[-2].startswith('Subquestion: ') or not lines[-1].startswith('Generated answer: '):
            return {'answer_text':'','status':'INVALID_MELLO_FORMAT','trace':trace,'route':'mello'}
        subq=lines[-2][len('Subquestion: '):]
        selected=retrieve(subq,memory,1)
        trace[-1]['retrieval']=selected;trace[-1]['subquestion']=subq
        transcript+=text.rstrip()+'\nRetrieved fact: '+memory[selected[0]['index']]+'\n'
    # Explicit final synthesis allows four completed retrievals; fixed protocol adaptation.
    final=yield neural_request([{'role':'system','content':instruction},{'role':'user','content':transcript+'Check the last retrieved fact, then give only Final answer: followed by one entity name.'}],64)
    trace.append({'round':'final_synthesis','generation':final})
    lines=[x.strip() for x in final['text'].strip().splitlines() if x.strip()]
    answer=lines[-1][len('Final answer: '):] if lines and lines[-1].startswith('Final answer: ') else ''
    return {'answer_text':answer,'status':'CAP_HIT' if (any_cap or final['cap_hit']) else ('GENERATED' if answer else 'INVALID_MELLO_FORMAT'),'trace':trace,'route':'mello'}

def drive_coroutines(coroutines, generate_batch):
    pending={};results={}
    for key,coroutine in coroutines.items():
        try:pending[key]=(coroutine,next(coroutine))
        except StopIteration as done:results[key]=done.value
    while pending:
        # Group only technically identical generation settings, never different evidence.
        groups={}
        for k,(_,r) in pending.items():groups.setdefault((r['max_new_tokens'],tuple(r['stop_strings'])),[]).append(k)
        for keys in groups.values():
            outputs=generate_batch([pending[k][1] for k in keys])
            assert len(outputs)==len(keys)
            for key,output in zip(keys,outputs):
                co,_=pending[key]
                try:pending[key]=(co,co.send(output))
                except StopIteration as done:results[key]=done.value;del pending[key]
    return results
