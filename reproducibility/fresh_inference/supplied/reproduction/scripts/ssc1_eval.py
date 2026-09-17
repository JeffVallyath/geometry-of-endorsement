"""Root-transactional compile/seal then read. Never adapts on final outcomes."""
from __future__ import annotations
from dataclasses import replace
import gzip
import json
from pathlib import Path
import time
import ssc1_common as C
import inputs as I
import ssc1_boundary as boundary
from ssc1_prepare import prepare, move_context

def cache_identity(bb, context):
    art=context._artifact
    return dict(hash=context.artifact_sha256,prefix_ids=art['prefix_ids'],
        prefix_length=art['prefix_len'],attention='NORMAL',
        layers=[dict(type=type(layer).__name__,length=int(layer.get_seq_length()),
                     key_shape=list(layer.keys.shape),value_shape=list(layer.values.shape),
                     dtype=str(layer.keys.dtype)) for layer in art['cache'].layers])

def descriptor(bb,context):
    art=context._artifact
    result=context.describe() | dict(text=context.text,cache_identity=cache_identity(bb,context),
        realized=art.get('realized'),prefix_sha256=art['prefix_sha256'])
    for field in ('_after_states','_workspace_states'):
        if field in art:
            result[field+'_sha256']=C.hashlib.sha256(art[field].tobytes()).hexdigest()
    return result

def compile_root(bb, source_inputs, actor):
    """Only updater envelopes are available here; no root/world/question files."""
    sources=sorted([s for s in source_inputs if s['actor']==actor],key=lambda s:s['origin'])
    if [s['origin'] for s in sources]!=list(range(8)): raise ValueError('Eight source origins required')
    commands=[I.commands_from_json(s['commands']) for s in sources]
    command_semantics=[]
    for source,cmds in zip(sources,commands,strict=True):
        I.validate(source['source_text'],cmds)
        command_semantics.append([(I.CLAUSE.fullmatch(source['source_text'][c.start:c.end])['actor'],
                                  I.CLAUSE.fullmatch(source['source_text'][c.start:c.end])['project'],c.value) for c in cmds])
    if not all(c==command_semantics[0] for c in command_semantics): raise ValueError('Assignments differ by origin')
    native_text=I.rewrite(sources[0]['source_text'],commands[0],'REBUILD')
    if native_text!=sources[0]['source_text']: raise ValueError('Origin zero is not the exact native final prefix')
    if any(I.rewrite(s['source_text'],c,'REBUILD') != native_text for s,c in zip(sources,commands,strict=True)):
        raise ValueError('Native final text differs across origins')
    contexts={}; descriptions={}; aliases={}; identities={}
    schedule=[('NATIVE_FINAL',sources[0],commands[0],'REBUILD',None)]
    schedule += [(f'o{origin}/{cond}',source,commands[origin],*C.method_seed(cond))
                 for origin,source in enumerate(sources) for cond in C.CONDITIONS]
    for key,source,cmds,method,seed in schedule:
        context=prepare(source['source_text'],cmds,method=method,backbone=bb,actor=actor,seed=seed)
        desc=descriptor(bb,context); descriptions[key]=desc
        # Alias only exact prefix/cache/metadata identity. Reader validates suffix
        # and public labels against the physical response before copying a row.
        identity=C.digest(desc['cache_identity'])
        if identity in identities:
            aliases[key]=identities[identity]
            del context
        else:
            identities[identity]=key
            move_context(context,'cpu')
            if bb.cache_hash(context._artifact['cache'])!=context.artifact_sha256:
                raise ValueError('CPU master transport changed bytes')
            contexts[key]=context
    if len(descriptions)!=57: raise ValueError('Missing context')
    if descriptions['NATIVE_FINAL']['cache_identity']!=descriptions['o0/SOURCE']['cache_identity']:
        raise ValueError('Native/source-zero identity failed')
    return contexts,descriptions,aliases

def cache_stamp(cache):
    """Per-call mutation guard; full byte hashes bracket the whole serving group."""
    return tuple((id(layer),int(layer.get_seq_length()),
        tuple((id(t),t.data_ptr(),t._version,tuple(t.shape),str(t.dtype),str(t.device)) for t in (layer.keys,layer.values)))
        for layer in cache.layers)

def make_journal(folder,key,privileged):
    def append(record):
        with (Path(folder)/'RAW_BATCHES.jsonl').open('a',encoding='utf8',newline='\n') as handle:
            handle.write(json.dumps(dict(context=key,privileged=privileged,**record),separators=(',',':'),allow_nan=False)+'\n')
            handle.flush();C.os.fsync(handle.fileno())
    return append

def serve(bb,context,questions,chunk_size=12,journal=None,qualify_batch=False):
    from canonical.tasks import suffix_text
    if any(set(q)!={'text','labels','query_id'} for q in questions): raise ValueError('Reader schema')
    art=context._artifact
    if bb.cache_hash(art['cache'])!=context.artifact_sha256: raise ValueError('Master changed before read')
    stamp=cache_stamp(art['cache']) if chunk_size==1 else None
    encoded=[]
    for q in questions:
        enc=bb.encode(context.text,suffix_text(q['text']))
        if enc['prefix_ids']!=art['prefix_ids']: raise ValueError('Suffix crossed prefix boundary')
        ids=bb.label_ids(enc['full_text'],q['labels'])
        encoded.append((q,enc,ids))
    # Length buckets avoid padding waste without changing question or attention.
    encoded.sort(key=lambda item:(len(item[1]['suffix_ids']),item[0]['query_id']))
    results={}; token_manifest={}; elapsed=0
    for start in range(0,len(encoded),chunk_size):
        chunk=encoded[start:start+chunk_size]
        I._sync(bb); t0=time.perf_counter()
        scored=bb.ask_batch(art,[x[1]['suffix_ids'] for x in chunk],[x[2] for x in chunk],verify_master=stamp is None)
        I._sync(bb); primary_seconds=time.perf_counter()-t0
        qualification_started=time.perf_counter()
        qualifications=[]
        if qualify_batch:
            from rso3_adapter import Backbone as Original
            reference=Original.ask_batch(bb,art,[x[1]['suffix_ids'] for x in chunk],[x[2] for x in chunk],verify_master=True)
            for slot,(item,actual,old) in enumerate(zip(chunk,scored,reference,strict=True)):
                single=bb.ask_batch(art,[item[1]['suffix_ids']],[item[2]],verify_master=True)[0]
                fields=('prediction','logps','answer_mass','argmax_in_labels','tie','finite')
                qualifications.append(dict(query_id=item[0]['query_id'],slot=slot,
                    exact_native=all(actual[k]==old[k] for k in fields),
                    single_same_choice=single['prediction']==actual['prediction'],single=single,
                    max_logp_delta=max(abs(a-b) for a,b in zip(single['logps'],actual['logps']))))
        if stamp is not None:
            if cache_stamp(art['cache'])!=stamp:raise ValueError('Master storage/version/metadata mutated during single serving')
            for row in scored:row['master_unchanged']=True
        I._sync(bb); qualification_seconds=time.perf_counter()-qualification_started if qualify_batch else 0.
        seconds=primary_seconds; elapsed+=seconds
        for slot,((q,enc,ids),r) in enumerate(zip(chunk,scored,strict=True)):
            if not r['master_unchanged']: raise ValueError('Master mutated during read')
            qid=q['query_id']
            token_manifest[qid]=dict(question_sha256=I.digest(q['text']),suffix_ids=enc['suffix_ids'],
                label_token_ids=ids,full_prompt_sha256=I.digest(enc['full_text']))
            results[qid]={k:r[k] for k in ('prediction','logps','answer_mass','argmax_in_labels','tie','finite')}
            results[qid].update(valid=bool(r['finite'] and not r['tie'] and r['argmax_in_labels']),physical_batch=start//chunk_size,
                physical_slot=slot,batch_size=len(chunk),batch_seconds=seconds,
                qualification_seconds=qualification_seconds,
                tokens_sha256=C.digest(token_manifest[qid]),master_unchanged=True)
        if journal is not None:
            journal(dict(batch=start//chunk_size,artifact_sha256=context.artifact_sha256,
                         results={x[0]['query_id']:results[x[0]['query_id']] for x in chunk},
                         tokens={x[0]['query_id']:token_manifest[x[0]['query_id']] for x in chunk},
                         qualifications=qualifications))
        if any(not r['exact_native'] or not r['single_same_choice'] for r in qualifications):
            raise ValueError('Native/single batch qualification failed; raw diagnostics retained')
    if bb.cache_hash(art['cache'])!=context.artifact_sha256:raise ValueError('Master bytes changed after serving group')
    return results,token_manifest,elapsed

def evaluate_root(bb,actor,shards,root_id,output,chunk_size=12,qualify_batch=False):
    """One complete immutable root. A partial prior root requires explicit recovery."""
    folder=Path(output)/actor/root_id
    if (folder/'DONE.json').exists():
        done=C.load(folder/'DONE.json')
        for name,hashed in done['files'].items():
            if C.sha(folder/name)!=hashed: raise ValueError('Completed root changed')
        return done
    if folder.exists(): raise ValueError('Partial root: retain and reconcile before runtime recovery')
    started=time.perf_counter()
    boundary.begin(actor,root_id)
    inputs=C.load(Path(shards)/root_id/'source_inputs.json')
    contexts,descriptions,aliases=compile_root(bb,inputs,actor)
    seal=dict(schema='SSC1_ROOT_CONTEXT_SEAL_V1',actor=actor,root_id=root_id,attention='NORMAL',
              contexts=descriptions,aliases=aliases,question_files_opened=False,
              source_inputs_sha256=C.sha(Path(shards)/root_id/'source_inputs.json'))
    C.save(folder/'CONTEXT_SEAL.json',seal)
    boundary.seal(actor,root_id,folder/'CONTEXT_SEAL.json')
    # This is the first read of the root's question file in the neural process.
    questions=C.load(Path(shards)/root_id/'questions.json')
    ordinary=[q for q in questions if q['family']!='given_facts']
    privileged=[q for q in questions if q['family']=='given_facts']
    if len(ordinary)!=34 or len(privileged)!=0: raise ValueError('Question inventory')
    outputs={}; manifests={}; seconds={}
    for key,context in contexts.items():
        move_context(context,bb.device)
        safe=[{k:q[k] for k in ('text','labels','query_id')} for q in ordinary]
        outputs[key],manifests[key],seconds[key]=serve(bb,context,safe,chunk_size,journal=make_journal(folder,key,False),qualify_batch=qualify_batch)
        if key=='NATIVE_FINAL' and privileged:
            extra,token_extra,elapsed=serve(bb,context,[{k:q[k] for k in ('text','labels','query_id')} for q in privileged],chunk_size,journal=make_journal(folder,key,True))
            for result in extra.values():
                result['physical_batch'] += (len(ordinary)+chunk_size-1)//chunk_size
            outputs[key].update(extra); manifests[key].update(token_extra); seconds[key]+=elapsed
        move_context(context,'cpu')
    physical=[]; logical=[]
    for key,results in outputs.items():
        for qid,result in results.items():
            physical.append(dict(physical_id=key+'|'+qid,context=key,query_id=qid,**result))
    for key in descriptions:
        source=aliases.get(key,key)
        active=ordinary+(privileged if key=='NATIVE_FINAL' else [])
        for q in active:
            qid=q['query_id']; result=outputs[source][qid]
            # Identity includes prefix IDs/bytes, cache bytes and metadata. The
            # identical public suffix/labels are looked up by exact query ID.
            if manifests[source][qid]['question_sha256']!=I.digest(q['text']): raise ValueError('Alias question mismatch')
            logical.append(dict(actor=actor,root_id=root_id,context=key,query_id=qid,
                family=q['family'],privileged=q['family']=='given_facts',
                physical_id=source+'|'+qid,alias=key!=source,**result))
    if len(logical)!=57*34: raise ValueError('Missing logical responses')
    for name,value in [('PHYSICAL.json',physical),('LOGICAL.json',logical),('TOKENS.json',manifests)]:
        with gzip.open(folder/(name+'.gz'),'xt',encoding='utf8') as handle:
            json.dump(value,handle,separators=(',',':'),allow_nan=False)
    summary=dict(status='COMPLETE',actor=actor,root_id=root_id,logical_responses=len(logical),
                 physical_responses=len(physical),physical_contexts=len(contexts),
                 seconds=time.perf_counter()-started,serving_seconds=seconds,
                 seal_sha256=C.sha(folder/'CONTEXT_SEAL.json'),
                 files={p.name:C.sha(p) for p in sorted(folder.iterdir())})
    qualifications={}
    for row in physical:
        qualifications[row['context'],row.get('physical_batch',0)]=row.get('qualification_seconds',0.)
    summary['qualification_seconds']=sum(qualifications.values())
    summary['forecast_seconds']=summary['seconds']-summary['qualification_seconds']
    C.save(folder/'DONE.json',summary)
    return summary
