"""Allowlisted, deterministic curation of local V10 archives; never run model code.

Writes a NEW directory. Does not import either analysis implementation.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

MAIN = 'STATE_SUFFICIENCY_CONFIRMATION_RESULTS_AND_REPRODUCTION.zip'
RAW = 'STATE_SUFFICIENCY_CONFIRMATION_RAW_DATA_01.zip'
SCORES = ('logps','prediction','answer_mass','argmax_in_labels','tie','finite','valid','tokens_sha256')
CONTEXT = ('method','source_sha256','updated_text_sha256','artifact_sha256','command_count','selected_weight_sha256','text','prefix_sha256')


def sha(data): return hashlib.sha256(data).hexdigest()
def encode(data): return (json.dumps(data,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def canonical(data): return sha(encode(data).rstrip(b'\n'))
def need(ok,message):
    if not ok: raise ValueError(message)


class Archive:
    def __init__(self,path):
        self.path=path
        with path.open('rb') as f: self.digest=hashlib.file_digest(f,'sha256').hexdigest()
        self.zip=zipfile.ZipFile(path)
        names=self.zip.namelist()
        need(len(names)==len(set(names)),'Duplicate ZIP names')
        for n in names:
            p=PurePosixPath(n)
            need(not p.is_absolute() and '..' not in p.parts and ':' not in n and '\\' not in n,'Unsafe member')
        self.manifest=json.loads(self.zip.read('MANIFEST.json'))['files']
        need(set(names)==set(self.manifest)|{'MANIFEST.json'},'Undeclared/missing archive members')
        for name,row in self.manifest.items():
            data=self.zip.read(name)
            need(len(data)==row['bytes'] and sha(data)==row['sha256'],'Source integrity: '+name)
        self.used={}

    def get(self,name):
        data=self.zip.read(name)
        info=self.manifest[name]
        self.used[name]={'archive':self.path.name,'member':name,'sha256':info['sha256'],'bytes':len(data)}
        return data

    def json(self,name):
        raw=self.get(name)
        return json.loads(gzip.decompress(raw) if name.endswith('.gz') else raw)


def extract(archives,output):
    need(not output.exists(),'Output already exists; extraction never overwrites')
    main=Archive(archives/MAIN); raw=Archive(archives/RAW)
    companions=main.json('RAW_COMPANIONS.json')
    record=next(r for r in companions['companions'] if r['file']==RAW)
    need(record['sha256']==raw.digest and record['bytes']==raw.path.stat().st_size,'Companion identity mismatch')
    for name,r in companions['members'].items():
        member=name if name.startswith('results/') else 'results/'+name
        need(raw.manifest[member]['sha256']==r['sha256'],'Companion member binding mismatch')
    frozen=main.json('FROZEN_SPECIFICATION.json')
    for row in frozen['science']['files']:
        name='reproduction/'+row['path']
        need(main.manifest[name]['sha256']==row['sha256'] and main.manifest[name]['bytes']==row['bytes'],'Final source capsule binding mismatch')
    output.mkdir(parents=True)
    manifest={'schema_version':1,'scope':'Saved per-example reconstruction, not neural execution attestation',
              'archives':{a.path.name:{'sha256':a.digest,'bytes':a.path.stat().st_size,
                   'manifest_sha256':sha(a.zip.read('MANIFEST.json')),'verified_members':len(a.manifest),
                   'source':'user-supplied local archive; not a remote download'} for a in (main,raw)},
              'files':[], 'extraction':{'implementation_sha256':sha(Path(__file__).read_bytes()),'python':sys.version.split()[0],
                  'encoding':'Sorted compact ASCII JSON with LF; gzip mtime=0, compression level 9.',
                  'omissions':'Only allowlisted scientific fields are emitted. No billing, credentials, host/device information, timings, training tensors, technical DEV outcomes or unrelated operations records.',
                  'integrity_boundary':'Archive and final-source member hashes checked; runtime receipts and physical forward passes are not independently attested.'}}
    def emit(name,data,sources,rule,rows,omitted):
        payload=encode(data)
        if name.endswith('.gz'): payload=gzip.compress(payload,compresslevel=9,mtime=0)
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(payload)
        manifest['files'].append({'path':name,'role':'expected' if name.startswith('expected/') else 'input',
             'sha256':sha(payload),'bytes':len(payload),'row_counts':rows,
             'sources':[a.used[n] for a,n in sources], 'transformation':rule,'omitted_fields':omitted})
    shard='reproduction/reports/state_sufficiency_confirmation/shards/'
    repeat_name=shard+'REPEATABILITY.json';repeat=main.json(repeat_name)
    protocol={'models':frozen['science']['models'],'inventory':frozen['science']['inventory'],
              'bootstrap':frozen['science']['bootstrap'],'repeatability':repeat,
              'source_commit':frozen['science']['source_commit'],
              'primary':'A and C and D; denominator 18 per root; learned seeds averaged within root.',
              'A':'Both direct operand responses correct and valid in every origin.',
              'C':'Native-final and same-method origin-zero/no-op joint responses correct and valid.',
              'D':'At least two different valid joint answers across origins; invalid rows cannot create disagreement.',
              'strong':'Primary witness; all joint answers valid; all direct correct probabilities >= .95; candidate mass >= .90 for operands, all joint answers and native; max joint correct probability >= .95 and min <= .05.',
              'argmax_certificate_slack':1e-6}
    emit('inputs/protocol.json',protocol,[(main,'FROZEN_SPECIFICATION.json'),(main,repeat_name)],
         'Copy scientific model/inventory/bootstrap fields and repeat specification; primary definitions transcribed from frozen experiment interface.',
         {'models':2,'repeat_roots_per_model':8},['technical timing forecasts','runtime/capsule fields unrelated to classification'])
    for ri in range(64):
        rid=f'SSC1-FINAL-{ri:04d}'
        world_name=shard+rid+'/root.json';qname=shard+rid+'/questions.json';sname=shard+rid+'/source_inputs.json'
        world=main.json(world_name);questions=main.json(qname);sources=main.json(sname)
        emit('inputs/designs/'+rid+'.json',{'root':world,'questions':questions,'source_inputs':sources},
             [(main,n) for n in (world_name,qname,sname)],'Copy FINAL shard root, question and rendered-source objects without changing their fields.',
             {'roots':1,'histories':8,'questions':len(questions),'rendered_sources':len(sources)},['precomputed Boolean labels; independently recomputed from records'])
        for model in ('gemma','qwen'):
            panels=[('core',f'results/roots/{model}/{rid}')]
            if rid in repeat['roots']:
                panels += [(f'pass{p}',f'results/repeatability/{model}/{rid}/pass{p}') for p in range(2)]
            for phase,folder in panels:
                seal_name=folder+'/CONTEXT_SEAL.json';seal=main.json(seal_name)
                contexts={key:{**{k:r[k] for k in CONTEXT},'cache_hash':r['cache_identity']['hash'],
                          'prefix_ids':r['cache_identity']['prefix_ids']} for key,r in seal['contexts'].items()}
                journal_name=folder+'/RAW_BATCHES.jsonl';journal=raw.get(journal_name)
                rows=[];seen=set()
                for line_number,line in enumerate(journal.splitlines(),1):
                    batch=json.loads(line)
                    need(not batch['privileged'] and not batch['qualifications'],'Unexpected qualification panel in selected final/repeat records')
                    for qid,s in batch['results'].items():
                        identity=(batch['context'],qid);need(identity not in seen,'Duplicate raw response');seen.add(identity)
                        tokens=batch['tokens'][qid]
                        need(canonical(tokens)==s['tokens_sha256'],'Raw token hash mismatch')
                        need(batch['artifact_sha256']==contexts[batch['context']]['cache_hash'],'Journal/context binding mismatch')
                        rows.append({'context':batch['context'],'query_id':qid,'journal_line':line_number,
                                     'logps':s['logps'],'tokens':tokens,
                                     'recorded_checks':{k:s[k] for k in SCORES if k!='logps'}})
                used=[(main,seal_name),(raw,journal_name)]
                if phase=='core':
                    logical_name=folder+'/LOGICAL.json.gz';logical=main.json(logical_name);used.append((main,logical_name))
                    bindings=[{k:r[k] for k in ('context','query_id','physical_id','alias')} for r in logical]
                    index={(r['context']+'|'+r['query_id']):r for r in rows}
                    for r in logical:
                        p=index[r['physical_id']]
                        need(r['logps']==p['logps'] and all(r[k]==p['recorded_checks'][k] for k in SCORES if k!='logps'), 'Logical/raw score mismatch')
                else:
                    score_name=folder+'/SCORES.json';scores=main.json(score_name);used.append((main,score_name))
                    bindings=[{'context':r['context'],'query_id':r['query_id'],'physical_id':r['context']+'|'+r['query_id'],'alias':False} for r in rows]
                    need(sum(len(v) for v in scores.values())==len(rows),'Repeat source inventory mismatch')
                    for r in rows:
                        s=scores[r['context']][r['query_id']]
                        need(s['logps']==r['logps'] and all(s[k]==r['recorded_checks'][k] for k in SCORES if k!='logps'),'Repeat raw/score mismatch')
                payload={'model':model,'root_id':rid,'phase':phase,'contexts':contexts,'aliases':seal['aliases'],
                         'physical':rows,'bindings':bindings}
                emit(f'inputs/responses/{model}/{rid}/{phase}.json.gz',payload,used,
                     'Flatten raw batches, retaining candidate logps, complete token objects and observed flags; attach one-based journal line and query key. Copy logical bindings (core), or identity bindings (repeats). Select CONTEXT fields plus cache hash and prefix IDs. No eligibility/witness fields are used.',
                     {'physical_responses':len(rows),'logical_responses':len(bindings),'contexts':len(contexts)},
                     ['batch/compile/preprocess timings','backend, model_checks and realized tensors','cache layer shapes/dtypes','privileged=false and empty qualifications (validated)','batch scheduling, master_unchanged (not classification inputs)'])
        print('Extracted '+rid,file=sys.stderr,flush=True) if ri%16==0 else None
    for name in ('PRIMARY.json','ROOT_TABLE.json','WITNESS_SUMMARY.json','QUERY_TABLE.json.gz'):
        member='results/analysis/'+name;data=main.json(member)
        emit('expected/'+name,data,[(main,member)],'Lossless JSON reserialization of reported output, for optional post-calculation comparison only.',{'rows':len(data)},[])
    with (output/'MANIFEST.json').open('xb') as f:f.write(encode(manifest))
    return {'status':'curated','files':len(manifest['files']),'bytes':sum(r['bytes'] for r in manifest['files']),
            'source_members_verified':sum(len(a.manifest) for a in (main,raw))}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archives',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(extract(a.archives,a.output),indent=2))
