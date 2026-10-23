"""Independent reconstruction of saved 32B records. No study-module imports."""
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
import argparse, json, hashlib, math
import numpy as np

def load(p):
    return json.loads(p.read_text(encoding='utf-8'))

def gold(q, values):
    k=q['spec']['kind']; a=bool(values[q['spec']['a']][q['spec']['p']]); b=bool(values[q['spec']['b']][q['spec']['p']])
    return int({'direct':a, 'opposes':not a, 'same':a==b, 'both':a and b,'either':a or b}[k])

def code(qid): return qid.split('-d')[1][0]
def sem(q):
    s=q['spec']; return tuple(sorted([s['a'],s['b']])),s['p'],s['kind']

def raw_load(path):
    idx={}
    for line in path.read_text(encoding='utf-8').splitlines():
        r=json.loads(line); k=(r['context'],r['query_id']); assert k not in idx; s=r['score']; lp=s['logps']
        assert all(math.isfinite(x) for x in lp)
        pr=int(lp[1]>lp[0]); mass=sum(math.exp(v) for v in lp)
        opts=s['label_token_ids']; top=max(lp)
        token=min(t for t,l in zip(opts,lp) if l==top)
        mx=s['maximum_non_candidate_logp']; ot=s['maximum_non_candidate_token_id']
        winner=token if top>mx else ot if top<mx else min(token,ot)
        valid=lp[0]!=lp[1] and winner in opts
        assert s['prediction']==pr and s['valid']==valid and winner==s['full_vocab_argmax_token_id']
        assert abs(mass-s['answer_mass'])<1e-6
        assert s['batch_size']==1 and not any(s[x] for x in ['use_cache','past_key_values_supplied','cache_returned'])
        assert s['last_token_index']==s['prompt_tokens']-1 and s['attention_mask_ones']==s['prompt_tokens']
        assert r['model_revision']=='9216db5781bf21249d130ec9da846c4624c16137'
        weights=[math.exp(v-max(lp)) for v in lp]; prob=[v/sum(weights) for v in weights]
        idx[k]={'pred':pr,'valid':valid,'mass':mass,'prob':prob,'raw':r}
    return idx

def check(root,out):
    runtime=root/'runtime'; expected=load(runtime/'analysis/SUMMARY.json')
    case_ids=load(runtime/'ANALYSIS_FREEZE.json')['case_order']; methods=list(expected)
    totals={m:Counter() for m in methods}; bycase={m:[] for m in methods}; casesets={m:defaultdict(set) for m in methods}
    single_examples=[]; matched_examples=[]; final_idx={}; checked_raw=0; semantic_fields=0
    for rid in case_ids:
        design=load(runtime/'cpu/designs'/f'{rid}.json'); vals=design['root']['terminal']['values']; qs={q['query_id']:q for q in design['questions']}
        native_keys=[('NATIVE_FINAL',q) for q in qs]
        joints=[q for q in qs if qs[q]['spec']['kind'] in ('same','both','either')]; direct=[q for q in qs if qs[q]['spec']['kind']=='direct']
        arr=raw_load(runtime/'results/qwen32_t4/final'/rid/'RAW_SCORES.jsonl'); assert len(arr)==578; checked_raw+=len(arr); final_idx[rid]=arr
        seal=load(runtime/'results/qwen32_t4/final'/rid/'CONTEXT_SEAL.json')['contexts']
        changed=[sum(s['values'][c['actor']][c['project']]!=c['value'] for c in design['root']['commands']) for s in design['root']['sources']]
        archived=load(runtime/'analysis/cases'/f'{rid}.json')
        for m in methods:
            def r(o,q):return arr[(f'o{o}/{m}',q)]
            def ok(o,q):return r(o,q)['valid'] and r(o,q)['pred']==gold(qs[q],vals)
            def native(q):return arr[('NATIVE_FINAL',q)]['valid'] and arr[('NATIVE_FINAL',q)]['pred']==gold(qs[q],vals)
            matches=[(a,b) for a,b in combinations(range(8),2) if changed[a]==changed[b] and len(seal[f'o{a}/{m}']['prefix_ids'])==len(seal[f'o{b}/{m}']['prefix_ids'])]
            cc=Counter(); coded={}; expected_coded={x['query_id']:x for x in archived[m]['inherited_coded']}
            for q in joints:
                pair,project,kind=sem(qs[q]); fmt=code(q)
                operands=[d for d in direct if qs[d]['spec']['a'] in pair and qs[d]['spec']['p']==project and code(d)==fmt]
                same=[j for j in joints if sem(qs[j])[:2]==(pair,project) and j!=q and code(j)==fmt]
                cross=[j for j in joints if sem(qs[j])[0]!=pair and sem(qs[j])[1]==project and code(j)==fmt]
                assert len(operands)==2 and len(same)==2 and len(cross)==6
                A=all(ok(o,d) for d in operands for o in range(8)); C=native(q) and ok(0,q)
                D=len({r(o,q)['pred'] for o in range(8) if r(o,q)['valid']})>1
                W=A and C and D
                G1=C and all(ok(o,d) for d in direct for o in range(8))
                gates={'G0':A and C,'G1':G1,'G2a':G1 and all(ok(o,d) for d in same for o in range(8)),'G2b':G1 and all(ok(o,d) for d in cross for o in range(8))}
                gates['G2ab']=gates['G2a'] and gates['G2b']
                pairs=[(a,b) for a,b in matches if r(a,q)['valid'] and r(b,q)['valid'] and r(a,q)['pred']!=r(b,q)['pred']]
                strong=W and all(r(o,q)['valid'] for o in range(8)) and all(r(o,d)['prob'][gold(qs[d],vals)]>=.95 for d in operands for o in range(8)) and min([r(o,q)['mass'] for o in range(8)]+[r(o,d)['mass'] for o in range(8) for d in operands]+[arr['NATIVE_FINAL',q]['mass']])>=.9 and min(r(o,q)['prob'][gold(qs[q],vals)] for o in range(8))<=.05 and max(r(o,q)['prob'][gold(qs[q],vals)] for o in range(8))>=.95
                extremes=[(a,b) for a,b in pairs if strong and min(r(o,q)['prob'][gold(qs[q],vals)] for o in (a,b))<=.05 and max(r(o,q)['prob'][gold(qs[q],vals)] for o in (a,b))>=.95]
                for g,v in gates.items():
                    assert v==expected_coded[q]['gates'][g]
                    assert (v and bool(pairs))==expected_coded[q][g+'_M_witness']
                    assert (v and bool(extremes))==expected_coded[q][g+'_strong_M']
                    cc[g+'_eligible']+=v; cc[g+'_matched']+=v and bool(pairs)
                coded[q]={'A':A,'C':C,'D':D,'W':W,'strong':strong,'gates':gates,'pairs':pairs,'extremes':extremes}
                cc.update({k:int(v) for k,v in [('A',A),('C',C),('E',A and C),('W',W),('strong',strong)]})
                cc['all_origin_joint_correct']+=all(ok(o,q) for o in range(8))
                if W:cc['W_'+kind]+=1
                if gates['G2ab'] and pairs:
                    twin=next(j for j in joints if sem(qs[j])==sem(qs[q]) and code(j)!=fmt)
                    single_examples.append({'root':rid,'method':m,'query_id':q,'question':qs[q]['text'],'code':fmt,'matched_pairs':pairs,'high_margin_pairs':extremes,'repeated_full':rid in load(runtime/'REPEAT_SELECTION.json')['roots'],'twin_id':twin})
            for q in joints:
                if code(q)!='0':continue
                twin=next(j for j in joints if sem(qs[j])==sem(qs[q]) and code(j)=='1')
                bothE=coded[q]['gates']['G0'] and coded[twin]['gates']['G0']
                common=[pair for pair in coded[q]['pairs'] if pair in coded[twin]['pairs'] and all(r(o,q)['pred']==r(o,twin)['pred'] for o in pair)]
                fullcert=coded[q]['gates']['G2ab'] and coded[twin]['gates']['G2ab']
                cc['semantic_base_eligible']+=bothE; cc['semantic_matched']+=bothE and bool(common);cc['semantic_logical_eligible']+=fullcert;cc['semantic_strongest']+=fullcert and bool(common)
                if fullcert:casesets[m]['full_logical'].add(rid)
                if bothE and common:
                    casesets[m]['matched_crossformat'].add(rid); matched_examples.append({'root':rid,'method':m,'target':q,'pairs':common})
                semantic_fields+=1
            cc['direct_correct']=sum(ok(o,d) for o in range(8) for d in direct)
            cc['joint_correct']=sum(ok(o,q) for o in range(8) for q in joints)
            cc['battery_correct']=sum(ok(o,q) for o in range(8) for q in qs)
            cc['complete_case']=all(ok(o,q) for o in range(8) for q in qs)
            if cc['W']:casesets[m]['primary'].add(rid)
            totals[m]+=cc; bycase[m].append({'root':rid,**dict(cc)})
    draws=np.random.default_rng(2609141002).integers(0,64,size=(10000,64))
    report={}
    for m,t in totals.items():
        ex=expected[m]; assert t['W']==ex['primary']['W']; assert t['E']==ex['primary']['conditional_witness']['denominator'];assert t['strong']==ex['primary']['strong']
        assert t['semantic_matched']==ex['semantic_counts']['matched_crossformat_G0_pairs']; assert t['semantic_strongest']==ex['semantic_counts']['strongest_pairs']
        assert t['semantic_logical_eligible']==80 and len(casesets[m]['full_logical'])==9
        vec=np.array([x['W']/18 for x in bycase[m]]); interval=np.quantile(vec[draws].mean(1),[(1-.99375)/2,1-(1-.99375)/2]).tolist()
        assert np.array_equal(interval,ex['primary']['primary_family8_interval99375'])
        report[m]={'counts':dict(t),'primary_rate':t['W']/1152,'conditional_primary_rate':t['W']/t['E'],'adjusted_interval':interval,'root_sets':{k:sorted(v) for k,v in casesets[m].items()}}
    # Independent score/token equality check for every retained fixed and full repeat.
    repeats=Counter();discrepancies=[]; max_delta=0.0
    for phase in ['repeat0','repeat1','repeat_full']:
        for path in (runtime/'results/qwen32_t4'/phase).glob('*/RAW_SCORES.jsonl'):
            rid=path.parent.name; rows=raw_load(path)
            for k,row in rows.items():
                old=final_idx[rid][k]; a=row['raw'];b=old['raw']
                delta=max(abs(x-y) for x,y in zip(a['score']['logps'],b['score']['logps']));max_delta=max(max_delta,delta)
                if delta>1e-6 or row['pred']!=old['pred'] or row['valid']!=old['valid'] or a['tokens']!=b['tokens']:discrepancies.append([phase,rid,k,delta])
                repeats[phase]+=1
    result={'status':'PASS','study_modules_imported':False,'raw_final_checked':checked_raw,'semantic_rows_checked':semantic_fields,'summary':report,'single_format_full_control_examples':single_examples,'matched_crossformat_examples':matched_examples,'repeat_rows':dict(repeats),'repeat_max_logp_delta':max_delta,'repeat_discrepancies':discrepancies,'new_pretrained_inference':False}
    out.mkdir(exist_ok=True,parents=True);(out/'INDEPENDENT_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');(out/'PER_CASE_RECONSTRUCTION.json').write_text(json.dumps(bycase,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['summary','matched_crossformat_examples']},indent=2))
    return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();check(args.source,args.output)
