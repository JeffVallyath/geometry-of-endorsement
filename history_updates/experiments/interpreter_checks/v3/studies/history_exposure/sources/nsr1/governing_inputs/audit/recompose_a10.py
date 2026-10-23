"""Retrospective supplied-plan recomposition of saved A10 direct predictions.
Not semantic parsing, not new inference and not an internal intervention.
Operation/operand identifiers come from the exposed query specification.
Actual saved direct predictions, never target values, supply operands.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,importlib.util,json,math,zipfile
from pathlib import Path
from collections import Counter,defaultdict

def main(archive:Path,out:Path):
 out.mkdir(parents=True,exist_ok=True)
 with archive.open('rb') as f:
  if hashlib.file_digest(f,'sha256').hexdigest() != 'b7775b696632ea15f3d173089dab8f70ae9cf0a06612ced46aae2b77ed07766f':
   raise ValueError('Unexpected source archive: do not execute a different replay module')
 totals=Counter(); by_seed=defaultdict(Counter); targets=[]; roots=set(); badcoverage=[]
 with zipfile.ZipFile(archive) as z:
  designs={d['root']['root_id']:d for d in map(json.loads,gzip.decompress(z.read('reproducibility/question_blind_repair/scores/designs.jsonl.gz')).splitlines())}
  # Use original frozen truth/validity/accounting functions unchanged.
  text=z.read('reproducibility/question_blind_repair/supplied/verification/tools/replay.py').decode()
  ns={};exec(compile(text,'original_replay.py','exec'),ns)
  files=sorted(n for n in z.namelist() if '/scores/primary-seed' in n and n.endswith('.jsonl.gz'))
  assert len(files)==16
  for name in files:
   for ln in gzip.decompress(z.read(name)).splitlines():
    panel=json.loads(ln);d=designs[panel['root_id']];seed=panel['seed'];root=d['root'];qs={q['query_id']:q for q in d['questions']}
    records={(r['context'],r['query_id']):ns['decode'](r) for r in panel['physical']}
    baseline=ns['metrics'](root,d['questions'],d['plan'],panel,False)
    base_targets={r['query_id']:r for r in baseline['targets']}
    directs={(int(qid.split('-d')[1][0]),q['spec']['a'],q['spec']['p']):qid for qid,q in qs.items() if q['spec']['kind']=='direct'}
    gold={qid:ns['truth'](root['terminal']['values'],q['spec']) for qid,q in qs.items()}
    # Recomposer receives prediction records and operand plan, not gold/root terminal.
    def recomposed(context,qid):
     q=qs[qid];s=q['spec'];kind=s['kind'];draw=int(qid.split('-d')[1][0])
     if kind=='direct':return records[context,qid]
     ids=[directs.get((draw,s['a'],s['p']))]
     if kind!='opposes':ids.append(directs.get((draw,s['b'],s['p'])))
     if None in ids:badcoverage.append([panel['root_id'],seed,qid]);return dict(valid=False,prediction=None)
     rs=[records[context,k] for k in ids]
     if not all(r['valid'] for r in rs):return dict(valid=False,prediction=None)
     v=[r['prediction'] for r in rs]
     pred={'opposes':lambda:1-v[0], 'same':lambda:int(v[0]==v[1]), 'both':lambda:int(v[0] and v[1]), 'either':lambda:int(v[0] or v[1])}[kind]()
     return dict(valid=True,prediction=pred)
    def good(r,qid):return r['valid'] and r['prediction']==gold[qid]
    for qid,q in qs.items():
     kind=q['spec']['kind'];fam='joint' if kind in {'same','both','either'} else kind
     for o in range(8):
      ctx=f'o{o}/INV_PAIR_NLL_s{seed}';old=records[ctx,qid];new=recomposed(ctx,qid)
      og=good(old,qid);ng=good(new,qid)
      for c in [totals,by_seed[seed]]:
       for k,v in {'scheduled':1,'old_error':not og,'new_error':not ng,'corrected':not og and ng,'introduced':og and not ng,'invalid':not new['valid']}.items():c[f'{fam}_{k}']+=int(v)
    for qid,m in d['plan']['gates']['targets'].items():
     ids=[qid]+d['plan']['gates']['direct']+m['same_pair']+m['cross_pair']
     ok=all(good(recomposed(f'o{o}/INV_PAIR_NLL_s{seed}',q),q) for o in range(8) for q in ids)
     targetok=all(good(recomposed(f'o{o}/INV_PAIR_NLL_s{seed}',qid),qid) for o in range(8))
     b=base_targets[qid]
     r={'root_id':root['root_id'],'seed':seed,'query_id':qid,'baseline_W':b['W'],'A10_B_repair':b['B_repair'],'recomposition_target_all_correct':targetok,'recomposition_bundle_all_correct':ok,'recomposition_repair':bool(b['W'] and ok)}
     targets.append(r)
     for c in [totals,by_seed[seed]]:
      c['target_opportunities']+=1;c['baseline_W']+=int(b['W']);c['A10_B_repairs']+=int(b['B_repair']);c['recomposition_repairs']+=int(b['W'] and ok)
     if b['W']:roots.add(root['root_id'])
    totals['case_seed_blocks']+=1
  result={'status':'RETROSPECTIVE_SUPPLIED_QUERY_PLAN_RECOMPOSITION','neural_inference_performed':False,'native_tensors_reverified':False,'query_plan_source':'exposed evaluator spec; not inferred from natural language','value_source':'original valid direct model predictions at same origin, seed and answer mapping','missing_direct_coverage':badcoverage,'totals':dict(totals),'by_seed':{str(k):dict(v) for k,v in by_seed.items()},'baseline_witness_roots':len(roots),'interpretation':'Qualified repair is expected by the original direct-correctness guard; all-scheduled errors/harm reveal limitations outside it.'}
 (out/'A10_RECOMPOSITION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 with (out/'A10_RECOMPOSITION_TARGETS.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(targets[0]));w.writeheader();w.writerows(targets)
 print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.archive,a.output)
