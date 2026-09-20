"""Preserve all repeat discrepancies; never rerun in pursuit of agreement."""
import argparse
from pathlib import Path
from text_scope.common import read,save,sha,METHODS
from qwen32_full_forward.analyze import load_raw,analyze_case


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);a=p.parse_args();runtime=a.runtime
    freeze=read(runtime/'ANALYSIS_FREEZE.json');selection=read(runtime/'REPEAT_SELECTION.json')['roots']
    discrepancies=[];compared=0;processes={}
    for phase in ('final','repeat0','repeat1','repeat_full'):
        processes[phase]=read(runtime/'results/qwen32_t4'/phase/'PHASE_DONE.json')['scorer_pid']
    assert len(set(processes.values()))==4, 'REPEATS_REQUIRE_FRESH_PROCESSES'
    for phase in ('repeat0','repeat1','repeat_full'):
        roots=selection if phase=='repeat_full' else freeze['fixed_repeat_roots']
        for rid in roots:
            original=load_raw(runtime/'results/qwen32_t4/final'/rid)
            repeated=load_raw(runtime/'results/qwen32_t4'/phase/rid)
            assert len(repeated)==(578 if phase=='repeat_full' else 136)
            for key,row in repeated.items():
                old=original[key];left,right=old['score'],row['score'];compared+=1
                assert old['tokens']==row['tokens'] and old['request_sha256']==row['request_sha256']
                flags=('prediction','valid','finite','tie','argmax_in_labels','full_vocab_argmax_token_id','label_token_ids')
                delta=max(abs(float(x)-float(y)) for x,y in zip(left['logps'],right['logps']))
                if any(left[k]!=right[k] for k in flags) or delta>1e-6:
                    discrepancies.append(dict(phase=phase,root_id=rid,context=key[0],query_id=key[1],
                        logp_delta=delta,original=left,repeat=right))
    assert compared==4488
    classification=[]
    for rid in selection:
        first=read(runtime/'analysis/cases'/(rid+'.json'));repeat=analyze_case(runtime,rid,'repeat_full')
        changes=[]
        for method in METHODS:
            for key in ('A','C','AC','D','W','strong','complete_case'):
                if first[method]['metrics'][key]!=repeat[method]['metrics'][key]:changes.append(method+'/'+key)
            for left,right in zip(first[method]['semantic'],repeat[method]['semantic']):
                for key in ('strongest_pairs','high_margin_pairs','matched_crossformat_G0_pairs','attrition_reason'):
                    if left[key]!=right[key]:changes.append(method+'/'+left['target_ids'][0]+'/'+key)
        classification.append(dict(root_id=rid,classification_identical=not changes,changes=changes))
        save(runtime/'analysis/full_repeats'/(rid+'.json'),repeat)
    save(runtime/'REPEAT_CHECKS.json',dict(status='PASS' if not discrepancies and all(r['classification_identical'] for r in classification)
        else 'DISCREPANCIES_RETAINED_DEFINITIVE_INFERENCE_SUSPENDED',compared_reads=compared,
        fixed_panel_reads=2176,complete_case_reads=2312,processes=processes,
        discrepancies=discrepancies,complete_case_classification=classification,
        repeats_not_added_to_population_estimates=True,reruns_until_favorable_or_stable=False))


if __name__=='__main__':main()
