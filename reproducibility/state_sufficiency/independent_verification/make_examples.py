"""Build fully traceable illustrations from independently computed query rows."""
import argparse
import json
from pathlib import Path

import reconstruct as independent


def examples(base,result):
    rows=result['query_rows']
    selected=[]
    rules=[('repeat_checked_positive',lambda r:r['actor']=='qwen' and r['root_id']=='SSC1-FINAL-0006' and r['condition']=='FREE_PAIR_CONSISTENCY_s0' and r['query_id'].endswith('-d0-02')),
           ('textual_positive',lambda r:r['condition']=='EXISTING_CORRECTION' and r['W'] and r['strong']),
           ('eligible_non_witness',lambda r:r['condition']!='SOURCE' and r['AC'] and not r['D']),
           ('excluded_direct_operand',lambda r:r['condition']!='SOURCE' and not r['A'] and r['C'] and r['D']),
           ('excluded_native_reference',lambda r:r['condition']!='SOURCE' and r['A'] and not r['native_correct']),
           ('excluded_noop_reference',lambda r:r['condition']!='SOURCE' and r['A'] and r['native_correct'] and not r['noop_correct'])]
    for category,predicate in rules:
        matches=[r for r in rows if predicate(r)]
        independent.require(bool(matches),'No observed example for '+category)
        r=matches[0];model,rid,method=r['actor'],r['root_id'],r['condition']
        design_path=f'inputs/designs/{rid}.json';response_path=f'inputs/responses/{model}/{rid}/core.json.gz'
        design=independent.load(base/design_path);panel=independent.load(base/response_path)
        qs=independent.validate_design(design,rid)
        protocol=independent.load(base/'inputs/protocol.json')
        index,raw=independent.index_panel(panel,design,qs,'core',protocol)
        def trace(context,qid):
            response=index[context,qid];saved=raw[context,qid];q=qs[qid]
            return {'context':context,'query_id':qid,'question':q['text'],'answer_mapping':q['labels'],
                    'logps':saved['logps'],'normalized_probabilities':response['probabilities'],
                    'answer_mass':response['mass'],'prediction_index':response['prediction'],
                    'answer_token':q['labels'][response['prediction']],
                    'semantic_answer':'Yes' if response['prediction'] else 'No','valid':response['valid'],
                    'physical_id':saved['context']+'|'+qid,'journal_line':saved['journal_line'],
                    'tokens':saved['tokens'],'response_file':response_path}
        histories=[]
        for o in range(8):
            ctx=f'o{o}/{method}'
            histories.append({'origin':o,'starting_values':design['root']['sources'][o]['values'],
                              'direct_operands':[trace(ctx,qid) for qid in r['operand_ids']],
                              'joint':trace(ctx,r['query_id'])})
        repeat=[]
        if rid in protocol['repeatability']['roots'] and method in protocol['repeatability']['conditions'] and any(r['query_id']==rid+s for s in protocol['repeatability']['query_suffixes']):
            for phase in ('pass0','pass1'):
                rp=f'inputs/responses/{model}/{rid}/{phase}.json.gz'
                p=independent.load(base/rp)
                for s in p['physical']:
                    if s['query_id']==r['query_id'] and s['context'].endswith('/'+method):
                        repeat.append({'response_file':rp,'context':s['context'],'query_id':s['query_id'],'journal_line':s['journal_line'],'logps':s['logps']})
        selected.append({'category':category,'selection':'First matching row in deterministic model/root/method/question order, except the named repeat-checked illustration. Selected examples are not prevalence estimates.',
                         'model':model,'root_id':rid,'method':method,'query_id':r['query_id'],
                         'design_file':design_path,'actors':design['root']['terminal']['actors'],
                         'projects':design['root']['terminal']['projects'],'commands':design['root']['commands'],
                         'common_final_values':design['root']['terminal']['values'],
                         'calculated_flags':{k:r[k] for k in ('A','C','D','W','strong','native_correct','noop_correct')},
                         'histories':histories,'native_reference':trace('NATIVE_FINAL',r['query_id']),
                         'noop_reference':trace('o0/'+method,r['query_id']), 'repeat_joint_rows':repeat})
    return selected


def markdown(rows):
    lines=['# Traceable saved-output examples','',
           'These are selected illustrations, not prevalence estimates. Each example is selected solely from the independent calculation, without consulting expected outputs. `examples.json` contains full-precision scores, token bindings and journal line references. Manifest entries identify each original source archive/member.','',
           'Relation tables use 1 for supports and 0 for opposes. Actor and project order is explicit. Candidate log probabilities are ordered by the displayed answer mapping. All displayed semantic Yes/No answers are decoded from that mapping.','']
    for e in rows:
        lines += [f"## {e['category']}",'',f"{e['model']} / `{e['method']}` / `{e['root_id']}` / `{e['query_id']}`",'',
                  f"Actors: {', '.join(e['actors'])}. Projects: {', '.join(e['projects'])}.",'',
                  f"Commands (actor/project indices and assigned values): `{json.dumps(e['commands'])}`.",'',
                  f"Complete common final table: `{json.dumps(e['common_final_values'])}`.",'',
                  '```text',e['histories'][0]['joint']['question'],'```','',
                  f"Candidate order: `{e['histories'][0]['joint']['answer_mapping']}`.",'',
                  '| History | Complete starting table | Direct operand answers; P(correct) | Joint answer | Joint candidate logps |',
                  '|---|---|---|---|---|']
        for h in e['histories']:
            atoms=[]
            for d in h['direct_operands']:
                # The question spec defines the factual gold; do not take the modal answer as gold.
                # Read the actor from the concrete direct question, which is shown below as well.
                text=d['question'].splitlines()[0]
                actor=next(name for name in e['actors'] if text.startswith('Does '+name+' favor '))
                project=next(p for p in e['projects'] if ' the '+p+' proposal?' in text)
                gold=e['common_final_values'][e['actors'].index(actor)][e['projects'].index(project)]
                atoms.append(f"{actor}: {d['semantic_answer']}; {d['normalized_probabilities'][gold]:.10f}")
            lines.append(f"| {h['origin']} | `{json.dumps(h['starting_values'])}` | {' / '.join(atoms)} | {h['joint']['semantic_answer']} | `{h['joint']['logps']}` |")
        lines += ['', 'Direct questions: '+ ' / '.join(d['question'].splitlines()[0] for d in e['histories'][0]['direct_operands']), '']
        for label,key in [('Native-final','native_reference'),('Same-method no-op','noop_reference')]:
            r=e[key];lines.append(f"{label}: {r['semantic_answer']}; logps `{r['logps']}`; source journal line {r['journal_line']}, physical ID `{r['physical_id']}`.")
        lines += ['',f"Independently computed flags: `{json.dumps(e['calculated_flags'])}`.",'',
                  f"Inputs: [{e['design_file']}](../{e['design_file']}) and the response shard named in `examples.json`.",'',
                  'The exact joint query has two saved fresh passes across all histories.' if e['repeat_joint_rows'] else 'No repeat claim is made for this particular example.', '']
    return '\n'.join(lines).rstrip()+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,default=independent.HERE)
    p.add_argument('--calculation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    independent.require(not a.output.exists(),'Example output exists')
    result={'query_rows':independent.load(a.calculation/'query_rows.json')};rows=examples(a.package,result)
    a.output.mkdir(parents=True)
    (a.output/'examples.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8',newline='\n')
    (a.output/'EXAMPLES.md').write_text(markdown(rows),encoding='utf8',newline='\n')
    print(json.dumps({'examples':len(rows),'categories':[r['category'] for r in rows]}))
