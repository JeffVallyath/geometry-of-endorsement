"""Paper-facing tables from completed scientific reconstructions.

No estimator, alias rule, eligibility rule or resampling procedure is defined
here. Those remain in the retained scientific analyzers.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys

def read(path):return json.loads(Path(path).read_bytes())

def emit(folder,name,title,rows,sources,unit):
    if not rows:raise ValueError('Empty mapped table: '+name)
    columns=list(rows[0])
    if any(list(row)!=columns for row in rows):raise ValueError('Inconsistent table columns')
    record={'title':title,'unit':unit,'columns':columns,'rows':rows,'sources':sources}
    (folder/(name+'.json')).write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    with (folder/(name+'.csv')).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    def display(value):
        if value is None:return 'Not reported'
        if isinstance(value,float):return f'{value:.6g}'
        return str(value)
    md=['# '+title,'',unit,'','| '+' | '.join(columns)+' |','|'+'|'.join('---' for _ in columns)+'|']
    md.extend('| '+' | '.join(display(row[k]).replace('|','\\|') for k in columns)+' |' for row in rows)
    (folder/(name+'.md')).write_text('\n'.join(md)+'\n',encoding='utf-8')
    def tex(value):
        return display(value).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('#',r'\#')
    lines=[r'\begin{tabular}{'+'l'*len(columns)+'}', ' & '.join(tex(c) for c in columns)+r' \\',r'\hline']
    lines.extend(' & '.join(tex(row[k]) for k in columns)+r' \\' for row in rows)
    lines.append(r'\end{tabular}')
    (folder/(name+'.tex')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return record

def export(source,out,sections,analysis_sources=None):
    source,out=Path(source),Path(out)
    destination=out/'tables';destination.mkdir(exist_ok=False)
    supplied=analysis_sources or {}
    def at(job,name):return Path(supplied.get(job,out/'analyses'/job))/name
    def available(job,name):return at(job,name).is_file()
    def load(job,name):return read(at(job,name))
    tables={};claims={};bindings={}
    def table(name,title,rows,refs,unit):
        tables[name]=emit(destination,name,title,rows,refs,unit)
    legacy=source/'v3/studies/original_public_history'
    if available('primary','state_sufficiency/v10.csv'):
        with at('primary','state_sufficiency/v10.csv').open(encoding='utf-8',newline='') as stream:
            primary=list(csv.DictReader(stream))
        update_names={'INV_PAIR_NLL':'Activation intervention','FREE_PAIR_CONSISTENCY':'Activation plus consistency',
                      'EXISTING_CORRECTION':'In-context, request order','LATEST_SAME_WORDING':'In-context, statement order'}
        model_names={'gemma':'Gemma-2-9B','qwen':'Qwen3-8B'}
        rows=[{'Model':model_names[r['model']],'Update':update_names[r['group']],'Source_condition':r['group'],'Rate_percent':100*float(r['rate']),
               'Lower_percent':100*float(r['lower']),'Upper_percent':100*float(r['upper']),
               'Affected_cases':int(r['roots']),'Cases':int(r['total_roots'])} for r in primary]
        table('table_03','Synthetic matched-history rates',rows,['primary/state_sufficiency/v10.csv'],
              'Scheduled-question percentages; 99.375% whole-case intervals. Seeds are averaged within each case.')
        claims['synthetic_primary']=rows
        bindings['figure_02']={'data':'table_03.json','approved_design':'Supplied paper, Figure 2','kind':'measured rates and intervals'}
    if available('fresh','tables/PRIMARY_WITNESSES.csv'):
        example=read(legacy/'reproducibility/fresh_history_validation/examples/EXAMPLE_0017.json')
        with at('fresh','tables/PRIMARY_WITNESSES.csv').open(encoding='utf-8',newline='') as stream:
            matches=[r for r in csv.DictReader(stream) if r['root_id']==example['root_id'] and r['query_id']==example['question']['query_id'] and r['seed']=='0']
        assert len(matches)==1 and json.loads(matches[0]['predictions'])==example['witness']['predictions']
        rows=[]
        for h in example['histories']:
            checks=sum(bool(c['valid']) and c['prediction']==c['gold'] for c in h['controls'])
            assert len(h['controls'])==20
            rows.append({'History':'H'+str(h['origin']),'Changed_facts':h['changed_facts'],'Tokens_before_question':h['prefix_length'],
                         'Checks_passed':checks,'Checks':len(h['controls']),'Answer':example['question']['labels'][h['target']['prediction']]})
        table('table_02','Synthetic worked example',rows,['fresh/tables/PRIMARY_WITNESSES.csv','fresh_history_validation/examples/EXAMPLE_0017.json'],
              'One selected question, eight histories, 20 checks per history.')
        claims['synthetic_worked_example']={'question':example['question'],'rows':rows,'witness':example['witness'],
                                            'histories_with_all_checks_and_scores':example['histories']}
        claims['fresh_confirmation']=load('fresh','replay/PRIMARY.json')
    if available('controls','summary.json'):claims['stricter_controls']=load('controls','summary.json')
    if available('qwen32','CLAIMS.json'):claims['synthetic_larger_model']=load('qwen32','CLAIMS.json')
    if available('original_public_history','analysis/summary.json'):
        native=load('original_public_history','analysis/summary.json');rows=[]
        for model,title in [('qwen','Qwen2.5-7B'),('gemma','Gemma-2-9B')]:
            r=native[model]['primary']
            rows.append({'Model':title,'Prompt':'Original','History_dependent':r['qualified'],'Questions':r['target_phrasings'],
                         'Related_minus_unrelated_pp':100*r['relevance_difference'],'Long_chain_witnesses':r['long_chain_qualified'],
                         'Long_chain_questions':r['long_chain_phrasings']})
        claims['public_matched_histories']=native
        if available('paired_native_scale','analysis/SUMMARY.json'):
            paired=load('paired_native_scale','analysis/SUMMARY.json');claims['paired_scale']=paired
            for key,title in [('small','Qwen2.5-7B'),('large','Qwen2.5-32B')]:
                r=paired['models'][key]['primary']
                rows.append({'Model':title,'Prompt':'Shared size-comparison format','History_dependent':r['qualified'],'Questions':r['target_phrasings'],
                             'Related_minus_unrelated_pp':100*r['relevance_difference'],'Long_chain_witnesses':None,'Long_chain_questions':None})
        if available('repair_gemma27','analysis/SUMMARY.json'):
            gemma=load('repair_gemma27','analysis/SUMMARY.json');a=gemma['additional']
            rows.append({'Model':'Gemma-2-27B','Prompt':'Own format','History_dependent':gemma['own_native_witness_count'],
                         'Questions':a['target_denominator'],'Related_minus_unrelated_pp':100*a['guarded_relevance_contrast'],
                         'Long_chain_witnesses':a['K_witnesses'],'Long_chain_questions':a['K_scheduled_denominator']})
            claims['gemma_native']=a
        table('table_01','Public multi-hop matched-history results',rows,
              ['original_public_history/analysis/summary.json','paired_native_scale/analysis/SUMMARY.json','repair_gemma27/analysis/SUMMARY.json'],
              'Question counts use 384 scheduled phrasings per model. Prompt-matched size comparisons are separate measurements.')
        example=load('original_public_history','analysis/case_1617.json');rows=[]
        for arm in ('H0','H1','S0','S1','C','C+'):
            row={'History':arm}
            for model,title in [('qwen','Qwen2.5-7B'),('gemma','Gemma-2-9B')]:
                selected=example['models'][model];qid=selected['target']['query_id']
                matches=[r for r in selected['responses'] if r['arm']==arm and r['query_id']==qid]
                assert len(matches)==1 and matches[0]['valid']
                row[title]=example['case']['canonical_labels'][matches[0]['entity_id']]
            rows.append(row)
        table('table_04','Public worked example',rows,['original_public_history/analysis/case_1617.json'],
              'Parsed target answers under four histories and two references.')
        bindings['figure_01']={'data':'table_04.json','approved_design':'Supplied paper, Figure 1',
                              'kind':'author-designed schematic with measured example; no neural trajectory is inferred'}
    if available('refresh','PRIMARY_RECOUNT.json'):
        refresh=load('refresh','PRIMARY_RECOUNT.json');claims['cache_refresh']=load('refresh','RESULTS.json')
        labels={'raw_rows':'Saved score rows','baseline_W':'Eligible witness observations before refresh',
                'B_repairs':'Witnesses repaired, blocks 17-24','C_repairs':'Witnesses repaired, control blocks 25-32',
                'treated_qualifying_witnesses':'Eligible witnesses after refresh','baseline_witnesses_retained':'Original witnesses still present',
                'new_qualifying_witnesses':'New witnesses after refresh','baseline_ceased_qualifying_without_repair':'Lost qualification without repair',
                'B_direct_damage':'Direct answers made wrong','B_direct_corrections':'Direct answers corrected',
                'A_joint_errors':'Combined-answer errors before refresh','B_joint_errors':'Combined-answer errors after refresh',
                'B_joint_damage':'Combined answers made wrong','B_joint_corrections':'Combined answers corrected',
                'C_joint_errors':'Combined-answer errors after control-region refresh','D_joint_errors':'Combined-answer errors in donor reference',
                'cases':'Cases','treated_witness_cases':'Cases with witnesses after refresh','repaired_designs':'Distinct designs with repaired witnesses'}
        assert set(labels)==set(refresh)
        table('table_05','Cache refresh accounting',[{'Measurement':labels[k],'Count':v,'Source_field':k} for k,v in refresh.items()],
              ['refresh/PRIMARY_RECOUNT.json'],'Counts retain both seeds, all scheduled questions and every correction or harm.')
        bindings['figure_08']={'data':'claims.json#/cache_refresh','approved_design':'Supplied paper, Figure 8','kind':'measured paired changes and intervals'}
    for job,path,key in [('crossover','cache_crossover/results.json','cache_crossover'),
                         ('crossover-extension','cache_crossover_extension/replay.json','cache_sites'),
                         ('mechanism','VERIFICATION.json','cache_reader_checks')]:
        if available(job,path):claims[key]=load(job,path)
    if 'cache_reader_checks' in claims:
        from experiments._shared.mechanism_outputs import reconstruct
        effects=reconstruct(legacy);claims['cache_reader_effects']=effects
        table('cache_reader_effects','Question-side reader interventions',
              [{'Heads':len(r['sites']),'Recipient_history':'H'+str(r['recipient']),
                'Answer_format':{'d0':'Words','d1':'Letters'}[r['format']],
                'Selection_role':r['partition'],'Blocked_fraction':r['block_fraction'],
                'Restored_fraction':r['inject_fraction'],'Minimum_fraction':r['mediation_score']}
               for r in effects['studies']['mediation']['reader_unions']],
              ['completed_followups/mediation/ALL_UNIQUE_FORWARD_SCORES.csv','completed_followups/mediation/READER_UNION_EFFECTS.json'],
              'Fraction of the recorded full-carrier log-odds effect. Selection uses the original training condition; all reverse/format checks remain separate.')
    if 'cache_crossover' in claims:bindings['figure_03']={'data':'claims.json#/cache_crossover/example','approved_design':'Supplied paper, Figure 3','kind':'saved-score cache crossover'}
    if 'cache_sites' in claims:
        for n in ('06','07'):bindings['figure_'+n]={'data':'claims.json#/cache_sites','approved_design':'Supplied paper, Figure '+str(int(n)),'kind':'individual-site and leave-one-out saved-score data'}
    if available('history_exposure','bridge/a10/A10_RECOMPOSITION.json'):
        bridge=load('history_exposure','bridge/a10/A10_RECOMPOSITION.json');compiler=load('history_exposure','bridge/NSR1_ANALYSIS.json')
        claims['synthetic_recomposition']=bridge;claims['synthetic_compiler']=compiler
        b=bridge['totals'];n=compiler['a10']['A10_TEXT_SOURCE']['totals']
        refresh=load('refresh','PRIMARY_RECOUNT.json') if available('refresh','PRIMARY_RECOUNT.json') else None
        rows=[{'Answering_system':'Model alone','Repaired_witness_observations':0,'Combined_answer_errors':b['joint_old_error'],'Introduced_errors':0}]
        if refresh:rows.append({'Answering_system':'Cache refresh','Repaired_witness_observations':refresh['B_repairs'],'Combined_answer_errors':refresh['B_joint_errors'],'Introduced_errors':refresh['B_joint_damage']})
        rows.extend([{'Answering_system':'Saved direct answers plus supplied operations','Repaired_witness_observations':b['recomposition_repairs'],'Combined_answer_errors':b['joint_new_error'],'Introduced_errors':b['joint_introduced']},
                     {'Answering_system':'External dictionary and neural compiler','Repaired_witness_observations':n['W_fully_preserved_repair'],'Combined_answer_errors':n['joint_new_error'],'Introduced_errors':n['joint_introduced']}])
        table('table_06','Answering on the cache-refresh cohort',rows,['history_exposure/bridge/a10/A10_RECOMPOSITION.json','history_exposure/bridge/NSR1_ANALYSIS.json','refresh/PRIMARY_RECOUNT.json'],
              '83 eligible seed-question observations; 73,728 combined-answer comparisons. Compiler observations are reused across editor seeds.')
        bindings['figure_04']={'data':'implementation and methods.md','approved_design':'Supplied paper, Figure 4','kind':'author-designed architecture schematic, not a measured trajectory'}
    core_jobs=('original_public_history','paired_native_scale','repair_qwen7','repair_qwen32','repair_gemma27')
    if all(available(job,'analysis/'+('summary.json' if job=='original_public_history' else 'SUMMARY.json')) for job in core_jobs):
        sys.path.insert(0,str(source/'v3/src'))
        from repro import tables as frozen
        values={'original_native':load('original_public_history','analysis/summary.json'),
                'paired_native':load('paired_native_scale','analysis/SUMMARY.json'),
                **{job:load(job,'analysis/SUMMARY.json') for job in core_jobs[2:]}}
        # Existing presentation accounting reads detailed outcome/harm ledgers.
        # During preview, map only the output-root path resolution, not data.
        if supplied:
            class OutputPaths:
                def __truediv__(self,name):
                    first,*rest=Path(name).parts
                    return Path(supplied[first]).joinpath(*rest)
            output_paths=OutputPaths()
        else:output_paths=out/'analyses'
        core=frozen.add_presentation(frozen.legacy_result(values),values,output_paths,{})
        claims['public_repair']=core
        conditions={'HISTORY':'Full history','NATIVE32_SAVED':'Full history','NATIVE_GEMMA':'Full history',
                    'REFLECT':'Note before question','CURRENT_TEXT':'Corrected facts, text','CURRENT_JSON':'Corrected facts, JSON',
                    'STATE_BASE':'Dictionary, untrained','STATE_0':'Dictionary, trained seed 0','STATE_1':'Dictionary, trained seed 1'}
        table('table_07','Complete cases under each repair condition',
              [{'Model':r['model'],'Condition':conditions[r['arm']],'Source_endpoint':r['arm'],'Complete_cases':r['complete_subjects'],'Cases':r['subjects']} for r in core['repair']],
              [job+'/analysis/SUMMARY.json' for job in core_jobs[2:]],'Every target and control must be correct under all four histories; each seed remains separate.')
        gains=[r for r in core['presentation']['paired_gains'] if r['baseline'] not in ('CURRENT_JSON','STATE_BASE')]
        table('table_08','Paired gains of the trained interpreter',
              [{'Model':r['model'],'Baseline':r['baseline'],'Gain_pp':100*r['difference'],'Lower_pp':100*r['interval'][0],'Upper_pp':100*r['interval'][1]} for r in gains],
              ['public_repair/presentation/paired_gains'],'97.5% intervals from 10,000 whole-subject resamples, with both repair seeds linked.')
        table('table_09','Remaining trained-interpreter errors',
              [{'Model':r['model'],'Seed':r['arm'][-1],'Refusals':r['refusals'],'Wrong_accepted_answers':r['wrong_accepted'],'Questions':r['queries']} for r in core['presentation']['trained_current_outcomes']],
              ['public_repair/presentation/trained_current_outcomes'],'Distinct current questions, out of 901 per model and seed.')
        trained=[r['complete_subjects']/r['subjects'] for r in core['repair'] if r['arm'] in ('STATE_0','STATE_1')]
        text=[r['complete_subjects']/r['subjects'] for r in core['repair'] if r['arm']=='CURRENT_TEXT']
        assert len(trained)==6 and len(text)==3
        claims['paper_averages']={'trained_complete_case_fraction':sum(trained)/6,'corrected_text_complete_case_fraction':sum(text)/3}
        bindings['figure_05']={'data':'table_07.json','approved_design':'Supplied paper, Figure 5','kind':'complete cases; trained bars average two seeds within model'}
    if available('history_exposure','analysis/METRICS.json'):
        from experiments._shared.record_counts import token_matching
        exposure=source/'v3/studies/history_exposure'
        token_counts=token_matching(exposure/'data/schedule/token_lengths.jsonl.gz')
        frozen_counts=read(exposure/'protocol/token_matching_audit.json')
        assert all(value==frozen_counts[key] for key,value in token_counts.items())
        claims['history_exposure_token_matching']=token_counts
        metrics=load('history_exposure','analysis/METRICS.json');claims['history_exposure']=metrics
        rows=[]
        for arm in ('ISO','HIST','CURRENT_EXTRA'):
            for seed in ('seed0','seed1'):
                r=metrics['metrics'][seed+'/'+arm]
                label={'ISO':'None (ISO)','HIST':'Original history (HIST)','CURRENT_EXTRA':'Final facts only (CURRENT_EXTRA)'}[arm]
                rows.append({'Additional_text':label,'Seed':seed[-1],'Complete_cases':r['whole_subject'],'Targets':r['robust_target'],
                             'Controls':r['robust_control'],'Answer_changes':r['answer_disagreement']})
        table('table_10','History access at the trained interpreter',rows,['history_exposure/analysis/METRICS.json'],
              'Cases /128, targets /384, controls /517 and changed answers /901. Correctness requires all four histories.')
    if available('same_case_baselines','analysis/SCIENTIFIC_RESULTS.json'):
        from experiments._shared.record_counts import training_counts
        baseline_source=source/'v3/studies/same_case_baselines'
        counts=training_counts(baseline_source/'training_join.jsonl.gz')
        frozen_counts=read(baseline_source/'TRAINING_PROVENANCE.json')
        assert all(value==frozen_counts[key] for key,value in counts.items())
        claims['training_sample_counts']=counts
        science=load('same_case_baselines','analysis/SCIENTIFIC_RESULTS.json');claims['common_state_readers']=science
        snapshots=load('same_case_baselines','analysis/MATCHED_SNAPSHOTS.json')
        labels={'REPAIR_0':'Ranking, seed 0','REPAIR_1':'Ranking, seed 1','PROGRAM_0':'Generates operation, seed 0',
                'PROGRAM_1':'Generates operation, seed 1','QA_0':'Generates answer, seed 0','QA_1':'Generates answer, seed 1',
                'MELLO_CURRENT':'MeLLo, adapted','GMELLO_LOCAL':'GMeLLo, superseded implementation'}
        def endpoint_rows(data,names):
            assert all(data[name]['pre_correct']==snapshots[name]['original_correct'] and snapshots[name]['matched_reads']==1536 for name in names)
            return [{'Reader':labels[name],'Source_endpoint':name,'Complete_cases':data[name]['complete_subjects'],'Targets':data[name]['robust_targets'],
                     'Controls':data[name]['robust_controls'],'Original_targets':snapshots[name]['original_correct']} for name in names]
        primary=[k for k in ('REPAIR_0','REPAIR_1','PROGRAM_0','PROGRAM_1','QA_0','QA_1','MELLO_CURRENT')]
        table('table_11','Common-state reader comparisons',endpoint_rows(science['endpoints'],primary),['same_case_baselines/analysis/SCIENTIFIC_RESULTS.json'],
              'Cases /128, targets /384, controls /517, original-snapshot targets /1536. Ranking and PROGRAM share exact execution.')
        table('legacy_gmello','Superseded GMeLLo observation from the supplied paper',endpoint_rows(science['endpoints'],['GMELLO_LOCAL']),
              ['same_case_baselines/analysis/SCIENTIFIC_RESULTS.json'],'Original implementation and rows, not relabeled as corrected results.')
        for name in ('MATCHED_SNAPSHOTS.json','TRAINING_EXPOSURE_JOIN.json','BEHAVIOR_COMPONENT_AUDIT.json','COST_TIMING_WORKLOAD.json'):
            claims['common_state_'+name.removesuffix('.json').lower()]=load('same_case_baselines','analysis/'+name)
    if available('gmello_comparison','SCIENTIFIC_RESULTS.json'):
        science=load('gmello_comparison','SCIENTIFIC_RESULTS.json');claims['corrected_gmello']=science
        rows=[]
        for name in ('REPAIR_0','REPAIR_1','G_CORE','G_FULL','G_TEXT'):
            r=science['endpoints'][name]
            rows.append({'Endpoint':name,'Complete_cases':r['complete_subjects'],'Targets':r['robust_targets'],'Controls':r['robust_controls'],
                         'Original_targets':r['original_targets']['correct'],'Matched_current_targets':r['current_same_targets']['correct']})
        assert [r['Complete_cases'] for r in rows]==[124,127,91,89,46]
        table('gmello_corrected','Corrected local-graph comparison',rows,['gmello_comparison/SCIENTIFIC_RESULTS.json'],
              'Cases /128, targets /384, controls /517; matched original/current targets /1536. CORE/FULL are primary; TEXT is secondary.')
        table('gmello_contrasts','Corrected primary paired contrasts',
              [{'Comparator':k,'Repair_minus_comparator_pp':100*r['repair_mean_minus_comparator'],'Lower_pp':100*r['interval'][0],'Upper_pp':100*r['interval'][1]} for k,r in science['primary_contrasts'].items()],
              ['gmello_comparison/SCIENTIFIC_RESULTS.json'],'97.5% intervals; 10,000 linked-subject draws, seed 2026101211.')
    (destination/'claims.json').write_text(json.dumps(claims,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (destination/'figure_bindings.json').write_text(json.dumps(bindings,indent=2)+'\n',encoding='utf-8')
    if set(sections)=={'matched_histories','cache_interventions','external_state','interpreter_checks','gmello_comparison'}:
        assert {f'table_{i:02}' for i in range(1,12)}<=set(tables),'INCOMPLETE_PAPER_TABLE_COVERAGE'
        assert {f'figure_{i:02}' for i in range(1,9)}==set(bindings),'INCOMPLETE_FIGURE_BINDING_COVERAGE'
        assert len(tables['table_01']['rows'])==5 and len(tables['table_06']['rows'])==4
    return {'status':'PASS','tables':sorted(tables),'figures_bound':sorted(bindings),
            'numerical_source':'Completed observation analyzers, not paper text','paper_alignment':'Corrected GMeLLo is an updated supplement to the unchanged PDF'}
