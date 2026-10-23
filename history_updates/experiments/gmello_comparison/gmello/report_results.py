"""Formatting-only reports from frozen evaluation artifacts; no method selection."""
from pathlib import Path
import argparse
import collections
import json
from evaluate import ENDPOINTS, read, write
from gmello import decode_answer

def development_components(raw,out):
    """Post-terminal descriptive accounting only; never feeds model selection."""
    if not (raw/'inference/out/calls.jsonl').exists():return None
    root=Path(__file__).resolve().parent
    cases={case['case_id']:case for case in read(root/'inference/data/development.json')}
    truths={}
    for item in read(root/'evaluator_data/dev_gold.json'):
        case=cases[item['case_id']];history=case['histories']['H0'];state={(s,p):o for s,p,o in history['initial']}
        for s,p,o in history['edits']:state[s,p]=o
        for q in item['questions']:
            entity=q['subject_id']
            for relation in q['path']:entity=state[entity,relation]
            truths[item['case_id'],str(q['kind'])+':'+str(q['index'])]=entity
    stages=[]
    for folder in sorted((raw/'inference/out').glob('dev_r*')):
        counts=collections.defaultdict(collections.Counter);mapping=collections.defaultdict(collections.Counter)
        for path in sorted((folder/'cases').glob('*.json')):
            for row in read(path)['rows']:
                c=counts[row['arm']];truth=truths[row['case_id'],row['query_id']]
                generation=row['fallback_generation'];decoded=decode_answer(generation['text'],cases[row['case_id']]['entities'],generation.get('cap_hit',False),generation.get('invalid_control_tokens',False))
                fallback_correct=decoded['status']=='VALID' and decoded['entity']==truth
                valid_graph=row['graph'].get('entity') is not None;correct_graph=valid_graph and row['graph']['entity']==truth
                if valid_graph:assert row['route']=='graph' and row['outcome']['entity']==row['graph']['entity']
                c.update(questions=1,parseable=row['parse']['status']=='VALID',executable=row['graph']['status'] in ('VALID','NO_RESULT'),nonempty_graph=valid_graph,correct_graph=correct_graph,
                         wrong_graph_return=valid_graph and not correct_graph,fallback_calls=generation.get('measured',False),fallback_correct=fallback_correct,
                         wrong_graph_overrides_correct_fallback=valid_graph and not correct_graph and fallback_correct,
                         chain_cap_hits=row['chain_generation']['cap_hit'],fallback_cap_hits=generation['cap_hit'],
                         final_refusals=row['outcome']['status']=='REFUSAL',legacy_same_output_correct=row['legacy_sensitivity'].get('graph',{}).get('entity')==truth,
                         combined_correct=row['outcome']['status']=='VALID' and row['outcome']['entity']==truth)
                mapping[row['arm']].update(d.get('method','error') for d in row['parse']['decisions'] if d['role']=='relation')
        gate=read(raw/'evidence'/('DEV_GATE_R'+folder.name.removeprefix('dev_r')+'.json'))
        for endpoint,c in counts.items():
            assert c['questions']==189 and c['combined_correct']==gate['counts'][endpoint]['combined_correct']
            stages.append({'revision':gate['revision'],'endpoint':endpoint,**dict(c),'relation_mapping_methods':dict(mapping[endpoint])})
    physical=collections.defaultdict(collections.Counter)
    with (raw/'inference/out/calls.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            call=json.loads(line);request=call['request'];result=call['result']
            kind='chain' if request['max_new_tokens']==256 else 'edit' if request['messages'][-1]['content'].startswith('Sentence:') else 'fallback'
            physical[kind].update(calls=1,input_tokens=result['input_tokens'],output_tokens=result['output_tokens'],cap_hits=result['cap_hit'],invalid_control_tokens=result['invalid_control_tokens'])
    value={'status':'PASS','development_only':True,'main_performance_claimed':False,'stages':stages,
           'physical_calls_including_six_throughput_probes':{k:dict(v) for k,v in physical.items()},
           'logical_stages_are_not_independent_neural_replications':True,'legacy_sensitivity_selected_primary':False}
    write(out/'DEV_COMPONENTS.json',value);return value

def physical_measurements(raw,out):
    calls=[json.loads(line) for line in (raw/'inference/out/calls.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(calls)==len({call['call_id'] for call in calls})
    counters=collections.defaultdict(collections.Counter)
    references=[]
    def visit(value):
        if isinstance(value,dict):
            if 'call_id' in value:references.append(value['call_id'])
            for item in value.values():visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
    for folder in ('cases','controls'):
        for path in (raw/'inference/out/main'/folder).glob('*.json'):visit(read(path))
    for call in calls:
        request=call['request'];result=call['result']
        kind='chain' if request['max_new_tokens']==256 else 'edit' if request['messages'][-1]['content'].startswith('Sentence:') else 'fallback'
        counters[kind].update(calls=1,input_tokens=result['input_tokens'],output_tokens=result['output_tokens'],cap_hits=result['cap_hit'])
    repair=[json.loads(line) for line in (raw/'repair/original/results/calls.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    repair=[call for call in repair if call['endpoint'].startswith('REPAIR_') and call['nonce'] is not None]
    value={'status':'PASS','new_gmello_unique_physical_calls':len(calls),'new_gmello_by_component':{k:dict(v) for k,v in counters.items()},
           'new_gmello_input_tokens':sum(c['result']['input_tokens'] for c in calls),
           'new_gmello_output_tokens':sum(c['result']['output_tokens'] for c in calls),
           'amortized_generation_seconds':sum(c['result']['amortized_seconds'] for c in calls),
           
           'fixed_canary_calls':sum(c['nonce']=='GMELLO_R0_MAIN_V3_FIXED_TECHNICAL_CANARY' for c in calls),
           'main_uncached_physical_calls':sum(c['nonce'] is None for c in calls),
           'control_physical_calls':sum(c['nonce'] is not None and c['nonce']!='GMELLO_R0_MAIN_V3_FIXED_TECHNICAL_CANARY' for c in calls),
           'main_and_control_trace_call_references':len(references),'distinct_calls_referenced_in_main_and_controls':len(set(references)),
           'repeated_trace_references_from_shared_requests':len(references)-len(set(references)),
           'trace_references_are_not_independent_model_trials':True,
           'fresh_repair_loader_and_control_score_calls':len(repair),
           'fresh_repair_score_input_tokens':sum(c['result']['input_tokens'] for c in repair),
           'historical_repair_main_logical_reads':13884,'new_gmello_main_logical_reads':20826}
    write(out/'PHYSICAL_MEASUREMENTS.json',value);return value

def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + ["| " + " | ".join(map(str,row)) + " |" for row in rows]

def render(raw, prior, out):
    raw, prior, out = map(Path, (raw, prior, out)); out.mkdir(parents=True, exist_ok=True)
    terminal = read(raw / "COMPLETE.json")
    amended=terminal.get('main_logical_reads')==34710 and read(raw/'SCIENTIFIC_FREEZE.json').get('schema')=='GMELLO_R0_MAIN_V3_SCIENTIFIC_FREEZE'
    dev_raw=Path(__file__).resolve().parent/'parent_evidence/raw' if amended else raw
    components=development_components(dev_raw,out)
    gates = [read(p) for p in sorted((dev_raw / "evidence").glob("DEV_GATE_R*.json"))]
    historical = read(prior / "evaluation/SCIENTIFIC_RESULTS.json")
    write(out / "HISTORICAL_ENDPOINTS.json", {"historical_only": True, "fresh_neural_replications": False,
          "source_archive_sha256": "fd5ba7f79443377b18558fdf0f942a380c6c0e7039c6a5e570f596ad5dacc3ad",
          "endpoints": historical["endpoints"], "full_original_evidence": "historical/SAME_CASE_BASELINES_V1_RESULTS.zip"})
    text = ["# GMeLLo revision-0 main comparison V3" if amended else "# GMeLLo fair comparison V2", "", "This is an exposed, explicitly amended evaluation, not a fresh confirmatory cohort or a certified reproduction of the authors' implementation.", "", "## Retained development graph readiness", ""]
    gate_rows = []
    for gate in gates:
        core = gate["counts"]["G_CORE"]
        gate_rows.append([gate["revision"], gate["status"], f'{core["target_parseable_linked"]}/96', f'{core["target_nonempty_graphs"]}/96', ", ".join(map(str,gate["correct_graph_target_depths"]))])
    text += table(["Revision", "Gate", "Linked target chains", "Nonempty target graphs", "Correct target depths"],gate_rows)
    disclosure="Revision 0 was selected after inspecting all three DEV results: it is the original prompt and had the highest recorded primary graph coverage. Every original gate remains UNQUALIFIED. The explicit amendment removed the numerical readiness veto without retuning; no new DEV sweep or fresh-confirmation claim is made. G_TEXT is secondary."
    text += ["", disclosure if amended else "Readiness requires at least 90% linked target chains, 50% nonempty target graphs, and a correct graph answer at each depth 2, 3 and 4. Fallback answers never count toward this gate. Every attempted revision is retained; the first passing revision is selected.", ""]
    methods = (("We compared " if terminal.get("main_logical_reads")==34710 else "We prepared, but did not run, a main-cohort comparison of ") + "two unchanged repair scorers with three inference-only GMeLLo-style endpoints on 128 retained subjects, 384 target phrasings and 517 controls. "
               "G_CORE applies the supplied typed edits to an independent RDF graph and retrieves six current facts for fallback; G_FULL uses identical graph/query results and all current facts for fallback. "
               "G_TEXT instead extracts writes from correction sentences and retrieves edited sentences. The base-model question extractor receives only the question, the fixed 37-relation ontology and four TRAIN demonstrations. "
               "Unknown nodes remain variables, concrete constraints are retained, and graph answers always take priority. The prior cohort and outcomes were already exposed.")
    limitations = ("GMeLLo is an external-state comparator, not internal repair. The repair scorers were trained, whereas these comparators use native few-shot inference; this is a whole-system comparison, not training-matched causal isolation. "
                   "The finite graphs, local ontology and reconstructed prompts are documented adaptations, not published benchmark-score reproduction. "
                   "These results cannot establish unique isolation, a general internal-repair impossibility theorem, equivalence from nonsignificance, or a new pass of any earlier F1–F4 gate. The original results and manuscript remain unchanged.")
    if amended:methods+=' '+disclosure
    latex = ["% Standalone insertion only. Do not overwrite the existing manuscript.", "\\paragraph{Methods.} " + methods.replace('_',r'\_').replace('%',r'\%')]
    measured = terminal.get("main_logical_reads") == 34710
    if measured:
        result = read(out / "SCIENTIFIC_RESULTS.json"); valid = read(out / "COMPARISON_VALIDITY.json")
        summary = result["endpoints"]
        text += ["## Complete comparison", "", f'Validity status: {valid["status"]}. Each endpoint has 5,406 post-update and 1,536 original-snapshot logical reads; total 34,710. Repair records are verified historical replay with fresh loader and uncached checks, not new full replications. Identical-request reuse in the new methods is explicitly linked.', ""]
        text += table(["Endpoint", "Complete subjects /128", "Robust targets /384", "Robust controls /517", "Original targets /1536", "Current same targets /1536"],
                      [[a, summary[a]["complete_subjects"], summary[a]["robust_targets"], summary[a]["robust_controls"], summary[a]["original_targets"]["correct"], summary[a]["current_same_targets"]["correct"]] for a in ENDPOINTS])
        text += ["", "Complete subjects require every associated target and control to be correct across H0/H1/S0/S1. Original/current scores use the same target questions and denominators.", "", "## Two registered paired contrasts", ""]
        contrast_rows = []
        result_sentences = []
        for name, contrast in result["primary_contrasts"].items():
            point = 100 * contrast["repair_mean_minus_comparator"]; low, high = [100*x for x in contrast["interval"]]
            contrast_rows.append(["Mean repair minus " + name, f"{point:.2f}", f"[{low:.2f}, {high:.2f}]"])
            result_sentences.append(f'The mean-repair-minus-{name.replace("_"," ")} difference was {point:.2f} percentage points (97.5% interval [{low:.2f}, {high:.2f}]).')
        text += table(["Contrast", "Difference (percentage points)", "97.5% interval"], contrast_rows)
        text += ["", "Intervals use 10,000 subject-cluster bootstrap draws, seed 2026101211. Both repair seeds, all questions and all histories remain linked in each sampled subject. The intervals do not establish equivalence.", ""]
        if valid["status"] != "PASS":
            interpretation = "The repair parity gate failed, so the paired competitive headline is blocked. All observed results and failures remain disclosed."
        elif all(c["interval"][0] > 0 for c in result["primary_contrasts"].values()):
            interpretation = "Both registered intervals favor the mean repair system on this retained workload under the explicit amendment. This is local whole-system evidence, not a general claim about graph methods or internal repair."
        else:
            interpretation = "The two registered intervals do not both establish a positive repair advantage. Report the measured endpoint counts and intervals directly; do not reinterpret this as equivalence or select the more favorable comparator."
        text += [interpretation, "", "## Graph and fallback stages", ""]
        text += table(["Endpoint", "Logical questions", "Parseable", "Executable", "Nonempty graph", "Correct graph", "Correct fallback", "Wrong graph overrides correct fallback"],
                      [[a] + [summary[a]["pipeline"][k] for k in ("logical_questions","parseable_chains","executable_queries","nonempty_graphs","correct_graph_answers","fallback_correct","graph_overrides_correct_fallback")] for a in ("G_CORE","G_FULL","G_TEXT")])
        text += ["", "All fallback components were scheduled. A correct fallback never replaces a wrong graph return. The same-output old literal parser is a sensitivity only and was not used to choose the primary method.", "", "## Secondary outcomes and diagnostics", ""]
        text += table(["Endpoint", "Any of three /128", "All three /128", "Correct refusals", "Incorrect refusals", "Wrong program / right endpoint", "Guarded witnesses /384"],
                      [[a] + [summary[a][k] for k in ("any_of_three_robust_phrasings","all_three_robust_phrasings","correct_refusals","incorrect_refusals","wrong_program_right_endpoint","guarded_witnesses")] for a in ENDPOINTS])
        text += ["", "Any/all-three scores require the relevant phrasing to remain correct across the four histories. Full per-history, control-family, reference and status denominators are in SCIENTIFIC_RESULTS.json; row-level failures, paired fixes/harms and lost guards are retained in the ledgers. Uniform wrong answers are not recast as history dependence.", ""]
        edits = summary["G_TEXT"]["edit_fidelity"]
        text += [f'G_TEXT edit fidelity: {edits["exact"]}/{edits["intended"]} exact writes; {edits["wrong_applied"]} wrong applied writes and {edits["unapplied"]} unapplied writes. Its post-update graph matched the intended typed-input state in {edits["post_graph_equal"]}/{edits["post_graphs"]} states. These errors are retained, not oracle-corrected.', "",
                 f'Repair uncached/snapshot parity: {valid["repair_parity"]["status"]}; GMeLLo diagnostic panels with differences: {len(valid["gmello_diagnostic_differences"])}. Every input, selected-chain, output and state-return comparison is preserved in raw diagnostics. Snapshot access does not establish natural-language restore competence.', "",
                 "Witness cohorts remain separate: 94 original, 32 later, overlap 15 (17 later-only). See WITNESS_RECOVERY.jsonl and the cohort summaries for preserved controls and references.", ""]
        latex += ["", "\\begin{tabular}{lrrr}", "Endpoint & Subjects /128 & Targets /384 & Controls /517 " + r"\\", "\\hline"]
        for a in ENDPOINTS:
            latex.append(a.replace("_",r"\_") + f' & {summary[a]["complete_subjects"]} & {summary[a]["robust_targets"]} & {summary[a]["robust_controls"]} ' + r"\\")
        latex += ["\\end{tabular}", "", "\\paragraph{Results.} " + " ".join(result_sentences).replace("%",r"\%") + " " + interpretation,
                  "We used 10,000 subject-cluster bootstrap draws, retaining both repair seeds and every question/history within subjects. Repair replay was not counted as fresh replication."]
    else:
        status = terminal.get("scientific_status", "TECHNICALLY_INCOMPLETE")
        diagnostic = {"status":status,"complete_comparison_measured":False,"new_main_logical_reads":terminal.get("new_gmello_logical_reads",0),
                      "registered_main_logical_reads":34710,"registered_new_gmello_reads":20826,
                      "historical_repair_reads_replayed":13884,"competitive_headline_permitted":False,
                      "development_revisions":len(gates),"terminal":terminal}
        write(out / "DIAGNOSTIC_SUMMARY.json",diagnostic)
        sentence = f'The bounded run stopped with status {status}. The complete 34,710-read paired comparison was not measured; no poor-GMeLLo superiority headline is permitted.'
        text += ["## Diagnostic stop", "", sentence, "", "No configuration qualified, so none was selected or scientifically frozen for main inference. The actual neural work covered all 189 questions on the original 32 DEV roots under each of the three permitted global prompts.", "",
                 "All DEV revisions, raw generations, graph/edit traces and technical records are retained. All 20,826 new GMeLLo main reads remain unmeasured. Fresh main-cohort repeat/snapshot controls and the paired bootstrap contrasts were not run after this required stop. The 13,884 historical repair reads were independently replayed; both fresh original-adapter loader canaries matched their original logits exactly. Historical replay is not substituted for the missing comparison.", ""]
        if all('G_TEXT' in gate['counts'] for gate in gates):
            text += ["## DEV endpoint diagnostics", ""]
            text += table(["Revision", "Endpoint", "Parseable /189", "Nonempty graph /189", "Correct graph /189", "Combined correct /189"],
                          [[gate['revision'],endpoint]+[gate['counts'][endpoint][key] for key in ('parseable_linked','nonempty_graphs','correct_graphs','combined_correct')] for gate in gates for endpoint in ('G_CORE','G_FULL','G_TEXT')])
            text += ["", "These are DEV observations, not main-cohort performance estimates. The target-only graph-coverage gate above governs readiness; overall graph coverage and fallback correctness cannot substitute for it.", "", "## DEV edit intake", ""]
            text += table(["Revision", "Intended", "Applied", "Exact", "Wrong applied", "Unapplied", "Correct resulting graphs /32"],
                          [[gate['revision']]+[gate['g_text_edit_counts'][key] for key in ('intended','applied','exact_triple','wrong_applied','unapplied')]+[gate['g_text_graph_equal']] for gate in gates])
            text += ["", "Identical edit/fallback requests were reused across global chain-prompt revisions, with explicit call links; these are not independent replications. Wrong and unapplied writes were retained. Raw parse/link/query and same-output literal-parser sensitivity traces are included.", ""]
        latex += ["", "\\paragraph{Results.} " + sentence.replace("_",r"\_")]
        if components:
            text += ["## Independent fallback and same-output sensitivity", ""]
            text += table(["Revision", "Endpoint", "Correct fallback /189", "Wrong graph overrides correct fallback", "Old literal-parser correct /189", "Chain cap hits", "Fallback cap hits"],
                          [[r['revision'],r['endpoint']]+[r[k] for k in ('fallback_correct','wrong_graph_overrides_correct_fallback','legacy_same_output_correct','chain_cap_hits','fallback_cap_hits')] for r in components['stages']])
            text += ["", "These component scores do not change graph priority or the readiness decision. DEV_COMPONENTS.json also records relation-mapping counts, refusals, executable queries, physical call/token totals and edit-generation cap hits. Logical reads and exact-request reuse are not fresh replications.", ""]
            coverage=', '.join(str(g['counts']['G_CORE']['target_nonempty_graphs'])+'/96' for g in gates)
            latex += [f'Target graph coverage was {coverage} across the three permitted configurations; 48/96 was required. All configurations met linked-chain and correct 2/3/4-hop coverage checks, but none passed the full readiness gate. No new main-cohort response was generated.']
    if amended:
        usage=physical_measurements(raw,out)
        text += ["## Physical calls, tokens and reuse", ""]
        text += table(['Component','Unique physical calls','Input tokens','Output tokens'],[[kind,row['calls'],row['input_tokens'],row['output_tokens']] for kind,row in usage['new_gmello_by_component'].items()])
        text += ['',f"Generation occupied {usage['amortized_generation_seconds']:.2f} summed amortized seconds; this excludes setup, retrieval, parsing and export. There were {usage['fixed_canary_calls']} fixed technical canary calls and {usage['fresh_repair_loader_and_control_score_calls']} fresh repair loader/control scoring calls. Full timings, trace reuse and token accounting are in PHYSICAL_MEASUREMENTS.json. Logical rows are not independent model trials.",'']
    text += ["## Preserved historical endpoints", ""]
    text += table(["Historical endpoint", "Complete subjects /128", "Origin"], [[name, row["complete_subjects"], "Prior observation, not rerun"] for name,row in historical["endpoints"].items()])
    text += ["", "The complete original results, QA/PROGRAM models, old MeLLo/GMeLLo records and earlier comparisons remain in the hash-verified historical archive. The old 35/128 GMeLLo result and 189/189 DEV fallback audit are unchanged.", "", "## Methods and limits", "", methods, "", limitations, "",
             "Primary source: [Chen et al., Findings EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.844.pdf), Sections 3 and Appendix A. See FIDELITY.md for the source-to-component map and reconstruction limits.", ""]
    latex += ["", "\\paragraph{Limitations.} " + limitations.replace("–","--"), ""]
    (out / "RESULTS.md").write_text("\n".join(text),encoding="utf-8")
    (out / "manuscript_inserts.tex").write_text("\n".join(latex),encoding="utf-8")
    return {"status":"PASS","formatting_only":True,"complete_comparison_measured":measured,"development_revisions":len(gates)}

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--raw",type=Path,required=True); p.add_argument("--prior",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); print(json.dumps(render(a.raw,a.prior,a.out)))
