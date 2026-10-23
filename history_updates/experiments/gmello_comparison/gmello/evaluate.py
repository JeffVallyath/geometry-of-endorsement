"""Local label-bearing evaluator; never deployed in the model worker.

All endpoints retain the prescribed subject/history/question linkage. Historical
repair reads are explicitly replay, not new inference or additional replication.
"""
from pathlib import Path
import argparse
import ast
import collections
import gzip
import hashlib
from itertools import combinations
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "inference"))
from gmello import decode_answer, digest
ENDPOINTS = ("REPAIR_0", "REPAIR_1", "G_CORE", "G_FULL", "G_TEXT")
HISTORIES = ("H0", "H1", "S0", "S1")
ALL = HISTORIES + ("C", "C+")
PRIOR_SHA = "fd5ba7f79443377b18558fdf0f942a380c6c0e7039c6a5e570f596ad5dacc3ad"

def read(path): return json.loads(Path(path).read_bytes())
def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream: yield json.loads(line)
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
def write_rows(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for value in values: stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
def key(row): return tuple(row[k] for k in ("case_id", "history", "phase", "query_id", "arm"))

def bootstrap(complete, case_ids):
    vectors = {endpoint: np.asarray([complete[endpoint][cid] for cid in case_ids], dtype=float) for endpoint in ENDPOINTS}
    repair = (vectors["REPAIR_0"] + vectors["REPAIR_1"]) / 2
    indices = np.random.default_rng(2026101211).integers(0, len(case_ids), size=(10000, len(case_ids)))
    result = {}
    for endpoint in ("G_CORE", "G_FULL"):
        difference = repair - vectors[endpoint]
        interval = np.quantile(difference[indices].mean(axis=1), [.0125, .9875])
        result[endpoint] = {"repair_mean_minus_comparator": float(difference.mean()), "interval": interval.tolist(), "confidence": .975,
                            "draws": 10000, "seed": 2026101211, "unit": "subject; both repair seeds, every question and every history remain linked",
                            "subject_differences": difference.tolist(), "equivalence_claim": False,
                            "per_repair_seed_points": {str(seed): float((vectors["REPAIR_" + str(seed)] - vectors[endpoint]).mean()) for seed in (0,1)}}
    return result

def source_guard_function(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "endpoints")
    namespace = {"ARMS": HISTORIES, "ALL_ARMS": ALL, "combinations": combinations}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["endpoints"]

def evaluate(raw, prior, out, *, software_fixture=False):
    raw, prior, out = Path(raw), Path(prior), Path(out)
    terminal = read(raw / "COMPLETE.json")
    assert terminal["main_logical_reads"] == 34710
    assert bool(terminal.get("software_fixture_only", False)) == software_fixture, "FIXTURE_NOT_SCIENTIFIC_EVIDENCE"
    freeze = read(raw / "SCIENTIFIC_FREEZE.json")
    assert digest({k:v for k,v in freeze.items() if k != "public_projection_sha256"}) == freeze["public_projection_sha256"]
    gold_rows = read(prior / "evaluator_data/EVALUATOR_ONLY.json")
    gold = {(g["case_id"], g["query_id"]): g for g in gold_rows}
    cases = read(prior / "evaluator_data/RUNTIME_CASES.json"); case_ids = [case["case_id"] for case in cases]
    by_id = {case["case_id"]: case for case in cases}; entities = {case["case_id"]: {k:[v] for k,v in case["labels"].items()} for case in cases}
    logical = []
    for row in rows(prior / "raw/results/logical.jsonl"):
        if row["arm"] in ("REPAIR_0", "REPAIR_1"):
            logical.append(dict(row, record_origin="verified_historical_repair_replay", source_archive_sha256=PRIOR_SHA))
    graph_records = []; updates = []
    for path in sorted((raw / "inference/out/main/cases").glob("*.json")):
        result = read(path)
        assert result["freeze_sha256"] == freeze["freeze_sha256"] and result["completed_after_scientific_freeze"] is True
        logical.extend(dict(row, record_origin="new_base_model_inference_or_exact_request_reuse") for row in result["rows"])
        graph_records.extend(result["graphs"]); updates.extend(result["updates"])
    expected_keys = {(case["case_id"], history, phase, q["query_id"], endpoint) for case in cases for endpoint in ENDPOINTS
                     for phase in ("post", "pre") for history in (ALL if phase == "post" else HISTORIES)
                     for q in case["queries"] if phase == "post" or q["query_id"].startswith("J")}
    lookup = {key(row): row for row in logical}
    assert len(lookup) == len(logical) == len(expected_keys) == 34710 and set(lookup) == expected_keys, "COMPLETE_SCHEDULE_REQUIRED"
    scored = []
    for row in logical:
        g = gold[row["case_id"], row["query_id"]]
        expected = g["post"] if row["phase"] == "post" else g["pre"][row["history"]]
        question = next(q["question"] for q in by_id[row["case_id"]]["queries"] if q["query_id"] == row["query_id"])
        assert row["question"] == question
        correct = row["outcome"]["status"] == "VALID" and row["outcome"]["entity"] == expected["entity"]
        program = row.get("program")
        faithful = None if program is None else program["start_id"] == expected["start"] and list(program["path"]) == expected["path"]
        if faithful and row["arm"].startswith("G_"):
            source_history = by_id[row["case_id"]]["histories"][row["history"]]
            source_state = {(s,p):o for s,p,o in source_history["initial"]}
            if row["phase"] == "post":
                for s,p,o in source_history["edits"]: source_state[s,p] = o
            expected_nodes = [expected["start"]]
            for relation in expected["path"]: expected_nodes.append(source_state[expected_nodes[-1],relation])
            bindings = {}
            for term, entity in zip(row["chain"]["nodes"],expected_nodes):
                if term.startswith("?"):
                    if term in bindings and bindings[term] != entity: faithful = False
                    bindings[term] = entity
                elif term != entity: faithful = False
        answer = row.get("answer", row.get("answer_text", ""))
        item = {**row, "answer_text": answer, "correct": correct, "expected_entity": expected["entity"], "branch": g["branch"],
                "program_faithful": faithful, "wrong_program_right_endpoint": bool(correct and faithful is False and (row["arm"].startswith("REPAIR") or row.get("route") == "graph")),
                "correct_refusal": row["outcome"]["status"] == "REFUSAL" and expected["entity"] is None,
                "incorrect_refusal": row["outcome"]["status"] == "REFUSAL" and expected["entity"] is not None}
        if row["arm"].startswith("G_"):
            fallback = row["fallback_generation"]
            fallback_out = decode_answer(fallback.get("text", ""), entities[row["case_id"]], fallback.get("cap_hit", False), fallback.get("invalid_control_tokens", False)) if fallback.get("measured") else None
            item["fallback_correct"] = None if fallback_out is None else fallback_out["status"] == "VALID" and fallback_out["entity"] == expected["entity"]
            item["graph_correct"] = row["graph"].get("entity") is not None and row["graph"]["entity"] == expected["entity"]
            assert (row["route"] == "graph") == (row["graph"].get("entity") is not None)
            if row["route"] == "graph": assert row["outcome"]["entity"] == row["graph"]["entity"]
            old = row["legacy_sensitivity"]
            legacy_graph = old.get("graph", {"entity": None})
            if row["chain_generation"].get("cap_hit") or row["chain_generation"].get("invalid_control_tokens"): legacy_graph = {"entity": None}
            legacy_entity = legacy_graph.get("entity")
            item["legacy_same_output_correct"] = legacy_entity == expected["entity"] if legacy_entity is not None else item["fallback_correct"]
        scored.append(item); lookup[key(row)] = item
    def get(cid, history, qid, endpoint, phase="post"): return lookup[cid, history, phase, qid, endpoint]
    complete = {}; robust = {}; summary = {}; guards = []
    source = json.loads(gzip.decompress((prior / "evaluator_data/cohort.json.gz").read_bytes()))
    source_by_id = {case["case_id"]: case for case in source["cases"]}; guard_function = source_guard_function(prior / "evaluator_data/mqmh.py")
    for endpoint in ENDPOINTS:
        complete[endpoint] = {}; robust[endpoint] = {}
        post = [r for r in scored if r["arm"] == endpoint and r["phase"] == "post"]
        pre = [r for r in scored if r["arm"] == endpoint and r["phase"] == "pre"]
        any_phrasings = 0; all_phrasings = 0; phrasings_per_history = {h: {"any": 0, "all": 0, "subjects": 128} for h in ALL}
        for case in cases:
            cid = case["case_id"]
            complete[endpoint][cid] = all(get(cid,h,q["query_id"],endpoint)["correct"] for h in HISTORIES for q in case["queries"])
            for q in case["queries"]: robust[endpoint][cid,q["query_id"]] = all(get(cid,h,q["query_id"],endpoint)["correct"] for h in HISTORIES)
            targets = [q["query_id"] for q in case["queries"] if q["query_id"].startswith("J")]; assert len(targets) == 3
            any_phrasings += any(robust[endpoint][cid,qid] for qid in targets); all_phrasings += all(robust[endpoint][cid,qid] for qid in targets)
            for h in ALL:
                values = [get(cid,h,qid,endpoint)["correct"] for qid in targets]
                phrasings_per_history[h]["any"] += any(values); phrasings_per_history[h]["all"] += all(values)
            values = {(cid,h,q["query_id"]): {"correct": get(cid,h,q["query_id"],endpoint)["correct"], "entity_id": get(cid,h,q["query_id"],endpoint)["outcome"]["entity"]} for h in ALL for q in case["queries"]}
            guards.extend(dict(g, endpoint=endpoint) for g in guard_function(source_by_id[cid], values)["targets"])
        branch = {name: {"questions": sum(g["branch"] == name for g in gold_rows),
                          "robust_correct": sum(robust[endpoint][g["case_id"],g["query_id"]] for g in gold_rows if g["branch"] == name)} for name in ("J", "F", "Fdecoy", "K")}
        current_targets = [r for r in post if r["branch"] == "J" and r["history"] in HISTORIES]
        assert len(current_targets) == len(pre) == 1536
        actual_guards = [g for g in guards if g["endpoint"] == endpoint]
        uniform_wrong = sum(len({get(g["case_id"],h,g["query_id"],endpoint)["outcome"]["entity"] for h in HISTORIES}) == 1 and
                            all(not get(g["case_id"],h,g["query_id"],endpoint)["correct"] for h in HISTORIES) for g in gold_rows)
        summary[endpoint] = {"subjects": 128, "complete_subjects": sum(complete[endpoint].values()), "targets": 384, "controls": 517,
                             "robust_targets": branch["J"]["robust_correct"], "robust_controls": sum(branch[b]["robust_correct"] for b in ("F","Fdecoy","K")),
                             "by_branch": branch, "post_reads": len(post), "post_correct": sum(r["correct"] for r in post),
                             "original_targets": {"correct": sum(r["correct"] for r in pre), "reads": 1536},
                             "current_same_targets": {"correct": sum(r["correct"] for r in current_targets), "reads": 1536},
                             "any_of_three_robust_phrasings": any_phrasings, "all_three_robust_phrasings": all_phrasings,
                             "target_phrasings_by_history": phrasings_per_history,
                             "status_counts": dict(collections.Counter(r["outcome"]["status"] for r in post + pre)),
                             "correct_refusals": sum(r["correct_refusal"] for r in post + pre), "incorrect_refusals": sum(r["incorrect_refusal"] for r in post + pre),
                             "wrong_program_right_endpoint": sum(r["wrong_program_right_endpoint"] for r in post),
                             "guard_denominator": 384, "guard_coverage": sum(g["common_guard"] for g in actual_guards),
                             "reference_coverage": sum(g["references"] for g in actual_guards), "guarded_witnesses": sum(g["qualified"] for g in actual_guards),
                             "uniformly_wrong_four_histories": uniform_wrong,
                             "reference_accuracy": {h: {"correct": sum(r["correct"] for r in post if r["history"] == h), "reads": 901} for h in ("C","C+")},
                             "origin": "verified historical replay, with fresh loader/uncached checks" if endpoint.startswith("REPAIR") else "new inference with explicitly linked identical-request reuse"}
        if endpoint.startswith("G_"):
            local = post + pre; graph_rows = [r for r in local if r["graph"].get("entity") is not None]
            summary[endpoint]["pipeline"] = {"logical_questions": len(local), "unique_chain_calls": len({r["chain_generation"]["call_id"] for r in local}),
                "parseable_chains": sum(r["parse"]["status"] == "VALID" for r in local),
                "uniquely_linked_starts": sum(bool(r["parse"]["decisions"] and r["parse"]["decisions"][0].get("normalized")) for r in local),
                "relation_mapping_methods": dict(collections.Counter(d.get("method", "error") for r in local for d in r["parse"]["decisions"] if d["role"] == "relation")),
                "executable_queries": sum(r["graph"]["status"] in ("VALID", "NO_RESULT") for r in local),
                "nonempty_graphs": len(graph_rows), "correct_graph_answers": sum(r["graph_correct"] for r in local),
                "wrong_graph_returns": sum(not r["graph_correct"] for r in graph_rows),
                "fallback_logical_calls": sum(r["fallback_generation"].get("measured", False) for r in local),
                "fallback_unique_calls": len({r["fallback_generation"]["call_id"] for r in local if r["fallback_generation"].get("measured")}),
                "fallback_correct": sum(r["fallback_correct"] is True for r in local),
                "graph_overrides_correct_fallback": sum(not r["graph_correct"] and r["fallback_correct"] is True for r in graph_rows),
                "chain_cap_hits": sum(r["chain_generation"].get("cap_hit", False) for r in local),
                "concrete_nonstart_constraints": sum(sum(not node.startswith("?") for node in r["chain"]["nodes"][1:]) for r in local if r["chain"] is not None),
                "fallback_cap_hits": sum(r["fallback_generation"].get("cap_hit", False) for r in local),
                "legacy_same_output_correct": sum(r["legacy_same_output_correct"] is True for r in local),
                "legacy_sensitivity_selected_primary": False}
    # Core/full identity is a mandatory invariant, not a correlation claim.
    for row in scored:
        if row["arm"] == "G_CORE":
            other = get(row["case_id"],row["history"],row["query_id"],"G_FULL",row["phase"])
            assert row["graph"] == other["graph"] and row["chain_generation"] == other["chain_generation"] and row["state_sha256"] == other["state_sha256"]
    edit_ledger = []
    for update in updates:
        intended = by_id[update["case_id"]]["histories"][update["history"]]["edits"][update["index"]]
        edit_ledger.append(dict(update, intended=intended, exact=update["parsed"].get("triple") == intended,
                                wrong_applied=update["parsed"]["status"] == "APPLIED" and update["parsed"]["triple"] != intended))
    core_graphs = {(g["case_id"],g["history"],g["phase"]): g for g in graph_records if g["endpoint"] == "G_CORE"}
    graph_fidelity = [dict(case_id=g["case_id"], history=g["history"], phase=g["phase"], equal_to_typed_input_state=g["triples"] == core_graphs[g["case_id"],g["history"],g["phase"]]["triples"]) for g in graph_records if g["endpoint"] == "G_TEXT"]
    summary["G_TEXT"]["edit_fidelity"] = {"intended": len(edit_ledger), "applied": sum(r["parsed"]["status"] == "APPLIED" for r in edit_ledger),
                                            "unapplied": sum(r["parsed"]["status"] != "APPLIED" for r in edit_ledger), "exact": sum(r["exact"] for r in edit_ledger),
                                            "wrong_applied": sum(r["wrong_applied"] for r in edit_ledger), "direction_swaps": sum(r["parsed"]["swap"] for r in edit_ledger),
                                            "post_graph_equal": sum(r["equal_to_typed_input_state"] for r in graph_fidelity if r["phase"] == "post"), "post_graphs": 768,
                                            "edit_cap_hits": sum(r["generation"].get("cap_hit", False) for r in edit_ledger)}
    paired = []
    for comparator in ("G_CORE", "G_FULL", "G_TEXT"):
        for seed in (0,1):
            repair = "REPAIR_" + str(seed)
            for g in gold_rows:
                cid, qid = g["case_id"], g["query_id"]; left, right = robust[repair][cid,qid], robust[comparator][cid,qid]
                paired.append({"repair": repair, "comparator": comparator, "case_id": cid, "query_id": qid, "branch": g["branch"],
                               "repair_correct_all_histories": left, "comparator_correct_all_histories": right,
                               "repair_fixes_comparator_error": left and not right, "comparator_fixes_repair_error": right and not left})
    # Preserve the original two witness cohorts and overlap, without converting
    # uniformly wrong answers into history-dependence claims.
    crosswalk = read(prior / "evaluator_data/HISTORICAL_CROSSWALK.json")
    old94 = {(r["case_id"],r["query_id"]) for r in crosswalk if r["historical_qwen_flag"]}
    new32 = {(r["case_id"],r["query_id"]) for r in crosswalk if r["new_HISTORY"]["qualified"]}
    assert (len(old94),len(new32),len(old94 & new32)) == (94,32,15)
    historic_old = json.loads(gzip.decompress((prior / "evaluator_data/qwen_responses.json.gz").read_bytes()))["primary"]
    old_lookup = {(r["case_id"],r["arm"],r["query_id"]): r for r in historic_old}
    native_lookup = {(r["case_id"],r["history"],r["query_id"]): r for r in rows(prior / "evaluator_data/HISTORICAL_SCORED.jsonl") if r["arm"] == "HISTORY" and r["phase"] == "post"}
    witness_ledger = []; witness_summary = {}
    for title, members in (("original_94",old94),("later_32",new32),("overlap_15",old94 & new32),("later_not_original_17",new32-old94)):
        witness_summary[title] = {"denominator": len(members), "endpoints": {}}
        for endpoint in ENDPOINTS:
            preserved = 0
            for cid,qid in sorted(members):
                checks = [(h,q["query_id"]) for h in ALL for q in by_id[cid]["queries"] if not q["query_id"].startswith("J")] + [("C",qid),("C+",qid)]
                losses = []
                for h,q in checks:
                    previous = (old_lookup[cid,h,q]["valid"] and old_lookup[cid,h,q]["entity_id"] == gold[cid,q]["post"]["entity"]) if title == "original_94" else native_lookup[cid,h,q]["correct"]
                    if previous and not get(cid,h,q,endpoint)["correct"]: losses.append([h,q])
                good = robust[endpoint][cid,qid] and not losses; preserved += good
                witness_ledger.append({"cohort": title, "endpoint": endpoint, "case_id": cid, "query_id": qid,
                                       "target_correct_all_histories": robust[endpoint][cid,qid], "lost_guards_or_controls": losses, "preserved_repair": good})
            witness_summary[title]["endpoints"][endpoint] = {"preserved_repair": preserved}
    repair_parity = read(raw / "evidence/REPAIR_PARITY.json")
    diagnostics = read(raw / "inference/out/main/DIAGNOSTICS.json")
    validity = {"status": "PASS" if repair_parity["status"] == "PASS" else "FAILED_REPAIR_PARITY", "schedule_complete": True,
                "logical_reads": 34710, "graph_gate_passed": freeze["graph_gate"]["status"] == "PASS", "core_full_graph_identity": True,
                "repair_parity": repair_parity, "gmello_diagnostic_differences": diagnostics["failures"],
                "g_text_intake_failures_are_measured_secondary_outcomes": True, "exposed_cohort": True, "training_matched_causal_ablation": False}
    result = {"status": "MEASURED" if validity["status"] == "PASS" else "MEASURED_WITH_FAILED_VALIDITY_GATE", "endpoints": summary,
              "primary_contrasts": bootstrap(complete, case_ids), "historical_witness_cohorts": witness_summary, "comparison_validity": validity["status"],
              "legacy_gates_unchanged": True, "nonsignificance_is_not_equivalence": True, "internal_repair_or_unique_isolation_theorem": False}
    if software_fixture:
        validity["status"] = result["comparison_validity"] = "SOFTWARE_FIXTURE_ONLY"
        result["status"] = "SOFTWARE_FIXTURE_VERIFIED_NOT_A_MODEL_RESULT"
    write(out / "SCIENTIFIC_RESULTS.json", result); write(out / "COMPARISON_VALIDITY.json", validity)
    write(out / "SUBJECT_VECTORS.json", {"case_ids": case_ids, "complete": complete})
    write_rows(out / "SCORED_LOGICAL.jsonl", scored); write_rows(out / "ERROR_LEDGER.jsonl", (r for r in scored if not r["correct"] or r["program_faithful"] is False))
    write_rows(out / "GUARD_LEDGER.jsonl", guards); write_rows(out / "PAIRED_FIXES_HARMS.jsonl", paired)
    write_rows(out / "WITNESS_RECOVERY.jsonl", witness_ledger); write_rows(out / "EDIT_FIDELITY.jsonl", edit_ledger)
    write_rows(out / "GRAPH_FIDELITY.jsonl", graph_fidelity)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--raw", type=Path, required=True); parser.add_argument("--prior", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); result = evaluate(args.raw, args.prior, args.out)
    print(json.dumps({"status": result["status"], "complete_subjects": {k:v["complete_subjects"] for k,v in result["endpoints"].items()}}))
