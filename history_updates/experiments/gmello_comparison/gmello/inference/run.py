"""Base-only DEV and frozen complete main schedules. Never loads evaluation labels."""
from pathlib import Path
import argparse
import json
import time
if __name__ == "__main__":
    from isolation_guard import install
    ACCESS_REPORT = install(Path(__file__).resolve().parent)
from base_runtime import ROOT, BaseEngine, Journal, PersonDetector, Retriever, check_time, publish, read, sha_file, verify_capsule
from gmello import ENDPOINTS, digest, chain_request, edit_request, fallback_request, graph_for_history, current_facts
from orchestration import evaluate_case
from authorization import validate_authorization

def jobs_for(case, diagnostic=None):
    queries = case["queries"]
    result = []
    for endpoint in ENDPOINTS:
        for history in case["histories"]:
            for question in queries:
                result.append({"case_id": case["case_id"], "history": history, "phase": "post", "query_id": question["query_id"], "arm": endpoint})
        if len(case["histories"]) == 6 and diagnostic is None:
            for history in ("H0", "H1", "S0", "S1"):
                for question in queries:
                    if question["query_id"].startswith("J"):
                        result.append({"case_id": case["case_id"], "history": history, "phase": "pre", "query_id": question["query_id"], "arm": endpoint})
    return result

def selected_signature(row):
    return {key: row.get(key) for key in ("question", "state_sha256", "chain", "answer", "route", "outcome")}

def benchmark(engine, cases, ontology, templates, examples):
    candidates = {"chain": [], "edit": [], "fallback": []}
    for case in cases:
        for query in case["queries"]:
            candidates["chain"].append(chain_request(query["question"], ontology, examples))
            graph = graph_for_history(case["histories"]["H0"], case["entities"])
            facts = current_facts(graph, ontology, templates)
            candidates["fallback"].append(fallback_request(query["question"], facts, examples))
        candidates["edit"].extend(edit_request(s, ontology, examples) for s in case["histories"]["H0"]["edit_sentences"])
    selected = []; kinds = []
    for kind, values in candidates.items():
        ordered = sorted(values, key=lambda value: len(engine.tokens(value)))
        for length in ("shortest", "longest"):
            selected.append(ordered[0 if length == "shortest" else -1]); kinds.append([kind, length])
    outputs = engine.calls(selected, nonce="DEV_THROUGHPUT_BENCHMARK")
    publish(ROOT / "out/DEV_BENCHMARK.json", {"measurements": [{"kind": kind, "length": length, "generation": generation} for (kind, length), generation in zip(kinds, outputs)],
                                              "selection": "shortest and longest actual DEV prompts, before main inference", "evaluation_labels_used": False})

def main(stage, revision):
    manifest = verify_capsule(); journal = Journal()
    ontology, templates, examples = [read(ROOT / "data" / name) for name in ("ontology.json", "fact_templates.json", "examples.json")]
    if stage == "main":
        freeze = read(ROOT / "SCIENTIFIC_FREEZE.json")
        assert digest({k:v for k,v in freeze.items() if k != "freeze_sha256"}) == freeze["freeze_sha256"]
        assert freeze["inference_manifest"] == manifest
        validate_authorization(freeze, manifest)
        revision = freeze["selected_revision"]
        assert freeze["created_epoch"] <= time.time()
        cases = read(ROOT / "data/main.json"); folder = ROOT / "out/main"
    else:
        assert revision in (0, 1, 2)
        assert not (ROOT / "SCIENTIFIC_FREEZE.json").exists(), "NO_DEV_AFTER_FREEZE"
        cases = read(ROOT / "data/development.json"); folder = ROOT / "out" / ("dev_r" + str(revision))
    folder.mkdir(parents=True, exist_ok=True)
    complete_path = folder / "COMPLETE.json"
    if complete_path.exists():
        assert read(complete_path)["technical_status"] == "COMPLETE"; return
    engine = BaseEngine(journal); retriever = Retriever(); detector = PersonDetector(); began = time.perf_counter()
    rows = []; graphs = []; edits = []
    try:
        if stage == "dev" and revision == 0 and not (ROOT / "out/DEV_BENCHMARK.json").exists(): benchmark(engine, cases, ontology, templates, examples)
        for case in cases:
            check_time(); path = folder / "cases" / (str(case["case_id"]).replace("/", "_") + ".json")
            if path.exists():
                result = read(path)
                assert result["case_input_sha256"] == digest(case) and result["revision"] == revision
            else:
                result = evaluate_case(case, jobs_for(case), engine, retriever, detector, ontology, templates, examples, revision,
                                       all_fallbacks=read(ROOT / "CONFIG.json")["all_fallbacks"])
                result.update(case_input_sha256=digest(case), revision=revision, completed_epoch=time.time(),
                              freeze_sha256=freeze["freeze_sha256"] if stage == "main" else None)
                publish(path, result)
            rows.extend(result["rows"]); graphs.extend(result["graphs"]); edits.extend(result["updates"])
            journal.status(stage.upper(), revision=revision, case_id=case["case_id"], logical_reads=len(rows))
        if stage == "main":
            assert len(rows) == 20826
            lookup = {(r["case_id"], r["history"], r["phase"], r["query_id"], r["arm"]): r for r in rows}
            selection = read(ROOT / "data/CONTROL_SELECTION.json"); by_id = {case["case_id"]: case for case in cases}
            diagnostics = []
            for cid in selection["fixed_uncached_case_ids"]:
                control_id = "repeat_" + str(cid); path = folder / "controls" / (control_id + ".json")
                if path.exists(): result = read(path)
                else:
                    result = evaluate_case(by_id[cid], jobs_for(by_id[cid], diagnostic=True), engine, retriever, detector, ontology, templates, examples,
                                           revision, nonce=control_id, all_fallbacks=True)
                    checks = []
                    for row in result["rows"]:
                        original = lookup[row["case_id"], row["history"], row["phase"], row["query_id"], row["arm"]]
                        checks.append({"arm": row["arm"], "history": row["history"], "query_id": row["query_id"],
                                       "fields": {field: selected_signature(row)[field] == selected_signature(original)[field] for field in selected_signature(row)}})
                    result.update(control_id=control_id, kind="fixed_uncached_case", comparisons=checks,
                                  all_equal=all(all(c["fields"].values()) for c in checks)); publish(path, result)
                diagnostics.append(result)
            for cid, qid in selection["uncached_pairs"]:
                control_id = "snapshot_" + str(cid) + "_" + qid; path = folder / "controls" / (control_id + ".json")
                if path.exists(): result = read(path)
                else:
                    jobs = [{"case_id": cid, "history": history, "phase": phase, "query_id": qid, "arm": endpoint, "slot": slot}
                            for endpoint in ENDPOINTS for slot, history, phase in (("current_before", "H0", "post"), ("original", "H0", "pre"), ("current_after", "H0", "post"), ("equivalent_current", "S0", "post"))]
                    result = evaluate_case(by_id[cid], jobs, engine, retriever, detector, ontology, templates, examples, revision, nonce=control_id, all_fallbacks=True)
                    checks = []
                    for endpoint in ENDPOINTS:
                        current = [r for r in result["rows"] if r["arm"] == endpoint and r["slot"] != "original"]
                        checks.append({"arm": endpoint, "fields": {field: all(selected_signature(r)[field] == selected_signature(current[0])[field] for r in current[1:]) for field in selected_signature(current[0])}})
                    result.update(control_id=control_id, kind="snapshot_return", comparisons=checks,
                                  all_equal=all(all(c["fields"].values()) for c in checks)); publish(path, result)
                diagnostics.append(result)
            publish(folder / "DIAGNOSTICS.json", {"panels": len(diagnostics), "endpoint_panels": len(diagnostics) * 3,
                                                   "failures": [d["control_id"] for d in diagnostics if not d["all_equal"]]})
        publish(complete_path, {"technical_status": "COMPLETE", "stage": stage, "revision": revision, "logical_reads": len(rows), "cases": len(cases),
                                "graph_records": len(graphs), "edit_records": len(edits), "seconds": time.perf_counter() - began,
                                "retrieval_seconds": retriever.seconds, "new_main_outcomes_scored": False, "completed_epoch": time.time()})
    finally:
        engine.close()
        if __name__ == "__main__": publish(folder / "ACCESS_AUDIT.json", ACCESS_REPORT())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=["dev", "main"], required=True); parser.add_argument("--revision", type=int, default=0)
    args = parser.parse_args(); main(args.stage, args.revision)
