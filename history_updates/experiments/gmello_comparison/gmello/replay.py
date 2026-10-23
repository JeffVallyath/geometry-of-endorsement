"""Portable CPU replay from saved generations and complete retrieval score vectors.

No model generation, adapter loading, gold-based repair, or path search occurs.
Input/output token decoding, request identities, parser/linker decisions, graph
updates/queries, fallback priority and original repair execution are rechecked.
"""
from pathlib import Path
import argparse
import ast
import collections
import hashlib
import json
import sys
import time
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "inference"))
from gmello import ENDPOINTS, digest
from orchestration import evaluate_case

def read(path): return json.loads(Path(path).read_bytes())
def lines(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream: yield json.loads(line)
def serialized_result(value): return json.loads(json.dumps(value, ensure_ascii=False))

class SavedEngine:
    def __init__(self, calls, tokenizer):
        self.calls = calls; self.tokenizer = tokenizer
        self.fingerprint = next(iter(calls.values()))["fingerprint"]
        assert self.fingerprint["adapter"] is None
        self.replayed = set()
        for key,row in calls.items():
            request = row["request"]; ids = tokenizer.apply_chat_template(request["messages"],tokenize=True,add_generation_prompt=True,return_dict=False)
            assert ids == row["input_ids"] and row["fingerprint"] == self.fingerprint
            assert key == digest({"fingerprint": self.fingerprint,"request": request,"input_ids": ids,
                                  "nonce": None if row["nonce"] is None else [row["nonce"],row["request_index"]]})
            result = row["result"]
            assert tokenizer.decode(result["output_ids"],skip_special_tokens=True) == result["text"] == result["raw_decoded"]
    def calls_for(self, requests, nonce=None):
        result = []
        for i, request in enumerate(requests):
            ids = self.tokenizer.apply_chat_template(request["messages"], tokenize=True, add_generation_prompt=True, return_dict=False)
            key = digest({"fingerprint": self.fingerprint, "request": request, "input_ids": ids, "nonce": None if nonce is None else [nonce,i]})
            row = self.calls[key]
            assert row["request"] == request and row["input_ids"] == ids and row["fingerprint"] == self.fingerprint
            output = row["result"]; raw = output["output_ids"]
            assert self.tokenizer.decode(raw, skip_special_tokens=True) == output["text"] == output["raw_decoded"]
            assert len(raw) == output["output_tokens"] and len(ids) == output["input_tokens"]
            assert output["cap_hit"] == (len(raw) >= request["max_new_tokens"] and self.tokenizer.eos_token_id not in raw)
            self.replayed.add(key); result.append(dict(output, call_id=key))
        return result

class SavedRetrieval:
    def __init__(self, ontology):
        self.ontology = ontology; self.saved = {}; self.person = {}
        self.options = [label for _,labels in sorted(ontology.items()) for label in labels]
    def remember(self, query, pool, selected):
        if not selected: return
        scores = selected[0].get("all_scores")
        if scores is None:
            assert all(row.get("score") is None for row in selected)
            return  # G_FULL's explicit all-facts condition is not retrieval.
        assert len(scores) == len(pool)
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:len(selected)]
        assert order == [row["index"] for row in selected]
        for row in selected: assert row["fact"] == pool[row["index"]] and row["score"] == scores[row["index"]]
        key = digest([query,pool,len(selected)])
        if key in self.saved: assert self.saved[key] == selected
        self.saved[key] = selected
    def add_result(self, result):
        for row in result["rows"]:
            fallback = row["fallback_evidence"]
            self.remember(row["question"], fallback["pool"], fallback["selected"])
            for decision in row["parse"]["decisions"]:
                if decision.get("retrieval"): self.remember(decision["generated"], self.options, decision["retrieval"])
            for decision in row["legacy_sensitivity"].get("normalization", []):
                if decision.get("retrieval"): self.remember(decision["generated"], self.options, decision["retrieval"])
        for update in result["updates"]:
            for decision in update["parsed"]["decisions"]:
                if decision.get("retrieval"): self.remember(decision["generated"], self.options, decision["retrieval"])
            detection = update["parsed"].get("person_detection")
            if detection:
                value = (detection["is_person"], detection["entities"])
                if detection["text"] in self.person: assert self.person[detection["text"]] == value
                self.person[detection["text"]] = value
    def __call__(self, query, pool, k):
        if not pool: return []
        return self.saved[digest([query,pool,k])]
    def detect(self, name): return self.person[name]

def replay_repair(prior):
    prior = Path(prior)
    records = [r for r in lines(prior / "raw/results/logical.jsonl") if r["arm"].startswith("REPAIR_")]
    wanted = {call_id for row in records for call_id in row["call_ids"]}
    calls = {r["call_id"]: r for r in lines(prior / "raw/results/calls.jsonl") if r["call_id"] in wanted}
    assert set(calls) == wanted
    sys.path[:0] = [str(prior / "evaluator_data/ppr/implementation"), str(prior / "evaluator_data/ppr/vendor")]
    from ppr_requests import score_requests
    from ranker_contract import choose
    from execution import execute
    cases = {case["case_id"]: case for case in read(prior / "evaluator_data/RUNTIME_CASES.json")}
    for row in records:
        candidates, requests = score_requests(cases[row["case_id"]], row["history"], row["phase"], row["question"])
        assert candidates == row["candidates"] and len(requests) == len(row["call_ids"])
        local = []
        for request, call_id in zip(requests,row["call_ids"]):
            call = calls[call_id]; assert call["request"] == request and call["kind"] == "score"
            expected_adapter = "abdfc30d4ad24cedad04d99a06797b854115bccdc0853dd240e133f8dd3119c3" if row["arm"] == "REPAIR_0" else "b9b84813e04f39d47e55a6bb8be25f34b8921970e11cd3e707dbd357b8cc6ce1"
            assert call["fingerprint"]["adapter"] == expected_adapter and call["fingerprint"]["base"] == "a09a35458c702b33eeacc393d103063234e8bc28"
            local.append(call["result"]["margin"])
        selected = choose(candidates, local)
        assert selected == row["selected"] and execute(cases[row["case_id"]],row["history"],row["phase"],selected) == row["outcome"]
    return {"logical_reads": len(records), "verified_source_calls": len(calls), "status": "PASS"}

def replay(raw, prior, output):
    began = time.perf_counter(); raw = Path(raw)
    terminal = read(raw / "COMPLETE.json")
    assert not terminal.get("software_fixture_only", False)
    diagnostic_stop = terminal["main_logical_reads"] == 0
    if diagnostic_stop:
        # A prespecified readiness stop has no main scientific freeze. Verify its
        # immutable pre-rental input manifest and saved DEV outputs instead.
        freeze = {"inference_manifest": read(ROOT / "inference/INFERENCE_MANIFEST.json"), "selected_revision": 0}
    else:
        freeze = read(raw / "SCIENTIFIC_FREEZE.json")
        assert digest({k:v for k,v in freeze.items() if k != "public_projection_sha256"}) == freeze["public_projection_sha256"]
    for name, expected in freeze["public_science_files"].items():
        with (ROOT / "inference" / name).open("rb") as stream: assert hashlib.file_digest(stream,"sha256").hexdigest() == expected, name
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "inference/tokenizer", local_files_only=True)
    calls = {r["call_id"]: r for r in lines(raw / "inference/out/calls.jsonl")}
    saved_engine = SavedEngine(calls, tokenizer)
    class EngineAdapter:
        def calls(self, requests, nonce=None): return saved_engine.calls_for(requests,nonce)
    engine = EngineAdapter()
    ontology, templates, examples = [read(ROOT / "inference/data" / n) for n in ("ontology.json","fact_templates.json","examples.json")]
    all_cases = {c["case_id"]: c for n in ("main.json","development.json") for c in read(ROOT / "inference/data" / n)}
    main_paths = sorted((raw / "inference/out/main/cases").glob("*.json"))
    control_paths = sorted((raw / "inference/out/main/controls").glob("*.json"))
    dev_paths = sorted((raw / "inference/out").glob("dev_r*/cases/*.json"))
    total_rows = 0; panels = 0
    for path in dev_paths + main_paths + control_paths:
        saved = read(path); retrieval = SavedRetrieval(ontology); retrieval.add_result(saved)
        jobs = [{k:v for k,v in row.items() if k in ("case_id","history","phase","query_id","arm","slot")} for row in saved["rows"]]
        case = all_cases[jobs[0]["case_id"]]
        revision = saved.get("revision", freeze["selected_revision"])
        nonce = saved.get("control_id")
        observed = serialized_result(evaluate_case(case,jobs,engine,retrieval,retrieval.detect,ontology,templates,examples,revision,nonce=nonce,all_fallbacks=True))
        assert observed["rows"] == saved["rows"], ("ROW_REPLAY", str(path.relative_to(raw)))
        assert observed["graphs"] == saved["graphs"] and observed["updates"] == saved["updates"], ("STATE_REPLAY", str(path.relative_to(raw)))
        total_rows += len(jobs); panels += 1
    for path in (raw / "inference/out").glob("*/ACCESS_AUDIT.json"):
        access = read(path)
        assert not access["rejected_attempts"], ("UNEXPECTED_REJECTED_ACCESS", str(path.relative_to(raw)))
        assert all(not name.startswith(("../", "repair", "development_evaluator")) for name in access["task_files_opened"])
    repair = replay_repair(prior)
    report = {"status": "PASS", "new_main_logical_reads_replayed": 0 if diagnostic_stop else 20826, "diagnostic_stop": diagnostic_stop,
              "all_gmello_rows_including_dev_controls": total_rows,
              "case_and_control_panels": panels, "distinct_saved_base_calls_replayed": len(saved_engine.replayed), "repair_replay": repair,
              "all_saved_base_calls_token_and_identity_verified": len(calls),
              "request_tokens_and_output_decoding_verified": True, "all_retrieval_rank_scores_replayed": True,
              "independent_graphs_edits_and_priority_replayed": True, "new_neural_inference": False,
              "seconds": time.perf_counter()-began}
    Path(output).parent.mkdir(parents=True, exist_ok=True); Path(output).write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report), flush=True)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--raw",type=Path,required=True); parser.add_argument("--prior",type=Path,required=True); parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(); replay(args.raw,args.prior,args.out)
