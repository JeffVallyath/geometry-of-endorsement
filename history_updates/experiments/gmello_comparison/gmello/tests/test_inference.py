from pathlib import Path
import ast
import inspect
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "inference"), str(ROOT / "protocol/tools")]
import gmello
from orchestration import evaluate_case

class FakeEngine:
    def __init__(self, edit="Aster->relation one->Cedar"):
        self.requests = []; self.edit = edit
    def calls(self, requests, nonce=None):
        outputs = []
        for request in requests:
            self.requests.append(request)
            user = request["messages"][-1]["content"]
            text = self.edit if user.startswith("Sentence:") else "Aster->relation one->?x" if user.endswith("Relation Chain:") else "Beryl"
            outputs.append({"text": text, "cap_hit": False, "invalid_control_tokens": False,
                            "call_id": gmello.digest([request, nonce, len(self.requests)])})
        return outputs

def retrieve(query, facts, k): return [{"index": i, "fact": fact, "score": float(-i)} for i, fact in enumerate(facts[:k])]
def ner(name): return False, []

class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.entities = {"Q1": ["Aster"], "Q2": ["Beryl"], "Q3": ["Cedar"], "Q4": ["Delta"]}
        self.ontology = {"P1": ["relation one"], "P2": ["relation two"], "P35": ["head of state"], "P6": ["head of government"], "P169": ["chief executive officer"]}
        self.examples = [{"question": "TRAIN " + str(i), "chain": "Training->relation one->?x", "facts": ["TRAIN FACT"], "answer": "TRAIN ANSWER",
                          "edit_sentence": "TRAIN EDIT", "edit_chain": "Training->relation one->Other"} for i in range(4)]
        self.history = {"initial": [["Q1", "P1", "Q2"]], "edits": [["Q1", "P1", "Q3"]], "edit_sentences": ["Aster changed to Cedar."]}
        self.case = {"case_id": "synthetic", "entities": self.entities, "histories": {"H0": self.history, "S0": self.history},
                     "queries": [{"query_id": "q", "question": "Which relation one of Aster?"}]}
    def job(self, endpoint, phase="post", history="H0"):
        return {"case_id": "synthetic", "history": history, "phase": phase, "query_id": "q", "arm": endpoint}
    def run_case(self, jobs, engine=None):
        return evaluate_case(self.case, jobs, engine or FakeEngine(), retrieve, ner, self.ontology, {}, self.examples)
    def test_extractor_input_boundary(self):
        self.assertEqual(list(inspect.signature(gmello.chain_request).parameters), ["question", "ontology", "examples", "revision"])
        request = gmello.chain_request("Question only", self.ontology, self.examples)
        content = gmello.canonical(request)
        for forbidden in ("current_records", "candidate_program", "expected", "Cedar", "H0"):
            self.assertNotIn(forbidden, content)
    def test_caps_and_global_revisions(self):
        for revision in range(3): self.assertEqual(gmello.chain_request("q", self.ontology, self.examples, revision)["max_new_tokens"], 256)
        with self.assertRaises(AssertionError): gmello.chain_request("q", self.ontology, self.examples, 3)
        self.assertEqual(gmello.edit_request("s", self.ontology, self.examples)["max_new_tokens"], 128)
        self.assertEqual(gmello.fallback_request("q", [], self.examples)["max_new_tokens"], 128)
    def test_core_full_identical_graph_and_chain(self):
        rows = self.run_case([self.job("G_CORE"), self.job("G_FULL")])["rows"]
        self.assertEqual(rows[0]["graph"], rows[1]["graph"])
        self.assertEqual(rows[0]["chain_generation"], rows[1]["chain_generation"])
        self.assertEqual(rows[0]["outcome"]["entity"], "Q3")
    def test_wrong_graph_overrides_different_fallback(self):
        row = self.run_case([self.job("G_CORE")])["rows"][0]
        self.assertEqual(row["fallback_generation"]["text"], "Beryl")
        self.assertEqual(row["answer"], "Cedar")
        self.assertEqual(row["route"], "graph")
    def test_wrong_text_update_not_oracle_corrected(self):
        result = self.run_case([self.job("G_TEXT")], FakeEngine("Aster->relation one->Delta"))
        self.assertEqual(result["rows"][0]["outcome"]["entity"], "Q4")
        self.assertEqual(result["updates"][0]["parsed"]["triple"], ["Q1", "P1", "Q4"])
    def test_unapplied_text_update_retains_initial(self):
        result = self.run_case([self.job("G_TEXT")], FakeEngine("Invented->relation one->Delta"))
        self.assertEqual(result["rows"][0]["outcome"]["entity"], "Q2")
        self.assertEqual(result["updates"][0]["parsed"]["status"], "UNAPPLIED")
    def test_text_pre_has_no_future_edited_facts(self):
        result = self.run_case([self.job("G_TEXT", "pre")])
        self.assertEqual(result["rows"][0]["fallback_evidence"]["pool"], [])
        self.assertEqual(result["rows"][0]["outcome"]["entity"], "Q2")
        self.assertEqual(result["updates"], [])
    def test_all_current_pool_includes_unchanged_branch(self):
        self.history["initial"].append(["Q2", "P2", "Q4"])
        result = self.run_case([self.job("G_CORE")])
        self.assertEqual(len(result["rows"][0]["fallback_evidence"]["pool"]), 2)
    def test_direction_rule_not_expanded_to_ceo(self):
        for property_id, swapped in (("P35", True), ("P6", True), ("P169", False)):
            output = gmello.text_update({"text": "Aster->" + property_id + "->Beryl"}, self.entities, self.ontology, retrieve, ner)
            self.assertEqual(output["swap"], swapped)
    def test_person_rule_logs_without_swap(self):
        result = gmello.text_update({"text": "Aster->P35->Beryl"}, self.entities, self.ontology, retrieve, lambda n: (True, [{"label": "PERSON", "text": n}]))
        self.assertFalse(result["swap"])
        self.assertTrue(result["person_detection"]["is_person"])
    def test_same_response_codec(self):
        for value in ("Beryl", "Q2", "Answer: Beryl.", '"Beryl."', "\u201cBeryl\u201d"):
            self.assertEqual(gmello.decode_answer(value, self.entities), {"entity": "Q2", "status": "VALID"})
        self.assertEqual(gmello.decode_answer("beryl", self.entities)["status"], "UNRECOGNIZED_OUTPUT")
        self.assertEqual(gmello.decode_answer("Beryl\nexplanation", self.entities)["status"], "MULTILINE_OUTPUT")
        self.assertEqual(gmello.decode_answer("UNSUPPORTED", self.entities)["status"], "REFUSAL")
        self.assertEqual(gmello.decode_answer("Beryl", self.entities, cap_hit=True)["status"], "INVALID_OUTPUT")
    def test_repeated_variable_and_constants_preserved(self):
        chain, trace = gmello.parse_generated({"text": "Aster->P1->?x->P2->?x"}, self.entities, self.ontology, retrieve)
        self.assertEqual(chain.nodes, ("Q1", "?x", "?x"))
        graph = gmello.graph_for_history(self.history, self.entities)
        self.assertIsNone(gmello.query_generated(graph, chain)["entity"])
    def test_legacy_sensitivity_rejects_id_start(self):
        graph = gmello.graph_for_history(self.history, self.entities)
        old = gmello.legacy_sensitivity("Q1->P1->?x", graph, self.entities, self.ontology, retrieve)
        self.assertEqual(old["error"], "UNLINKED_OR_AMBIGUOUS_NAME")
        new, _ = gmello.parse_generated({"text": "Q1->P1->?x"}, self.entities, self.ontology, retrieve)
        self.assertEqual(gmello.query_generated(graph, new)["entity"], "Q3")
    def test_independent_histories(self):
        a = gmello.graph_for_history(self.history, self.entities)
        b = gmello.graph_for_history(self.history, self.entities)
        b.apply([["Q1", "P1", "Q4"]])
        self.assertNotEqual(a.triples(), b.triples())
    def test_no_forbidden_inference_imports(self):
        forbidden = {"baseline_core", "worker", "model_runtime", "peft", "ppr_requests", "pst1_core", "ranker_contract", "state_grounding", "execution", "prepare_inputs", "evaluate"}
        for path in (ROOT / "inference").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else [node.module.split(".")[0]] if isinstance(node, ast.ImportFrom) and node.module else []
                self.assertFalse(set(names) & forbidden, path.name)

if __name__ == "__main__": unittest.main()
