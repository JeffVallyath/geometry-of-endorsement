"""Inference-only pinned base backend, adapted from the prior verified engine.

There is deliberately no adapter loader, scoring interface, training operation,
candidate import or evaluator input in this module.
"""
from pathlib import Path
import gc
import hashlib
import json
import os
import random
import time
import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from gmello import canonical, digest

ROOT = Path(__file__).resolve().parent

def read(path): return json.loads(Path(path).read_bytes())
def sha_file(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()
def publish(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".pending")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def check_time():
    contract = read(ROOT / "contract.json")
    if time.time() >= contract["clock"]["science_cutoff_epoch_utc"]: raise TimeoutError("SCIENCE_CUTOFF")

def verify_capsule():
    manifest = read(ROOT / "INFERENCE_MANIFEST.json")
    for name, expected in manifest.items():
        assert sha_file(ROOT / name) == expected, ("FROZEN_INPUT_CHANGED", name)
    return manifest

class Journal:
    def __init__(self):
        self.folder = ROOT / "out"; self.folder.mkdir(exist_ok=True)
        self.calls = {}
        path = self.folder / "calls.jsonl"
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line); assert row["call_id"] not in self.calls
                    self.calls[row["call_id"]] = row

    def add_call(self, row):
        assert row["call_id"] not in self.calls
        with (self.folder / "calls.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical(row) + "\n"); stream.flush(); os.fsync(stream.fileno())
        self.calls[row["call_id"]] = row

    def status(self, stage, **fields):
        value = {"stage": stage, "epoch": time.time(), "unique_calls": len(self.calls), **fields}
        publish(ROOT / "WORKER_STATUS.json", value)
        print(canonical(value), flush=True)

class BaseEngine:
    def __init__(self, journal):
        check_time(); self.cfg = read(ROOT / "CONFIG.json"); self.journal = journal
        manifest = verify_capsule()
        random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        assert transformers.__version__ == "4.56.2" and torch.__version__ == "2.8.0+cu128"
        assert torch.cuda.device_count() == 1 and "H100" in torch.cuda.get_device_name(0)
        assert torch.cuda.get_device_properties(0).total_memory >= 79_000_000_000
        model_path = Path("/workspace/models/qwen")
        self.tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                         attn_implementation="flash_attention_2", device_map={"": "cuda:0"}, local_files_only=True)
        assert self.model.config._attn_implementation == "flash_attention_2"
        assert not hasattr(self.model, "peft_config")
        assert not any("lora_" in name for name, _ in self.model.named_parameters())
        self.model.eval(); self.model.config.use_cache = True
        for parameter in self.model.parameters(): parameter.requires_grad_(False)
        pins = read(ROOT / "model_pins.json")
        self.fingerprint = {"base": pins["revision"], "adapter": None, "model_file_hashes": {k: v["expected"] for k, v in pins["files"].items()},
                            "numerics": "BF16-FA2-transformers-4.56.2", "torch": torch.__version__, "code_and_data": manifest,
                            "configuration": self.cfg, "retriever": ["facebook/contriever-msmarco", "abe8c1493371369031bcb1e02acb754cf4e162fa"]}

    def tokens(self, request):
        result = self.tok.apply_chat_template(request["messages"], tokenize=True, add_generation_prompt=True, return_dict=False)
        assert len(result) + request["max_new_tokens"] <= self.cfg["max_sequence_length"], "NO_SEMANTIC_TRUNCATION"
        return result

    def calls(self, requests, nonce=None):
        if not requests: return []
        seqs = [self.tokens(request) for request in requests]
        keys = [digest({"fingerprint": self.fingerprint, "request": req, "input_ids": seq, "nonce": None if nonce is None else [nonce, i]})
                for i, (req, seq) in enumerate(zip(requests, seqs))]
        request_indices = {key: i for i,key in enumerate(keys)}
        missing = {key: (req, seq) for key, req, seq in zip(keys, requests, seqs) if key not in self.journal.calls}
        groups = {}
        for key, (req, _) in missing.items():
            assert req["do_sample"] is False and not req["stop_strings"]
            groups.setdefault(req["max_new_tokens"], []).append(key)
        for cap, group in groups.items():
            for offset in range(0, len(group), self.cfg["generation_batch"]):
                check_time(); chunk = group[offset:offset + self.cfg["generation_batch"]]
                items = [missing[key] for key in chunk]; width = max(len(seq) for _, seq in items)
                ids = torch.tensor([[self.tok.pad_token_id] * (width - len(seq)) + seq for _, seq in items], device="cuda")
                mask = torch.tensor([[0] * (width - len(seq)) + [1] * len(seq) for _, seq in items], device="cuda")
                started_epoch = time.time(); began = time.perf_counter()
                with torch.inference_mode():
                    outputs = self.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=cap, do_sample=False,
                              pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id, use_cache=True)[:, width:].cpu().tolist()
                torch.cuda.synchronize(); elapsed = time.perf_counter() - began
                for key, (req, seq), raw in zip(chunk, items, outputs):
                    if self.tok.eos_token_id in raw: raw = raw[:raw.index(self.tok.eos_token_id) + 1]
                    else:
                        while raw and raw[-1] == self.tok.pad_token_id: raw.pop()
                    text = self.tok.decode(raw, skip_special_tokens=True)
                    result = {"text": text, "raw_decoded": text, "output_ids": raw, "output_tokens": len(raw), "input_tokens": len(seq),
                              "invalid_control_tokens": any(token in self.tok.all_special_ids for token in (raw[:-1] if raw and raw[-1] == self.tok.eos_token_id else raw)),
                              "cap_hit": len(raw) >= cap and self.tok.eos_token_id not in raw,
                              "stopped_on": "eos" if self.tok.eos_token_id in raw else None,
                              "batch_size": len(chunk), "batch_seconds": elapsed, "amortized_seconds": elapsed / len(chunk)}
                    self.journal.add_call({"call_id": key, "kind": "generate", "endpoint": "BASE_ONLY", "request": req, "input_ids": seq,
                                           "fingerprint": self.fingerprint, "result": result, "nonce": nonce,
                                           "request_index": request_indices[key],
                                           "started_epoch": started_epoch, "completed_epoch": time.time()})
        return [dict(self.journal.calls[key]["result"], call_id=key) for key in keys]

    def close(self):
        del self.model; gc.collect(); torch.cuda.empty_cache()

class Retriever:
    def __init__(self):
        path = "/workspace/models/contriever"
        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).eval().to("cpu")
        self.cache = {}; self.seconds = 0.; torch.set_num_threads(4)
    def embed(self, texts):
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        for offset in range(0, len(missing), 32):
            chunk = missing[offset:offset + 32]
            inputs = self.tok(chunk, padding=True, truncation=False, return_tensors="pt")
            assert inputs["input_ids"].shape[1] <= 512, "NO_RETRIEVER_TRUNCATION"
            started = time.perf_counter()
            with torch.inference_mode():
                output = self.model(**inputs)[0]; mask = inputs["attention_mask"]
                output = output.masked_fill(~mask[..., None].bool(), 0.)
                embeddings = output.sum(dim=1) / mask.sum(dim=1)[..., None]
                assert torch.isfinite(embeddings).all(), "RETRIEVER_NUMERICAL_FAILURE"
            self.seconds += time.perf_counter() - started
            self.cache.update(zip(chunk, embeddings))
        return torch.stack([self.cache[text] for text in texts])
    def __call__(self, query, facts, k):
        if not facts: return []
        query_vector = self.embed([query])[0]; memory = self.embed(facts)
        scores = (query_vector @ memory.T).tolist()
        indices = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        result = [{"index": i, "score": scores[i], "fact": facts[i]} for i in indices]
        if result: result[0]["all_scores"] = scores
        return result

class PersonDetector:
    def __init__(self):
        import spacy
        import en_core_web_sm
        assert spacy.__version__ == "3.8.7" and en_core_web_sm.__version__ == "3.8.0"
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer", "attribute_ruler"])
    def __call__(self, text):
        entities = [{"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char} for ent in self.nlp(text).ents]
        return any(ent["label"] == "PERSON" for ent in entities), entities
