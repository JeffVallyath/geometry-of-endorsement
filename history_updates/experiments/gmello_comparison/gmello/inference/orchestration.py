"""Independent source-history execution and all three GMeLLo endpoints.

No expected values, candidate lists, repair modules or annotations are accepted.
The caller supplies a pure base-model engine, pinned retrieval and pinned NER.
"""
from gmello import (ENDPOINTS, chain_request, edit_request, fallback_request,
                   parse_generated, query_generated, text_update, graph_for_history,
                   fallback_evidence, combined_answer, legacy_sensitivity, digest)

def evaluate_case(case, jobs, engine, retrieve, person_detector, ontology, templates, examples,
                  revision=0, nonce=None, all_fallbacks=True):
    assert all(job["arm"] in ENDPOINTS and job["case_id"] == case["case_id"] for job in jobs)
    questions = {q["query_id"]: q["question"] for q in case["queries"]}
    entities = case["entities"]
    # The primary core/full pair shares the identical freshly generated chain.
    # Uncached controls use distinct calls across logical (history, phase, query)
    # reads while still preserving this mandated within-pair sharing.
    key = lambda job: (job["history"], job["phase"], job["query_id"], job.get("slot")) if nonce else job["query_id"]
    chain_keys = list(dict.fromkeys(key(job) for job in jobs))
    first_job = {key(job): job for job in reversed(jobs)}
    chains_out = engine.calls([chain_request(questions[first_job[k]["query_id"]], ontology, examples, revision) for k in chain_keys],
                             nonce=None if nonce is None else [nonce, "chains"])
    generated_by_key = dict(zip(chain_keys, chains_out))
    parsed_by_key = {k: parse_generated(generated_by_key[k], entities, ontology, retrieve) for k in chain_keys}
    histories = list(dict.fromkeys(job["history"] for job in jobs))
    need_text = any(job["arm"] == "G_TEXT" for job in jobs)
    states = {}; graph_records = []; update_records = []
    for history_name in histories:
        h = case["histories"][history_name]
        updates = []
        if need_text and any(j["history"] == history_name and j["phase"] == "post" and j["arm"] == "G_TEXT" for j in jobs):
            generated = engine.calls([edit_request(sentence, ontology, examples) for sentence in h["edit_sentences"]],
                                     nonce=None if nonce is None else [nonce, "edits", history_name])
            for index, (sentence, output) in enumerate(zip(h["edit_sentences"], generated)):
                update = text_update(output, entities, ontology, retrieve, person_detector)
                updates.append(update)
                update_records.append({"case_id": case["case_id"], "history": history_name, "index": index, "sentence": sentence,
                                       "generation": output, "parsed": update})
        for phase in dict.fromkeys(j["phase"] for j in jobs if j["history"] == history_name):
            core = graph_for_history(h, entities, phase=phase)
            text = graph_for_history(h, entities, text_updates=updates, phase=phase) if need_text else None
            states[history_name, phase] = {"G_CORE": core, "G_FULL": core, "G_TEXT": text}
            # No history is constructed by copying C or another history's state.
            for endpoint in ("G_CORE", "G_TEXT") if need_text else ("G_CORE",):
                graph = states[history_name, phase][endpoint]
                graph_records.append({"case_id": case["case_id"], "history": history_name, "phase": phase, "endpoint": endpoint,
                                      "initial_sha256": digest(h["initial"]), "state_sha256": digest(graph.triples()), "triples": graph.triples()})
    prepared = []; fallback_requests = []; fallback_positions = []
    shared_queries = {}
    for position, job in enumerate(jobs):
        history_name, phase, endpoint = job["history"], job["phase"], job["arm"]
        graph = states[history_name, phase][endpoint]
        chain, parsing = parsed_by_key[key(job)]
        shared_key = (history_name, phase, key(job), "G_CORE" if endpoint == "G_FULL" else endpoint)
        if shared_key not in shared_queries: shared_queries[shared_key] = query_generated(graph, chain)
        graph_result = shared_queries[shared_key]
        question = questions[job["query_id"]]
        facts, retrieval = fallback_evidence(endpoint, question, graph, case["histories"][history_name]["edit_sentences"], ontology, templates, retrieve, phase)
        row = {**job, "question": question, "chain_generation": generated_by_key[key(job)], "parse": parsing,
               "chain": parsing["chain"], "graph": graph_result, "state_sha256": digest(graph.triples()),
               "fallback_evidence": retrieval,
               "legacy_sensitivity": legacy_sensitivity(generated_by_key[key(job)]["text"], graph, entities, ontology, retrieve)}
        if all_fallbacks or graph_result.get("entity") is None:
            fallback_positions.append(position); fallback_requests.append(fallback_request(question, facts, examples))
        prepared.append(row)
    generations = engine.calls(fallback_requests, nonce=None if nonce is None else [nonce, "fallbacks"])
    for position, generation in zip(fallback_positions, generations): prepared[position]["fallback_generation"] = generation
    for row in prepared:
        fallback = row.get("fallback_generation")
        if fallback is None:
            row["fallback_generation"] = {"measured": False, "reason": "predeclared_lazy_graph_priority"}
            fallback = {"text": "", "cap_hit": False}
        else: row["fallback_generation"]["measured"] = True
        row.update(combined_answer(row["graph"], fallback, entities))
        row["program"] = None if row["chain"] is None else {"start_id": row["chain"]["nodes"][0], "path": row["chain"]["relations"]}
        row["method_id"] = digest({"endpoint": row["arm"], "state": row["state_sha256"], "question": row["question"],
                                   "chain_call": row["chain_generation"].get("call_id"), "fallback_call": fallback.get("call_id"), "nonce": nonce})
    return {"rows": prepared, "graphs": graph_records, "updates": update_records}
