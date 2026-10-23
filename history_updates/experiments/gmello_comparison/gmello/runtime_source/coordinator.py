"""Retained development-gate calculation; operational coordinator excluded."""
def development_gate(revision):
    folder = ROOT / "inference/out" / ("dev_r" + str(revision))
    complete = read(folder / "COMPLETE.json"); assert complete["cases"] == 32
    cases = {c["case_id"]: c for c in read(ROOT / "inference/data/development.json")}
    gold = {g["case_id"]: g["questions"] for g in read(ROOT / "development_evaluator/dev_gold.json")}
    truth = {}; depth = {}
    for cid, questions in gold.items():
        history = cases[cid]["histories"]["H0"]
        state = {(s,p): o for s,p,o in history["initial"]}
        for s,p,o in history["edits"]: state[s,p] = o
        for q in questions:
            qid = str(q["kind"]) + ":" + str(q["index"]); current = q["subject_id"]
            for relation in q["path"]: current = state[current, relation]
            truth[cid,qid] = current; depth[cid,qid] = len(q["path"])
    counts = collections.defaultdict(collections.Counter); correct_depths = set(); rows = []; update_counts = collections.Counter(); graph_equal = 0; text_graphs = 0
    for path in sorted((folder / "cases").glob("*.json")):
        result = read(path)
        for row in result["rows"]:
            expected = truth[row["case_id"], row["query_id"]]; endpoint = row["arm"]
            is_target = row["query_id"].startswith("target:")
            graph_valid = row["graph"].get("entity") is not None
            graph_correct = graph_valid and row["graph"]["entity"] == expected
            counts[endpoint]["questions"] += 1
            counts[endpoint]["parseable_linked"] += row["parse"]["status"] == "VALID"
            counts[endpoint]["nonempty_graphs"] += graph_valid
            counts[endpoint]["correct_graphs"] += graph_correct
            counts[endpoint]["combined_correct"] += row["outcome"]["status"] == "VALID" and row["outcome"]["entity"] == expected
            counts[endpoint]["fallback_measured"] += row["fallback_generation"].get("measured", False)
            if is_target:
                counts[endpoint]["targets"] += 1
                counts[endpoint]["target_parseable_linked"] += row["parse"]["status"] == "VALID"
                counts[endpoint]["target_nonempty_graphs"] += graph_valid
                counts[endpoint]["target_correct_graphs"] += graph_correct
                if endpoint == "G_CORE" and graph_correct: correct_depths.add(depth[row["case_id"], row["query_id"]])
            rows.append({"case_id": row["case_id"], "query_id": row["query_id"], "arm": endpoint, "target": is_target,
                         "depth": depth[row["case_id"], row["query_id"]], "graph_correct": graph_correct,
                         "combined_correct": row["outcome"]["status"] == "VALID" and row["outcome"]["entity"] == expected})
        for update in result["updates"]:
            intended = cases[update["case_id"]]["histories"][update["history"]]["edits"][update["index"]]
            parsed = update["parsed"]
            update_counts["intended"] += 1
            update_counts["applied"] += parsed["status"] == "APPLIED"
            update_counts["unapplied"] += parsed["status"] != "APPLIED"
            update_counts["exact_triple"] += parsed.get("triple") == intended
            update_counts["wrong_applied"] += parsed["status"] == "APPLIED" and parsed["triple"] != intended
            update_counts["direction_swaps"] += parsed["swap"]
        core_graphs = {(g["history"], g["phase"]): g for g in result["graphs"] if g["endpoint"] == "G_CORE"}
        for graph in result["graphs"]:
            if graph["endpoint"] == "G_TEXT":
                text_graphs += 1; graph_equal += graph["triples"] == core_graphs[graph["history"], graph["phase"]]["triples"]
    core = counts["G_CORE"]; assert core["targets"] == 96 and core["questions"] == 189
    passed = core["target_parseable_linked"] / 96 >= .9 and core["target_nonempty_graphs"] / 96 >= .5 and {2,3,4} <= correct_depths
    report = {"status": "PASS" if passed else "UNQUALIFIED", "revision": revision, "counts": {k: dict(v) for k,v in counts.items()},
              "correct_graph_target_depths": sorted(correct_depths), "thresholds": {"parseable_linked": .9, "nonempty_graph": .5, "correct_depths": [2,3,4]},
              "fallback_counts_toward_gate": False, "g_text_edit_counts": dict(update_counts), "g_text_graph_equal": graph_equal,
              "g_text_graph_denominator": text_graphs, "rows": rows, "development_only": True, "main_outcomes_observed": False}
    publish(ROOT / "evidence" / ("DEV_GATE_R" + str(revision) + ".json"), report)
    return report
