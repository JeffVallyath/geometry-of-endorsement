"""Original secondary tables for the completed matched-origin panel."""
from collections import defaultdict

def per_scene(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["actor"], r["condition"], r.get("scene_id"), r.get("input_panel"))].append(r)
    out = []
    for (actor, cond, scene, panel), rr in sorted(g.items(), key=lambda x: str(x[0])):
        valid = [x for x in rr if x["valid"]]
        out.append(dict(actor=actor, condition=cond, scene_id=scene, panel=panel,
                        questions=len(rr), valid=len(valid),
                        correct=sum(1 for x in rr if x["correct"]),
                        accuracy=(sum(1 for x in rr if x["correct"]) / len(rr)) if rr else None,
                        all_correct=int(bool(rr) and all(x["correct"] for x in rr)),
                        disagreement=sum(1 for x in rr if x["valid"] and x["source_valid"]
                                         and x["prediction"] != x["source_prediction"])))
    return out


def per_root(rows):
    g = defaultdict(list)
    for r in rows:
        if r.get("input_panel") != "ORIGIN":
            continue
        g[(r["actor"], r["condition"], r["root_id"])].append(r)
    out = []
    for (actor, cond, root), rr in sorted(g.items(), key=lambda x: str(x[0])):
        origins = sorted({x["origin"] for x in rr if x["origin"] is not None})
        out.append(dict(actor=actor, condition=cond, root_id=root,
                        origins_present=len(origins), questions=len(rr),
                        correct=sum(1 for x in rr if x["correct"]),
                        bundle_complete=int(len(origins) == 4 and all(x["correct"] for x in rr)),
                        mean_question_accuracy=(sum(1 for x in rr if x["correct"]) / len(rr)) if rr else None))
    return out


def headline(rows, primary):
    g = defaultdict(list)
    for r in rows:
        g[(r["actor"], r["condition"])].append(r)
    eff = {}
    for entry in (primary or []):
        eff[(entry["actor"], entry["recipe"])] = entry

    out = []
    for (actor, cond), rr in sorted(g.items(), key=lambda x: str(x[0])):
        base = cond.rsplit("_s", 1)[0] if "_s" in cond else cond
        e = eff.get((actor, base))
        origin_rows = [x for x in rr if x.get("input_panel") == "ORIGIN"]
        main_rows = [x for x in rr if x.get("input_panel") == "MAIN"]
        out.append(dict(
            actor=actor, condition=cond, method=base,
            questions=len(rr),
            all_origin_question_accuracy=(sum(1 for x in origin_rows if x["correct"]) / len(origin_rows)) if origin_rows else None,
            main_question_accuracy=(sum(1 for x in main_rows if x["correct"]) / len(main_rows)) if main_rows else None,
            invalid=sum(1 for x in rr if not x["valid"]),
            disagreement_vs_source=sum(1 for x in rr if x["valid"] and x["source_valid"]
                                       and x["prediction"] != x["source_prediction"]),
            primary_effect_vs_baseline=e["effect"] if e else None,
            primary_ci_low=e["lower"] if e else None,
            primary_ci_high=e["upper"] if e else None,
        ))
    return out


def timing_memory(telemetry):
    g = defaultdict(list)
    for t in telemetry:
        g[(t["actor"], t["condition"])].append(t)
    out = []
    for (actor, cond), tt in sorted(g.items(), key=lambda x: str(x[0])):
        def mean(key):
            vals = [x[key] for x in tt if x.get(key) is not None]
            return sum(vals) / len(vals) if vals else None
        attempted = sum(1 for x in tt if x.get("refresh_attempted"))
        fallback = sum(1 for x in tt if x.get("refresh_fallback"))
        out.append(dict(actor=actor, condition=cond, cases=len(tt),
                        mean_preprocess_seconds=mean("preprocess_seconds"),
                        mean_source_prefill_seconds=mean("source_prefill_seconds"),
                        mean_update_refresh_seconds=mean("refresh_seconds"),
                        mean_compile_seconds=mean("compile_seconds"),
                        refresh_attempted=attempted, refresh_fallback=fallback,
                        refresh_fallback_rate=(fallback / attempted) if attempted else None))
    return out

