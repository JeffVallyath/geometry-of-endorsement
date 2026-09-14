from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def _mapping(item_id: str, scheme: str, symbols: tuple[str, str]) -> tuple[str, str, int, str]:
    reversed_map = bool(hashlib.sha256(f"answer-map:{item_id}".encode()).digest()[0] & 1)
    positive, negative = (symbols[1], symbols[0]) if reversed_map else symbols
    physical = -1 if reversed_map else 1
    instruction = f"Use {positive} when the answer is positive, and {negative} when it is negative. Reply with only {symbols[0]} or {symbols[1]}."
    return positive, negative, physical, instruction


def _actor_messages(actor: str, system: str, user: str) -> list[dict[str, str]]:
    if actor == "llama":
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if actor == "gemma":
        return [{"role": "user", "content": f"{system}\n\n{user}"}]
    raise RuntimeError(f"Unsupported actor chat policy: {actor}")

def _messages(actor: str, system: str, template: str, text: str, instruction: str) -> list[dict[str, str]]:
    user = template.format(text=text, mapping_instruction=instruction)
    return _actor_messages(actor, system, user)


def _valid(path: Path, sha256_file) -> bool:
    receipt = path.with_suffix(".json")
    if not path.is_file() or not receipt.is_file():
        return False
    try:
        record = json.loads(receipt.read_text(encoding="utf-8"))
        return record.get("complete") is True and record.get("sha256") == sha256_file(path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False


def _save(path: Path, sha256_file, write_json, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); partial = path.with_suffix(".part.npz")
    np.savez_compressed(partial, **arrays); partial.replace(path)
    write_json(path.with_suffix(".json"), {"sha256": sha256_file(path), "complete": True, "rows": len(next(iter(arrays.values())))})


def _extract_controls(actor: str, cfg: dict, work: Path, persistent: Path, load_actor, score_ids_and_extract, sha256_file, write_json, semantic_and_token_directions, residualize) -> None:
    spec = cfg["models"][actor]; model, tokenizer = load_actor(spec["id"], spec["revision"], work / "cache/models", os.environ.get("HF_TOKEN"))
    root = persistent / actor / "controls" / "extraction"; root.mkdir(parents=True, exist_ok=True)
    factual = list(csv.DictReader((work / "setup-v1.1/dataset_tables/factual_true_false_freeze.csv").open(encoding="utf-8", newline="")))
    sentiment = list(csv.DictReader((work / "setup-v1.1/dataset_tables/sentiment_uci_331_freeze.csv").open(encoding="utf-8", newline="")))
    # Exact prompt text is reconstructed from the pinned Truth-v2 config shipped in base v1.
    import yaml
    tp = yaml.safe_load((work / "configs/truth_control_v2.yaml").read_text(encoding="utf-8"))["prompt"]
    for split in ("train", "dev", "test"):
        for scheme in ("primary", "transfer"):
            target = root / f"factual__{split}__{scheme}.npz"
            if not _valid(target, sha256_file):
                acts=[]; labels=[]; physical=[]; ids=[]; margins=[]
                symbols=tuple(str(x) for x in tp["schemes"][scheme]["symbols"])
                for row in (r for r in factual if r["split"] == split and r["scheme"] == scheme):
                    positive, negative, token_sign, instruction = _mapping(row["item_id"], scheme, symbols)
                    orientation = "reversed" if token_sign == -1 else "standard"
                    if row["mapping"] != orientation:
                        raise RuntimeError(f"Frozen factual answer mapping drift: {row['record_id']}")
                    body = tp["schemes"][scheme]["user_template"].format(statement=row["statement"], mapping_instruction=instruction.replace("answer is positive", "statement is factually true").replace("answer is negative", "statement is factually false"))
                    pids=list(tokenizer.apply_chat_template(_actor_messages(actor,tp["system_template"],body), tokenize=True, add_generation_prompt=True))
                    result=score_ids_and_extract(model,tokenizer,pids,(positive,negative),spec["layer"])
                    label=1 if int(row["label"]) else -1; chosen=positive if label==1 else negative
                    acts.append(result["activation"]); labels.append(label); physical.append(1 if chosen in {"A","1"} else -1); ids.append(row["record_id"]); margins.append(result["candidate_log_probabilities"][positive]-result["candidate_log_probabilities"][negative])
                _save(target,sha256_file,write_json,activations=np.asarray(acts,dtype=np.float32),semantic=np.asarray(labels,dtype=np.int8),physical=np.asarray(physical,dtype=np.int8),row_ids=np.asarray(ids),semantic_margin=np.asarray(margins,dtype=np.float32))
            target = root / f"sentiment__{split}__{scheme}.npz"
            if not _valid(target, sha256_file):
                acts=[]; labels=[]; physical=[]; ids=[]; margins=[]; prompt=cfg["sentiment_control"]["schemes"][scheme]; symbols=tuple(prompt["symbols"])
                for row in (r for r in sentiment if r["split"] == split):
                    positive, negative, token_sign, instruction = _mapping(row["row_id"], scheme, symbols)
                    pids=list(tokenizer.apply_chat_template(_messages(actor,prompt["system"],prompt["user"],row["text"],instruction),tokenize=True,add_generation_prompt=True))
                    result=score_ids_and_extract(model,tokenizer,pids,(positive,negative),spec["layer"])
                    label=1 if int(row["label"]) else -1; chosen=positive if label==1 else negative
                    acts.append(result["activation"]); labels.append(label); physical.append(1 if chosen in {"A","1"} else -1); ids.append(f"{row['row_id']}:{scheme}"); margins.append(result["candidate_log_probabilities"][positive]-result["candidate_log_probabilities"][negative])
                _save(target,sha256_file,write_json,activations=np.asarray(acts,dtype=np.float32),semantic=np.asarray(labels,dtype=np.int8),physical=np.asarray(physical,dtype=np.int8),row_ids=np.asarray(ids),semantic_margin=np.asarray(margins,dtype=np.float32))
    directions={}; scales={}
    for lane in ("factual","sentiment"):
        xs=[]; ys=[]; ps=[]
        for path in sorted(root.glob(f"{lane}__train__*.npz")):
            with np.load(path,allow_pickle=False) as z: xs.append(z["activations"]); ys.append(z["semantic"]); ps.append(z["physical"])
        x=np.concatenate(xs); y=np.concatenate(ys); p=np.concatenate(ps); raw,token=semantic_and_token_directions(x,y,p); _,clean=residualize(raw,np.stack([token])); clean/=np.linalg.norm(clean)
        key="factual_true_false" if lane=="factual" else "sentiment_valence"; directions[key]=clean.astype(np.float32); directions[key+"_raw"]=raw.astype(np.float32); directions[key+"_physical_token"]=token.astype(np.float32); scales[key]=float(np.std(x@clean,ddof=1))
    bundle=persistent/actor/"controls"/"control_directions.npz"; _save(bundle,sha256_file,write_json,**directions,scales_json=np.asarray([json.dumps(scales)]))
    del model,tokenizer; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass


def _load_v1_rows(work: Path):
    import pandas as pd
    from datasets import Dataset
    from geometry_endorsement.direction_identity.dataset_adapters import load_amperepp
    from geometry_of_truth.m1.support.build_confirmatory_split import row_id as vp_row_id
    freeze=json.loads((work/"setup/DATASET_FREEZE.json").read_text()); candidates=load_amperepp(work/"datasets/release_modified"); selected={r["row_id"]:r for r in freeze["datasets"]["AMPERE++"]["rows"]}; amp=[r for r in candidates if r.row_id in selected]
    frames=[Dataset.from_file(str(path)).to_pandas() for path in sorted((work/"datasets/valueprism-arrow").glob("*.arrow"))]; source=pd.concat(frames,ignore_index=True); source["valence"]=source["valence"].astype(str).str.strip().str.lower(); source=source[source["valence"].isin(["supports","opposes"])].copy(); source["row_id"]=[vp_row_id(str(r.situation),str(r.text),str(r.valence),str(r.vrd)) for r in source.itertuples()]; by=source.drop_duplicates("row_id").set_index("row_id",drop=False)
    vp={}
    for split in ("pilot_select","pilot_eval"):
        manifest=pd.read_csv(work/f"datasets/valueprism-pilot-manifests/{split}.csv",dtype=str); frame=by.loc[manifest.row_id].reset_index(drop=True)
        for column in manifest.columns: frame[column]=manifest[column].to_numpy()
        vp[split]=frame
    return freeze,amp,vp


def _control_intervention(actor: str, cfg: dict, work: Path, persistent: Path, load_actor, score_and_extract, score_ids_and_extract, sha256_file, write_json) -> None:
    from geometry_endorsement.direction_identity.mapping_factorial import MAPPINGS
    from geometry_endorsement.direction_identity.prompt_contracts import TEMPLATES
    from geometry_of_truth.m1.prompts import AnswerMapping as M1Mapping, messages_for
    import yaml
    spec=cfg["models"][actor]; directions_path=persistent/actor/"controls/control_directions.npz"
    if not _valid(directions_path,sha256_file): raise RuntimeError("Control extraction gate failed")
    with np.load(directions_path,allow_pickle=False) as z: vectors={key:z[key] for key in ("factual_true_false","sentiment_valence")}; scales=json.loads(str(z["scales_json"][0]))
    model,tokenizer=load_actor(spec["id"],spec["revision"],work/"cache/models",os.environ.get("HF_TOKEN")); freeze,amp,vp=_load_v1_rows(work); out=persistent/actor/"controls/intervention"; out.mkdir(parents=True,exist_ok=True)
    amp_ids=set(freeze["intervention_subsets"]["AMPERE++"]["support_row_ids"]+freeze["intervention_subsets"]["AMPERE++"]["attack_row_ids"]); amp=[r for r in amp if r.row_id in amp_ids]
    boards=set(freeze["intervention_subsets"]["ValuePrism"]["board_ids"]); vp=vp["pilot_eval"]; vp=vp[vp.board_id.astype(str).isin(boards)]
    m1_prompt=yaml.safe_load((work/"configs/m1_development.yaml").read_text())["prompt"]
    vp_maps=(M1Mapping("MAPPING_AB_STANDARD","A","B","Use A when the named consideration supports the action in this situation, and B when it opposes the action. Reply with only A or B."),M1Mapping("MAPPING_AB_REVERSED","B","A","Use B when the named consideration supports the action in this situation, and A when it opposes the action. Reply with only A or B."))
    amp_maps=tuple(m for m in MAPPINGS if m.mapping_id in {"MAPPING_AB_STANDARD","MAPPING_AB_REVERSED"})
    for dataset,items,maps in (("ValuePrism",list(vp.itertuples()),vp_maps),("AMPERE++",amp,amp_maps)):
        for mapping in maps:
            for name,vector in vectors.items():
                for dose in (-1.,0.,1.):
                    mapping_id = mapping.name if dataset == "ValuePrism" else mapping.mapping_id
                    target=out/f"{dataset}__{mapping_id}__{name}__{dose:+.1f}.npz"
                    if _valid(target,sha256_file): continue
                    margins=[]; ids=[]; delta=vector*(float(scales[name])*dose/float(vector@vector))
                    for item in items:
                        if dataset=="ValuePrism":
                            messages=messages_for(str(item.situation),str(item.text),mapping,"primary","joint",m1_prompt,model_id=spec["id"]); pids=list(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)); candidates=(mapping.supports,mapping.opposes); result=score_ids_and_extract(model,tokenizer,pids,candidates,spec["layer"],delta); rid=str(item.row_id)
                        else:
                            rendered=TEMPLATES["EXPLICIT_RELATION"].format(source=item.source_text,target=item.target_text,mapping_instruction=mapping.instruction); candidates=(mapping.positive_token,mapping.negative_token); result=score_and_extract(model,tokenizer,rendered,candidates,spec["layer"],delta); rid=item.row_id
                        margins.append(result["candidate_log_probabilities"][candidates[0]]-result["candidate_log_probabilities"][candidates[1]]); ids.append(rid)
                    _save(target,sha256_file,write_json,row_ids=np.asarray(ids),semantic_margin=np.asarray(margins,dtype=np.float32),dose=np.full(len(ids),dose,dtype=np.float32))
    write_json(persistent/actor/"controls/control_intervention_complete.json",{"status":"COMPLETE","directions_sha256":sha256_file(directions_path),"files":len(list(out.glob("*.npz")))})
    del model,tokenizer; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass


def _auc(scores, labels):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score((np.asarray(labels)==1).astype(int),scores))


def _paired_slopes(folder: Path, dataset: str, direction: str) -> dict[str, float]:
    by_row: dict[str,list[float]]=defaultdict(list)
    for mapping in ("MAPPING_AB_STANDARD","MAPPING_AB_REVERSED"):
        files=sorted(folder.glob(f"{dataset}__{mapping}__{direction}__*.npz"),key=lambda p:float(p.stem.split("__")[-1])); doses=[]; matrix=[]; ids=None
        for path in files:
            with np.load(path,allow_pickle=False) as z: doses.append(float(np.asarray(z["dose"]).reshape(-1)[0])); matrix.append(z["semantic_margin"].astype(float)); current=z["row_ids"].astype(str)
            if ids is None: ids=current
            elif not np.array_equal(ids,current): raise RuntimeError(f"Intervention row drift: {path}")
        slopes=np.polyfit(np.asarray(doses),np.stack(matrix),1)[0]
        for rid,value in zip(ids,slopes,strict=True): by_row[rid].append(float(value))
    return {rid:float(np.mean(values)) for rid,values in by_row.items()}


def _analyze(cfg: dict, persistent: Path, sha256_file, write_json) -> None:
    from geometry_endorsement.direction_identity.decision_logic import PERMITTED_CLAIMS, decide
    from geometry_endorsement.direction_identity.analysis_runtime import _board_summary, _external_summary
    from geometry_endorsement.direction_identity.grouped_statistics import cluster_bootstrap_metric
    from geometry_endorsement.direction_identity.output_channel import residualize
    base=json.loads((persistent/"analysis_results.json").read_text(encoding="utf-8")); actors={}
    work=Path(os.environ["V1_1_WORK_ROOT"]); freeze,amp,vp_rows=_load_v1_rows(work); v1_cfg=json.loads((work/"configs/relation_direction_identity_v1.yaml").read_text())
    amp_groups={row["row_id"]:row["group_id"] for row in freeze["datasets"]["AMPERE++"]["rows"]}; amp_test={row["row_id"] for row in freeze["datasets"]["AMPERE++"]["rows"] if row["split"]=="test"}; vp_groups=vp_rows["pilot_eval"].set_index("row_id")["board_id"].astype(str).to_dict()
    for actor in ("llama","gemma"):
        root=persistent/actor/"controls"; direction=root/"control_directions.npz"
        if not _valid(direction,sha256_file) or len(list((root/"intervention").glob("*.npz")))!=24: raise RuntimeError(f"V1.1 control completeness gate failed: {actor}")
        with np.load(direction,allow_pickle=False) as z: vectors={key:z[key] for key in ("factual_true_false","sentiment_valence")}
        tests={}
        for lane,key in (("factual","factual_true_false"),("sentiment","sentiment_valence")):
            scores=[]; labels=[]
            for path in sorted((root/"extraction").glob(f"{lane}__test__*.npz")):
                with np.load(path,allow_pickle=False) as z: scores.extend(z["activations"]@vectors[key]); labels.extend(z["semantic"])
            tests[key]={"test_auroc":_auc(np.asarray(scores),np.asarray(labels)),"rows":len(scores)}
        relation=np.load(persistent/actor/"directions.npz",allow_pickle=False)["residual_dim"]; _,relation_without_sentiment=residualize(relation,np.stack([vectors["sentiment_valence"]])); relation_without_sentiment/=np.linalg.norm(relation_without_sentiment)
        tests["geometry"]={"relation_sentiment_abs_cosine":float(abs(relation@vectors["sentiment_valence"])/(np.linalg.norm(relation)*np.linalg.norm(vectors["sentiment_valence"]))),"relation_factual_abs_cosine":float(abs(relation@vectors["factual_true_false"])/(np.linalg.norm(relation)*np.linalg.norm(vectors["factual_true_false"])))}
        extraction=persistent/actor/"extraction"; amp_files=sorted(extraction.glob("AMPERE++__*.npz")); vp_select=sorted(extraction.glob("ValuePrism_PILOT_SELECT__*.npz")); vp_eval=sorted(extraction.glob("ValuePrism_PILOT_EVAL__*.npz"))
        residual_endpoints={"AMPERE++":_external_summary(amp_files,relation_without_sentiment,amp_test,amp_groups,v1_cfg,20260822),"ValuePrism":_board_summary(vp_select,vp_eval,relation_without_sentiment,vp_rows,work/"datasets",v1_cfg,20260823)}
        slope={}
        for dataset in ("ValuePrism","AMPERE++"):
            slope[dataset]={}
            for key in ("factual_true_false","sentiment_valence"):
                control=_paired_slopes(root/"intervention",dataset,key); relation_slopes=_paired_slopes(persistent/actor/"intervention",dataset,"residual_dim"); ids=np.asarray(sorted(set(control)&set(relation_slopes))); delta=np.asarray([relation_slopes[rid]-control[rid] for rid in ids]); group_map=vp_groups if dataset=="ValuePrism" else amp_groups; groups=np.asarray([group_map[rid] for rid in ids]); ci=cluster_bootstrap_metric(delta,np.zeros(len(delta)),groups,lambda values,_labels:float(np.mean(values)),replicates=int(v1_cfg["statistics"]["bootstrap_replicates"]),seed=20260822)
                slope[dataset][key]={"relation_minus_control_mean_slope":float(delta.mean()),"grouped_ci":ci,"rows":len(ids),"groups":int(len(np.unique(groups)))}
        actors[actor]={"control_test":tests,"sentiment_residualized_relation_endpoints":residual_endpoints,"control_intervention":slope}
    evidence=dict(base["decision_evidence"])
    sentiment_resolved=all(actors[a]["control_test"]["sentiment_valence"]["test_auroc"]>.5 and actors[a]["sentiment_residualized_relation_endpoints"]["AMPERE++"]["auroc_ci"]["low"]>.5 and actors[a]["sentiment_residualized_relation_endpoints"]["ValuePrism"]["interaction_ci"]["low"]>0 and all(actors[a]["control_intervention"][dataset]["sentiment_valence"]["grouped_ci"]["low"]>0 for dataset in ("ValuePrism","AMPERE++")) for a in actors)
    factual_specific=all(actors[a]["control_test"]["factual_true_false"]["test_auroc"]>.5 and all(actors[a]["control_intervention"][dataset]["factual_true_false"]["grouped_ci"]["low"]>0 for dataset in ("ValuePrism","AMPERE++")) for a in actors)
    evidence["sentiment_alternative_unresolved"]=not sentiment_resolved; evidence["sentiment_not_complete_explanation"]=sentiment_resolved
    result={"schema_version":"1.1","study_id":cfg["study_id"],"base_v1_analysis_sha256":sha256_file(persistent/"analysis_results.json"),"actors":actors,"decision_evidence":evidence,"sentiment_alternative_resolved":sentiment_resolved,"factual_steering_specificity_established":factual_specific,"terminal_disposition":decide(evidence),"permitted_claim":PERMITTED_CLAIMS[decide(evidence)],"claim_boundary":"Specificity upgrades fail closed until prespecified grouped paired intervention confidence intervals are present; control competence alone is insufficient."}
    write_json(persistent/"analysis_results_v1_1.json",result)


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--mode",required=True); parser.add_argument("--persistent-root",type=Path,required=True); parser.add_argument("--work-root",type=Path,required=True); args=parser.parse_args()
    work=args.work_root.resolve(); persistent=args.persistent_root.resolve()/"runtime"; persistent.mkdir(parents=True,exist_ok=True)
    if os.name=="nt" and persistent.drive.upper()=="C:": raise RuntimeError("Heavyweight persistent output resolves to C:")
    os.environ.update({"HF_HOME":str(work/"cache/huggingface"),"HF_HUB_CACHE":str(work/"cache/huggingface/hub"),"HF_DATASETS_CACHE":str(work/"cache/huggingface/datasets"),"TRANSFORMERS_CACHE":str(work/"cache/transformers"),"TORCH_HOME":str(work/"cache/torch"),"TMPDIR":str(work/"tmp"),"TEMP":str(work/"tmp"),"TMP":str(work/"tmp"),"V1_1_WORK_ROOT":str(work)})
    sys.path.insert(0,str(work/"src")); from geometry_endorsement.direction_identity.activation_extraction import load_actor,score_and_extract,score_ids_and_extract; from geometry_endorsement.direction_identity.artifact_io import sha256_file,write_json; from geometry_endorsement.direction_identity.output_channel import semantic_and_token_directions,residualize
    cfg=json.loads((work/"configs/relation_direction_identity_v1_1.yaml").read_text()); actor="llama" if "LLAMA" in args.mode else "gemma" if "GEMMA" in args.mode else None
    for path in (persistent,work/"cache/models",work/"cache/huggingface",work/"tmp"): path.mkdir(parents=True,exist_ok=True)
    print("STORAGE_PREFLIGHT_PASS")
    if args.mode in {"RUN_LLAMA_CONTROLS","RUN_GEMMA_CONTROLS"}: _extract_controls(actor,cfg,work,persistent,load_actor,score_ids_and_extract,sha256_file,write_json,semantic_and_token_directions,residualize)
    elif args.mode in {"RUN_LLAMA_CONTROL_INTERVENTION","RUN_GEMMA_CONTROL_INTERVENTION"}: _control_intervention(actor,cfg,work,persistent,load_actor,score_and_extract,score_ids_and_extract,sha256_file,write_json)
    elif args.mode=="ANALYZE_CPU_V1_1": _analyze(cfg,persistent,sha256_file,write_json)
    else: raise RuntimeError(f"Unsupported v1.1 mode: {args.mode}")
    return 0


if __name__=="__main__": raise SystemExit(main())
