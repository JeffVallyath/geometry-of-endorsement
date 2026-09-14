from __future__ import annotations

import argparse, csv, gc, hashlib, json, os, shutil, sys, zipfile
from pathlib import Path
import numpy as np

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--mode",required=True); parser.add_argument("--persistent-root",type=Path,required=True); parser.add_argument("--work-root",type=Path,required=True); args=parser.parse_args()
    work_resolved=args.work_root.resolve(); persistent_resolved=args.persistent_root.resolve()
    runtime_roots=(("PERSISTENT_ROOT",persistent_resolved),("WORK_ROOT",work_resolved),("MODEL_CACHE",args.work_root/"cache/models"),("HF_HOME",args.work_root/"cache/huggingface"),("HF_HUB_CACHE",args.work_root/"cache/huggingface/hub"),("HF_DATASETS_CACHE",args.work_root/"cache/huggingface/datasets"),("TRANSFORMERS_CACHE",args.work_root/"cache/transformers"),("TORCH_HOME",args.work_root/"cache/torch"),("TEMP_ROOT",args.work_root/"tmp"))
    os.environ.update({name:str(Path(path).resolve()) for name,path in runtime_roots if name in {"HF_HOME","HF_HUB_CACHE","HF_DATASETS_CACHE","TRANSFORMERS_CACHE","TORCH_HOME"}})
    os.environ.update({"TMPDIR":str((args.work_root/"tmp").resolve()),"TEMP":str((args.work_root/"tmp").resolve()),"TMP":str((args.work_root/"tmp").resolve())})
    sys.path.insert(0,str(args.work_root/"src"))
    from geometry_endorsement.direction_identity.activation_extraction import candidate_ids, load_actor, load_direction_npz, score_and_extract, score_ids_and_extract
    from geometry_endorsement.direction_identity.artifact_io import sha256_file, write_json, write_text
    from geometry_endorsement.direction_identity.dataset_adapters import load_abstrct, load_amperepp, iter_aries_binary
    from geometry_endorsement.direction_identity.mapping_factorial import MAPPINGS
    from geometry_endorsement.direction_identity.output_channel import residualize, semantic_and_token_directions
    from geometry_endorsement.direction_identity.prompt_contracts import TEMPLATES
    from geometry_of_truth.m1.prompts import AnswerMapping as M1Mapping, messages_for
    from geometry_of_truth.m1.support.build_confirmatory_split import row_id as vp_row_id
    import pandas as pd
    from datasets import Dataset
    import yaml
    cfg=json.loads((args.work_root/"configs/relation_direction_identity_v1.yaml").read_text())
    freeze=json.loads((args.work_root/"setup/DATASET_FREEZE.json").read_text())
    mode=args.mode; actor="llama" if "LLAMA" in mode else "gemma" if "GEMMA" in mode else None
    persistent=args.persistent_root/"runtime"; persistent.mkdir(parents=True,exist_ok=True)
    if os.name=="nt" and persistent_resolved.drive.upper()=="C:": raise RuntimeError("Heavyweight persistent output resolves to C:")
    for label,path in runtime_roots:
        path=Path(path); path.mkdir(parents=True,exist_ok=True); probe=path/".write_probe"; probe.write_text("ok",encoding="utf-8"); probe.unlink(); print(f"{label}={path.resolve()}")
    print(f"PERSISTENT_FREE_BYTES={shutil.disk_usage(persistent_resolved).free}")
    print(f"WORK_FREE_BYTES={shutil.disk_usage(work_resolved).free}")
    print("STORAGE_PREFLIGHT_PASS")
    data=args.work_root/"datasets"; amp=data/"release_modified"
    if not amp.exists():
        with zipfile.ZipFile(data/"release_modified.zip") as archive: archive.extractall(data)
        amp=data/"release_modified"
    all_rows={"AMPERE++":load_amperepp(amp),"AbstRCT":load_abstrct(data),"US2016":list(iter_aries_binary(data/"aries_data.csv","US2016","US2016"))}
    rows={}
    for dataset,candidates in all_rows.items():
        selected={r["row_id"]:r for r in freeze["datasets"][dataset]["rows"]}; rows[dataset]=[row for row in candidates if row.row_id in selected]
        rows[dataset]=[type(row)(row.row_id,row.dataset_id,selected[row.row_id]["split"],row.group_id,row.source_text,row.target_text,row.semantic_label,row.source_ref) for row in rows[dataset]]
        if {r.row_id for r in rows[dataset]} != set(selected): raise RuntimeError(f"Frozen rows do not replay for {dataset}")
    vp_frames=[Dataset.from_file(str(path)).to_pandas() for path in sorted((data/"valueprism-arrow").glob("*.arrow"))]
    vp_source=pd.concat(vp_frames,ignore_index=True); vp_source["valence"]=vp_source["valence"].astype(str).str.strip().str.lower(); vp_source=vp_source[vp_source["valence"].isin(["supports","opposes"])].copy()
    vp_source["row_id"]=[vp_row_id(str(r.situation),str(r.text),str(r.valence),str(r.vrd)) for r in vp_source.itertuples()]; vp_by_id=vp_source.drop_duplicates("row_id").set_index("row_id",drop=False)
    vp_rows={}
    for split in ("pilot_select","pilot_eval"):
        manifest=pd.read_csv(data/"valueprism-pilot-manifests"/f"{split}.csv",dtype=str)
        missing=set(manifest.row_id)-set(vp_by_id.index)
        if missing: raise RuntimeError(f"ValuePrism frozen manifest does not resolve: {len(missing)}")
        frame=vp_by_id.loc[manifest.row_id].reset_index(drop=True)
        for column in manifest.columns: frame[column]=manifest[column].to_numpy()
        vp_rows[split]=frame
    m1_prompt=yaml.safe_load((args.work_root/"configs/m1_development.yaml").read_text())["prompt"]
    vp_mappings=(
        ("MAPPING_AB_STANDARD","primary",M1Mapping("MAPPING_AB_STANDARD","A","B","Use A when the named consideration supports the action in this situation, and B when it opposes the action. Reply with only A or B.")),
        ("MAPPING_AB_REVERSED","primary",M1Mapping("MAPPING_AB_REVERSED","B","A","Use B when the named consideration supports the action in this situation, and A when it opposes the action. Reply with only A or B.")),
        ("MAPPING_12_STANDARD","transfer",M1Mapping("MAPPING_12_STANDARD","1","2","Use 1 when the named consideration supports the action in this situation, and 2 when it opposes the action. Reply with only 1 or 2.")),
        ("MAPPING_12_REVERSED","transfer",M1Mapping("MAPPING_12_REVERSED","2","1","Use 2 when the named consideration supports the action in this situation, and 1 when it opposes the action. Reply with only 1 or 2.")))
    if mode.endswith("EXTRACTION"):
        spec=cfg["models"][actor]; model,tokenizer=load_actor(spec["id"],spec["revision"],args.work_root/"cache/models",os.environ.get("HF_TOKEN"))
        actor_root=persistent/actor/"extraction"; actor_root.mkdir(parents=True,exist_ok=True)
        for dataset,dataset_rows in rows.items():
            for template_id,template in TEMPLATES.items():
                for mapping in MAPPINGS:
                    target=actor_root/f"{dataset}__{template_id}__{mapping.mapping_id}.npz"
                    receipt=target.with_suffix(".json")
                    if target.is_file() and receipt.is_file() and json.loads(receipt.read_text())["sha256"]==sha256_file(target): print("RESUME",target.name); continue
                    activations=[]; semantic=[]; physical=[]; row_ids=[]; margins=[]; token_sequences={}
                    for row in dataset_rows:
                        positive=row.semantic_label=="support"; semantic_label="positive" if positive else "negative"
                        instruction=mapping.instruction; rendered=template.format(source=row.source_text,target=row.target_text,mapping_instruction=instruction)
                        result=score_and_extract(model,tokenizer,rendered,(mapping.positive_token,mapping.negative_token),spec["layer"])
                        activations.append(result["activation"]); semantic.append(1 if positive else -1); physical.append(1 if (mapping.positive_token if positive else mapping.negative_token) in {"A","1"} else -1); row_ids.append(row.row_id)
                        margins.append(result["candidate_log_probabilities"][mapping.positive_token]-result["candidate_log_probabilities"][mapping.negative_token]); token_sequences=result["candidate_token_ids"]
                    partial=target.with_suffix(".part.npz"); np.savez_compressed(partial,activations=np.asarray(activations,dtype=np.float32),semantic=np.asarray(semantic,dtype=np.int8),physical=np.asarray(physical,dtype=np.int8),row_ids=np.asarray(row_ids),semantic_margin=np.asarray(margins,dtype=np.float32)); partial.replace(target)
                    write_json(receipt,{"sha256":sha256_file(target),"dataset":dataset,"template":template_id,"mapping":mapping.mapping_id,"candidate_token_ids":token_sequences,"batch_size":1,"complete":True})
                    print("SAVED",target)
        for vp_split,frame in vp_rows.items():
            for mapping_id,scheme,mapping in vp_mappings:
                target=actor_root/f"ValuePrism_{vp_split.upper()}__FROZEN_M1__{mapping_id}.npz"; receipt=target.with_suffix(".json")
                if target.is_file() and receipt.is_file() and json.loads(receipt.read_text())["sha256"]==sha256_file(target): print("RESUME",target.name); continue
                activations=[]; semantic=[]; physical=[]; row_ids=[]; board_ids=[]; margins=[]; token_sequences={}
                for row in frame.itertuples():
                    messages=messages_for(str(row.situation),str(row.text),mapping,scheme,"joint",m1_prompt,model_id=spec["id"])
                    pids=list(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
                    result=score_ids_and_extract(model,tokenizer,pids,(mapping.supports,mapping.opposes),spec["layer"])
                    positive=str(row.valence)=="supports"; activations.append(result["activation"]); semantic.append(1 if positive else -1); chosen=mapping.supports if positive else mapping.opposes; physical.append(1 if chosen in {"A","1"} else -1); row_ids.append(str(row.row_id)); board_ids.append(str(row.board_id)); margins.append(result["candidate_log_probabilities"][mapping.supports]-result["candidate_log_probabilities"][mapping.opposes]); token_sequences=result["candidate_token_ids"]
                partial=target.with_suffix(".part.npz"); np.savez_compressed(partial,activations=np.asarray(activations,dtype=np.float32),semantic=np.asarray(semantic,dtype=np.int8),physical=np.asarray(physical,dtype=np.int8),row_ids=np.asarray(row_ids),board_ids=np.asarray(board_ids),semantic_margin=np.asarray(margins,dtype=np.float32)); partial.replace(target); write_json(receipt,{"sha256":sha256_file(target),"dataset":vp_split,"mapping":mapping_id,"candidate_token_ids":token_sequences,"batch_size":1,"complete":True})
        # Outcome-independent controls are derived only from training shards at the frozen layer.
        arrays=[]
        for path in sorted(actor_root.glob("*.npz")):
            with np.load(path,allow_pickle=False) as z: arrays.append((path.name,z["activations"],z["semantic"],z["physical"],z["row_ids"]))
        train_ids=set(vp_rows["pilot_select"].row_id.astype(str))
        x=[]; sem=[]; phy=[]
        for name,a,s,p,ids in arrays:
            if not name.startswith("ValuePrism_PILOT_SELECT"): continue
            mask=np.asarray([str(i) in train_ids for i in ids]); x.append(a[mask]); sem.append(s[mask]); phy.append(p[mask])
        matrix=np.concatenate(x); sem=np.concatenate(sem); phy=np.concatenate(phy); dsem,dtok=semantic_and_token_directions(matrix,sem,phy)
        family_token={}
        for family,marker in (("ab","MAPPING_AB"),("12","MAPPING_12")):
            fx=[]; fs=[]; fp=[]
            for name,a,s,p,ids in arrays:
                if not name.startswith("ValuePrism_PILOT_SELECT") or marker not in name: continue
                mask=np.asarray([str(i) in train_ids for i in ids]); fx.append(a[mask]); fs.append(s[mask]); fp.append(p[mask])
            family_token[family]=semantic_and_token_directions(np.concatenate(fx),np.concatenate(fs),np.concatenate(fp))[1]
        dim=load_direction_npz(Path('external-artifacts/llama/m1_probe_parameters.npz') if actor=='llama' else Path('external-artifacts/gemma/m1_probe_parameters.npz'),spec["probe_bundle_sha256"],spec["direction_sha256"])
        component,residual=residualize(dim,np.stack([family_token["ab"],family_token["12"]])); external={}
        output_head=model.get_output_embeddings().weight
        def output_row_contrast(positive,negative):
            positive_row=output_head[candidate_ids(tokenizer,positive)].detach().float().mean(0).cpu().numpy()
            negative_row=output_head[candidate_ids(tokenizer,negative)].detach().float().mean(0).cpu().numpy()
            value=positive_row-negative_row; return value/np.linalg.norm(value)
        unembedding_ab=output_row_contrast("A","B"); unembedding_12=output_row_contrast("1","2")
        for dataset in ("AMPERE++","AbstRCT","US2016"):
            dataset_train={r.row_id for r in rows[dataset] if r.split=="train"}; parts=[]; labels=[]
            for name,a,s,_p,ids in arrays:
                if not name.startswith(dataset+"__"): continue
                mask=np.asarray([str(i) in dataset_train for i in ids]); parts.append(a[mask]); labels.append(s[mask])
            matrix_d=np.concatenate(parts); labels_d=np.concatenate(labels); external[dataset]=semantic_and_token_directions(matrix_d,labels_d,labels_d)[0]
        pooled_matrix=[]; pooled_labels=[]
        for dataset in ("AMPERE++","AbstRCT","US2016"):
            dataset_train={r.row_id for r in rows[dataset] if r.split=="train"}
            for name,a,s,_p,ids in arrays:
                if name.startswith(dataset+"__"):
                    mask=np.asarray([str(i) in dataset_train for i in ids]); pooled_matrix.append(a[mask]); pooled_labels.append(s[mask])
        pooled=semantic_and_token_directions(np.concatenate(pooled_matrix),np.concatenate(pooled_labels),np.concatenate(pooled_labels))[0]
        scales={"frozen_dim":float(np.std(matrix@dim,ddof=1)),"residual_dim":float(np.std(matrix@residual,ddof=1)),"physical_token":float(np.std(matrix@dtok,ddof=1)),"amperepp_relation":float(np.std(matrix@external["AMPERE++"],ddof=1)),"pooled_external":float(np.std(matrix@pooled,ddof=1))}
        controls=actor_root.parent/"directions.npz"; np.savez_compressed(controls,frozen_dim=dim,semantic_diagnostic=dsem,physical_token=dtok,physical_token_ab=family_token["ab"],physical_token_12=family_token["12"],token_component=component,residual_dim=residual,secondary_unembedding_ab=unembedding_ab,secondary_unembedding_12=unembedding_12,amperepp_relation=external["AMPERE++"],abstrct_relation=external["AbstRCT"],us2016_relation=external["US2016"],pooled_external=pooled,scales_json=np.asarray(json.dumps(scales)))
        write_json(controls.with_suffix(".json"),{"sha256":sha256_file(controls),"source":"training shards only","layer":spec["layer"],"primary_reporting_span":["physical_token_ab","physical_token_12"],"secondary_reporting_span":"output-head row contrasts are diagnostic only; final normalization/downstream layers prevent equivalence to the frozen intermediate representation","factual_control":"UNAVAILABLE_NO_FROZEN_VECTOR_IN_SAME_ACTOR_LAYER_SPACE","sentiment_control":"SENTIMENT_ALTERNATIVE_UNRESOLVED"})
        del model,tokenizer; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass
        return 0
    if mode.endswith("INTERVENTION"):
        extraction=persistent/actor/"extraction"; directions=persistent/actor/"directions.npz"
        if not directions.is_file() or not list(extraction.glob("*.npz")): raise RuntimeError("Extraction verification gate failed")
        spec=cfg["models"][actor]; model,tokenizer=load_actor(spec["id"],spec["revision"],args.work_root/"cache/models",os.environ.get("HF_TOKEN")); output=persistent/actor/"intervention"; output.mkdir(parents=True,exist_ok=True)
        with np.load(directions,allow_pickle=False) as z: vectors={name:z[name] for name in ("frozen_dim","residual_dim","physical_token","amperepp_relation","pooled_external")}; scales=json.loads(str(z["scales_json"]))
        basis=np.stack(list(vectors.values())); rng=np.random.default_rng(cfg["seed"]); random=rng.normal(size=spec["hidden_size"]); random-=basis.T@np.linalg.pinv(basis@basis.T)@(basis@random); random/=np.linalg.norm(random); vectors["random_orthogonal"]=random; scales["random_orthogonal"]=scales["frozen_dim"]
        direction_names=("frozen_dim","residual_dim","physical_token","amperepp_relation","pooled_external","random_orthogonal")
        full={"frozen_dim","residual_dim","physical_token","amperepp_relation"}; vp_boards=set(freeze["intervention_subsets"]["ValuePrism"]["board_ids"]); vp=vp_rows["pilot_eval"][vp_rows["pilot_eval"].board_id.astype(str).isin(vp_boards)]
        amp_ids=set(freeze["intervention_subsets"]["AMPERE++"]["support_row_ids"]+freeze["intervention_subsets"]["AMPERE++"]["attack_row_ids"]); amp_items=[r for r in rows["AMPERE++"] if r.row_id in amp_ids]
        for dataset,items in (("ValuePrism",list(vp.itertuples())),("AMPERE++",amp_items)):
            for mapping_id in ("MAPPING_AB_STANDARD","MAPPING_AB_REVERSED"):
                if dataset=="ValuePrism": _,scheme,mapping=next(v for v in vp_mappings if v[0]==mapping_id)
                else: mapping=next(v for v in MAPPINGS if v.mapping_id==mapping_id)
                for name in direction_names:
                    doses=(-2.,-1.,-.5,0.,.5,1.,2.) if name in full else (-1.,0.,1.)
                    for dose in doses:
                        target=output/f"{dataset}__{mapping_id}__{name}__{dose:+.1f}.npz"; receipt=target.with_suffix(".json")
                        if target.is_file() and receipt.is_file() and json.loads(receipt.read_text())["sha256"]==sha256_file(target): print("RESUME",target.name); continue
                        margins=[]; ids=[]
                        for item in items:
                            if dataset=="ValuePrism":
                                messages=messages_for(str(item.situation),str(item.text),mapping,scheme,"joint",m1_prompt,model_id=spec["id"]); pids=list(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)); candidates=(mapping.supports,mapping.opposes); identifier=str(item.row_id)
                            else:
                                rendered=TEMPLATES["EXPLICIT_RELATION"].format(source=item.source_text,target=item.target_text,mapping_instruction=mapping.instruction); pids=None; candidates=(mapping.positive_token,mapping.negative_token); identifier=item.row_id
                            vector=np.asarray(vectors[name],dtype=np.float32)
                            delta=vector*(float(scales[name])*dose/float(np.dot(vector,vector)))
                            result=score_ids_and_extract(model,tokenizer,pids,candidates,spec["layer"],delta) if pids is not None else score_and_extract(model,tokenizer,rendered,candidates,spec["layer"],delta)
                            margins.append(result["candidate_log_probabilities"][candidates[0]]-result["candidate_log_probabilities"][candidates[1]]); ids.append(identifier)
                        part=target.with_suffix(".part.npz"); np.savez_compressed(part,row_ids=np.asarray(ids),semantic_margin=np.asarray(margins,dtype=np.float32),dose=np.asarray(dose)); part.replace(target); write_json(receipt,{"sha256":sha256_file(target),"complete":True,"dataset":dataset,"mapping":mapping_id,"direction":name,"dose":dose})
        del model,tokenizer; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass
        write_json(persistent/actor/"intervention_complete.json",{"status":"COMPLETE","directions_sha256":sha256_file(directions),"batch_size":1})
        return 0
    if mode=="ANALYZE_CPU":
        from geometry_endorsement.direction_identity.analysis_runtime import analyze
        analyze(persistent,rows,vp_rows,data,cfg)
        return 0
    if mode=="BUILD_FINAL_REPORT":
        from geometry_endorsement.direction_identity.analysis_runtime import build_reports
        build_reports(persistent)
        return 0
    raise RuntimeError(f"Unsupported runtime mode: {mode}")

if __name__=="__main__": raise SystemExit(main())
