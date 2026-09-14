"""Frozen CPU analysis and post-inference report builder for the identity study."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .artifact_io import sha256_file, write_json, write_text
from .decision_logic import PERMITTED_CLAIMS, decide
from .grouped_statistics import cluster_bootstrap_metric, cluster_sign_permutation, grouped_direction_cosine_null
from .output_channel import cosine


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    binary=(np.asarray(labels)==1).astype(int)
    return float(roc_auc_score(binary,np.asarray(scores))) if len(np.unique(binary))==2 else .5


def _separation(scores: np.ndarray, labels: np.ndarray) -> float:
    scores=np.asarray(scores,dtype=float); labels=np.asarray(labels); sigma=float(scores.std(ddof=1))
    return float((scores[labels==1].mean()-scores[labels==-1].mean())/sigma) if sigma else 0.


def _safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    return cosine(left,right) if np.linalg.norm(left)>0 and np.linalg.norm(right)>0 else 0.


def _clean_board_metric(value: dict[str, Any]) -> dict[str, Any]:
    return {key:item for key,item in value.items() if not key.startswith("_")}


def _verified_npz_files(folder: Path, expected: int) -> list[Path]:
    files=sorted(folder.glob("*.npz"))
    if len(files)!=expected: raise RuntimeError(f"Completeness gate failed for {folder}: {len(files)} != {expected}")
    for path in files:
        receipt=path.with_suffix(".json")
        if not receipt.is_file() or json.loads(receipt.read_text())["sha256"]!=sha256_file(path):
            raise RuntimeError(f"Hash receipt failed: {path}")
    return files


def _condition(path: Path, vector: np.ndarray | None) -> dict[str, Any]:
    with np.load(path,allow_pickle=False) as z:
        score=z["semantic_margin"].astype(float) if vector is None else z["activations"].astype(float)@vector
        return {"ids":z["row_ids"].astype(str),"scores":score,"semantic":z["semantic"].astype(int),
                "physical":z["physical"].astype(int),"activations":z["activations"].astype(float)}


def _aggregate(files: list[Path], vector: np.ndarray | None) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    scores: dict[str,list[float]]=defaultdict(list); labels={}; physical: dict[str,list[int]]=defaultdict(list)
    for path in files:
        item=_condition(path,vector)
        for rid,score,semantic,token in zip(item["ids"],item["scores"],item["semantic"],item["physical"],strict=True):
            scores[rid].append(float(score)); labels[rid]=int(semantic); physical[rid].append(int(token))
    ids=np.asarray(sorted(scores)); return ids,np.asarray([np.mean(scores[rid]) for rid in ids]),np.asarray([labels[rid] for rid in ids]),np.asarray([np.mean(physical[rid]) for rid in ids])


def _aggregate_activations(files: list[Path]) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    vectors: dict[str,list[np.ndarray]]=defaultdict(list); labels={}
    for path in files:
        item=_condition(path,None)
        for rid,activation,label in zip(item["ids"],item["activations"],item["semantic"],strict=True):
            vectors[rid].append(activation); labels[rid]=int(label)
    ids=np.asarray(sorted(vectors)); return ids,np.stack([np.mean(vectors[rid],axis=0) for rid in ids]),np.asarray([labels[rid] for rid in ids])


def _uncertainty(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray, cfg: dict[str,Any], seed: int) -> dict[str,Any]:
    reps=int(cfg["statistics"]["bootstrap_replicates"]); perms=int(cfg["statistics"]["permutation_replicates"])
    ci=cluster_bootstrap_metric(scores,labels,groups,_auc,replicates=reps,seed=seed,confidence=float(cfg["statistics"]["interval"]))
    permutation=cluster_sign_permutation(scores,labels,groups,_auc,replicates=perms,seed=seed+1,null_center=.5)
    return {"auroc":_auc(scores,labels),"standardized_separation":_separation(scores,labels),"auroc_ci":ci,"group_sign_permutation":permutation,"n":int(len(scores)),"groups":int(len(np.unique(groups)))}


def _external_summary(files: list[Path], vector: np.ndarray | None, test_ids: set[str], group_by_id: dict[str,str], cfg: dict[str,Any], seed: int) -> dict[str,Any]:
    ids,scores,labels,_=_aggregate(files,vector); mask=np.asarray([rid in test_ids for rid in ids]); ids=ids[mask]; scores=scores[mask]; labels=labels[mask]; groups=np.asarray([group_by_id[rid] for rid in ids])
    result=_uncertainty(scores,labels,groups,cfg,seed); conditions={}; predictions: dict[str,list[int]]=defaultdict(list); token_rates={}
    for index,path in enumerate(files):
        item=_condition(path,vector); take=np.asarray([rid in test_ids for rid in item["ids"]]); cids=item["ids"][take]; cs=item["scores"][take]; cy=item["semantic"][take]
        conditions[path.stem]={"auroc":_auc(cs,cy),"standardized_separation":_separation(cs,cy),"n":int(len(cs))}
        for rid,value in zip(cids,cs,strict=True): predictions[rid].append(1 if value>=0 else -1)
        mapping=path.stem.split("__")[-1]; physical_first=(cs>=0) if mapping.endswith("STANDARD") else (cs<0); token_rates[path.stem]=float(np.mean(physical_first))
    result["conditions"]=conditions; result["mapping_consistency_rate"]=float(np.mean([abs(np.mean(value)) for value in predictions.values()])); result["physical_first_choice_rates"]=token_rates
    mapping_auc={}; template_auc={}
    for mapping in ("MAPPING_AB_STANDARD","MAPPING_AB_REVERSED","MAPPING_12_STANDARD","MAPPING_12_REVERSED"):
        subset=[path for path in files if path.stem.endswith(mapping)]; _,s,y,_=_aggregate(subset,vector); take=np.asarray([rid in test_ids for rid in _aggregate(subset,vector)[0]]); mapping_auc[mapping]=_auc(s[take],y[take])
    for template in ("EXPLICIT_RELATION","LEXICALLY_ABLATED_RELATION"):
        subset=[path for path in files if f"__{template}__" in path.stem]; sid,s,y,_=_aggregate(subset,vector); take=np.asarray([rid in test_ids for rid in sid]); template_auc[template]=_auc(s[take],y[take])
    result["mapping_aggregate_auroc"]=mapping_auc; result["template_aggregate_auroc"]=template_auc; result["template_auroc_gap"]=float(abs(template_auc["EXPLICIT_RELATION"]-template_auc["LEXICALLY_ABLATED_RELATION"]))
    return result


def _board_summary(select_files: list[Path], eval_files: list[Path], vector: np.ndarray, vp_rows: dict[str,pd.DataFrame], data: Path, cfg: dict[str,Any], seed: int) -> dict[str,Any]:
    from geometry_of_truth.m1.metrics import board_ci, board_metrics, boards_from_manifest
    from geometry_of_truth.m1.support.metrics import fit_scaler
    select_ids,select_scores,_,_=_aggregate(select_files,vector); eval_ids,eval_scores,eval_labels,_=_aggregate(eval_files,vector)
    select_frame=vp_rows["pilot_select"].set_index("row_id",drop=False).loc[select_ids].reset_index(drop=True)
    eval_frame=vp_rows["pilot_eval"].set_index("row_id",drop=False).loc[eval_ids].reset_index(drop=True)
    board_manifest=pd.read_csv(data/"valueprism-pilot-manifests"/"pilot_eval_boards.csv",dtype=str); boards=boards_from_manifest(board_manifest,eval_frame)
    metric=board_metrics(boards,eval_scores,fit_scaler(select_scores,"pilot_select")); reps=int(cfg["statistics"]["bootstrap_replicates"]); perms=int(cfg["statistics"]["permutation_replicates"])
    ci=board_ci(metric,field="interaction_contrast",replicates=reps,confidence_level=float(cfg["statistics"]["interval"]),seed=seed)
    values=np.asarray([row["interaction_contrast"] for row in metric["_per_board"]]); rng=np.random.default_rng(seed+1); null=np.mean(values[None,:]*rng.choice(np.asarray([-1.,1.]),size=(perms,len(values))),axis=1); observed=float(values.mean()); p=(1+int(np.count_nonzero(np.abs(null)>=abs(observed))))/(perms+1)
    return {**_clean_board_metric(metric),"interaction_ci":ci,"board_sign_permutation":{"p_two_sided":float(p),"replicates":perms},"semantic_auroc":_auc(eval_scores,eval_labels),"semantic_auroc_ci":cluster_bootstrap_metric(eval_scores,eval_labels,eval_frame["board_id"].astype(str).to_numpy(),_auc,replicates=reps,seed=seed+2)}


def _factorial(files: list[Path], directions: dict[str,np.ndarray]) -> dict[str,Any]:
    output={}
    for name,vector in directions.items():
        x=[]; sem=[]; token=[]
        for path in files:
            item=_condition(path,vector); x.append(item["scores"]); sem.append(item["semantic"]); token.append(item["physical"])
        score=np.concatenate(x); semantic=np.concatenate(sem); physical=np.concatenate(token); design=np.column_stack([np.ones(len(score)),semantic,physical,semantic*physical]); beta=np.linalg.lstsq(design,score,rcond=None)[0]
        output[name]={"semantic_auroc":_auc(score,semantic),"physical_token_auroc":_auc(score,physical),"factorial_coefficients":{"intercept":float(beta[0]),"semantic":float(beta[1]),"physical_token":float(beta[2]),"semantic_by_token":float(beta[3])}}
    return output


def _intervention_summary(folder: Path, dataset: str, groups_by_id: dict[str,str], cfg: dict[str,Any], seed: int) -> tuple[dict[str,Any],dict[str,dict[str,float]]]:
    directions=("frozen_dim","residual_dim","physical_token","amperepp_relation","pooled_external","random_orthogonal"); result={}; slopes_by_direction={}
    for direction in directions:
        mapping_rows={}; combined: dict[str,list[float]]=defaultdict(list)
        for mapping in ("MAPPING_AB_STANDARD","MAPPING_AB_REVERSED"):
            files=sorted(folder.glob(f"{dataset}__{mapping}__{direction}__*.npz"),key=lambda p:float(p.stem.split("__")[-1])); doses=[]; matrices=[]; ids=None
            for path in files:
                with np.load(path,allow_pickle=False) as z: doses.append(float(z["dose"])); matrices.append(z["semantic_margin"].astype(float)); current=z["row_ids"].astype(str)
                if ids is None: ids=current
                elif not np.array_equal(ids,current): raise RuntimeError(f"Intervention row drift: {path}")
            doses_a=np.asarray(doses); matrix=np.stack(matrices); slopes=np.polyfit(doses_a,matrix,1)[0]; zero=matrix[np.flatnonzero(doses_a==0)[0]]; order=np.argsort(doses_a); sorted_matrix=matrix[order]; diffs=np.diff(sorted_matrix,axis=0); monotonic=float(np.mean(np.all(diffs>=0,axis=0)|np.all(diffs<=0,axis=0)))
            symmetry=[]; hard={}
            for dose in sorted(set(abs(value) for value in doses if value)):
                if dose in doses and -dose in doses:
                    plus=matrix[doses.index(dose)]-zero; minus=matrix[doses.index(-dose)]-zero; symmetry.append(float(np.mean(np.abs(plus+minus)/(np.abs(plus)+np.abs(minus)+1e-12))))
            for dose,row in zip(doses,matrix,strict=True): hard[f"{dose:+.1f}"]=float(np.mean(np.sign(row)!=np.sign(zero)))
            physical_sign=1. if mapping.endswith("STANDARD") else -1.; mapping_rows[mapping]={"mean_semantic_slope":float(np.mean(slopes)),"mean_physical_token_slope":float(np.mean(slopes*physical_sign)),"monotonic_row_rate":monotonic,"sign_symmetry_relative_error":float(np.mean(symmetry)) if symmetry else 0.,"hard_verdict_change_rate":hard,"rows":int(len(ids))}
            for rid,value in zip(ids,slopes,strict=True): combined[rid].append(float(value))
        slopes_by_direction[direction]={rid:float(np.mean(values)) for rid,values in combined.items()}; result[direction]=mapping_rows
    left=slopes_by_direction["residual_dim"]; right=slopes_by_direction["physical_token"]; ids=np.asarray(sorted(set(left)&set(right))); delta=np.asarray([left[rid]-right[rid] for rid in ids]); groups=np.asarray([groups_by_id[rid] for rid in ids]); reps=int(cfg["statistics"]["bootstrap_replicates"])
    ci=cluster_bootstrap_metric(delta,np.zeros(len(delta)),groups,lambda values,_labels:float(np.mean(values)),replicates=reps,seed=seed)
    result["primary_paired_residual_minus_physical"]={"mean_slope_difference":float(delta.mean()),"grouped_ci":ci,"rows":int(len(delta)),"groups":int(len(np.unique(groups)))}; result["no_intervention"]={"definition":"dose_zero_reference_in_every_active_direction_family"}
    return result,slopes_by_direction


def analyze(persistent: Path, rows: dict[str,list[Any]], vp_rows: dict[str,pd.DataFrame], data: Path, cfg: dict[str,Any]) -> dict[str,Any]:
    seed=int(cfg["seed"]); thresholds=cfg["decision_thresholds"]; actor_results={}
    for actor_index,actor in enumerate(("llama","gemma")):
        extraction=persistent/actor/"extraction"; intervention=persistent/actor/"intervention"; extraction_files=_verified_npz_files(extraction,32); _verified_npz_files(intervention,136)
        if not (persistent/actor/"intervention_complete.json").is_file(): raise RuntimeError(f"Missing intervention completion: {actor}")
        direction_path=persistent/actor/"directions.npz"; direction_receipt=direction_path.with_suffix(".json")
        if not direction_path.is_file() or not direction_receipt.is_file() or json.loads(direction_receipt.read_text())["sha256"]!=sha256_file(direction_path): raise RuntimeError(f"Direction receipt failed: {actor}")
        if json.loads((persistent/actor/"intervention_complete.json").read_text()).get("directions_sha256")!=sha256_file(direction_path): raise RuntimeError(f"Intervention used a different direction bundle: {actor}")
        with np.load(direction_path,allow_pickle=False) as z: directions={key:z[key] for key in z.files if key!="scales_json"}
        vp_select=[p for p in extraction_files if p.name.startswith("ValuePrism_PILOT_SELECT")]; vp_eval=[p for p in extraction_files if p.name.startswith("ValuePrism_PILOT_EVAL")]
        output_channel={key:_board_summary(vp_select,vp_eval,directions[key],vp_rows,data,cfg,seed+actor_index*100+index) for index,key in enumerate(("frozen_dim","token_component","residual_dim","semantic_diagnostic","physical_token"))}
        output_channel["factorial"]=_factorial(vp_eval,{key:directions[key] for key in ("frozen_dim","residual_dim","semantic_diagnostic","physical_token")})
        output_channel["geometry"]={"token_component_cosine":_safe_cosine(directions["frozen_dim"],directions["token_component"]),"residual_retained_norm_fraction":float(np.linalg.norm(directions["residual_dim"])/np.linalg.norm(directions["frozen_dim"])),"cosines":{key:_safe_cosine(directions["frozen_dim"],directions[key]) for key in ("semantic_diagnostic","physical_token","physical_token_ab","physical_token_12","secondary_unembedding_ab","secondary_unembedding_12")},"secondary_unembedding_caveat":"Final-head row contrasts are diagnostic only and are not identified with the frozen intermediate representation."}
        external={}; reverse={}; nulls={}
        for dataset_index,(dataset,direction_key) in enumerate((("AMPERE++","amperepp_relation"),("AbstRCT","abstrct_relation"),("US2016","us2016_relation"))):
            files=[p for p in extraction_files if p.name.startswith(dataset+"__")]; group_by={row.row_id:row.group_id for row in rows[dataset]}; test_ids={row.row_id for row in rows[dataset] if row.split=="test"}; train_ids={row.row_id for row in rows[dataset] if row.split=="train"}
            native=_external_summary(files,None,test_ids,group_by,cfg,seed+1000+actor_index*100+dataset_index); frozen=_external_summary(files,directions["frozen_dim"],test_ids,group_by,cfg,seed+1100+actor_index*100+dataset_index); own=_external_summary(files,directions[direction_key],test_ids,group_by,cfg,seed+1200+actor_index*100+dataset_index); pooled=_external_summary(files,directions["pooled_external"],test_ids,group_by,cfg,seed+1300+actor_index*100+dataset_index)
            floor=float(thresholds["fixed_token_collapse_floor"]); ceiling=float(thresholds["fixed_token_collapse_ceiling"]); native["fixed_token_collapse"]=any(rate<=floor or rate>=ceiling for rate in native["physical_first_choice_rates"].values()); native["competence_established"]=native["auroc_ci"]["low"]>.5 and native["mapping_consistency_rate"]>=float(thresholds["mapping_consistency_minimum"]) and not native["fixed_token_collapse"]
            external[dataset]={"native":native,"frozen_valueprism_dim":frozen,"within_task_relation":own,"pooled_external":pooled}
            ids,x,y=_aggregate_activations(files); mask=np.asarray([rid in train_ids for rid in ids]); groups=np.asarray([group_by[rid] for rid in ids[mask]]); nulls[dataset]=grouped_direction_cosine_null(x[mask],y[mask],groups,directions["frozen_dim"],replicates=int(cfg["statistics"]["permutation_replicates"]),seed=seed+1400+actor_index*100+dataset_index)
            reverse[dataset]={"direction_cosine_with_frozen_dim":_safe_cosine(directions["frozen_dim"],directions[direction_key]),"grouped_null_cosine":nulls[dataset]}
        reverse["ValuePrism_from_AMPERE++"]=_board_summary(vp_select,vp_eval,directions["amperepp_relation"],vp_rows,data,cfg,seed+2000+actor_index)
        reverse["ValuePrism_from_pooled_external"]=_board_summary(vp_select,vp_eval,directions["pooled_external"],vp_rows,data,cfg,seed+2100+actor_index)
        vp_groups=vp_rows["pilot_eval"].set_index("row_id")["board_id"].astype(str).to_dict(); amp_groups={row.row_id:row.group_id for row in rows["AMPERE++"]}
        vp_intervention,_=_intervention_summary(intervention,"ValuePrism",vp_groups,cfg,seed+3000+actor_index); amp_intervention,_=_intervention_summary(intervention,"AMPERE++",amp_groups,cfg,seed+3100+actor_index)
        actor_results[actor]={"output_channel":output_channel,"external_transfer":external,"reverse_transfer":reverse,"intervention":{"ValuePrism":vp_intervention,"AMPERE++":amp_intervention},"controls":{"factual":"UNAVAILABLE_NO_FROZEN_VECTOR_IN_SAME_ACTOR_LAYER_SPACE","sentiment":"SENTIMENT_ALTERNATIVE_UNRESOLVED","random_orthogonal":"EXECUTED_IN_INTERVENTION_STAGE","no_intervention":"DOSE_ZERO_REFERENCE"}}
    amp_established={actor:actor_results[actor]["external_transfer"]["AMPERE++"]["frozen_valueprism_dim"]["auroc_ci"]["low"]>.5 and actor_results[actor]["external_transfer"]["AMPERE++"]["frozen_valueprism_dim"]["group_sign_permutation"]["p_two_sided"]<=float(thresholds["alpha"]) for actor in actor_results}
    residual_signal={actor:actor_results[actor]["output_channel"]["residual_dim"]["interaction_ci"]["low"]>0 and actor_results[actor]["output_channel"]["residual_dim"]["semantic_auroc_ci"]["low"]>.5 for actor in actor_results}
    reverse_established={actor:actor_results[actor]["reverse_transfer"]["ValuePrism_from_AMPERE++"]["interaction_ci"]["low"]>0 and actor_results[actor]["reverse_transfer"]["ValuePrism_from_AMPERE++"]["board_sign_permutation"]["p_two_sided"]<=float(thresholds["alpha"]) for actor in actor_results}
    evidence={"models_materially_disagree":len(set(amp_established.values()))>1,"native_external_competence":all(value["external_transfer"][dataset]["native"]["competence_established"] for value in actor_results.values() for dataset in ("AMPERE++","AbstRCT","US2016")),"sentiment_alternative_unresolved":True,"broad_intervals":any((value["external_transfer"]["AMPERE++"]["frozen_valueprism_dim"]["auroc_ci"]["high"]-value["external_transfer"]["AMPERE++"]["frozen_valueprism_dim"]["auroc_ci"]["low"])>float(thresholds["broad_primary_auroc_interval_width"]) for value in actor_results.values()),"semantic_mapping_invariance":all(min(value["external_transfer"]["AMPERE++"]["frozen_valueprism_dim"]["mapping_aggregate_auroc"].values())>.5 for value in actor_results.values()),"strong_template_dependence":any(value["external_transfer"]["AMPERE++"]["frozen_valueprism_dim"]["template_auroc_gap"]>float(thresholds["strong_template_auroc_gap"]) for value in actor_results.values()),"residual_checkerboard_signal":all(residual_signal.values()),"zero_shot_amperepp_both_models":all(amp_established.values()),"secondary_directionally_consistent":all(any(value["external_transfer"][dataset]["frozen_valueprism_dim"]["auroc"]>.5 for dataset in ("AbstRCT","US2016")) for value in actor_results.values()),"external_within_task_valid":all(value["external_transfer"][dataset]["within_task_relation"]["auroc_ci"]["low"]>.5 for value in actor_results.values() for dataset in ("AMPERE++","AbstRCT","US2016")),"reverse_transfer_both_models":all(reverse_established.values()),"cosines_exceed_grouped_null":all(abs(value["reverse_transfer"]["AMPERE++"]["direction_cosine_with_frozen_dim"])>value["reverse_transfer"]["AMPERE++"]["grouped_null_cosine"]["absolute_95_envelope"] for value in actor_results.values()),"sentiment_not_complete_explanation":False,"direct_token_explanation_weakened":all(residual_signal.values()),"physical_token_alignment_dominates":all(abs(value["output_channel"]["geometry"]["cosines"]["physical_token"])>abs(value["output_channel"]["geometry"]["cosines"]["semantic_diagnostic"]) for value in actor_results.values())}
    disposition=decide(evidence)
    result={"schema_version":1,"study_id":cfg["study_id"],"actors":actor_results,"decision_evidence":evidence,"terminal_disposition":disposition,"permitted_claim":PERMITTED_CLAIMS[disposition],"sentiment_lane":"SENTIMENT_ALTERNATIVE_UNRESOLVED","model_combination":"reported separately","claim_boundary":"Transfer nulls do not establish moral specificity; output-head contrasts are not intermediate-layer identities."}
    write_json(persistent/"analysis_results.json",result); write_json(persistent/"analysis_status.json",{"status":"ANALYSIS_COMPLETE","actors":["llama","gemma"],"scientific_report_allowed":True,"analysis_sha256":sha256_file(persistent/"analysis_results.json")})
    return result


def build_reports(persistent: Path) -> dict[str,str]:
    status_path=persistent/"analysis_status.json"; results_path=persistent/"analysis_results.json"
    if not status_path.is_file() or not results_path.is_file() or json.loads(status_path.read_text()).get("status")!="ANALYSIS_COMPLETE" or json.loads(status_path.read_text()).get("analysis_sha256")!=sha256_file(results_path): raise RuntimeError("FINAL_REPORT_BLOCKED_UNTIL_ALL_COMPLETENESS_GATES_PASS")
    results=json.loads(results_path.read_text()); disposition=results["terminal_disposition"]
    cross={actor:{"external_transfer":value["external_transfer"],"reverse_transfer":value["reverse_transfer"]} for actor,value in results["actors"].items()}; intervention={actor:value["intervention"] for actor,value in results["actors"].items()}
    common=["Generated only after both frozen actors passed extraction, intervention, and hash-completeness gates.","Llama and Gemma are reported separately. Transfer nulls never establish moral specificity."]
    reports={"CROSS_TASK_TRANSFER_REPORT.md":"# Cross-task transfer report\n\n"+"\n\n".join(common)+"\n\n```json\n"+json.dumps(cross,indent=2,sort_keys=True)+"\n```\n","INTERVENTION_SPECIFICITY_REPORT.md":"# Intervention specificity report\n\n"+"\n\n".join(common)+"\n\n```json\n"+json.dumps(intervention,indent=2,sort_keys=True)+"\n```\n","FINAL_DIRECTION_IDENTITY_REPORT.md":f"# Final relation-direction identity report\n\n{' '.join(common)}\n\n## Terminal disposition\n\n`{disposition}`\n\n{results['permitted_claim']}\n\nSentiment remains unresolved, so the frozen rules prohibit upgrading this disposition.\n\n```json\n{json.dumps(results,indent=2,sort_keys=True)}\n```\n"}
    hashes={}
    for name,text in reports.items(): path=persistent/name; write_text(path,text); hashes[name]=sha256_file(path)
    write_json(persistent/"final_report_receipt.json",{"status":"COMPLETE","analysis_sha256":sha256_file(results_path),"report_sha256":hashes}); return hashes
