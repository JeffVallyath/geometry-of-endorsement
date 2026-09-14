"""CPU-only replay of retained representation claims from saved scientific records.

Loads selected pure function definitions by AST, not their model-loading modules.
The output records source hashes, exact function line spans and every comparison.
"""
from __future__ import annotations

import ast
import argparse
import collections
import dataclasses
import gzip
import hashlib
import io
import json
import types
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from .common import ROOT as REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT / "reproducibility/representation"
SOURCES = []
CHECKS = []


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    path = Path(path)
    if not path.exists() and path.with_name(path.name + ".gz").exists():
        path = path.with_name(path.name + ".gz")
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(data)


def definitions(path, names, env=None):
    path = Path(path)
    data = path.read_bytes()
    tree = ast.parse(data.decode("utf-8-sig"))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f"pure function selection incomplete: {path}")
    ns = {"np": np, "pd": pd, "Path": Path, "Any": Any, "dataclass": dataclasses.dataclass,
          "asdict": dataclasses.asdict, "defaultdict": collections.defaultdict, "collections": collections,
          "__name__": "__main__", **(env or {})}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    SOURCES.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest(data), "functions": {n.name: [n.lineno, n.end_lineno] for n in nodes}})
    return ns


def compare(name, got, expected, atol=2e-10):
    a, b = np.asarray(got, dtype=float), np.asarray(expected, dtype=float)
    error = float(np.max(np.abs(a-b))) if a.size else 0.0
    passed = a.shape == b.shape and bool(np.allclose(a, b, rtol=0, atol=atol, equal_nan=True))
    CHECKS.append({"claim": name, "max_absolute_error": error, "tolerance": atol, "passed": passed})
    return passed


def triple(x, center="mean"):
    return [x[center], x["ci_low"], x["ci_high"]]


def fragility():
    root = ROOT / "rewrite_fragility"
    result = {}
    source = ROOT/"source/claim2_diagnostics/predictor_models.py"
    pure = definitions(source, ["item_bootstrap_metric_delta"])
    for actor in ("llama", "gemma"):
        pred = pd.read_csv(root/actor/"nested_predictions.csv") if actor == "llama" else pd.read_parquet(root/actor/"nested_predictions.parquet")
        pred = pred[pred.population == "representative_primary"]
        avg = pred.groupby(["base_item_id", "model"], as_index=False).agg(k_i=("k_i", "first"), n_i=("n_i", "first"), predicted_probability=("predicted_probability", "mean"))
        models = [avg[avg.model == m] for m in ("M2", "M3")]
        losses = []
        for df in models:
            p = np.clip(df.predicted_probability.to_numpy(), 1e-8, 1-1e-8)
            k, n = df.k_i.to_numpy(), df.n_i.to_numpy()
            losses.append(float(-(k*np.log(p)+(n-k)*np.log(1-p)).sum()/n.sum()))
        if actor == "llama":
            # Original saved bootstrap array is itself a replayable uncertainty witness.
            boot = np.load(root/actor/"bootstrap.npy", allow_pickle=False)
            expected = load(root/actor/"summary.json")
            want = [expected["delta_log_loss_m2_minus_m3"]["point_estimate"], *expected["delta_log_loss_m2_minus_m3"]["interval_95"]]
            compare("fragility/llama/M2", losses[0], expected["models"]["M2"]["binomial_log_loss_per_rephrasing"])
            compare("fragility/llama/M3", losses[1], expected["models"]["M3"]["binomial_log_loss_per_rephrasing"])
        else:
            config = yaml.safe_load((root/actor/"config.yaml").read_text())
            seed = config["inference"]["random_seed"] + 1000 + sum(map(ord, "M3"))
            boot = pure["item_bootstrap_metric_delta"](*models, config["inference"]["bootstrap_replicates"], seed)
            ladder = pd.read_csv(root/actor/"predictor_ladder.csv")
            row = ladder[(ladder.population=="representative_primary") & (ladder.model=="M3")].iloc[0]
            want = [row.delta_log_loss_vs_M2,row.delta_vs_M2_ci_low,row.delta_vs_M2_ci_high]
        got = [losses[0]-losses[1], *np.quantile(boot,[.025,.975])]
        compare(f"fragility/{actor}/increment",got,want)
        result[actor] = {"baseline_logloss":losses[0],"augmented_logloss":losses[1],"increment_ci":list(map(float,got)),"bootstrap_draws":len(boot),"qualification":"no reliable positive scalar increment"}
    return result


def full_state():
    root=ROOT/"full_state_report/replication"
    source=ROOT/"full_state_report/source/rrrd_v1_no_inference.py"
    boot_class=definitions(source,["WorldBootstrap"])["WorldBootstrap"]
    output={}
    for actor in ("llama","gemma"):
        expected=load(root/f"{actor}.json")
        for pop,short in (("primary_replication","primary"),("secondary_board_population","secondary")):
            df=pd.read_csv(root/f"{actor}_{short}_predictions.csv")
            y=df.label.to_numpy()
            loss=[]
            for col in ("L0_prob","L1_prob"):
                p=np.clip(df[col].to_numpy(),1e-7,1-1e-7)
                loss.append(-(y*np.log(p)+(1-y)*np.log1p(-p)))
            units=pd.Series(loss[0]-loss[1],index=df.unit_id).groupby(level=0,sort=True).mean().to_dict()
            got=boot_class(sorted(units),10000,2026090217).summary(units)
            want=expected["populations"][pop]["primary"]["L"]["delta_logloss"]
            compare(f"full_state/{actor}/{pop}",triple(got,"point"),triple(want,"point"))
            output[f"{actor}/{pop}"]=got|{"rows":len(df),"qualification":"prospectively frozen internal replication on pre-existing development rows; not untouched confirmation"}
    return output


def fingerprint():
    source=ROOT/"causal_fingerprint/source"
    math=definitions(source/"numerical_resolution.py",["Diagnostics","diagnostics"])
    rel=definitions(source/"reliability_v2.py",["pearson","per_direction_fingerprint_similarity","causal_gram_reliability"])
    root=ROOT/"causal_fingerprint"
    families={"relation":slice(0,8),"language":slice(8,12),"sentiment":slice(12,20),"truth":slice(20,28),"token":slice(28,32),"random":slice(32,40)}
    mappings=[types.SimpleNamespace(mapping_id=m) for m in ("MAPPING_AB_STANDARD","MAPPING_AB_REVERSED","MAPPING_12_STANDARD","MAPPING_12_REVERSED")]
    output={}
    for actor in ("llama","gemma"):
        rows=[]
        for path in sorted((root/f"scores/{actor}").glob("*.npz")):
            with np.load(path,allow_pickle=False) as z:rows.append({k:z[k] for k in z.files})
        assert len(rows)==256
        def verified(*args):
            assert len({(str(r['item_id']),str(r['template']),str(r['mapping'])) for r in rows})==256
        env={"diagnostics":math["diagnostics"],"per_direction_fingerprint_similarity":rel["per_direction_fingerprint_similarity"],"causal_gram_reliability":rel["causal_gram_reliability"],"FAMILY_SLICES":families,"MAPPINGS":mappings,"_verify_actor_completion":verified,"_load_stage_rows":lambda ignored:rows}
        fn=definitions(source/"runtime_causal_fingerprint_v2.py",["finalize_actor_validation"],env)["finalize_actor_validation"]
        got=fn(root,actor);want=load(root/f"{actor}_validation.json")
        for key in ("correlation","sign_agreement","median_absolute_relative_error","median_per_direction_fingerprint_similarity","causal_gram_reliability","mapping_family_agreement","semantic_token_contrast_correlation","random_direction_reliability"):
            compare(f"fingerprint/{actor}/{key}",got[key],want[key])
        for family in got["family_fingerprint_similarity"]:
            compare(f"fingerprint/{actor}/{family}",got["family_fingerprint_similarity"][family],want["family_fingerprint_similarity"][family])
        output[actor]=got
    output["qualification"]="Frozen cross-model instrument INVALID: Gemma fails cross-item reliability. Both random-direction numerical controls fail; do not infer causal compression or its absence."
    return output


def counterfactual():
    root=ROOT/"counterfactual_fidelity"
    source=root/"source"
    env={"CONSEQUENCE_QUERIES":("Q2","Q3","Q4","Q5","Q6"),"FOCAL_QUERIES":("Q2","Q3","Q4"),"GROUP_WEIGHTS":{"Q2":1/3,"Q3":1/3,"Q4":1/3,"Q5":.5,"Q6":.5},"BOOTSTRAP_REPLICATES":10000,"BOOTSTRAP_SEED":202609041720}
    math=definitions(source/"crif_v1_common.py",["group_weights","squared_distance","csr","bootstrap_indices","bootstrap_mean"],env)
    C=types.SimpleNamespace(**env,QUERIES=("Q1","Q2","Q3","Q4","Q5","Q6"),TAUS=(.25,.5,.75,1.),LOGIT_CONTROL="MATCHED_FINAL_LOGIT_CONTROL",**{k:math[k] for k in ("csr","bootstrap_indices","bootstrap_mean")})
    fit=definitions(source/"stcrf_v2_fit.py",["natural_signature"],{"C":C})
    funcs=definitions(source/"stcrf_v2_analysis.py",["world_signatures","applied_mask","csr_for"],{"C":C,"F":types.SimpleNamespace(natural_signature=fit["natural_signature"])})
    summaries=[json.loads(line) for line in gzip.open(root/"scores/final_gemma.jsonl.gz","rt")]
    summaries.sort(key=lambda s:s["world_id"])
    assert len(summaries)==96 and len({s['world_id'] for s in summaries})==96
    expected=load(root/"gemma_analysis.json")
    freeze=load(root/"analysis_freeze.json")
    scales=np.asarray(freeze["actors"]["gemma"]["scales_Q2_to_Q6"])
    output={}
    for fmt,real,label in (("F2","A","primary"),("F2","B","cross_template"),("F1","A","mapped_format")):
        sigs=[funcs["world_signatures"](s,fmt,real)|{"world_id":s["world_id"]} for s in summaries]
        cache={}
        for name,record in expected["csr"].items():
            wanted=record.get("applied" if label=="primary" else label)
            if wanted is None or name in {"NATURAL","NO_INTERVENTION"}:continue
            arm,tau=name.split("@") if "@" in name else (name,"")
            mask=funcs["applied_mask"](summaries,fmt,arm,tau)
            # Whole-state patch lacks an arm calibration row; its complete query records are the witness.
            if arm=="FULL_STATE_SOURCE_PATCH":mask=np.ones(96,dtype=bool)
            got=funcs["csr_for"](sigs,(arm,tau),scales,mask,cache)
            if got is None:
                CHECKS.append({"claim":f"counterfactual/{label}/{name}","passed":False,"reason":"no complete applied score rows"});continue
            for metric in ("csr","csr_focal","csr_invariance"):
                compare(f"counterfactual/{label}/{name}/{metric}",triple(got[metric]),triple(wanted[metric]))
            output[f"{label}/{name}"]={k:got[k] for k in ("csr","csr_focal","csr_invariance","n","mean_signature")}
    hits={}
    for arm in ("ORIGINAL_FROZEN_DIM","REGIME_MATCHED_DIM"):
        rows=[s['arms']['F2'][arm]['applied']['0.75'] for s in summaries]
        hits[arm]={"within_tolerance":float(np.mean([r['within_tolerance'] for r in rows])),"hard_answer_counterfactual":float(np.mean([r['hard_answer_counterfactual'] for r in rows]))}
        for k,v in hits[arm].items():compare(f"counterfactual/hits/{arm}/{k}",v,expected['dose'][arm]['0.75']['criteria'][k])
    output['hits']=hits
    output['qualification']='Gemma rank-1 direct-answer success does not reproduce semantic consequences; Llama natural-reference qualification fails. Regime-matched and original frozen directions remain distinct.'
    return output


def specificity():
    root=ROOT/"relation_transfer"
    fn=definitions(root/"runtime_executed.py",["_paired_slopes"])["_paired_slopes"]
    expected=load(root/"corrected_analysis.json")
    output={}
    for actor in ("llama","gemma"):
        for dataset in ("ValuePrism","AMPERE++"):
            relation=fn(root/f"scores/{actor}/intervention",dataset,"residual_dim")
            for control in ("factual_true_false","sentiment_valence"):
                comparator=fn(root/f"scores/{actor}/controls/intervention",dataset,control)
                assert set(relation)==set(comparator)
                delta=float(np.mean([relation[i]-comparator[i] for i in sorted(relation)]))
                want=expected['actors'][actor]['control_intervention'][dataset][control]['relation_minus_control_mean_slope']
                compare(f"specificity/{actor}/{dataset}/{control}",delta,want)
                output[f"{actor}/{dataset}/{control}"]=delta
    return output


def transfer():
    root=ROOT/"relation_transfer"
    metadata=load(root/'transfer_scores.metadata.json')
    witness=pd.read_csv(root/'transfer_scores.csv',dtype={'row_id':str,'group_id':str})
    expected=load(root/"analysis.json")
    math=definitions(root/"source/grouped_statistics.py",["interval","cluster_bootstrap_metric"])
    output={}
    assert len(witness)==metadata['rows']
    for ai,actor in enumerate(metadata['actor_order']):
        for di,dataset in enumerate([*metadata['external_dataset_order'],'ValuePrism_PILOT_EVAL']):
            frame=witness[(witness.actor==actor)&(witness.dataset==dataset)].sort_values('row_id')
            assert len(frame)>0 and not frame.row_id.duplicated().any()
            scores=frame.score.to_numpy();y=frame.label.to_numpy();groups=frame.group_id.to_numpy()
            auc=float(roc_auc_score(y,scores))
            want=expected['actors'][actor]['reverse_transfer']['ValuePrism_from_AMPERE++'] if di==3 else expected['actors'][actor]['external_transfer'][dataset]['frozen_valueprism_dim']
            compare(f'transfer/{actor}/{dataset}/auroc',auc,want['semantic_auroc' if di==3 else 'auroc'])
            item={'auroc':auc,'rows':len(frame)}
            if dataset=='AMPERE++' or di==3:
                seed=metadata['base_seed']+(2000+ai+2 if di==3 else 1100+ai*100+di)
                ci=math['cluster_bootstrap_metric'](scores,y,groups,lambda s,l:float(roc_auc_score(l,s)),replicates=metadata['bootstrap_replicates'],seed=seed,confidence=metadata['confidence'])
                wantci=want['semantic_auroc_ci' if di==3 else 'auroc_ci']
                compare(f'transfer/{actor}/{dataset}/ci',[ci['low'],ci['high']],[wantci['low'],wantci['high']])
                item['ci']=ci
            output[f'{actor}/{dataset}']=item
    return output


def equal_norm():
    root=ROOT/'relation_transfer'
    expected=load(root/'equal_norm_results.json')
    contract=load(root/'equal_norm_contract.json')
    witness=pd.read_csv(root/'transfer_scores.csv',dtype={'row_id':str,'group_id':str})
    def dose_file(path):
        with np.load(path,allow_pickle=False) as z:
            return z['row_ids'].astype(str),z['semantic_margin'].astype(float),float(np.asarray(z['dose']).reshape(-1)[0])
    funcs=definitions(root/'source/intervention_normalization.py', ['unit_displacement_norm','equal_norm_central_slope','_folder_for_direction','_direction_matrices','_central_by_row'], {'_load_dose_file':dose_file})
    stats=definitions(root/'source/postmortem_grouped_statistics.py',['_interval','paired_group_bootstrap_mean'])
    output={}
    controls=('factual_true_false','sentiment_valence','physical_token','random_orthogonal')
    for ai,actor in enumerate(('llama','gemma')):
        with np.load(root/f'{actor}_directions.npz',allow_pickle=False) as z:
            vectors={k:z[k].astype(float) for k in ('frozen_dim','residual_dim','physical_token','amperepp_relation','pooled_external')}
            scales=json.loads(str(z['scales_json']))
        basis=np.stack(list(vectors.values()))
        random=np.random.default_rng(20260822).normal(size=len(vectors['frozen_dim']))
        random-=basis.T@np.linalg.pinv(basis@basis.T)@(basis@random)
        random/=np.linalg.norm(random)
        vectors['random_orthogonal']=random;scales['random_orthogonal']=scales['frozen_dim']
        with np.load(root/f'{actor}_control_directions.npz',allow_pickle=False) as z:
            cscales=json.loads(str(z['scales_json'][0]))
            for k in ('factual_true_false','sentiment_valence'):vectors[k]=z[k].astype(float);scales[k]=cscales[k]
        units={k:funcs['unit_displacement_norm'](vectors[k],scales[k]) for k in ('residual_dim',)+controls}
        for k,v in units.items():compare(f'equal_norm/{actor}/displacement/{k}',v,expected['actors'][actor]['scale_audit'][k]['actual_l2_delta_at_plus_one'])
        for di,dataset in enumerate(('ValuePrism','AMPERE++')):
            subset=witness[(witness.actor==actor)&(witness.dataset==('ValuePrism_PILOT_EVAL' if dataset=='ValuePrism' else dataset))]
            groups_by_id=dict(zip(subset.row_id,subset.group_id))
            slopes={k:funcs['_central_by_row'](root/'scores',actor,dataset,k,units[k]) for k in ('residual_dim',)+controls}
            ids=sorted(slopes['residual_dim'])
            assert all(set(slopes[k])==set(ids) for k in controls)
            groups=np.asarray([groups_by_id[i] for i in ids])
            for ci,control in enumerate(controls):
                result={}
                for lane,field,offset in [('per_sd','original_per_sd_relation_minus_control',0),('equal_norm','equal_norm_relation_minus_control',10)]:
                    delta=np.asarray([slopes['residual_dim'][i][lane]-slopes[control][i][lane] for i in ids])
                    got=stats['paired_group_bootstrap_mean'](delta,groups,replicates=contract['statistics']['bootstrap_replicates'],seed=contract['statistics']['base_seed']+ai*100+di*20+ci+offset)
                    want=expected['actors'][actor]['datasets'][dataset]['controls'][control][field]
                    compare(f'equal_norm/{actor}/{dataset}/{control}/{lane}',[got['point'],got['low'],got['high']],[want['point'],want['low'],want['high']])
                    result[lane]=got
                output[f'{actor}/{dataset}/{control}']=result
    output['qualification']='Exploratory equal-L2 re-expression of central +/-1 slopes; mixed result, some rankings reverse; does not upgrade the frozen result.'
    return output


def replay():
    """Recompute all retained representation endpoints from repository evidence."""
    SOURCES.clear()
    CHECKS.clear()
    output = {}
    for name, fn in [('fragility', fragility), ('full_state', full_state),
                     ('fingerprint', fingerprint), ('counterfactual', counterfactual),
                     ('specificity', specificity), ('transfer', transfer), ('equal_norm', equal_norm)]:
        output[name] = fn()
    if not CHECKS or any(not row['passed'] for row in CHECKS):
        failed = [r['claim'] for r in CHECKS if not r['passed']]
        raise ValueError('Representation replay mismatch: ' + ', '.join(failed))
    output.update(formula_sources=list(SOURCES), comparisons=list(CHECKS), all_passed=True)
    return output
