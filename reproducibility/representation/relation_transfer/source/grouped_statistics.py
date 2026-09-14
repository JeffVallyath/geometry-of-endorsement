from __future__ import annotations

import numpy as np


def grouped_bootstrap(values: np.ndarray, groups: np.ndarray, statistic, *, replicates: int,
                      seed: int) -> np.ndarray:
    values, groups = np.asarray(values), np.asarray(groups)
    unique = np.unique(groups); rng = np.random.default_rng(seed); output = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        output.append(float(statistic(values[indices])))
    return np.asarray(output)


def grouped_label_permutation(labels: np.ndarray, groups: np.ndarray, statistic, *,
                              replicates: int, seed: int) -> np.ndarray:
    labels, groups = np.asarray(labels), np.asarray(groups)
    unique = np.unique(groups); rng = np.random.default_rng(seed); output=[]
    group_labels = {group: labels[np.flatnonzero(groups == group)[0]] for group in unique}
    base = np.asarray([group_labels[group] for group in unique])
    for _ in range(replicates):
        shuffled = rng.permutation(base); mapping = dict(zip(unique, shuffled, strict=True))
        output.append(float(statistic(np.asarray([mapping[group] for group in groups]))))
    return np.asarray(output)


def interval(samples: np.ndarray, confidence: float = .95) -> tuple[float, float]:
    alpha=(1-confidence)/2; return tuple(float(x) for x in np.quantile(samples,[alpha,1-alpha]))


def cluster_bootstrap_metric(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                             statistic, *, replicates: int, seed: int,
                             confidence: float = .95) -> dict[str, float]:
    """Resample independent groups and keep every row in each sampled group."""
    scores=np.asarray(scores); labels=np.asarray(labels); groups=np.asarray(groups)
    unique=np.unique(groups); indices={group:np.flatnonzero(groups==group) for group in unique}
    rng=np.random.default_rng(seed); draws=[]
    for _ in range(replicates):
        chosen=rng.choice(unique,size=len(unique),replace=True)
        take=np.concatenate([indices[group] for group in chosen])
        draws.append(float(statistic(scores[take],labels[take])))
    low,high=interval(np.asarray(draws),confidence)
    return {"low":low,"high":high,"replicates":int(replicates)}


def cluster_sign_permutation(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                             statistic, *, replicates: int, seed: int,
                             null_center: float) -> dict[str, float]:
    """Flip the complete binary label vector of each independent group."""
    scores=np.asarray(scores); labels=np.asarray(labels); groups=np.asarray(groups)
    unique,inverse=np.unique(groups,return_inverse=True); rng=np.random.default_rng(seed)
    observed=float(statistic(scores,labels)); null=np.empty(replicates,dtype=float)
    for index in range(replicates):
        signs=rng.choice(np.asarray([-1,1],dtype=np.int8),size=len(unique))
        null[index]=float(statistic(scores,labels*signs[inverse]))
    distance=abs(observed-null_center)
    p=(1+int(np.count_nonzero(np.abs(null-null_center)>=distance)))/(replicates+1)
    low,high=interval(null)
    return {"observed":observed,"p_two_sided":float(p),"null_low":low,"null_high":high,"replicates":int(replicates)}


def grouped_direction_cosine_null(activations: np.ndarray, labels: np.ndarray,
                                  groups: np.ndarray, reference: np.ndarray, *,
                                  replicates: int, seed: int,
                                  batch_size: int = 64) -> dict[str, float]:
    """Refit mean-difference directions under whole-group label sign flips."""
    x=np.asarray(activations,dtype=np.float64); y=np.asarray(labels,dtype=np.float64)
    groups=np.asarray(groups); reference=np.asarray(reference,dtype=np.float64)
    unique,inverse=np.unique(groups,return_inverse=True); group_yx=np.zeros((len(unique),x.shape[1]),dtype=np.float64); group_y=np.zeros(len(unique),dtype=np.float64)
    np.add.at(group_yx,inverse,x*y[:,None]); np.add.at(group_y,inverse,y)
    total_x=x.sum(axis=0); n=float(len(x)); ref=reference/np.linalg.norm(reference)
    rng=np.random.default_rng(seed); values=[]
    for start in range(0,replicates,batch_size):
        count=min(batch_size,replicates-start); signs=rng.choice(np.asarray([-1.,1.]),size=(count,len(unique)))
        signed_y=signs@group_y; signed_yx=signs@group_yx
        n_pos=(n+signed_y)/2.; n_neg=(n-signed_y)/2.
        valid=(n_pos>0)&(n_neg>0); directions=np.full_like(signed_yx,np.nan)
        directions[valid]=(total_x+signed_yx[valid])/(2*n_pos[valid,None])-(total_x-signed_yx[valid])/(2*n_neg[valid,None])
        norms=np.linalg.norm(directions,axis=1); values.extend(((directions@ref)/norms).tolist())
    values=np.asarray(values,dtype=float); values=values[np.isfinite(values)]; low,high=interval(values)
    return {"low":low,"high":high,"absolute_95_envelope":float(np.quantile(np.abs(values),.95)),"replicates":int(len(values))}
