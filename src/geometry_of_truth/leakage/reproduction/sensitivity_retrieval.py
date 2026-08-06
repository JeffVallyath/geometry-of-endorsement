"""Deterministic text-only leakage sensitivity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .build_confirmatory_split import manifest_hash, row_id
from .split_stress_test import CON, OPPOSES, SIT, SUPPORTS, VAL, l1_normalize, l2_normalize, load_binary

ROOT = Path(os.environ.get("GEOMETRY_OF_TRUTH_WORKDIR", Path(__file__).resolve().parents[4])).resolve()
DEFAULT_CONFIG = ROOT / "configs" / "valueprism_sensitivity.json"
QUEUE = ROOT / "audits" / "residual_near_duplicate_queue.csv"
OUTPUTS = {
    "U0": ROOT / "manifests" / "strict_U0_current.csv",
    "U1": ROOT / "manifests" / "strict_U1_high_precision_clean.csv",
    "U2": ROOT / "manifests" / "strict_U2_conservative_clean.csv",
    "U3": ROOT / "manifests" / "strict_U3_human_confirmed_clean.csv",
}
ANNOTATION_FIELDS = ["same_identity", "distinct", "uncertain", "notes", "annotator_id", "adjudicated_label"]
IRREGULAR = {"children": "child", "people": "person", "women": "woman", "men": "man",
             "mice": "mouse", "duties": "duty", "rights": "right", "lives": "life"}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def light_lemma(text: str) -> str:
    out = []
    for token in l2_normalize(text).split():
        token = IRREGULAR.get(token, token)
        if len(token) > 5 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
            if len(token) > 3 and token[-1] == token[-2]:
                token = token[:-1]
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("es"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
            token = token[:-1]
        out.append(token)
    return " ".join(out)


def tokens(text: str) -> frozenset[str]:
    return frozenset(x for x in l2_normalize(text).split() if len(x) > 1)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def verify_inputs(config: dict[str, Any]) -> dict[str, str]:
    allowed = {"manifest_train_common.csv", "manifest_strict_train.csv",
               "manifest_strict_test.csv", "manifest_confirmatory.csv"}
    actual = {}
    for name, expected in config["frozen_inputs"].items():
        if name not in allowed:
            raise RuntimeError(f"unapproved input {name}")
        actual[name] = sha_file(ROOT / name)
        if actual[name] != expected:
            raise RuntimeError(f"frozen artifact drift {name}: {actual[name]} != {expected}")
    return actual


def manifest_ids(name: str) -> pd.Series:
    frame = pd.read_csv(ROOT / name, dtype={"row_id": str})
    if list(frame.columns) != ["row_id"] or frame.row_id.duplicated().any():
        raise RuntimeError(f"invalid row manifest {name}")
    return frame.row_id


def load_frames(config: dict[str, Any]):
    df = load_binary().reset_index(drop=True)
    df["row_id"] = [row_id(s, c, v, k) for s, c, v, k in zip(df[SIT], df[CON], df[VAL], df.vrd)]
    duplicate_rows = int(df.row_id.duplicated().sum())
    df = df.drop_duplicates("row_id").reset_index(drop=True)
    if len(df) != config["dataset"]["expected_deduplicated_binary_rows"]:
        raise RuntimeError("dataset count drift")
    tr_ids, te_ids = manifest_ids("manifest_strict_train.csv"), manifest_ids("manifest_strict_test.csv")
    if manifest_hash(tr_ids) != config["frozen_row_hashes"]["strict_train"]:
        raise RuntimeError("strict train row hash drift")
    if manifest_hash(te_ids) != config["frozen_row_hashes"]["strict_test"]:
        raise RuntimeError("strict test row hash drift")
    known = set(df.row_id)
    if set(tr_ids) - known or set(te_ids) - known:
        raise RuntimeError("manifest row does not resolve")
    train = df[df.row_id.isin(set(tr_ids))].copy()
    test = df[df.row_id.isin(set(te_ids))].copy()
    order = {rid: i for i, rid in enumerate(te_ids)}
    test["_order"] = test.row_id.map(order)
    test = test.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    overlap = {
        "row_id": len(set(train.row_id) & set(test.row_id)),
        "situation": len(set(train[SIT]) & set(test[SIT])),
        "exact_consideration": len(set(train[CON]) & set(test[CON])),
        "l1_normalized": len(set(train.l1) & set(test.l1)),
        "l2_prefix_stripped": len(set(train.l2) & set(test.l2)),
        "l3_semantic_cluster": len(set(train.l3) & set(test.l3)),
    }
    for key in ("row_id", "situation", "exact_consideration", "l2_prefix_stripped", "l3_semantic_cluster"):
        if overlap[key]:
            raise RuntimeError(f"frozen overlap contract failed: {key}")
    reproduction = {
        "dataset_binary_rows": len(df), "duplicate_rows_dropped": duplicate_rows,
        "strict_train_rows": len(train), "strict_test_rows": len(test),
        "strict_train_situations": int(train[SIT].nunique()),
        "strict_test_situations": int(test[SIT].nunique()),
        "strict_train_consideration_clusters": int(train.l3.nunique()),
        "strict_test_consideration_clusters": int(test.l3.nunique()),
        "strict_train_row_hash": manifest_hash(tr_ids),
        "strict_test_row_hash": manifest_hash(te_ids),
        "overlaps": overlap, "assertions_pass": True,
    }
    return df, train, test, reproduction


def form_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for form, g in frame.groupby("l2", sort=True):
        rows.append({"l2": str(form), "lemma": light_lemma(str(form)), "l3": int(g.l3.iloc[0])})
    return pd.DataFrame(rows)


def profiles(frame: pd.DataFrame) -> dict[int, dict[str, Any]]:
    out = {}
    for cluster, g in frame.groupby("l3", sort=True):
        counts = g.l2.value_counts()
        out[int(cluster)] = {
            "raw": sorted(str(x) for x in g[CON].unique()),
            "l1": sorted(str(x) for x in g.l1.unique()),
            "l2": sorted(str(x) for x in g.l2.unique()),
            "leader": str(sorted(counts.index, key=lambda x: (-counts[x], str(x)))[0]),
        }
    return out


def nearest(train, test, k: int):
    n = min(k, train.shape[0])
    nn = NearestNeighbors(n_neighbors=n, metric="cosine", algorithm="brute", n_jobs=-1).fit(train)
    distance, index = nn.kneighbors(test)
    return 1.0 - distance, index


def retained_pairs(method: str, score, index, tr_forms: pd.DataFrame, te_forms: pd.DataFrame, keep: int):
    by_test = defaultdict(dict)
    for i in range(len(te_forms)):
        tc = int(te_forms.iloc[i].l3)
        for s, j in zip(score[i], index[i]):
            rc = int(tr_forms.iloc[int(j)].l3)
            by_test[tc][rc] = max(by_test[tc].get(rc, -1.0), float(s))
    out = defaultdict(set)
    for tc, values in by_test.items():
        for rc, _ in sorted(values.items(), key=lambda x: (-x[1], x[0]))[:keep]:
            out[(tc, rc)].add(method)
    return out


def sentence_embeddings(config: dict[str, Any], train_text: list[str], test_text: list[str]):
    os.environ.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                       "TRANSFORMERS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""})
    from sentence_transformers import SentenceTransformer
    r = config["retrieval"]
    model = SentenceTransformer(r["sentence_embedding_model"], revision=r["sentence_embedding_revision"],
                                local_files_only=True, device="cpu")
    all_text = train_text + test_text
    encoded = model.encode(all_text, batch_size=r["batch_size"], convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=True, device="cpu").astype(np.float32)
    return encoded[:len(train_text)], encoded[len(train_text):]


def retrieve(config: dict[str, Any], train: pd.DataFrame, test: pd.DataFrame):
    tr, te = form_table(train), form_table(test)
    all_text, n = tr.l2.tolist() + te.l2.tolist(), len(tr)
    top, keep = config["retrieval"]["search_top_k_forms_per_view"], config["retrieval"]["retain_top_k_clusters_per_view"]
    ngram = tuple(config["retrieval"]["character_ngram_range"])
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram).fit_transform(all_text)
    cs, ci = nearest(char[:n], char[n:], top)
    token = TfidfVectorizer(binary=True, use_idf=False, token_pattern=r"(?u)\b\w+\b").fit_transform(all_text)
    ts, ti = nearest(token[:n], token[n:], top)
    lemmas = tr.lemma.tolist() + te.lemma.tolist()
    lemma = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram).fit_transform(lemmas)
    ls, li = nearest(lemma[:n], lemma[n:], top)
    emb_tr, emb_te = sentence_embeddings(config, tr.l2.tolist(), te.l2.tolist())
    es, ei = nearest(emb_tr, emb_te, top)
    maps = [retained_pairs("char", cs, ci, tr, te, keep),
            retained_pairs("token", ts, ti, tr, te, keep),
            retained_pairs("lemma", ls, li, tr, te, keep),
            retained_pairs("embedding", es, ei, tr, te, keep)]
    candidates = defaultdict(set)
    for mapping in maps:
        for pair, methods in mapping.items():
            candidates[pair].update(methods)
    tr_idx, te_idx = defaultdict(list), defaultdict(list)
    for i, cluster in enumerate(tr.l3): tr_idx[int(cluster)].append(i)
    for i, cluster in enumerate(te.l3): te_idx[int(cluster)].append(i)
    tr_profile, te_profile = profiles(train), profiles(test)
    tr_tokens, te_tokens = [tokens(x) for x in tr.l2], [tokens(x) for x in te.l2]
    high, amb = config["risk_bands"]["automatic_high_confidence"], config["risk_bands"]["ambiguous_high_risk"]
    rows = []
    for tc, rc in sorted(candidates):
        ii, jj = te_idx[tc], tr_idx[rc]
        ch = float(char[n:][ii].dot(char[:n][jj].T).toarray().max())
        le = float(lemma[n:][ii].dot(lemma[:n][jj].T).toarray().max())
        em = float(np.max(emb_te[ii] @ emb_tr[jj].T))
        to = max(jaccard(te_tokens[i], tr_tokens[j]) for i in ii for j in jj)
        tp, rp = te_profile[tc], tr_profile[rc]
        raw_exact, l1_exact, l2_exact = bool(set(tp["raw"]) & set(rp["raw"])), bool(set(tp["l1"]) & set(rp["l1"])), bool(set(tp["l2"]) & set(rp["l2"]))
        exact = raw_exact or l1_exact or l2_exact
        consensus = ch >= high["char_min"] and to >= high["token_jaccard_min"] and le >= high["lemma_char_min"]
        emb_high = em >= high["embedding_alt_min"] and to >= high["embedding_alt_token_min"] and le >= high["embedding_alt_lemma_min"]
        if exact or consensus or emb_high:
            band = "automatic_high_confidence"
        elif ch >= amb["char_min"] or to >= amb["token_jaccard_min"] or le >= amb["lemma_char_min"] or em >= amb["embedding_min"]:
            band = "ambiguous_high_risk"
        else:
            band = "safe_outside"
        rows.append({
            "pair_id": sha_text(f"{tc}|{rc}")[:20], "risk_band": band,
            "retrieved_by": "|".join(sorted(candidates[(tc, rc)])),
            "test_cluster": tc, "train_cluster": rc,
            "test_raw_forms": " || ".join(tp["raw"]), "train_raw_forms": " || ".join(rp["raw"]),
            "test_normalized_forms": " || ".join(tp["l2"]), "train_normalized_forms": " || ".join(rp["l2"]),
            "raw_exact": raw_exact, "l1_normalized_exact": l1_exact, "l2_prefix_stripped_exact": l2_exact,
            "char_similarity": ch, "token_set_jaccard": to, "lemma_char_similarity": le, "embedding_cosine": em,
        })
    ledger = pd.DataFrame(rows)
    if ledger.test_cluster.nunique() != test.l3.nunique():
        raise RuntimeError("not every test cluster has retrieved candidates")
    meta = {
        "train_forms": len(tr), "test_forms": len(te), "train_clusters": int(train.l3.nunique()),
        "test_clusters": int(test.l3.nunique()), "candidate_cluster_pairs": len(ledger),
        "views": config["retrieval"]["views"], "risk_band_pair_counts": ledger.risk_band.value_counts().to_dict(),
        "sentence_embedding": {"model": config["retrieval"]["sentence_embedding_model"],
          "revision": config["retrieval"]["sentence_embedding_revision"],
          "weight_sha256": config["retrieval"]["sentence_embedding_weight_sha256"], "device": "cpu"},
    }
    return ledger, meta
