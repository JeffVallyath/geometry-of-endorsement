import copy
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from repro.relation_readout import check, tail_check
from relation_study.m1 import text_baseline
from relation_study.m1.support.split_stress_test import SIT, CON, VAL


def test_saved_null_tails_and_development_scope():
    result = check()
    assert result['development_only'] and not result['audited_confirmatory_result_available']
    assert len(result['relation']) == 18
    assert result['permutation']['llama_10000']['difference_in_means'] == dict(draws=10000, exceedances=131, p=132 / 10001)
    assert result['permutation']['llama_10000']['logistic'] == dict(draws=10000, exceedances=0, p=1 / 10001)
    assert result['llama_truth']['exact_permutation_p'] == 1 / 1001


def test_permutation_tail_corruption_is_rejected():
    with pytest.raises(ValueError, match='tail accounting'):
        tail_check(dict(values=[-1., 1.], observed=0., B=2, k_ge_observed=0, p=1/3), 'p')


def test_text_feature_construction_without_loading_an_encoder(monkeypatch):
    calls, fitted = [], {}
    embeddings = {'s1': [1., 0.], 's2': [0., 1.], 'c1': [.6, .8], 'c2': [.8, .6]}
    class Encoder:
        def __init__(self, model, device):
            assert model == 'sentence-transformers/all-MiniLM-L6-v2' and device == 'cpu'
        def encode(self, strings, **kwargs):
            assert kwargs == dict(batch_size=256, show_progress_bar=False, normalize_embeddings=True)
            calls.append(strings)
            return np.asarray([embeddings[s] for s in strings])
    class Classifier:
        def __init__(self, **kwargs):
            assert kwargs == dict(max_iter=3000, C=1.0)
        def fit(self, features, labels):
            fitted.update(features=features, labels=labels)
            return self
        def predict_proba(self, features):
            return np.tile([.5, .5], (len(features), 1))
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=Encoder))
    monkeypatch.setattr(text_baseline, 'LogisticRegression', Classifier)
    frame = pd.DataFrame({SIT: ['s1', 's2'], CON: ['c1', 'c2'], VAL: ['supports', 'opposes']})
    scores, metadata = text_baseline.fit_sbert_interaction(frame, frame)
    s, c = np.asarray([[1., 0.], [0., 1.]]), np.asarray([[.6, .8], [.8, .6]])
    np.testing.assert_array_equal(fitted['features'], np.hstack([s, c, s*c, abs(s-c)]))
    np.testing.assert_array_equal(fitted['labels'], [1, 0])
    assert calls == [['s1', 's2'], ['c1', 'c2']] * 2
    assert metadata['encoder_revision'] is None
