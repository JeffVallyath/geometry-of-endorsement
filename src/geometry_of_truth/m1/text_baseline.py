from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .support.split_stress_test import CON, SIT, SUPPORTS, VAL

SBERT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def fit_sbert_interaction(
    train: pd.DataFrame,
    *evaluation_frames: pd.DataFrame,
    device: str = "cpu",
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Fit the frozen M0 SBERT interaction baseline once and score each frame."""

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(SBERT_MODEL_ID, device=device)

    def features(frame: pd.DataFrame) -> np.ndarray:
        situations = model.encode(
            frame[SIT].astype(str).tolist(),
            batch_size=256,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        considerations = model.encode(
            frame[CON].astype(str).tolist(),
            batch_size=256,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.hstack(
            [
                situations,
                considerations,
                situations * considerations,
                np.abs(situations - considerations),
            ]
        )

    train_features = features(train)
    labels = (train[VAL] == SUPPORTS).astype(int).to_numpy()
    classifier = LogisticRegression(max_iter=3000, C=1.0).fit(
        train_features, labels
    )
    scores = [
        classifier.predict_proba(features(frame))[:, 1]
        for frame in evaluation_frames
    ]
    metadata = {
        "name": "sbert_interaction",
        "encoder": SBERT_MODEL_ID,
        "encoder_revision": getattr(model, "model_card_data", None)
        and getattr(model.model_card_data, "base_model_revision", None),
        "device": device,
        "feature_construction": [
            "situation_embedding",
            "consideration_embedding",
            "elementwise_product",
            "absolute_difference",
        ],
        "classifier": "LogisticRegression(C=1.0,max_iter=3000)",
        "trained_rows": int(len(train)),
    }
    return scores, metadata
