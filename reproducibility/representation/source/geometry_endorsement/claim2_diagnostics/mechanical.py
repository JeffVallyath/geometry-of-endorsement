from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_mechanical_features(scores: pd.DataFrame) -> pd.DataFrame:
    required = {"base_item_id", "trial_type", "trial_index", "mapping", "semantic_margin"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Missing mechanical columns: {sorted(missing)}")
    rows = []
    for base_item_id, frame in scores.groupby("base_item_id", sort=False):
        exact = frame[frame["trial_type"].eq("exact_repeat")]
        irrelevant = frame[frame["trial_type"].eq("irrelevant_string")]
        formats = frame[frame["trial_type"].eq("format")]
        exact_sign = np.sign(exact["semantic_margin"].to_numpy(dtype=float))
        irrelevant_sign = np.sign(irrelevant["semantic_margin"].to_numpy(dtype=float))
        reference = exact_sign[0] if len(exact_sign) else np.nan
        mapping = frame[frame["mapping"].isin(["ab_standard", "ab_reversed", "12_standard", "12_reversed"])]
        mapping_sign = mapping.groupby("mapping")["semantic_margin"].mean().map(np.sign)
        rows.append({
            "base_item_id": base_item_id,
            "exact_repeat_verdict_retention": float((exact_sign == reference).mean()) if len(exact_sign) else np.nan,
            "exact_repeat_margin_std": float(exact["semantic_margin"].std(ddof=0)) if len(exact) else np.nan,
            "irrelevant_string_verdict_retention": float((irrelevant_sign == reference).mean()) if len(irrelevant_sign) else np.nan,
            "irrelevant_string_margin_std": float(irrelevant["semantic_margin"].std(ddof=0)) if len(irrelevant) else np.nan,
            "irrelevant_string_max_abs_movement": float((irrelevant["semantic_margin"] - exact["semantic_margin"].iloc[0]).abs().max()) if len(irrelevant) and len(exact) else np.nan,
            "ab_mapping_consistency": bool(mapping_sign.get("ab_standard", np.nan) == mapping_sign.get("ab_reversed", np.nan)),
            "one_two_mapping_consistency": bool(mapping_sign.get("12_standard", np.nan) == mapping_sign.get("12_reversed", np.nan)),
            "cross_format_consistency": float((np.sign(formats["semantic_margin"]) == reference).mean()) if len(formats) else np.nan,
            "valid_semantic_judgments": int(frame["semantic_margin"].notna().sum()),
        })
    return pd.DataFrame(rows)
