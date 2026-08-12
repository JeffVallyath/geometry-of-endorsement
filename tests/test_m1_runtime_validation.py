from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from geometry_of_truth.m1.config import load_config
from geometry_of_truth.m1.model import ExtractionResult
from geometry_of_truth.m1.runner import CON, SIT, _runtime_validation, _signature


def _fake_result(records: list[dict[str, str]]) -> ExtractionResult:
    batch_offset = 0.0 if len(records) == 1 else 0.5
    rows = []
    margins = []
    positions = []
    prompts = []
    mappings = []
    for record in records:
        index = int(record["item_id"].removeprefix("item-"))
        rows.append(np.full((2, 3), index + batch_offset, dtype=np.float16))
        margins.append(-2.0 if index % 2 == 0 else 2.0)
        positions.append(10 + index)
        prompts.append(f"prompt-{record['item_id']}")
        mappings.append("primary:standard" if index % 2 == 0 else "primary:reversed")
    native_margin = np.asarray(margins, dtype=np.float32)
    return ExtractionResult(
        activations=np.stack(rows),
        supports_logp=native_margin.copy(),
        opposes_logp=np.zeros(len(records), dtype=np.float32),
        native_margin=native_margin,
        predicted_labels=(native_margin >= 0).astype(np.int8),
        prompt_token_indices=np.asarray(positions, dtype=np.int32),
        rendered_prompts=prompts,
        mapping_names=mappings,
        candidate_token_lengths=np.ones((len(records), 2), dtype=np.int16),
        peak_vram_bytes=123,
    )


def _fixture(monkeypatch):
    frame = pd.DataFrame(
        {
            "item_id": ["item-0", "item-1"],
            SIT: ["s0", "s1"],
            CON: ["c0", "c1"],
        }
    )
    config = SimpleNamespace(
        raw={"cache": {"batch_size": 2, "activation_dtype": "float16"}},
        section=lambda name: {"prompt": True},
    )
    loaded = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(num_hidden_layers=2))
    )

    def fake_extract(loaded, records, prompt_config, activation_dtype):
        return _fake_result(records)

    monkeypatch.setattr("geometry_of_truth.m1.runner.extract_batch", fake_extract)
    batch = _fake_result(
        [
            {
                "item_id": "item-0",
                "situation": "s0",
                "consideration": "c0",
                "scheme": "primary",
                "representation": "joint",
            },
            {
                "item_id": "item-1",
                "situation": "s1",
                "consideration": "c1",
                "scheme": "primary",
                "representation": "joint",
            },
        ]
    )
    cached = {
        "activations": batch.activations.copy(),
        "supports_logp": batch.supports_logp.copy(),
        "opposes_logp": batch.opposes_logp.copy(),
        "native_margin": batch.native_margin.copy(),
        "predicted_labels": batch.predicted_labels.copy(),
        "prompt_token_indices": batch.prompt_token_indices.copy(),
        "prompt_sha256": np.asarray(
            [
                __import__("hashlib").sha256(prompt.encode()).hexdigest()
                for prompt in batch.rendered_prompts
            ]
        ),
        "mapping_names": np.asarray(batch.mapping_names),
    }
    return loaded, frame, config, cached


def test_runtime_validation_accepts_fixed_batch_reproduction_not_cross_batch_bits(
    monkeypatch,
) -> None:
    loaded, frame, config, cached = _fixture(monkeypatch)
    result = _runtime_validation(
        loaded=loaded,
        frame=frame,
        config=config,
        cached_primary=cached,
    )
    assert result["status"] == "PASS"
    assert all(result["checks"].values())
    assert result["validation_contract"] == (
        "m1_runtime_validation_v2_fixed_batch_reproduction"
    )
    assert result["cross_batch_activation_max_abs"] == 0.5
    assert "descriptive_only" in result["cross_batch_activation_comparison_role"]


def test_runtime_validation_rejects_same_batch_cache_drift(monkeypatch) -> None:
    loaded, frame, config, cached = _fixture(monkeypatch)
    cached["activations"][0, 0, 0] += 1.0
    result = _runtime_validation(
        loaded=loaded,
        frame=frame,
        config=config,
        cached_primary=cached,
    )
    assert result["status"] == "FAIL"
    assert result["checks"]["same_batch_cache_activations_atol_0_002"] is False


def test_cache_signature_binds_repository_commit() -> None:
    config = load_config("configs/m1_development_smoke.yaml")
    kwargs = {
        "key": "primary_joint",
        "model_revision": "model",
        "tokenizer_revision": "tokenizer",
        "chat_template_sha256": "chat",
        "pilot_manifest_sha256": "pilot",
        "record_ids": ["one", "two"],
    }
    first = _signature(config, repository_commit="a" * 40, **kwargs)
    second = _signature(config, repository_commit="b" * 40, **kwargs)
    assert first != second
