from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from geometry_of_truth.m1.config import ConfigError, load_config
from geometry_of_truth.m1.model import LoadedModel, _score_candidate_sequences
from geometry_of_truth.m1.prompts import (
    mapping_for_item,
    messages_for,
    prompt_contract_hash,
)


ROOT_CONFIG = "configs/m1_development_smoke.yaml"


def test_frozen_config_and_mapping_are_deterministic_and_label_free() -> None:
    config = load_config(ROOT_CONFIG)
    prompt = config.section("prompt")
    mapping = mapping_for_item("item-with-no-label", "primary", prompt)
    assert mapping == mapping_for_item("item-with-no-label", "primary", prompt)
    messages = messages_for(
        "A situation",
        "A consideration",
        mapping,
        "primary",
        "joint",
        prompt,
    )
    rendered = "\n".join(message["content"] for message in messages)
    assert "A situation" in rendered
    assert "A consideration" in rendered
    assert prompt_contract_hash(prompt) == prompt_contract_hash(prompt)
    assert set(mapping_for_item(f"item-{i}", "primary", prompt).name for i in range(100)) == {
        "primary:standard",
        "primary:reversed",
    }


def test_model_substitution_and_quantization_fail_closed(tmp_path) -> None:
    text = open(ROOT_CONFIG, encoding="utf-8").read()
    bad_model = tmp_path / "bad-model.yaml"
    bad_model.write_text(
        text.replace(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "some/other-model",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="substitution"):
        load_config(bad_model)
    bad_quant = tmp_path / "bad-quant.yaml"
    bad_quant.write_text(text.replace("quantization: none", "quantization: int8"), encoding="utf-8")
    with pytest.raises(ConfigError, match="Quantization"):
        load_config(bad_quant)


class FakeTokenizer:
    pad_token_id = 0
    padding_side = "right"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return {"A": [7, 8], "B": [9]}[text]

    def pad(self, payload, padding=True, return_tensors="pt"):
        rows = payload["input_ids"]
        width = max(map(len, rows))
        input_ids = torch.tensor(
            [row + [0] * (width - len(row)) for row in rows], dtype=torch.long
        )
        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != 0).to(torch.long),
        }


class FakeLM:
    def __init__(self) -> None:
        self.embedding = torch.nn.Embedding(16, 4)

    def get_input_embeddings(self):
        return self.embedding

    def __call__(self, input_ids, attention_mask, **kwargs):
        logits = torch.zeros((*input_ids.shape, 16), dtype=torch.float32)
        for row, sequence in enumerate(input_ids.tolist()):
            start = next(i for i, token in enumerate(sequence) if token in {7, 9})
            logits[row, start - 1, 7] = 2.0
            logits[row, start - 1, 9] = 1.0
            if sequence[start] == 7:
                logits[row, start, 8] = 3.0
        return SimpleNamespace(logits=logits)


def test_sequence_candidate_scoring_handles_multi_token_answers() -> None:
    loaded = LoadedModel(
        model=FakeLM(),
        tokenizer=FakeTokenizer(),
        model_revision="model-rev",
        tokenizer_revision="tokenizer-rev",
        device_name="cpu-test-only",
        gpu_memory_mib=0,
        activation_torch_dtype="float32",
    )
    supports, opposes, lengths = _score_candidate_sequences(
        loaded,
        [[1, 2, 3]],
        [("A", "B")],
    )
    first = torch.log_softmax(
        torch.tensor([0.0] * 7 + [2.0, 0.0, 1.0] + [0.0] * 6), dim=0
    )
    second_logits = torch.zeros(16)
    second_logits[8] = 3.0
    expected_supports = float(first[7] + torch.log_softmax(second_logits, dim=0)[8])
    expected_opposes = float(first[9])
    assert supports[0] == pytest.approx(expected_supports)
    assert opposes[0] == pytest.approx(expected_opposes)
    assert lengths.tolist() == [[2, 1]]
    assert np.isfinite(supports).all()
