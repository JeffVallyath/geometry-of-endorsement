from __future__ import annotations

import csv
import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DatasetSpec
from .prompts import mapping_for_item


@dataclass(frozen=True)
class DatasetArtifact:
    name: str
    path: Path
    sha256: str
    rows: int


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def fetch_dataset(spec: DatasetSpec, data_dir: str | Path) -> DatasetArtifact:
    target_dir = Path(data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{spec.name}.csv"
    expected = spec.sha256.lower()
    if target.exists() and sha256_file(target) != expected:
        raise RuntimeError(f"Cached {spec.name} hash does not match the frozen manifest: {target}")
    if not target.exists():
        partial = target.with_suffix(".csv.part")
        request = urllib.request.Request(spec.url, headers={"User-Agent": "geometry-of-endorsement/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
        if sha256_file(partial) != expected:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded {spec.name} hash does not match the frozen manifest.")
        os.replace(partial, target)
    with target.open("r", encoding="utf-8", newline="") as handle:
        rows = sum(1 for _ in csv.reader(handle)) - 1
    return DatasetArtifact(spec.name, target, expected, rows)


def _base_frame(
    artifact: DatasetArtifact,
    statement_column: str,
    label_column: str,
    group_columns: list[str],
) -> pd.DataFrame:
    frame = pd.read_csv(artifact.path).reset_index(names="source_row")
    required = {statement_column, label_column, *group_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{artifact.name} is missing columns: {missing}")
    frame["dataset"] = artifact.name
    frame["statement"] = frame[statement_column].astype(str)
    frame["label"] = frame[label_column].astype(int)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise RuntimeError(f"{artifact.name} labels must be binary 0/1.")
    frame["group_id"] = frame[group_columns].astype(str).agg("|".join, axis=1)
    frame["item_id"] = [
        hashlib.sha256(f"truth-v2-item:{artifact.name}:{row}".encode()).hexdigest()[:24]
        for row in frame["source_row"]
    ]
    return frame[["dataset", "source_row", "group_id", "item_id", "statement", "label"]]


def _paired_sample(frames: dict[str, pd.DataFrame], limit: int | None) -> dict[str, pd.DataFrame]:
    cities = frames["cities"].set_index("source_row", drop=False)
    negated = frames["neg_cities"].set_index("source_row", drop=False)
    if set(cities.index) != set(negated.index):
        raise RuntimeError("cities and neg_cities source-row sets differ.")
    aligned = sorted(cities.index)
    if not np.array_equal(
        cities.loc[aligned, "label"].to_numpy(dtype=np.int8),
        1 - negated.loc[aligned, "label"].to_numpy(dtype=np.int8),
    ):
        raise RuntimeError("cities and neg_cities labels are not complementary by source_row.")
    if not np.array_equal(
        cities.loc[aligned, "group_id"].astype(str).to_numpy(),
        negated.loc[aligned, "group_id"].astype(str).to_numpy(),
    ):
        raise RuntimeError("cities and neg_cities groups are not aligned by source_row.")
    selected = aligned
    if limit is not None:
        if limit < 2:
            raise ValueError("Smoke sample must contain at least two paired source rows.")
        per_label = max(1, limit // 2)
        picked: list[int] = []
        for label in (0, 1):
            picked.extend(cities[cities["label"] == label]["source_row"].head(per_label).astype(int))
        if len(picked) < limit:
            remainder = [row for row in aligned if row not in set(picked)]
            picked.extend(remainder[: limit - len(picked)])
        selected = sorted(picked[:limit])
    return {
        "cities": cities.loc[selected].reset_index(drop=True),
        "neg_cities": negated.loc[selected].reset_index(drop=True),
    }


def load_examples(
    artifacts: Iterable[DatasetArtifact],
    statement_column: str,
    label_column: str,
    group_columns: list[str],
    prompt_config: dict,
    max_examples_per_dataset: int | None = None,
) -> pd.DataFrame:
    frames = {
        artifact.name: _base_frame(artifact, statement_column, label_column, group_columns)
        for artifact in artifacts
    }
    if set(frames) != {"cities", "neg_cities"}:
        raise RuntimeError("Exactly cities and neg_cities are required.")
    frames = _paired_sample(frames, max_examples_per_dataset)
    base = pd.concat([frames["cities"], frames["neg_cities"]], ignore_index=True)

    expanded: list[pd.DataFrame] = []
    for scheme in ("primary", "transfer"):
        copy = base.copy()
        copy["scheme"] = scheme
        copy["mapping"] = [
            mapping_for_item(item_id, scheme, prompt_config).name.split(":", 1)[1]
            for item_id in copy["item_id"]
        ]
        copy["record_id"] = [
            hashlib.sha256(f"{item_id}:{scheme}".encode()).hexdigest()[:20]
            for item_id in copy["item_id"]
        ]
        expanded.append(copy)
    result = pd.concat(expanded, ignore_index=True)
    if result["record_id"].duplicated().any():
        raise RuntimeError("record_id collision detected.")
    return result


def assign_grouped_splits(
    examples: pd.DataFrame,
    seed: int,
    train_fraction: float,
    dev_fraction: float,
    test_fraction: float,
) -> pd.DataFrame:
    del test_fraction
    groups = sorted(examples["group_id"].astype(str).unique())
    if len(groups) < 3:
        raise RuntimeError("At least three groups are required for train/dev/test.")
    rng = np.random.default_rng(seed)
    shuffled = list(np.asarray(groups)[rng.permutation(len(groups))])
    n_groups = len(shuffled)
    n_train = max(1, int(round(n_groups * train_fraction)))
    n_dev = max(1, int(round(n_groups * dev_fraction)))
    if n_train + n_dev >= n_groups:
        n_train = max(1, n_groups - 2)
        n_dev = 1
    assignments = {group: "train" for group in shuffled[:n_train]}
    assignments.update({group: "dev" for group in shuffled[n_train : n_train + n_dev]})
    assignments.update({group: "test" for group in shuffled[n_train + n_dev :]})
    out = examples.copy()
    out["split"] = out["group_id"].astype(str).map(assignments)
    validate_split(out)
    return out


def validate_split(examples: pd.DataFrame) -> None:
    if examples["split"].isna().any():
        raise RuntimeError("Some examples lack split assignments.")
    if set(examples["split"].unique()) != {"train", "dev", "test"}:
        raise RuntimeError("All train/dev/test splits must be non-empty.")
    overlap = examples.groupby("group_id")["split"].nunique()
    if int(overlap.max()) != 1:
        raise RuntimeError("Group leakage across train/dev/test detected.")

    for _, rows in examples.groupby(["dataset", "source_row"], sort=False):
        if (
            set(rows["scheme"]) != {"primary", "transfer"}
            or rows["label"].nunique() != 1
            or rows["split"].nunique() != 1
            or rows["group_id"].nunique() != 1
            or rows["mapping"].nunique() != 1
        ):
            raise RuntimeError("Verbalizer mirrors differ in label, mapping, split, or group.")
    for _, rows in examples.groupby(["scheme", "source_row"], sort=False):
        labels = dict(zip(rows["dataset"], rows["label"], strict=True)) if len(rows) == 2 else {}
        if set(labels) != {"cities", "neg_cities"} or int(labels["cities"]) + int(labels["neg_cities"]) != 1:
            raise RuntimeError("Affirmative/negated proposition pairs are not complementary.")
    for split_name in ("train", "dev", "test"):
        for scheme in ("primary", "transfer"):
            subset = examples[(examples["split"] == split_name) & (examples["scheme"] == scheme)]
            if set(subset["mapping"]) != {"standard", "reversed"}:
                raise RuntimeError(f"{split_name}/{scheme} lacks one answer-map orientation.")


def write_split_manifest(examples: pd.DataFrame, path: str | Path) -> None:
    columns = [
        "record_id", "item_id", "dataset", "source_row", "group_id",
        "scheme", "mapping", "label", "split",
    ]
    output = examples[columns].copy()
    output["statement_sha256"] = [
        hashlib.sha256(statement.encode()).hexdigest() for statement in examples["statement"]
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
