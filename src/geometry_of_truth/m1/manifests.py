from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .support.build_confirmatory_split import manifest_hash, row_id
from .support.split_stress_test import CON, OPPOSES, SIT, SUPPORTS, VAL, load_binary
from .config import canonical_digest


def pilot_protocol_digest(config: Any) -> str:
    return canonical_digest(
        {
            "seed": config.raw["run"]["seed"],
            "data": config.section("data"),
            "pilot": config.section("pilot"),
        }
    )


def short_hash(value: str, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def source_frame() -> pd.DataFrame:
    frame = load_binary().reset_index(drop=True)
    frame["row_id"] = [
        row_id(situation, consideration, valence, vrd)
        for situation, consideration, valence, vrd in zip(
            frame[SIT], frame[CON], frame[VAL], frame["vrd"], strict=True
        )
    ]
    return frame.drop_duplicates("row_id").reset_index(drop=True)


def common_text_training_frame(config: Any, repo_root: Path) -> pd.DataFrame:
    """Resolve the frozen M0 common-training manifest to licensed source text."""
    source = source_frame()
    ids = pd.read_csv(
        repo_root / config.raw["data"]["m0_common_train_manifest"]
    )["row_id"].astype(str)
    by_id = source.set_index("row_id", drop=False)
    missing = set(ids) - set(by_id.index)
    if missing:
        raise RuntimeError(f"{len(missing)} common-training rows do not resolve.")
    frame = by_id.loc[ids.tolist()].reset_index(drop=True)
    if len(frame) != len(ids):
        raise RuntimeError("Common-training manifest did not resolve one-to-one.")
    return frame


def _row_lookup(frame: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for _, row in frame.sort_values("row_id").iterrows():
        key = (str(row[SIT]), str(row[CON]), str(row[VAL]))
        lookup.setdefault(key, row.to_dict())
    return lookup


def _confirmatory_ids(path: Path) -> tuple[set[str], set[str]]:
    columns = list(pd.read_csv(path, nrows=0).columns)
    row_columns = [name for name in columns if name.startswith("row_id_")]
    usecols = ["board_id", *row_columns]
    frame = pd.read_csv(path, usecols=usecols)
    return (
        set(frame["board_id"].astype(str)),
        set(frame[row_columns].astype(str).to_numpy().ravel()),
    )


def _development_boards(
    dev_path: Path,
    source: pd.DataFrame,
    confirmatory_board_ids: set[str],
) -> list[dict[str, Any]]:
    lookup = _row_lookup(source)
    dev = pd.read_csv(dev_path)
    boards: list[dict[str, Any]] = []
    for _, record in dev.iterrows():
        board_id = str(record["board_id"])
        if board_id in confirmatory_board_ids:
            raise RuntimeError("Development and confirmatory board IDs overlap.")
        a = str(record["consideration_A"])
        b = str(record["consideration_B"])
        s1 = str(record["situation_1"])
        s2 = str(record["situation_2"])
        keys = [
            (s1, a, SUPPORTS),
            (s1, b, OPPOSES),
            (s2, a, OPPOSES),
            (s2, b, SUPPORTS),
        ]
        try:
            cells = [lookup[key] for key in keys]
        except KeyError as exc:
            raise RuntimeError(f"Development board {board_id} does not resolve to four source rows.") from exc
        boards.append(
            {
                "board_id": board_id,
                "situation_ids": {short_hash(f"situation:{s1}"), short_hash(f"situation:{s2}")},
                "clusters": {str(cells[0]["l3"]), str(cells[1]["l3"])},
                "row_ids": [str(cell["row_id"]) for cell in cells],
                "cell_roles": [
                    "s1_A_supports",
                    "s1_B_opposes",
                    "s2_A_opposes",
                    "s2_B_supports",
                ],
            }
        )
    return boards


def _cluster_bucket(cluster: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:cluster:{cluster}".encode()).hexdigest()
    # Two preregistered lanes maximize usable development boards while keeping
    # every authoritative M0 l3 cluster exclusive to select or eval. Training
    # is drawn separately from M0 common-train after both reserved lanes are
    # removed, so a third hash lane is unnecessary.
    return int(digest, 16) % 2


def _choose_boards(
    boards: list[dict[str, Any]],
    *,
    bucket: int,
    target: int,
    seed: int,
    forbidden_clusters: set[str] | None = None,
    forbidden_situations: set[str] | None = None,
) -> list[dict[str, Any]]:
    forbidden_clusters = forbidden_clusters or set()
    forbidden_situations = forbidden_situations or set()
    candidates = [
        board
        for board in boards
        if all(_cluster_bucket(cluster, seed) == bucket for cluster in board["clusters"])
    ]
    candidates.sort(
        key=lambda board: hashlib.sha256(
            f"{seed}:board:{board['board_id']}".encode()
        ).hexdigest()
    )
    chosen: list[dict[str, Any]] = []
    used_situations: set[str] = set()
    used_pairs: set[tuple[str, str]] = set()
    for board in candidates:
        pair = tuple(sorted(board["clusters"]))
        if board["clusters"] & forbidden_clusters:
            continue
        if board["situation_ids"] & (forbidden_situations | used_situations):
            continue
        if pair in used_pairs:
            continue
        chosen.append(board)
        used_situations |= board["situation_ids"]
        used_pairs.add(pair)
        if len(chosen) == target:
            break
    if len(chosen) < target:
        raise RuntimeError(
            f"Only {len(chosen)} development boards satisfy the frozen "
            f"bucket/isolation rules; need {target}. Do not loosen the contract "
            "after activations have been inspected."
        )
    return chosen


def _rows_for_boards(
    split: str,
    boards: list[dict[str, Any]],
    source_by_id: pd.DataFrame,
) -> pd.DataFrame:
    board_for_row: dict[str, str] = {}
    role_for_row: dict[str, str] = {}
    for board in boards:
        for row_identifier, role in zip(board["row_ids"], board["cell_roles"], strict=True):
            if row_identifier in board_for_row:
                raise RuntimeError(f"{split} reuses one source row across boards.")
            board_for_row[row_identifier] = board["board_id"]
            role_for_row[row_identifier] = role
    rows = source_by_id.loc[list(board_for_row)].copy()
    rows["split"] = split
    rows["board_id"] = rows["row_id"].map(board_for_row)
    rows["cell_role"] = rows["row_id"].map(role_for_row)
    return rows


def _public_rows(frame: pd.DataFrame) -> pd.DataFrame:
    board = frame["board_id"] if "board_id" in frame else ""
    role = frame["cell_role"] if "cell_role" in frame else ""
    situation_ids = frame[SIT].map(lambda value: short_hash(f"situation:{value}"))
    consideration_ids = frame[CON].map(
        lambda value: short_hash(f"consideration:{value}")
    )
    result = pd.DataFrame(
        {
            # The answer mapping is hashed from item_id, so item_id must not
            # contain valence. Fail uniqueness below if the source contains
            # contradictory duplicate identities rather than leaking the label
            # into the mapping seed.
            "item_id": [
                short_hash(
                    f"m1-item:{situation_id}:{consideration_id}:{vrd}",
                    24,
                )
                for situation_id, consideration_id, vrd in zip(
                    situation_ids,
                    consideration_ids,
                    frame["vrd"].astype(str),
                    strict=True,
                )
            ],
            "row_id": frame["row_id"].astype(str),
            "split": frame["split"].astype(str),
            "label": (frame[VAL] == SUPPORTS).astype(int),
            "situation_id": situation_ids,
            "consideration_id": consideration_ids,
            "consideration_cluster_id": frame["l3"].astype(str),
            "board_id": board,
            "cell_role": role,
        }
    )
    return result.sort_values("item_id").reset_index(drop=True)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def _write_immutable(path: Path, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Immutable pilot artifact differs: {path}")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_bytes(payload)
    os.replace(partial, path)
    return digest


def _board_frame(boards: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "board_id": board["board_id"],
                "row_id_s1_A": board["row_ids"][0],
                "row_id_s1_B": board["row_ids"][1],
                "row_id_s2_A": board["row_ids"][2],
                "row_id_s2_B": board["row_ids"][3],
            }
            for board in boards
        ]
    ).sort_values("board_id")


def build_pilot_manifests(config: Any, repo_root: Path, output_dir: Path) -> dict[str, Any]:
    pilot = config.section("pilot")
    seed = int(config.raw["run"]["seed"])
    source = source_frame()
    source_by_id = source.set_index("row_id", drop=False)
    data_config = config.section("data")
    confirmatory_boards, confirmatory_rows = _confirmatory_ids(
        repo_root / data_config["m0_confirmatory_manifest"]
    )
    boards = _development_boards(
        repo_root / data_config["m0_dev_manifest"], source, confirmatory_boards
    )

    evaluation = _choose_boards(
        boards, bucket=1, target=int(pilot["eval_boards"]), seed=seed
    )
    eval_clusters = {cluster for board in evaluation for cluster in board["clusters"]}
    eval_situations = {
        situation for board in evaluation for situation in board["situation_ids"]
    }
    selection = _choose_boards(
        boards,
        bucket=0,
        target=int(pilot["select_boards"]),
        seed=seed,
        forbidden_clusters=eval_clusters,
        forbidden_situations=eval_situations,
    )
    select_clusters = {cluster for board in selection for cluster in board["clusters"]}
    select_rows = _rows_for_boards("pilot_select", selection, source_by_id)
    eval_rows = _rows_for_boards("pilot_eval", evaluation, source_by_id)

    common_ids = set(
        pd.read_csv(repo_root / data_config["m0_common_train_manifest"])["row_id"].astype(str)
    )
    reserved_clusters = eval_clusters | select_clusters
    reserved_situations = set(select_rows[SIT]) | set(eval_rows[SIT])
    train_pool = source[
        source["row_id"].isin(common_ids)
        & ~source["row_id"].isin(confirmatory_rows)
        & ~source[SIT].isin(reserved_situations)
        & ~source["l3"].astype(str).isin(reserved_clusters)
    ].copy()
    target = int(pilot["train_rows"])
    if target % 2:
        raise RuntimeError("pilot.train_rows must be even for exact label balance.")
    pieces: list[pd.DataFrame] = []
    for label in (SUPPORTS, OPPOSES):
        part = train_pool[train_pool[VAL] == label].copy()
        part["_order"] = part["row_id"].map(
            lambda value: hashlib.sha256(f"{seed}:train:{value}".encode()).hexdigest()
        )
        pieces.append(part.sort_values("_order").head(target // 2))
    train = pd.concat(pieces, ignore_index=True).drop(columns="_order")
    if len(train) != target:
        raise RuntimeError("Insufficient balanced rows for the frozen pilot_train target.")
    train["split"] = "pilot_train"
    train["board_id"] = ""
    train["cell_role"] = ""

    public = {
        "pilot_train": _public_rows(train),
        "pilot_select": _public_rows(select_rows),
        "pilot_eval": _public_rows(eval_rows),
    }
    train_clusters = set(public["pilot_train"]["consideration_cluster_id"])
    select_cluster_ids = set(public["pilot_select"]["consideration_cluster_id"])
    eval_cluster_ids = set(public["pilot_eval"]["consideration_cluster_id"])
    all_rows = set().union(*(set(frame["row_id"]) for frame in public.values()))
    checks = {
        "item_id_unique_and_label_independent": sum(
            len(frame) for frame in public.values()
        )
        == len(
            set().union(*(set(frame["item_id"]) for frame in public.values()))
        ),
        "row_disjoint": sum(len(frame) for frame in public.values()) == len(all_rows),
        "situation_disjoint": not (
            set(public["pilot_train"]["situation_id"]) & set(public["pilot_select"]["situation_id"])
            or set(public["pilot_train"]["situation_id"]) & set(public["pilot_eval"]["situation_id"])
            or set(public["pilot_select"]["situation_id"]) & set(public["pilot_eval"]["situation_id"])
        ),
        "cluster_disjoint": not (
            train_clusters & select_cluster_ids
            or train_clusters & eval_cluster_ids
            or select_cluster_ids & eval_cluster_ids
        ),
        "confirmatory_rows_absent": not (confirmatory_rows & all_rows),
        "confirmatory_boards_absent": not (
            confirmatory_boards
            & {board["board_id"] for board in selection + evaluation}
        ),
        "select_eval_board_disjoint": not (
            {board["board_id"] for board in selection}
            & {board["board_id"] for board in evaluation}
        ),
        "labels_balanced": all(
            set(frame["label"].unique()) == {0, 1}
            and frame["label"].value_counts().nunique() == 1
            for frame in public.values()
        ),
        "common_training_subset": set(public["pilot_train"]["row_id"]) <= common_ids,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Pilot manifest contract failed: {checks}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_digests = {
        name: _write_immutable(output_dir / f"{name}.csv", _csv_bytes(frame))
        for name, frame in public.items()
    }
    board_digests = {
        name: _write_immutable(
            output_dir / f"{name}_boards.csv", _csv_bytes(_board_frame(chosen))
        )
        for name, chosen in (("pilot_select", selection), ("pilot_eval", evaluation))
    }
    m0_meta = json.loads((repo_root / "MANIFEST.json").read_text(encoding="utf-8"))
    metadata = {
        "schema_version": 2,
        "config_sha256": config.digest,
        "pilot_protocol_sha256": pilot_protocol_digest(config),
        "source_row_id_sha256": manifest_hash(set(source["row_id"])),
        "m0_created_from_commit": m0_meta["created_from_commit"],
        "m0_manifest_sha256": file_sha256(repo_root / "MANIFEST.json"),
        "counts": {name: len(frame) for name, frame in public.items()},
        "board_counts": {
            "pilot_select": len(selection),
            "pilot_eval": len(evaluation),
        },
        "manifest_sha256": manifest_digests,
        "board_manifest_sha256": board_digests,
        "combined_row_manifest_hash": manifest_hash(all_rows),
        "checks": checks,
        "contains_source_text": False,
        "construction": (
            "development-only; pilot_train is an M0 common-training subset; "
            "hash-ordered; situations and authoritative M0 l3 clusters isolated"
        ),
    }
    meta_bytes = (
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _write_immutable(output_dir / "pilot_manifest.json", meta_bytes)
    return metadata


def load_materialized_pilot(
    config: Any,
    repo_root: Path,
    manifest_dir: Path,
    *,
    allow_analysis_only_config_mismatch: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = json.loads((manifest_dir / "pilot_manifest.json").read_text(encoding="utf-8"))
    protocol_hash = metadata.get("pilot_protocol_sha256")
    if protocol_hash is not None:
        if protocol_hash != pilot_protocol_digest(config):
            raise RuntimeError("Pilot manifest construction protocol differs from the run config.")
    elif metadata["config_sha256"] != config.digest:
        if not allow_analysis_only_config_mismatch:
            raise RuntimeError("Pilot manifest config hash differs from the run config.")
        expected_counts = {
            "pilot_train": int(config.raw["pilot"]["train_rows"]),
            "pilot_select": 4 * int(config.raw["pilot"]["select_boards"]),
            "pilot_eval": 4 * int(config.raw["pilot"]["eval_boards"]),
        }
        if metadata.get("counts") != expected_counts or not all(
            bool(value) for value in metadata.get("checks", {}).values()
        ):
            raise RuntimeError(
                "Legacy pilot metadata cannot be accepted for analysis-only replay."
            )
        metadata = dict(metadata)
        metadata["legacy_analysis_only_config_mismatch_accepted"] = True
    manifests: list[pd.DataFrame] = []
    for name in ("pilot_train", "pilot_select", "pilot_eval"):
        path = manifest_dir / f"{name}.csv"
        if file_sha256(path) != metadata["manifest_sha256"][name]:
            raise RuntimeError(f"Pilot manifest hash mismatch: {name}")
        manifests.append(pd.read_csv(path, dtype={"consideration_cluster_id": str}))
    manifest = pd.concat(manifests, ignore_index=True)
    source = source_frame().set_index("row_id", drop=False)
    missing = set(manifest["row_id"]) - set(source.index)
    if missing:
        raise RuntimeError(f"{len(missing)} pilot row IDs do not resolve in ValuePrism.")
    rows = source.loc[manifest["row_id"]].reset_index(drop=True)
    for column in manifest.columns:
        rows[column] = manifest[column].to_numpy()
    if not (
        (rows[VAL] == SUPPORTS).astype(int).to_numpy() == rows["label"].to_numpy()
    ).all():
        raise RuntimeError("Pilot labels disagree with the gated source dataset.")
    if not (
        rows["l3"].astype(str).to_numpy()
        == rows["consideration_cluster_id"].astype(str).to_numpy()
    ).all():
        raise RuntimeError("Pilot consideration clusters disagree with M0.")
    return rows, metadata
