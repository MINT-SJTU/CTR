"""Verify the exported arm loss mask, its statistics and preview values."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .stats import feature_statistics

MASK = "observation.arm_active_mask"
REMOVED = {"retime.left_idle", "retime.right_idle"}


def validate_arm_mask_dataset(root: Path, *, preview: bool = False) -> dict:
    info = json.loads((root / "meta/info.json").read_text())
    stats = json.loads((root / "meta/stats.json").read_text())
    if REMOVED.intersection(info["features"]) or REMOVED.intersection(stats):
        raise ValueError("removed idle fields remain in metadata")
    if info["features"][MASK] != {
        "dtype": "float32",
        "shape": [2],
        "names": ["left", "right"],
    }:
        raise ValueError("arm mask feature must be float32[2] in [left, right] order")
    chunks = []
    for path in sorted((root / "data").rglob("*.parquet")):
        if REMOVED.intersection(pq.read_schema(path).names):
            raise ValueError(f"removed idle fields remain in {path}")
        column = pq.read_table(path, columns=[MASK])[MASK].combine_chunks()
        if (
            column.type != pa.list_(pa.float32(), 2)
            or column.null_count
            or column.values.null_count
        ):
            raise ValueError(f"arm mask must be non-null fixed float32[2] in {path}")
        chunks.append(column.values.to_numpy().reshape(-1, 2))
    values = np.concatenate(chunks)
    if len(values) != info["total_frames"] or not np.isin(values, [0, 1]).all():
        raise ValueError("arm mask frame count or binary values differ")

    def check_stats(actual, expected):
        for name, value in expected.items():
            if np.shape(actual[name]) != np.shape(value):
                raise ValueError(f"arm mask {name} statistics shape differs")
            np.testing.assert_allclose(actual[name], value, rtol=1e-7, atol=1e-9)

    check_stats(stats[MASK], feature_statistics(values))
    episodes = []
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        if any(
            name.startswith(f"stats/{old}/")
            for name in table.column_names
            for old in REMOVED
        ):
            raise ValueError("removed idle statistics remain in episode metadata")
        episodes.extend(table.to_pylist())
    if len(episodes) != info["total_episodes"]:
        raise ValueError("episode count differs")
    end = 0
    for row in episodes:
        start, stop = row["dataset_from_index"], row["dataset_to_index"]
        if start != end or stop - start != row["length"] or stop <= start:
            raise ValueError("episode mask ranges do not cover the dataset")
        end = stop
        expected = feature_statistics(values[start:stop])
        check_stats({name: row[f"stats/{MASK}/{name}"] for name in expected}, expected)
        if preview:
            timeline = json.loads(
                (
                    root / f"preview/timelines/{row['episode_index']:03d}.json"
                ).read_text()
            )
            np.testing.assert_array_equal(timeline[MASK], values[start:stop])
    if end != len(values):
        raise ValueError("episode masks do not cover all frames")
    return {
        "feature": MASK,
        "dtype": "float32",
        "shape": [2],
        "order": ["left", "right"],
        "binary_values": True,
        "removed_fields_absent": True,
        "global_and_episode_statistics": "exact",
        "preview_matches_parquet": preview,
        "episodes": len(episodes),
        "frames": len(values),
    }
