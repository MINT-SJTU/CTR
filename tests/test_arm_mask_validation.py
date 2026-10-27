import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from real_robot_data_retime.arm_mask_validation import MASK, validate_arm_mask_dataset
from real_robot_data_retime.stats import feature_statistics


@pytest.fixture
def exported(tmp_path):
    for folder in ("data", "meta/episodes", "preview/timelines"):
        (tmp_path / folder).mkdir(parents=True)
    arrays = [
        np.array([[0, 1], [1, 0]], np.float32),
        np.array([[1, 1], [0, 0], [1, 0]], np.float32),
    ]
    rows = []
    start = 0
    for ep, values in enumerate(arrays):
        column = pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), 2)
        pq.write_table(pa.table({MASK: column}), tmp_path / f"data/{ep:03d}.parquet")
        row = {
            "episode_index": ep,
            "length": len(values),
            "dataset_from_index": start,
            "dataset_to_index": start + len(values),
        }
        row.update(
            {
                f"stats/{MASK}/{key}": value
                for key, value in feature_statistics(values).items()
            }
        )
        rows.append(row)
        start += len(values)
        (tmp_path / f"preview/timelines/{ep:03d}.json").write_text(
            json.dumps({MASK: values.tolist()})
        )
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "meta/episodes/000.parquet")
    (tmp_path / "meta/info.json").write_text(
        json.dumps(
            {
                "total_frames": start,
                "total_episodes": 2,
                "features": {
                    MASK: {"dtype": "float32", "shape": [2], "names": ["left", "right"]}
                },
            }
        )
    )
    (tmp_path / "meta/stats.json").write_text(
        json.dumps({MASK: feature_statistics(np.concatenate(arrays))})
    )
    return tmp_path


def test_checks_unequal_episode_lengths_and_preview(exported):
    result = validate_arm_mask_dataset(exported, preview=True)
    assert result["frames"] == 5 and result["episodes"] == 2
    assert result["preview_matches_parquet"]


@pytest.mark.parametrize(
    "fault",
    [
        "dtype",
        "width",
        "nonbinary",
        "null",
        "legacy_column",
        "legacy_info",
        "legacy_stats",
        "legacy_episode_stats",
        "episode_stats",
        "preview",
        "stats_width",
    ],
)
def test_rejects_broken_contract(exported, fault):
    data = exported / "data/000.parquet"
    if fault in {"dtype", "width", "nonbinary", "null", "legacy_column"}:
        values = [[0, 1], [1, 0]]
        dtype = pa.list_(pa.float32(), 2)
        if fault == "dtype":
            dtype = pa.list_(pa.float64(), 2)
        if fault == "width":
            dtype, values = pa.list_(pa.float32(), 1), [[0], [1]]
        if fault == "nonbinary":
            values[0][0] = 0.5
        if fault == "null":
            values[0][0] = None
        table = pa.table({MASK: pa.array(values, type=dtype)})
        if fault == "legacy_column":
            table = table.append_column("retime.left_idle", pa.array([True, False]))
        pq.write_table(table, data)
    elif fault in {"legacy_info", "legacy_stats", "stats_width"}:
        path = exported / (
            "meta/info.json" if fault == "legacy_info" else "meta/stats.json"
        )
        value = json.loads(path.read_text())
        if fault == "legacy_info":
            value["features"]["retime.right_idle"] = {}
        elif fault == "legacy_stats":
            value["retime.right_idle"] = {}
        else:
            value[MASK]["mean"] = [0.5]
        path.write_text(json.dumps(value))
    elif fault in {"episode_stats", "legacy_episode_stats"}:
        path = exported / "meta/episodes/000.parquet"
        rows = pq.read_table(path).to_pylist()
        if fault == "episode_stats":
            rows[1][f"stats/{MASK}/mean"][1] = 0.5
        else:
            for row in rows:
                row["stats/retime.left_idle/mean"] = [0.5]
        pq.write_table(pa.Table.from_pylist(rows), path)
    else:
        (exported / "preview/timelines/000.json").write_text(
            json.dumps({MASK: [[1, 0], [0, 1]]})
        )
    with pytest.raises((ValueError, AssertionError)):
        validate_arm_mask_dataset(exported, preview=True)
