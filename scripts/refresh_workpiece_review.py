"""Refresh the shared review UI and export the dataset's actual Idle Mask columns."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from real_robot_data_retime.retime import boolean_ranges


def refresh(dataset):
    preview = dataset / "preview"
    manifest = json.loads((preview / "manifest.json").read_text())
    records = []
    for row in manifest["episodes_data"]:
        table = pq.read_table(
            dataset / f"data/chunk-000/file-{row['output']:03}.parquet"
        )
        n = len(table)
        if n != row["frames"]:
            raise ValueError("video/idle mask frame mismatch")
        record = {"output": row["output"], "source": row["source"], "frames": n}
        active = np.asarray(
            table["observation.arm_active_mask"].to_pylist(), dtype=np.float32
        )
        timeline_path = preview / f"timelines/{row['output']:03}.json"
        timeline = json.loads(timeline_path.read_text())
        timeline["observation.arm_active_mask"] = active.tolist()
        timeline_path.write_text(json.dumps(timeline))
        for side, arm in enumerate(["left", "right"]):
            values = active[:, side] == 0
            clock = np.asarray(table[f"retime.{arm}_source_frame"], float)
            record[arm] = [[span.start, span.end] for span in boolean_ranges(values)]
            reconstructed = np.zeros(n, bool)
            for start, end in record[arm]:
                reconstructed[start:end] = True
            np.testing.assert_array_equal(reconstructed, values)
            changes = np.flatnonzero(np.diff(clock) > 0)
            start = int(changes[-1] + 1) if len(changes) else 0
            retained = int((~values[start:]).sum())
            unmasked = np.flatnonzero(~values[start:])
            cutoff = start + int(unmasked[-1]) + 1 if len(unmasked) else start
            record[arm + "_terminal_rest"] = {
                "start": start,
                "mask_start": cutoff,
                "supervised_frames": retained,
                "max_supervised_frames": retained,
            }
        records.append(record)
    (preview / "idle-masks.json").write_text(
        json.dumps(
            {
                "fields": ["observation.arm_active_mask"],
                "intervals": "zero-based half-open",
                "episodes": records,
            },
            indent=2,
        )
    )
    repo = Path(__file__).resolve().parents[1]
    for name in ["multi-camera.js", "idle-timeline.js", "idle-timeline.css"]:
        shutil.copy2(repo / "web/review" / name, preview / name)
    shutil.copy2(repo / "web/review/workpiece.html", preview / "index.html")
    report = {
        "passed": True,
        "episodes": len(records),
        "exact_parquet_mask_roundtrip": True,
    }
    (preview / "idle-validation.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    print(json.dumps(refresh(p.parse_args().dataset)))
