"""Water source views and rendering; reuse the repository's LeRobot exporter."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .automatic_dataset import finalize, write_retimed_episode
from .compositing.layers import composite
from .compositing.ownership import arm_foreground
from .interaction.video import read_video
from .stats import feature_statistics
from .trim import read_episode
from .video import remap_video

MAIN_CAMERA = "observation.images.right_environment_1"
ALIAS = "observation.images.top"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def source_view(source, info, row, work):
    """Normalize one source episode for the existing top/wrist exporter."""
    view = work / "source"
    view.mkdir()
    table = read_episode(source, info, row)
    n, fps = len(table), info["fps"]
    normalized = copy.deepcopy(info)
    normalized["features"] = {
        k.replace(MAIN_CAMERA, ALIAS): v for k, v in info["features"].items()
    }
    normalized.update(total_episodes=1, total_frames=n, splits={"train": "0:1"})
    metadata = {k.replace(MAIN_CAMERA, ALIAS): v for k, v in row.items()}
    metadata.update(
        episode_index=0,
        dataset_from_index=0,
        dataset_to_index=n,
        **{"data/chunk_index": 0, "data/file_index": 0},
    )
    for key, feature in info["features"].items():
        if feature["dtype"] != "video":
            continue
        prefix = "videos/" + key
        original = source / info["video_path"].format(
            video_key=key,
            chunk_index=row[prefix + "/chunk_index"],
            file_index=row[prefix + "/file_index"],
        )
        first = round(row[prefix + "/from_timestamp"] * fps)
        dest = view / f"videos/{key.replace(MAIN_CAMERA, ALIAS)}/chunk-000/file-000.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                f"{first / fps:.9f}",
                "-i",
                str(original),
                "-frames:v",
                str(n),
                "-c:v",
                "libx264",
                "-crf",
                "0",
                "-threads",
                "4",
                "-an",
                str(dest),
            ],
            check=True,
        )
        prefix = prefix.replace(MAIN_CAMERA, ALIAS)
        metadata.update(
            {
                prefix + "/chunk_index": 0,
                prefix + "/file_index": 0,
                prefix + "/from_timestamp": 0.0,
                prefix + "/to_timestamp": n / fps,
            }
        )
    for key, values in [("episode_index", np.zeros(n, int)), ("index", np.arange(n))]:
        table = table.set_column(
            table.schema.get_field_index(key),
            key,
            pa.array(values, type=table.schema.field(key).type),
        )
    for folder in ["data/chunk-000", "meta/episodes/chunk-000"]:
        (view / folder).mkdir(parents=True)
    pq.write_table(table, view / "data/chunk-000/file-000.parquet")
    pq.write_table(
        pa.Table.from_pylist([metadata]),
        view / "meta/episodes/chunk-000/file-000.parquet",
    )
    pq.write_table(
        pq.read_table(source / "meta/tasks.parquet"), view / "meta/tasks.parquet"
    )
    dump(view / "meta/info.json", normalized)
    return view, table, metadata


def render_source(view, work, output, source_id, phase, schedules, first_output):
    frames, fps = read_video(view / f"videos/{ALIAS}/chunk-000/file-000.mp4")
    if len(frames) != phase["source_frames"] or abs(fps - phase["fps"]) > 1e-6:
        raise ValueError("video and frozen phase metadata differ")
    n, h, w = frames.shape[:3]
    table = pq.read_table(view / "data/chunk-000/file-000.parquet")
    row = pq.read_table(view / "meta/episodes/chunk-000/file-000.parquet").to_pylist()[
        0
    ]
    identity = hashlib.sha256(json.dumps(phase, sort_keys=True).encode())
    for path in [
        view / "data/chunk-000/file-000.parquet",
        work / "pre-segmentation.npz",
        work / "post-segmentation.npz",
    ]:
        with path.open("rb") as stream:
            identity.update(hashlib.file_digest(stream, "sha256").digest())
    segments = {}
    for label, start, stop in [
        ("pre", 0, phase["shared_start"] + 1),
        ("post", phase["shared_end"], n),
    ]:
        with np.load(work / f"{label}-segmentation.npz") as packed:
            if (
                int(packed["source_start"]) != start
                or int(packed["source_stop"]) != stop
            ):
                raise ValueError("segmentation does not cover the current phase")
            segments[label] = {
                key: packed[key] for key in ("robots", "objects", "frame_shape")
            }
    middle = work / "shared.mp4"
    remap_video(
        view / f"videos/{ALIAS}/chunk-000/file-000.mp4",
        middle,
        np.arange(phase["shared_start"] + 1, phase["shared_end"] + 1),
        fps,
        w,
        h,
    )
    checks = []
    for variant, schedule in enumerate(schedules):
        ep = first_output + variant
        clocks, stage = schedule["clocks"], schedule["stage"]
        parts = []
        reports = {}
        for label, stage_id, start, stop in [
            ("pre", 0, 0, phase["shared_start"] + 1),
            ("post", 2, phase["shared_end"], n),
        ]:
            local = clocks[stage == stage_id] - start
            segment = segments[label]
            robots = np.unpackbits(segment["robots"], axis=-1, count=w).astype(bool)
            objects = np.unpackbits(segment["objects"], axis=-1, count=w).astype(bool)
            events = phase["bottle_events"] if label == "pre" else []
            if len(objects):
                robots &= ~objects.any(axis=0)[:, None]
            overlap = 0
            for left, right in local:
                if left != right:
                    lm = arm_foreground(robots, objects, events, 0, int(left))
                    rm = arm_foreground(robots, objects, events, 1, int(right))
                    overlap = max(overlap, int((lm & rm).sum()))
            if overlap:
                raise ValueError(
                    f"{source_id}/{variant}/{label}: conflicting foregrounds ({overlap} pixels)"
                )
            segment = dict(segment, robots=np.packbits(robots, axis=-1))
            path = work / f"{variant}-{label}.mp4"
            reports[label] = composite(
                frames[start:stop],
                dict(task="water", fps=fps, episodes=events),
                segment,
                local[:, 0],
                local[:, 1],
                path,
                work / f"{variant}-{label}",
            )
            parts.append(path)
        concat = work / f"{variant}-concat.txt"

        def escaped(path):
            return str(path).replace("'", "'\\''")

        concat.write_text(
            "".join("file '" + escaped(p) + "'\n" for p in (parts[0], middle, parts[1]))
        )
        dest = output / f"videos/{ALIAS}/chunk-000/file-{ep:03d}.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tail = int(schedule["synthetic"].sum())
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
        ]
        if tail:
            command += ["-vf", f"tpad=stop_mode=clone:stop={tail}"]
        subprocess.run(
            command
            + [
                "-c:v",
                "libx264",
                "-crf",
                "20",
                "-threads",
                "4",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(dest),
            ],
            check=True,
        )
        plan = dict(
            task="water",
            source_episode=source_id,
            timing=schedule["timing"],
            uniform=schedule.get("uniform"),
            shared_source_interval=[phase["shared_start"], phase["shared_end"]],
            head_removed=phase["head_removed"],
            arm_head_removed=phase["arm_head_removed"],
            synthetic_terminal_hold=dict(start_frame=len(clocks) - tail, frames=tail),
            mask_policy="Only optional onset waits on the same delayed arm; all required waits supervised",
        )
        receipt = write_retimed_episode(
            view,
            output,
            row,
            ep,
            table,
            *clocks.T,
            dict(start=0, stop=n, source_frames=n),
            plan,
            dict(training_ready=False, parts=reports, visual_review="pending"),
            identity.hexdigest(),
            dict(phase=phase),
        )
        receipt["source_episode_index"] = source_id
        path = output / f"data/chunk-000/file-{ep:03d}.parquet"
        mapped = pq.read_table(path).append_column(
            "observation.arm_active_mask",
            pa.array(schedule["active"].tolist(), type=pa.list_(pa.float32(), 2)),
        )
        pq.write_table(mapped, path)
        receipt["statistics"]["observation.arm_active_mask"] = feature_statistics(
            schedule["active"]
        )
        dump(output / f"meta/retime_receipts/episode_{ep:03d}.json", receipt)
        checks.append(
            dict(
                episode=ep,
                source=source_id,
                frames=len(clocks),
                idle_frames=(schedule["active"] == 0).sum(axis=0).tolist(),
                origin_audit_passed=all(
                    report["automatic_origin_audit"]["passed"]
                    for report in reports.values()
                ),
                timing=schedule["timing"],
                visual_review="pending",
            )
        )
    return checks


def finish_export(view, output, checks):
    result = finalize(
        view, output, "local/water-retimed", episode_indices=range(len(checks))
    )
    (output / "videos" / ALIAS).rename(output / "videos" / MAIN_CAMERA)
    for path in output.rglob("*.json"):
        path.write_text(path.read_text().replace(ALIAS, MAIN_CAMERA))
    path = output / "meta/episodes/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    pq.write_table(
        table.rename_columns(
            [k.replace(ALIAS, MAIN_CAMERA) for k in table.column_names]
        ),
        path,
    )
    dump(
        output / "VALIDATION.json",
        dict(
            **result,
            variants=checks,
            training_ready=False,
            numeric_export_passed=True,
            origin_audit_passed=all(check["origin_audit_passed"] for check in checks),
            visual_review="pending",
        ),
    )
    return result
