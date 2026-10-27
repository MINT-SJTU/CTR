"""Extend a packaged screw dataset to 1.5 seconds of supervised final rest."""

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from real_robot_data_retime.arm_mask_validation import validate_arm_mask_dataset
from real_robot_data_retime.retime import boolean_ranges
from real_robot_data_retime.screw import write_json
from real_robot_data_retime.stats import feature_statistics
from real_robot_data_retime.timeline.terminal import terminal_hold_plan
from real_robot_data_retime.trim import aggregate_stats, image_statistics
from real_robot_data_retime.validate import mae


def extend_video(source, dest, n, added, fps, encoded=None):
    if encoded is not None:
        shutil.copy2(encoded, dest)
    elif added:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(source),
                "-an",
                "-vf",
                f"tpad=stop_mode=clone:stop={added}",
                "-c:v",
                "libx264",
                "-preset",
                "medium"
                if "observation.images.right_environment_1" in source.parts
                else "veryfast",
                "-crf",
                "20",
                "-threads",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(dest),
            ],
            check=True,
        )
    else:
        shutil.copy2(source, dest)
    a, b = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(dest))
    max_mae = 0.0
    for i in range(n + added):
        if i < n:
            ok, original = a.read()
            if not ok:
                raise ValueError(f"source video ended early: {source}")
        ok, frame = b.read()
        if not ok:
            raise ValueError(f"video ended early at {i}: {dest}")
        error = mae(original, frame)
        max_mae = max(max_mae, error)
        if error > 8.0:
            raise ValueError(f"video compression MAE {error} exceeds 8 at {i}: {dest}")
    if a.read()[0] or b.read()[0]:
        raise ValueError("unexpected extra video frames")
    a.release()
    b.release()
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "json",
                str(dest),
            ]
        )
    )
    pts = [float(f["best_effort_timestamp_time"]) for f in probe["frames"]]
    np.testing.assert_allclose(pts, np.arange(n + added) / fps, atol=1e-6, rtol=0)
    return image_statistics(dest, n + added), max_mae


def complete(source, output, revision, encoded_videos=None):
    pool = ThreadPoolExecutor(max_workers=9)
    try:
        return _complete(source, output, revision, pool, encoded_videos)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _complete(source, output, revision, pool, encoded_videos):
    if output.exists():
        raise FileExistsError(output)
    shutil.copytree(
        source, output, ignore=shutil.ignore_patterns(".cache", "SHA256SUMS")
    )
    info = json.loads((source / "meta/info.json").read_text())
    if "retime.synthetic_terminal_hold" in info["features"]:
        raise ValueError("source already contains synthetic terminal holds")
    fps = info["fps"]
    episodes = pq.read_table(
        source / "meta/episodes/chunk-000/file-000.parquet"
    ).to_pylist()
    manifest = json.loads((source / "preview/manifest.json").read_text())
    labels = json.loads((source / "preview/idle-masks.json").read_text())
    checks, numeric, all_stats = [], {}, []
    total = 0
    cameras = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    video_jobs = {}
    for entry in episodes:
        ep = entry["episode_index"]
        table = pq.read_table(source / f"data/chunk-000/file-{ep:03d}.parquet")
        plan = terminal_hold_plan(
            np.asarray(table["observation.state"].to_pylist()),
            np.asarray(table["action"].to_pylist()),
            fps,
        )
        for camera in cameras:
            rel = Path(f"videos/{camera}/chunk-000/file-{ep:03d}.mp4")
            video_jobs[ep, camera] = pool.submit(
                extend_video,
                source / rel,
                output / rel,
                len(table),
                plan["added_frames"],
                fps,
                encoded_videos / rel if encoded_videos is not None else None,
            )
    for entry, row, label in zip(
        episodes, manifest["episodes_data"], labels["episodes"], strict=True
    ):
        ep = entry["episode_index"]
        assert row["output"] == label["output"] == ep
        path = Path(f"data/chunk-000/file-{ep:03d}.parquet")
        table = pq.read_table(source / path)
        n = len(table)
        state, action = [
            np.asarray(table[k].to_pylist()) for k in ["observation.state", "action"]
        ]
        plan = terminal_hold_plan(state, action, fps)
        added, start = plan["added_frames"], plan["recorded_static_start"]
        size = n + added
        arrays = {}
        for key in table.column_names:
            values = np.asarray(table[key].to_pylist())
            arrays[key] = np.concatenate(
                [values, np.repeat(values[-1:], added, axis=0)]
            )
        arrays["timestamp"] = np.arange(size, dtype=np.float32) / fps
        arrays["frame_index"] = np.arange(size, dtype=np.int64)
        arrays["index"] = np.arange(total, total + size, dtype=np.int64)
        arrays["observation.arm_active_mask"][start:] = 1
        for key in [
            "preview.left_required_wait",
            "preview.right_required_wait",
            "preview.synchronized",
            "preview.right_mandatory_retreat",
            "retime.interpolated",
        ]:
            arrays[key][n:] = False
        arrays["retime.synthetic_terminal_hold"] = np.arange(size) >= n
        columns = {}
        for field in table.schema:
            columns[field.name] = pa.array(arrays[field.name].tolist(), type=field.type)
        columns["retime.synthetic_terminal_hold"] = pa.array(
            arrays["retime.synthetic_terminal_hold"]
        )
        exported = pa.table(columns)
        pq.write_table(exported, output / path)
        assert pq.read_table(output / path).equals(exported)
        for key in [
            "action",
            "observation.state",
            "retime.left_source_frame",
            "retime.right_source_frame",
        ]:
            np.testing.assert_array_equal(
                arrays[key][:n], np.asarray(table[key].to_pylist())
            )
        stats = {k: feature_statistics(v.reshape(size, -1)) for k, v in arrays.items()}
        for k, v in arrays.items():
            numeric.setdefault(k, []).append(v.reshape(size, -1))
        video_errors = {}
        for camera in cameras:
            entry[f"videos/{camera}/to_timestamp"] = size / fps
            stats[camera], video_errors[camera] = video_jobs[ep, camera].result()
        for k, v in stats.items():
            for stat, value in v.items():
                entry[f"stats/{k}/{stat}"] = value
        all_stats.append(stats)
        entry.update(
            length=size, dataset_from_index=total, dataset_to_index=total + size
        )
        entry["preview/synthetic_terminal_frames"] = added
        entry["preview/recorded_terminal_static_frames"] = plan[
            "recorded_static_frames"
        ]
        row["frames"] = label["frames"] = size
        row["quality_notes"] += (
            f"；末尾共同静止补足1.5秒，新增{added}帧合成静止，双臂参与监督。"
        )
        for side, arm in enumerate(["left", "right"]):
            label[arm] = [
                [r.start, r.end]
                for r in boolean_ranges(
                    arrays["observation.arm_active_mask"][:, side] == 0
                )
            ]
            label[arm + "_terminal_rest"] = {
                "start": start,
                "mask_start": size,
                "end": size,
                "supervised_frames": size - start,
                "max_supervised_frames": size - start,
            }
        timeline_path = Path(f"preview/timelines/{ep:03d}.json")
        timeline = json.loads((source / timeline_path).read_text())
        for key in ["left", "right"]:
            timeline[key] += [timeline[key][-1]] * added
        timeline["video_pts"] = (np.arange(size) / fps).tolist()
        timeline["observation.arm_active_mask"] = arrays[
            "observation.arm_active_mask"
        ].tolist()
        timeline["retime.synthetic_terminal_hold"] = arrays[
            "retime.synthetic_terminal_hold"
        ].tolist()
        write_json(output / timeline_path, timeline)
        check = dict(
            episode=ep,
            original_frames=n,
            output_frames=size,
            video_max_mae=video_errors,
            **plan,
        )
        checks.append(check)
        total += size
        print("COMPLETE", ep, "added", added, "total", total, flush=True)
    info["total_frames"] = total
    info["features"]["retime.synthetic_terminal_hold"] = {
        "dtype": "bool",
        "shape": [1],
        "names": None,
    }
    write_json(output / "meta/info.json", info)
    pq.write_table(
        pa.Table.from_pylist(episodes),
        output / "meta/episodes/chunk-000/file-000.parquet",
    )
    stats = aggregate_stats(all_stats)
    stats.update({k: feature_statistics(np.concatenate(v)) for k, v in numeric.items()})
    write_json(output / "meta/stats.json", stats)
    write_json(output / "preview/manifest.json", manifest)
    labels["terminal_policy"] = (
        "At least 1.5 seconds common final rest; count recorded state/action rest first, append only shortage, supervise both arms."
    )
    for path in ["idle-masks.json", "preview/idle-masks.json"]:
        write_json(output / path, labels)
    result = {
        "source_revision": revision,
        "episodes": checks,
        "frames": total,
        "added_frames": sum(c["added_frames"] for c in checks),
        "video_validation": "H264 CRF20 yuv420p as original; every decoded frame compared to original or repeated final source frame with MAE <= 8/255; all PTS verified",
        "mask_validation": validate_arm_mask_dataset(output, preview=True),
    }
    write_json(output / "terminal-hold-validation.json", result)
    # Preserve earlier reports explicitly as historical evidence, not new counts.
    archive = output / "checks/before-terminal-extension"
    archive.mkdir(exist_ok=True)
    for name in [
        "packaging-validation.json",
        "lerobot-reader-validation.json",
        "active-mask-migration.json",
    ]:
        if (output / name).exists():
            shutil.move(output / name, archive / name)
    card = output / "README.md"
    card.write_text(
        card.read_text()
        .replace("168,847 frames", f"{total:,} frames")
        .replace(
            "onset delays and excess terminal rest are masked.",
            "onset delays are masked; final common rest remains supervised.",
        )
        .replace(
            "Videos are byte-identical to the generated previews.",
            "Videos retain H264 CRF20 yuv420p encoding; terminal frames repeat the final source frame before encoding. Re-encoding introduces bounded compression differences.",
        )
        .replace(
            "Packaging checks are in `packaging-validation.json`; reader checks are in `lerobot-reader-validation.json`.",
            "Current checks are in `terminal-hold-validation.json` and `lerobot-reader-validation.json`; earlier packaging receipts are archived under `checks/before-terminal-extension/`.",
        )
        + f"\n\n## Supervised terminal hold update\n\nCurrent export: **{total:,} frames**, 75 episodes at 30 FPS. Every episode has at least 1.5 seconds of final joint-command/state rest, counting recorded rest first. Added {result['added_frames']} synthetic frames in total. `retime.synthetic_terminal_hold` explicitly identifies appended frames. These repeat the last recorded command, measured state and three camera frames; they are synthetic, not evidence of physical settling. Both entries of `observation.arm_active_mask` are 1 throughout the final common rest. Original action/state values and motion timing are unchanged; video compression MAE is verified for every frame. `terminal-hold-validation.json` is the current validation; earlier receipts/source mappings and `meta/original_trajectories` describe the pre-extension generation.\n"
    )
    for asset in ["idle-timeline.js", "idle-timeline.css", "multi-camera.js"]:
        shutil.copy2(Path("web/review") / asset, output / "preview" / asset)
    shutil.copy2("web/review/screw.html", output / "preview/index.html")
    page = output / "preview/index.html"
    page.write_text(
        page.read_text().replace(
            "人为延后的启动和超过 1.5 秒的末尾静止标为 idle；",
            "人为延后的启动标为 idle；任务末尾共同静止补足 1.5 秒并保留双臂监督；",
        )
    )
    page.write_text(
        page.read_text()
        .replace("原逐集预览与画面复核项", "补齐前生成回执")
        .replace('<a href="../">全量目录</a>', '<a href="./">预览目录</a>')
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--revision", required=True)
    p.add_argument(
        "--encoded-videos",
        type=Path,
        help="Reuse encoded files, still validating every frame and PTS",
    )
    a = p.parse_args()
    complete(a.source, a.output, a.revision, a.encoded_videos)
