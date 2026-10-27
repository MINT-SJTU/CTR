#!/usr/bin/env python3
"""Export a frame-addressable PiperX link6 trajectory for the live viewer."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from robo_visualize.arms.piperx.live_server import PiperXLiveModel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str) -> dict[str, Any]:
    if path == "-":
        payload = json.load(sys.stdin)
    else:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PiperX action JSON must be an object")
    return payload


def _actions(payload: dict[str, Any]) -> np.ndarray:
    actions = np.asarray(payload.get("actions"), dtype=np.float64)
    if (
        actions.ndim != 2
        or actions.shape[1] != 7
        or len(actions) < 2
        or not np.all(np.isfinite(actions))
    ):
        raise ValueError("PiperX action JSON must contain finite [frames, 7] actions")
    return actions


def _resample_actions(actions: np.ndarray, frame_count: int) -> np.ndarray:
    if frame_count < 2:
        raise ValueError("display trajectory requires at least two frames")
    source_time = np.linspace(0.0, 1.0, len(actions), dtype=np.float64)
    target_time = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)
    return np.column_stack(
        [
            np.interp(target_time, source_time, actions[:, column])
            for column in range(actions.shape[1])
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions-json", required=True)
    parser.add_argument("--rgbd-manifest", type=Path, required=True)
    parser.add_argument("--replay-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    action_payload = _read_json(args.actions_json)
    actions = _actions(action_payload)
    rgbd_manifest = json.loads(args.rgbd_manifest.read_text(encoding="utf-8"))
    replay_manifest = json.loads(args.replay_manifest.read_text(encoding="utf-8"))
    frame_count = int(rgbd_manifest["wrist"]["frame_count"])
    display_actions = _resample_actions(actions, frame_count)
    assets_root = (
        Path(__file__).resolve().parents[1]
        / "src/robo_visualize/arms/piperx/assets"
    )
    model = PiperXLiveModel(assets_root)
    poses = [
        model.link6_pose_base_wxyz(action)
        for action in display_actions
    ]
    payload = {
        "schema": "robo_visualize.piperx_live_expert_trajectory.v1",
        "coordinate_frame": "base_link",
        "controlled_frame": "link6",
        "frame_indices": list(range(frame_count)),
        "link6_poses_base_wxyz": poses,
        "source": {
            "variant_id": args.rgbd_manifest.parent.name,
            "rgbd_manifest_sha256": _sha256(args.rgbd_manifest),
            "action_dataset": replay_manifest["repo_id"],
            "action_sha256": replay_manifest["action_sha256"],
            "source_parquet_sha256": replay_manifest["source_parquet_sha256"],
            "source_action_frames": len(actions),
            "display_rgbd_frames": frame_count,
            "frame_alignment": (
                "normalized-time interpolation from action frames to wrist RGB-D frames"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
