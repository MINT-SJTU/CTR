"""Dependency-light live PiperX viewer for an already-running SID policy."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from flask import Response, jsonify

from robo_visualize import web as rv_web
from robo_visualize.web.server import create_app as _create_app


TCP_OFFSET_M = 0.1358
GRIPPER_STROKE_M = 0.035
JOINT_ORIGINS = (
    ((0.0, 0.0, 0.123), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.5708, -0.13586, 3.1416)),
    ((0.28503, 0.0, 0.0), (0.0, 0.0, 2.8377)),
    ((0.27264, 0.0, 0.0), (0.0, 0.0, 0.075129)),
    ((0.076857, 0.00062395, 0.0), (-1.5708, 0.0, 0.0)),
    ((0.035, 0.0, 0.0), (1.5708, 0.0, 1.5708)),
)


def _rotation_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _transform(
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = _rotation_rpy(*rpy)
    value[:3, 3] = translation
    return value


def _translation(value: tuple[float, float, float]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = value
    return result


def _quaternion_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("live pose quaternion must be a finite four-vector")
    norm = float(np.linalg.norm(quaternion))
    if not np.isclose(norm, 1.0, atol=1e-5):
        raise ValueError("live pose quaternion must be normalized")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError("rotation matrix must be a finite 3x3 array")
    trace = float(np.trace(value))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        quaternion = [
            0.25 * scale,
            (value[2, 1] - value[1, 2]) / scale,
            (value[0, 2] - value[2, 0]) / scale,
            (value[1, 0] - value[0, 1]) / scale,
        ]
    else:
        axis = int(np.argmax(np.diag(value)))
        next_axis = (axis + 1) % 3
        last_axis = (axis + 2) % 3
        scale = 2.0 * math.sqrt(
            1.0
            + value[axis, axis]
            - value[next_axis, next_axis]
            - value[last_axis, last_axis]
        )
        quaternion = [0.0, 0.0, 0.0, 0.0]
        quaternion[axis + 1] = 0.25 * scale
        quaternion[0] = (
            value[last_axis, next_axis] - value[next_axis, last_axis]
        ) / scale
        quaternion[next_axis + 1] = (
            value[next_axis, axis] + value[axis, next_axis]
        ) / scale
        quaternion[last_axis + 1] = (
            value[last_axis, axis] + value[axis, last_axis]
        ) / scale
    result = np.asarray(quaternion, dtype=np.float64)
    return result / np.linalg.norm(result)


def _pose(value: Any, *, name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must be a finite [xyz,qwxyz] pose")
    _quaternion_rotation(pose[3:])
    return pose


def _tcp_pose(link6_pose: Any) -> list[float]:
    pose = _pose(link6_pose, name="link6 pose")
    position = pose[:3] + _quaternion_rotation(pose[3:]) @ np.asarray(
        [0.0, 0.0, TCP_OFFSET_M], dtype=np.float64
    )
    return np.r_[position, pose[3:]].tolist()


class PiperXLiveModel:
    def __init__(self, assets_root: str | Path) -> None:
        root = Path(assets_root).resolve()
        specs = (
            ("base_link", root / "meshes/dae/base_link.dae", "base_link"),
            ("link1", root / "meshes/dae/J1.dae", "link1"),
            ("link2", root / "meshes/dae/J2.dae", "link2"),
            ("link3", root / "meshes/dae/J3.dae", "link3"),
            ("link4", root / "meshes/dae/J4.dae", "link4"),
            ("link5", root / "meshes/dae/J5.dae", "link5"),
            ("link6", root / "meshes/Link6.STL", "link6"),
            ("gripper_base", root / "gripper/gripper_base.STL", "gripper_base"),
            ("link7", root / "gripper/link7.STL", "link7"),
            ("link8", root / "gripper/link8.STL", "link8"),
        )
        missing = [str(path) for _, path, _ in specs if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"PiperX live viewer meshes are missing: {missing}")
        self.meshes = tuple(specs)

    @staticmethod
    def _state_and_placements(action: Any) -> tuple[np.ndarray, list[np.ndarray]]:
        state = np.asarray(action, dtype=np.float64)
        if state.shape != (7,) or not np.all(np.isfinite(state)):
            raise ValueError("PiperX live action must be a finite seven-vector")
        placements = [np.eye(4, dtype=np.float64)]
        current = placements[0]
        for (translation, rpy), angle_deg in zip(JOINT_ORIGINS, state[:6], strict=True):
            current = current @ _transform(translation, rpy) @ _transform(
                rpy=(0.0, 0.0, math.radians(float(angle_deg)))
            )
            placements.append(current)
        return state, placements

    def link6_pose_base_wxyz(self, action: Any) -> list[float]:
        _, placements = self._state_and_placements(action)
        link6 = placements[6]
        quaternion = _rotation_quaternion_wxyz(link6[:3, :3])
        return np.r_[link6[:3, 3], quaternion].tolist()

    def transforms(self, action: Any) -> list[np.ndarray]:
        state, placements = self._state_and_placements(action)
        gripper = placements[6] @ _transform(rpy=(0.0, 0.0, math.pi / 2.0))
        opening_m = float(np.clip(state[6] / 1000.0, 0.0, GRIPPER_STROKE_M))
        link7 = (
            gripper
            @ _transform((0.0, 0.0, TCP_OFFSET_M), (math.pi / 2.0, 0.0, 0.0))
            @ _translation((0.0, 0.0, opening_m))
        )
        link8 = (
            gripper
            @ _transform((0.0, 0.0, TCP_OFFSET_M), (math.pi / 2.0, 0.0, -math.pi))
            @ _translation((0.0, 0.0, opening_m))
        )
        return [
            placements[0],
            *placements[1:6],
            placements[6] @ _translation((0.0, 0.0, -0.035)),
            gripper,
            link7,
            link8,
        ]


class LiveViewerBackend:
    def __init__(
        self,
        state_path: str | Path,
        expert_trajectory_path: str | Path,
        *,
        assets_root: str | Path | None = None,
    ) -> None:
        self.state_path = Path(state_path).resolve()
        source = Path(expert_trajectory_path).resolve()
        expert = json.loads(source.read_text(encoding="utf-8"))
        if expert.get("schema") != "robo_visualize.piperx_live_expert_trajectory.v1":
            raise ValueError("unexpected live expert trajectory schema")
        frame_indices = expert.get("frame_indices")
        poses = expert.get("link6_poses_base_wxyz")
        if not isinstance(frame_indices, list) or not frame_indices:
            raise ValueError("live expert trajectory has no frame indices")
        if not isinstance(poses, list) or not poses:
            raise ValueError("live expert trajectory has no link6 poses")
        if len(frame_indices) != len(poses):
            raise ValueError("live expert frame indices and poses have different lengths")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in frame_indices
        ) or len(set(frame_indices)) != len(frame_indices):
            raise ValueError("live expert frame indices must be unique nonnegative integers")
        self.expert_link6_poses = tuple(
            _pose(value, name=f"expert frame {frame_index}")
            for frame_index, value in zip(frame_indices, poses, strict=True)
        )
        self.expert_link6_pose_by_frame = dict(
            zip(frame_indices, self.expert_link6_poses, strict=True)
        )
        self.expert_source = expert.get("source")
        if assets_root is None:
            from robo_visualize.assets import assets_root as default_assets_root
            assets_root = default_assets_root()
        self.model = PiperXLiveModel(assets_root)
        tcp_positions = np.asarray(
            [_tcp_pose(pose)[:3] for pose in self.expert_link6_poses],
            dtype="<f4",
        )
        self._expert_tcp_bytes = tcp_positions.tobytes()

    @property
    def mesh_count(self) -> int:
        return len(self.model.meshes)

    def _read_state(self) -> dict[str, Any]:
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if payload.get("schema") != "up_vla.sid_live_visualization.v2":
            raise ValueError("unexpected SID live visualization schema")
        action = np.asarray(payload.get("current_action_deg_mm"), dtype=np.float64)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError("SID live visualization action is invalid")
        payload["current_action_deg_mm"] = action.tolist()
        payload["current_link6_pose_base_wxyz"] = _pose(
            payload.get("current_link6_pose_base_wxyz"),
            name="current live link6 pose",
        ).tolist()
        endpoint = payload.get("predicted_endpoint_link6_pose_base_wxyz")
        if endpoint is not None:
            payload["predicted_endpoint_link6_pose_base_wxyz"] = _pose(
                endpoint,
                name="predicted live link6 endpoint",
            ).tolist()
        return payload

    def config(self) -> dict[str, Any]:
        return {
            "loader": "mixed",
            "sideLabels": ["live Piper X"],
            "sides": ["piper_x"],
            "cameras": [],
            "trajectory_signals": [],
            "preferred_episode_index": 0,
            "datasetScanner": False,
            "live_mode": True,
            "live_state_url": "/api/live_state.json",
            "live_poll_interval_ms": 100,
            "live_stale_after_s": 2.0,
            "overlay": {
                "has_ee_trajectory": True,
                "has_perturbations": False,
                "has_sid_recovery": False,
            },
            "mesh_count": self.mesh_count,
            "meshes": [
                {
                    "index": index,
                    "name": name,
                    "link_name": link_name,
                    "scale": [1.0, 1.0, 1.0],
                    "url": f"/api/mesh/{index}{path.suffix.lower()}",
                }
                for index, (name, path, link_name) in enumerate(self.model.meshes)
            ],
            "datasets": {"live": {"episode_count": 1, "fps": 20.0}},
        }

    def episode_payload(
        self,
        dataset: str,
        episode_index: int,
        signal: str | None = None,
    ) -> dict[str, Any]:
        del signal
        self._require_episode(dataset, episode_index)
        return {
            "dataset": "live",
            "episode_index": 0,
            "frame_count": 1,
            "fps": 20.0,
            "duration_s": 0.05,
            "task": ["SID live recovery"],
            "videos": {},
            "transforms_url": "/api/transforms/live/0.bin",
            "ee_trajectory_url": "/api/ee_trajectory/live/0.bin",
        }

    @staticmethod
    def _require_episode(dataset: str, episode_index: int) -> None:
        if dataset != "live" or episode_index != 0:
            raise ValueError("PiperX live viewer only exposes live episode 0")

    @staticmethod
    def _packed_transforms(transforms: list[np.ndarray]) -> bytes:
        packed = np.asarray([value.T.reshape(16) for value in transforms], dtype="<f4")
        return packed.tobytes()

    def transforms(
        self,
        dataset: str,
        episode_index: int,
        signal: str | None = None,
    ) -> bytes:
        del signal
        self._require_episode(dataset, episode_index)
        state = self._read_state()
        return self._packed_transforms(self.model.transforms(state["current_action_deg_mm"]))

    def ee_trajectory(
        self,
        dataset: str,
        episode_index: int,
        signal: str | None = None,
    ) -> bytes:
        del signal
        self._require_episode(dataset, episode_index)
        return self._expert_tcp_bytes

    def live_state(self) -> dict[str, Any]:
        payload = self._read_state()
        transforms = self.model.transforms(payload["current_action_deg_mm"])
        nearest_frame = payload.get("nearest_expert_frame")
        nearest_pose = None
        if nearest_frame is not None:
            if (
                isinstance(nearest_frame, bool)
                or not isinstance(nearest_frame, int)
                or nearest_frame not in self.expert_link6_pose_by_frame
            ):
                raise ValueError("nearest expert frame is outside the displayed trajectory")
            nearest_pose = self.expert_link6_pose_by_frame[nearest_frame]
        endpoint = payload.get("predicted_endpoint_link6_pose_base_wxyz")
        wall_time = float(payload["wall_time_unix_s"])
        return {
            **payload,
            "source_age_s": max(0.0, time.time() - wall_time),
            "mesh_transforms_column_major": [
                value.T.reshape(16).tolist() for value in transforms
            ],
            "current_tcp_pose_base_wxyz": _tcp_pose(
                payload["current_link6_pose_base_wxyz"]
            ),
            "predicted_endpoint_tcp_pose_base_wxyz": (
                None if endpoint is None else _tcp_pose(endpoint)
            ),
            "nearest_expert_tcp_pose_base_wxyz": (
                None if nearest_pose is None else _tcp_pose(nearest_pose)
            ),
            "expert_frame_count": len(self.expert_link6_poses),
        }

    def mesh_path(self, mesh_index: int) -> Path:
        if not 0 <= mesh_index < self.mesh_count:
            raise ValueError(f"PiperX live mesh index out of range: {mesh_index}")
        return self.model.meshes[mesh_index][1]

    def mesh_asset_path(self, filename: str) -> Path:
        raise FileNotFoundError(f"PiperX live viewer has no auxiliary mesh: {filename}")

    def video_path(self, dataset: str, episode_index: int, camera: str) -> Path:
        raise FileNotFoundError(
            f"PiperX live viewer has no recorded video: {dataset}/{episode_index}/{camera}"
        )


def create_app(
    state_path: str | Path,
    expert_trajectory_path: str | Path,
    *,
    assets_root: str | Path | None = None,
):
    backend = LiveViewerBackend(
        state_path,
        expert_trajectory_path,
        assets_root=assets_root,
    )
    app = _create_app(
        backend,
        Path(rv_web.__file__).parent / "static",
        title="Piper X · SID live recovery",
    )

    @app.get("/api/live_state.json")
    def live_state() -> Response:
        response = jsonify(backend.live_state())
        response.headers["Cache-Control"] = "no-store"
        return response

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18876)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--expert-trajectory", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = create_app(args.state_file, args.expert_trajectory)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
