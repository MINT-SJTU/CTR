from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from robo_visualize import web as rv_web
from robo_visualize.arms.piperx.dataset import (
    CAMERAS,
    TRAJECTORY_SIGNAL,
    TRAJECTORY_SIGNALS,
    PiperXDatasetView,
    PiperXTrajectory,
)
from robo_visualize.arms.piperx.model import PiperXModel, pack_visual_transforms
from robo_visualize.web.server import create_app as _create_app


@dataclass(frozen=True)
class PerturbationPoint:
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    residual_m: float | None = None
    point_id: str = ""


def _load_perturbation_points(path: str | Path | None) -> tuple[PerturbationPoint, ...]:
    if path is None:
        return ()
    source = Path(path).resolve()
    if source.is_file() and source.suffix.lower() == ".jsonl":
        points = []
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                pose = row.get(
                    "augmentation.p_pose_base_f64",
                    row.get("pose", row.get("position", row.get("position_m"))),
                )
                if pose is None or len(pose) < 3:
                    position = row.get("position")
                    quaternion = row.get("quaternion")
                else:
                    position = pose[:3]
                    quaternion = pose[3:7] if len(pose) >= 7 else None
                if position is None:
                    raise ValueError(f"perturbation point row lacks position: {line[:200]}")
                if quaternion is None:
                    quaternion = (1.0, 0.0, 0.0, 0.0)
                points.append(
                    PerturbationPoint(
                        position=tuple(float(value) for value in position),
                        quaternion=tuple(float(value) for value in quaternion),
                        residual_m=(
                            float(row["augmentation.projection_residual_m_f64"])
                            if "augmentation.projection_residual_m_f64" in row
                            else None
                        ),
                        point_id=str(row.get("augmentation.point_id", row.get("point_id", ""))),
                    )
                )
        return tuple(points)
    if source.is_dir():
        parquet_files = sorted(
            path for path in source.rglob("file-*.parquet") if path.is_file()
        )
        points = []
        for parquet_file in parquet_files:
            table = pq.read_table(
                parquet_file,
                columns=[
                    "augmentation.p_pose_base_f64",
                    "augmentation.q_pose_base_f64",
                    "augmentation.point_id",
                    "augmentation.projection_residual_m_f64",
                ],
            )
            for row in table.to_pylist():
                pose = row["augmentation.p_pose_base_f64"]
                points.append(
                    PerturbationPoint(
                        position=tuple(float(value) for value in pose[:3]),
                        quaternion=tuple(float(value) for value in pose[3:7]),
                        residual_m=float(row["augmentation.projection_residual_m_f64"])
                        if row["augmentation.projection_residual_m_f64"] is not None
                        else None,
                        point_id=str(row["augmentation.point_id"]),
                    )
                )
        return tuple(points)
    raise ValueError(f"unsupported perturbation point source: {source}")


def _pose(value: object, *, name: str) -> list[float]:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"{name} must be a finite [xyz,qwxyz] pose")
    if not np.isclose(np.linalg.norm(pose[3:]), 1.0, atol=1e-5):
        raise ValueError(f"{name} quaternion must be normalized")
    return pose.tolist()


def _trajectory(value: object, *, name: str) -> list[list[float]]:
    trajectory = np.asarray(value, dtype=np.float64)
    if (
        trajectory.ndim != 2
        or trajectory.shape[1] != 7
        or not len(trajectory)
        or not np.isfinite(trajectory).all()
    ):
        raise ValueError(f"{name} must be a non-empty finite [N,7] pose trajectory")
    norms = np.linalg.norm(trajectory[:, 3:], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError(f"{name} quaternions must be normalized")
    return trajectory.tolist()


def _load_recovery_vectors(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"SID recovery vectors are missing: {source}")
    payload = json.loads(source.read_text())
    if payload.get("schema") != "robo_visualize.sid_recovery_vectors.v2":
        raise ValueError("unexpected SID recovery vectors schema")
    if payload.get("coordinate_frame") != "base_link":
        raise ValueError("SID recovery vectors must use base_link coordinates")
    if payload.get("controlled_frame") != "link6":
        raise ValueError("SID recovery vectors must control PiperX link6")
    episode_index = payload.get("expert_episode_index")
    if not isinstance(episode_index, int) or episode_index < 0:
        raise ValueError("SID recovery vectors require an expert episode index")
    manifold = payload.get("manifold")
    samples = payload.get("samples")
    aggregate = payload.get("aggregate_metrics")
    if not isinstance(manifold, dict) or not isinstance(samples, list) or not samples:
        raise ValueError("SID recovery vectors require manifold and sample data")
    if (
        not isinstance(aggregate, dict)
        or aggregate.get("sample_count") != len(samples)
    ):
        raise ValueError("SID aggregate sample count differs from the sample list")
    manifold["full_trajectory_poses_base_link6_wxyz"] = _trajectory(
        manifold.get("full_trajectory_poses_base_link6_wxyz"),
        name="full SID trajectory",
    )
    manifold["approach_trajectory_poses_base_link6_wxyz"] = _trajectory(
        manifold.get("approach_trajectory_poses_base_link6_wxyz"),
        name="SID approach trajectory",
    )
    sample_indices = set()
    metric_names = (
        "base_translation_direction_cosine",
        "nearest_manifold_distance_before",
        "nearest_manifold_distance_after",
        "nearest_manifold_distance_reduction_fraction",
    )
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("SID recovery samples must be objects")
        sample_index = sample.get("sample_index")
        if (
            not isinstance(sample_index, int)
            or sample_index < 0
            or sample_index in sample_indices
        ):
            raise ValueError(
                "SID recovery sample indices must be unique nonnegative integers"
            )
        sample_indices.add(sample_index)
        for key in (
            "p_pose_base_link6_wxyz",
            "attached_episode_q_pose_base_link6_wxyz",
            "predicted_endpoint_pose_base_link6_wxyz",
        ):
            sample[key] = _pose(
                sample.get(key),
                name=f"sample {sample_index} {key}",
            )
        metrics = sample.get("metrics")
        if not isinstance(metrics, dict) or any(
            not isinstance(metrics.get(name), (int, float))
            or not np.isfinite(metrics[name])
            for name in metric_names
        ):
            raise ValueError(f"SID recovery sample {sample_index} has invalid metrics")
    return payload


class ViewerBackend:
    def __init__(
        self,
        datasets: dict[str, str | Path] | None = None,
        datasets_root: str | Path | None = None,
        dataset_path: str | Path | None = None,
        perturbation_points_path: str | Path | None = None,
        recovery_vectors_path: str | Path | None = None,
    ) -> None:
        self.datasets_root = Path(datasets_root).resolve() if datasets_root else None
        configured: dict[str, str | Path] = dict(datasets or {})
        if dataset_path:
            path = Path(dataset_path).resolve()
            configured.setdefault("piperx", path)
        self.datasets = {
            name: PiperXDatasetView.load(name, root)
            for name, root in configured.items()
        }
        self._dataset_cache: dict[str, tuple[tuple[int, int], PiperXDatasetView]] = {}
        self.model = PiperXModel()
        self._transform_lock = threading.Lock()
        self._perturbation_points = _load_perturbation_points(perturbation_points_path)
        self._recovery_vectors = _load_recovery_vectors(recovery_vectors_path)

    @property
    def mesh_count(self) -> int:
        return len(self.model.meshes)

    def available_datasets(self) -> dict[str, PiperXDatasetView]:
        if self.datasets_root is None:
            return self.datasets
        discovered: list[tuple[int, str, PiperXDatasetView]] = []
        if not self.datasets_root.is_dir():
            return {}
        for path in self.datasets_root.iterdir():
            info_path = path / "meta" / "info.json"
            if not path.is_dir() or not info_path.is_file():
                continue
            signature = (info_path.stat().st_mtime_ns, info_path.stat().st_mtime_ns)
            cached = self._dataset_cache.get(path.name)
            if cached is None or cached[0] != signature:
                cached = (signature, PiperXDatasetView.load(path.name, path))
                self._dataset_cache[path.name] = cached
            discovered.append((path.stat().st_mtime_ns, path.name, cached[1]))
        return {
            name: dataset
            for _, name, dataset in sorted(discovered, reverse=True)
        }

    def config(self) -> dict[str, Any]:
        datasets = self.available_datasets()
        meshes = []
        canonical_meshes: dict[Path, int] = {}
        for mesh in self.model.meshes:
            canonical_index = canonical_meshes.setdefault(mesh.mesh_path, mesh.output_index)
            meshes.append(
                {
                    "index": mesh.output_index,
                    "name": mesh.name,
                    "link_name": mesh.link_name,
                    "scale": mesh.scale,
                    "url": f"/api/mesh/{canonical_index}{mesh.mesh_path.suffix.lower() or '.stl'}",
                }
            )
        return {
            "loader": "mixed",
            "sideLabels": ["piper_x"],
            "sides": ["piper_x"],
            "cameras": list(CAMERAS),
            "master_camera": "left_wrist",
            "trajectory_signals": list(TRAJECTORY_SIGNALS),
            "default_trajectory_signal": TRAJECTORY_SIGNAL,
            "preferred_episode_index": (
                self._recovery_vectors["expert_episode_index"]
                if self._recovery_vectors
                else 0
            ),
            "datasetScanner": self.datasets_root is not None,
            "overlay": {
                "has_ee_trajectory": True,
                "has_perturbations": len(self._perturbation_points) > 0,
                "points_url": "/api/perturbation_points.json"
                if self._perturbation_points
                else None,
                "has_sid_recovery": self._recovery_vectors is not None,
                "sid_recovery_url": (
                    "/api/recovery_vectors.json" if self._recovery_vectors else None
                ),
                "sid_display_frame": "tcp" if self._recovery_vectors else None,
                "sid_controlled_frame_to_display_pose_wxyz": (
                    list(self.model.relative_frame_pose("link6", "tcp"))
                    if self._recovery_vectors
                    else None
                ),
            },
            "mesh_count": self.mesh_count,
            "meshes": meshes,
            "datasets": {
                name: {"episode_count": len(dataset.episodes), "fps": dataset.fps}
                for name, dataset in datasets.items()
            },
        }

    def episode_payload(
        self,
        dataset_name: str,
        episode_index: int,
        signal: str | None = None,
    ) -> dict[str, Any]:
        del signal
        dataset = self.dataset(dataset_name)
        episode = dataset.episode(episode_index)
        videos = {}
        for camera in CAMERAS:
            timing = dataset.video_timing(episode_index, camera)
            videos[camera] = {
                **timing,
                "url": f"/api/video/{dataset_name}/{episode_index}/{camera}.mp4",
            }
        duration = min(item["to_s"] - item["from_s"] for item in videos.values())
        payload = {
            "dataset": dataset_name,
            "episode_index": episode_index,
            "frame_count": int(episode["length"]),
            "fps": dataset.fps,
            "duration_s": max(0.0, float(duration)),
            "task": list(episode.get("tasks", [])),
            "videos": videos,
            "transforms_url": f"/api/transforms/{dataset_name}/{episode_index}.bin",
        }
        payload["ee_trajectory_url"] = (
            f"/api/ee_trajectory/{dataset_name}/{episode_index}.bin"
        )
        return payload

    def dataset(self, dataset_name: str) -> PiperXDatasetView:
        datasets = self.available_datasets()
        if dataset_name not in datasets:
            raise ValueError(f"unknown dataset mode: {dataset_name}")
        return datasets[dataset_name]

    def mesh_path(self, mesh_index: int) -> Path:
        if mesh_index < 0 or mesh_index >= self.mesh_count:
            raise ValueError(f"mesh index out of range: {mesh_index}")
        paths = []
        seen = set()
        for mesh in self.model.meshes:
            if mesh.mesh_path in seen:
                continue
            seen.add(mesh.mesh_path)
            paths.append(mesh.mesh_path)
        return paths[mesh_index]

    def mesh_asset_path(self, filename: str) -> Path:
        raise FileNotFoundError(f"Piper X has no auxiliary mesh asset: {filename}")

    def transforms(
        self,
        dataset_name: str,
        episode_index: int,
        signal: str | None = None,
    ) -> bytes:
        if signal is None:
            signal = TRAJECTORY_SIGNAL
        if signal not in TRAJECTORY_SIGNALS:
            raise ValueError(f"unknown trajectory signal: {signal}")
        dataset = self.dataset(dataset_name)
        trajectory = PiperXTrajectory.from_lerobot_parquet(
            dataset.data_path(episode_index),
            episode_index,
            signal=signal,
        )
        with self._transform_lock:
            return pack_visual_transforms(self.model, trajectory.state_values)

    def video_path(self, dataset_name: str, episode_index: int, camera: str) -> Path:
        path = self.dataset(dataset_name).video_path(episode_index, camera)
        if not path.is_file():
            raise FileNotFoundError(f"video missing: {path}")
        return path

    def ee_trajectory(
        self,
        dataset_name: str,
        episode_index: int,
        signal: str | None = None,
    ) -> bytes:
        if signal is None:
            signal = TRAJECTORY_SIGNAL
        dataset = self.dataset(dataset_name)
        trajectory = PiperXTrajectory.from_lerobot_parquet(
            dataset.data_path(episode_index),
            episode_index,
            signal=signal,
        )
        with self._transform_lock:
            positions = np.empty((trajectory.frame_count, 3), dtype="<f4")
            for frame_index, state in enumerate(trajectory.state_values):
                self.model.update(state)
                positions[frame_index] = self.model.tcp_transform().translation
            return positions.tobytes()

    def perturbation_points(self) -> dict[str, Any]:
        return {
            "points": [
                {
                    "point_id": point.point_id,
                    "position": list(point.position),
                    "quaternion": list(point.quaternion),
                    "residual_m": point.residual_m,
                }
                for point in self._perturbation_points
            ]
        }

    def recovery_vectors(self) -> dict[str, Any]:
        if self._recovery_vectors is None:
            raise ValueError("SID recovery vectors were not configured")
        return self._recovery_vectors


def create_app(
    datasets: dict[str, str | Path] | None = None,
    datasets_root: str | Path | None = None,
    dataset_path: str | Path | None = None,
    perturbation_points_path: str | Path | None = None,
    recovery_vectors_path: str | Path | None = None,
):
    backend = ViewerBackend(
        datasets,
        datasets_root,
        dataset_path,
        perturbation_points_path,
        recovery_vectors_path,
    )
    return _create_app(
        backend,
        Path(rv_web.__file__).parent / "static",
        title="Piper X viewer",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the interactive Piper X LeRobot dataset viewer"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Direct LeRobot dataset root (contains meta/, data/, videos/).",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        help="Scan immediate child folders for LeRobot datasets and allow switching at runtime.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Explicit NAME=PATH dataset entries; repeatable.",
    )
    parser.add_argument(
        "--perturbation-points",
        type=Path,
        help="JSONL sidecar or extracted LeRobot directory with augmenting P poses.",
    )
    parser.add_argument(
        "--recovery-vectors",
        type=Path,
        help="SID recovery_vectors.json with Episode 3 link6 trajectory and sampled arrows.",
    )
    return parser


def _parse_dataset_entries(values: list[str]) -> dict[str, Path]:
    entries: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--dataset entries must be NAME=PATH")
        name, raw_path = value.split("=", 1)
        entries[name] = Path(raw_path)
    return entries


def main() -> None:
    args = build_parser().parse_args()
    datasets = _parse_dataset_entries(args.dataset)
    app = create_app(
        datasets=datasets or None,
        datasets_root=args.datasets_root,
        dataset_path=args.dataset_path,
        perturbation_points_path=args.perturbation_points,
        recovery_vectors_path=args.recovery_vectors,
    )
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
