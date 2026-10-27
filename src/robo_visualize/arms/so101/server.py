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
from robo_visualize.arms.so101.model import SO101Model
from robo_visualize.arms.so101.trajectory import SO101Trajectory
from robo_visualize.web.server import create_app as _create_app

CAMERAS = ("left_wrist", "right_front", "right_wrist")
SIDES = ("left", "right")
TRAJECTORY_SIGNAL = "observation.state"
TRAJECTORY_SIGNALS = (TRAJECTORY_SIGNAL, "action")


def discover_datasets(root: str | Path) -> dict[str, Path]:
    path = Path(root).resolve()
    discovered: dict[str, Path] = {}
    if not path.is_dir():
        return discovered
    for child in path.iterdir():
        if not child.is_dir():
            continue
        if not (child / "meta" / "info.json").is_file():
            continue
        if not (child / "meta" / "episodes" / "chunk-000" / "file-000.parquet").is_file():
            continue
        discovered[child.name] = child.resolve()
    return discovered


@dataclass(frozen=True)
class DatasetView:
    name: str
    root: Path
    fps: float
    episodes: tuple[dict[str, Any], ...]

    @classmethod
    def load(cls, name: str, root: str | Path) -> "DatasetView":
        path = Path(root).resolve()
        info_path = path / "meta" / "info.json"
        episodes_path = path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        if not info_path.is_file():
            raise FileNotFoundError(f"dataset metadata missing: {info_path}")
        if not episodes_path.is_file():
            raise FileNotFoundError(f"episode metadata missing: {episodes_path}")
        info = json.loads(info_path.read_text())
        episodes = tuple(pq.read_table(episodes_path).to_pylist())
        return cls(name, path, float(info["fps"]), episodes)

    def episode(self, episode_index: int) -> dict[str, Any]:
        if episode_index < 0 or episode_index >= len(self.episodes):
            raise ValueError(f"episode {episode_index} is outside {self.name} dataset")
        episode = self.episodes[episode_index]
        if int(episode["episode_index"]) != episode_index:
            raise ValueError(f"episode metadata is not ordered at index {episode_index}")
        return episode

    def data_path(self, episode_index: int) -> Path:
        episode = self.episode(episode_index)
        return (
            self.root
            / "data"
            / f"chunk-{int(episode['data/chunk_index']):03d}"
            / (f"file-{int(episode['data/file_index']):03d}.parquet")
        )

    def video_path(self, episode_index: int, camera: str) -> Path:
        if camera not in CAMERAS:
            raise ValueError(f"unknown camera: {camera}")
        episode = self.episode(episode_index)
        prefix = f"videos/observation.images.{camera}"
        return (
            self.root
            / "videos"
            / f"observation.images.{camera}"
            / (f"chunk-{int(episode[f'{prefix}/chunk_index']):03d}")
            / f"file-{int(episode[f'{prefix}/file_index']):03d}.mp4"
        )

    def video_timing(self, episode_index: int, camera: str) -> dict[str, float]:
        if camera not in CAMERAS:
            raise ValueError(f"unknown camera: {camera}")
        episode = self.episode(episode_index)
        prefix = f"videos/observation.images.{camera}"
        return {
            "from_s": float(episode[f"{prefix}/from_timestamp"]),
            "to_s": float(episode[f"{prefix}/to_timestamp"]),
        }


def pack_visual_transforms(model: SO101Model, joint_degrees: np.ndarray) -> bytes:
    values = np.asarray(joint_degrees, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (2, 6):
        raise ValueError(
            f"expected joint trajectory shape (frames, 2, 6), got {values.shape}"
        )
    arms = (model.arm("left"), model.arm("right"))
    mesh_count = len(model.visual_model.geometryObjects)
    transforms = np.empty((len(values), len(SIDES), mesh_count, 16), dtype="<f4")
    for frame_index, frame in enumerate(values):
        for side_index, arm in enumerate(arms):
            arm.update(frame[side_index])
            for mesh_index in range(mesh_count):
                transforms[frame_index, side_index, mesh_index] = (
                    arm.visual_transform_matrix(mesh_index).T.reshape(16)
                )
    return transforms.tobytes()


class ViewerBackend:
    def __init__(
        self,
        datasets: dict[str, str | Path] | None = None,
        datasets_root: str | Path | None = None,
    ) -> None:
        self.datasets_root = Path(datasets_root).resolve() if datasets_root else None
        self.datasets = {
            name: DatasetView.load(name, root)
            for name, root in (datasets or {}).items()
        }
        self._dataset_cache: dict[str, tuple[tuple[int, int], DatasetView]] = {}
        self.model = SO101Model()
        self._transform_lock = threading.Lock()

    @property
    def mesh_count(self) -> int:
        return len(self.model.visual_model.geometryObjects)

    def available_datasets(self) -> dict[str, DatasetView]:
        if self.datasets_root is None:
            return self.datasets
        discovered: list[tuple[int, str, DatasetView]] = []
        if not self.datasets_root.is_dir():
            return {}
        for path in self.datasets_root.iterdir():
            info_path = path / "meta" / "info.json"
            episodes_path = path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
            if not path.is_dir() or not info_path.is_file() or not episodes_path.is_file():
                continue
            signature = (info_path.stat().st_mtime_ns, episodes_path.stat().st_mtime_ns)
            cached = self._dataset_cache.get(path.name)
            if cached is None or cached[0] != signature:
                cached = (signature, DatasetView.load(path.name, path))
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
        for index, geometry in enumerate(self.model.visual_model.geometryObjects):
            mesh_path = Path(geometry.meshPath)
            canonical_index = canonical_meshes.setdefault(mesh_path, index)
            meshes.append(
                {
                    "index": index,
                    "name": geometry.name,
                    "scale": np.asarray(geometry.meshScale, dtype=float).tolist(),
                    "url": f"/api/mesh/{canonical_index}.stl",
                }
            )
        return {
            "loader": "stl",
            "sideLabels": ["left", "right"],
            "sides": list(SIDES),
            "cameras": list(CAMERAS),
            "master_camera": "left_wrist",
            "trajectory_signals": list(TRAJECTORY_SIGNALS),
            "default_trajectory_signal": TRAJECTORY_SIGNAL,
            "datasetScanner": self.datasets_root is not None,
            "mesh_count": self.mesh_count,
            "meshes": meshes,
            "datasets": {
                name: {"episode_count": len(dataset.episodes), "fps": dataset.fps}
                for name, dataset in datasets.items()
            },
        }

    def episode_payload(
        self,
        mode: str,
        episode_index: int,
        signal: str | None = None,
    ) -> dict[str, Any]:
        del signal
        dataset = self.dataset(mode)
        episode = dataset.episode(episode_index)
        videos = {}
        for camera in CAMERAS:
            timing = dataset.video_timing(episode_index, camera)
            videos[camera] = {
                **timing,
                "url": f"/api/video/{mode}/{episode_index}/{camera}.mp4",
            }
        duration = min(item["to_s"] - item["from_s"] for item in videos.values())
        crop_count = 0
        if episode.get("retime/crop_frames_json"):
            crop_count = len(json.loads(episode["retime/crop_frames_json"]))
        return {
            "dataset": mode,
            "episode_index": episode_index,
            "frame_count": int(episode["length"]),
            "fps": dataset.fps,
            "duration_s": duration,
            "task": list(episode.get("tasks", [])),
            "crop_count": crop_count,
            "videos": videos,
            "transforms_url": f"/api/transforms/{mode}/{episode_index}.bin",
        }

    def dataset(self, mode: str) -> DatasetView:
        datasets = self.available_datasets()
        if mode not in datasets:
            raise ValueError(f"unknown dataset mode: {mode}")
        return datasets[mode]

    def mesh_path(self, mesh_index: int) -> Path:
        if mesh_index < 0 or mesh_index >= self.mesh_count:
            raise ValueError(f"mesh index out of range: {mesh_index}")
        path = Path(self.model.visual_model.geometryObjects[mesh_index].meshPath)
        if not path.is_file():
            raise FileNotFoundError(f"visual mesh missing: {path}")
        return path

    def mesh_asset_path(self, filename: str) -> Path:
        raise FileNotFoundError(f"SO101 has no auxiliary mesh asset: {filename}")

    def transforms(
        self,
        mode: str,
        episode_index: int,
        signal: str | None = None,
    ) -> bytes:
        if signal is None:
            signal = TRAJECTORY_SIGNAL
        if signal not in TRAJECTORY_SIGNALS:
            raise ValueError(f"unknown trajectory signal: {signal}")
        dataset = self.dataset(mode)
        trajectory = SO101Trajectory.from_lerobot_parquet(
            dataset.data_path(episode_index),
            episode_index,
            signal=signal,
        )
        with self._transform_lock:
            return pack_visual_transforms(self.model, trajectory.joint_degrees)

    def video_path(self, mode: str, episode_index: int, camera: str) -> Path:
        path = self.dataset(mode).video_path(episode_index, camera)
        if not path.is_file():
            raise FileNotFoundError(f"video missing: {path}")
        return path


def create_app(
    datasets: dict[str, str | Path] | None = None,
    datasets_root: str | Path | None = None,
):
    backend = ViewerBackend(datasets, datasets_root)
    return _create_app(
        backend,
        Path(rv_web.__file__).parent / "static",
        title="SO-101 viewer",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the interactive SO101 dataset viewer"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--original-root", type=Path)
    parser.add_argument("--retimed-root", type=Path)
    parser.add_argument(
        "--datasets-root",
        type=Path,
        help="Scan immediate child folders for LeRobot datasets and allow switching at runtime.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.datasets_root is None and (args.original_root is None or args.retimed_root is None):
        parser.error("provide --datasets-root or both --original-root and --retimed-root")
    if args.datasets_root is not None and (args.original_root is not None or args.retimed_root is not None):
        parser.error("use either --datasets-root or explicit dataset roots")
    datasets = None
    if args.datasets_root is None:
        datasets = {"original": args.original_root, "retimed": args.retimed_root}
    app = create_app(datasets, datasets_root=args.datasets_root)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
