from __future__ import annotations

import argparse
import subprocess
import threading
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from robo_visualize.arms.robotwin.dataset import RoboTwinDataset
from robo_visualize.arms.robotwin.model import (
    AlohaActionModel,
    pack_visual_transforms,
)
from robo_visualize import web as rv_web
from robo_visualize.web.server import create_app as _create_app

FPS = 30.0
CAMERA_DATASETS = {
    "left_wrist": "observation/left_camera/rgb",
    "right_wrist": "observation/right_camera/rgb",
}
VIDEO_CACHE_ROOT = Path("/tmp/robo_visualize_robotwin_videos")


def deployment_phase(frame: int, async_steps: int, boundary_context_steps: int) -> str:
    if frame < 0 or async_steps < 0 or boundary_context_steps < 0:
        raise ValueError("deployment router steps must be non-negative")
    if frame >= async_steps:
        return "sync"
    boundary_start = max(0, async_steps - boundary_context_steps)
    if frame >= boundary_start:
        return "boundary"
    return "async"


def optional_timeline(
    root: h5py.File,
    path: str,
    frame_count: int,
    dtype: np.dtype,
) -> np.ndarray | None:
    if path not in root:
        return None
    values = np.asarray(root[path][:], dtype=dtype)
    if values.shape != (frame_count,):
        raise ValueError(
            f"Episode timeline {path} has shape {values.shape}, expected {(frame_count,)}"
        )
    return values


class ViewerBackend:
    def __init__(
        self,
        datasets: dict[str, str | Path],
        urdf_path: str | Path | None = None,
        video_cache_root: str | Path = VIDEO_CACHE_ROOT,
        deploy_async_steps: int = 0,
        deploy_boundary_context_steps: int = 0,
    ) -> None:
        self.datasets = {
            name: RoboTwinDataset.load(name, root)
            for name, root in datasets.items()
            if Path(root).is_dir()
        }
        if not self.datasets:
            raise ValueError("no configured RoboTwin dataset is available")
        self.model = AlohaActionModel(urdf_path)
        self.video_cache_root = Path(video_cache_root)
        deployment_phase(0, deploy_async_steps, deploy_boundary_context_steps)
        self.deployment_router = {
            "async_steps": deploy_async_steps,
            "boundary_context_steps": deploy_boundary_context_steps,
        }
        self._transform_lock = threading.Lock()
        self._video_lock = threading.Lock()

    def config(self) -> dict[str, Any]:
        canonical_meshes: dict[Path, int] = {}
        meshes = []
        for mesh in self.model.meshes:
            canonical_index = canonical_meshes.setdefault(
                mesh.mesh_path, mesh.output_index
            )
            meshes.append(
                {
                    "index": mesh.output_index,
                    "name": mesh.name,
                    "link_name": mesh.link_name,
                    "scale": mesh.scale,
                    "url": f"/api/mesh/{canonical_index}.dae",
                }
            )
        return {
            "fps": FPS,
            "loader": "collada",
            "sideLabels": ["left", "right"],
            "sides": ["left", "right"],
            "cameras": ["head", "left_wrist", "right_wrist"],
            "master_camera": "head",
            "mesh_count": len(self.model.meshes),
            "meshes": meshes,
            "datasets": {
                name: {
                    "episode_indices": dataset.episode_indices,
                    "episode_count": len(dataset.episode_indices),
                }
                for name, dataset in self.datasets.items()
            },
        }

    def episode_payload(
        self,
        dataset_name: str,
        episode_index: int,
        signal: str | None = None,
    ) -> dict[str, Any]:
        del signal
        dataset = self.datasets[dataset_name]
        with h5py.File(dataset.episode_path(episode_index), "r") as root:
            frame_count = int(root["joint_action/vector"].shape[0])
            camera_frame_counts = {
                camera: int(root[path].shape[0])
                for camera, path in CAMERA_DATASETS.items()
                if path in root
            }
            phase_ids = optional_timeline(
                root, "observation/phase_type_id", frame_count, np.uint8
            )
            async_probabilities = optional_timeline(
                root, "observation/async_probability", frame_count, np.float32
            )
            inference_indices = optional_timeline(
                root, "observation/policy_inference_index", frame_count, np.int32
            )
            fps = float(root.attrs.get("parallelvla_fps", FPS))
            attributes = {
                str(key): _json_value(value)
                for key, value in root.attrs.items()
                if str(key).startswith("parallelvla_")
            }
        mismatched = {
            camera: count
            for camera, count in camera_frame_counts.items()
            if count != frame_count
        }
        if mismatched:
            raise ValueError(
                f"camera/action frame counts differ for episode {episode_index}: "
                f"actions={frame_count}, cameras={mismatched}"
            )
        if fps <= 0:
            raise ValueError(f"Episode FPS must be positive, got {fps}")
        if phase_ids is not None and not np.all(np.isin(phase_ids, (0, 1))):
            raise ValueError("Episode phase timeline must contain sync=0 and async=1")
        probability_values = None
        if async_probabilities is not None:
            probability_values = [
                None if np.isnan(value) else float(value)
                for value in async_probabilities
            ]
        return {
            "dataset": dataset_name,
            "episode_index": episode_index,
            "frame_count": frame_count,
            "fps": fps,
            "duration_s": frame_count / fps,
            "attributes": attributes,
            "deployment_router": self.deployment_router,
            "phase_timeline": {
                "has_gate_records": phase_ids is not None,
                "phase_ids": None if phase_ids is None else phase_ids.tolist(),
                "async_probabilities": probability_values,
                "inference_indices": (
                    None if inference_indices is None else inference_indices.tolist()
                ),
            },
            "videos": {
                "head": f"/api/video/{dataset_name}/{episode_index}/head.mp4",
                "left_wrist": f"/api/video/{dataset_name}/{episode_index}/left_wrist.mp4",
                "right_wrist": f"/api/video/{dataset_name}/{episode_index}/right_wrist.mp4",
            },
            "transforms_url": f"/api/transforms/{dataset_name}/{episode_index}.bin",
        }

    def transforms(
        self,
        dataset_name: str,
        episode_index: int,
        signal: str | None = None,
    ) -> bytes:
        del signal
        dataset = self.datasets[dataset_name]
        with h5py.File(dataset.episode_path(episode_index), "r") as root:
            actions = np.asarray(root["joint_action/vector"][:], dtype=np.float64)
        with self._transform_lock:
            return pack_visual_transforms(self.model, actions)

    def mesh_path(self, output_index: int) -> Path:
        if output_index < 0 or output_index >= len(self.model.meshes):
            raise ValueError(f"mesh index out of range: {output_index}")
        return self.model.meshes[output_index].mesh_path

    def mesh_asset_path(self, filename: str) -> Path:
        if Path(filename).name != filename:
            raise ValueError(f"invalid mesh asset name: {filename}")
        mesh_directories = {mesh.mesh_path.parent for mesh in self.model.meshes}
        existing = [directory / filename for directory in mesh_directories]
        matches = [path for path in existing if path.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(f"Aloha mesh asset not uniquely found: {filename}")
        return matches[0]

    def video_path(self, dataset_name: str, episode_index: int, camera: str) -> Path:
        dataset = self.datasets[dataset_name]
        if camera not in {"head", *CAMERA_DATASETS}:
            raise ValueError(f"unknown RoboTwin camera: {camera}")
        recorded_path = dataset.root / "videos" / camera / f"episode{episode_index}.mp4"
        if recorded_path.is_file():
            return recorded_path
        if camera == "head":
            return dataset.recorded_video_path(episode_index, camera)
        output_path = (
            self.video_cache_root
            / dataset_name
            / f"episode{episode_index}"
            / f"{camera}.mp4"
        )
        if output_path.is_file():
            return output_path
        with self._video_lock:
            if not output_path.is_file():
                self._build_camera_video(
                    dataset.episode_path(episode_index),
                    CAMERA_DATASETS[camera],
                    output_path,
                )
        return output_path

    @staticmethod
    def _build_camera_video(
        episode_path: Path,
        camera_dataset: str,
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(".building.mp4")
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-r",
            str(FPS),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary_path),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None or process.stderr is None:
            raise RuntimeError("ffmpeg pipes were not created")
        with h5py.File(episode_path, "r") as root:
            for frame in root[camera_dataset]:
                process.stdin.write(bytes(frame).rstrip(b"\x00"))
        process.stdin.close()
        stderr = process.stderr.read()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command, stderr=stderr)
        temporary_path.replace(output_path)


def create_app(
    datasets: dict[str, str | Path] | None = None,
    urdf_path: str | Path | None = None,
    deploy_async_steps: int = 0,
    deploy_boundary_context_steps: int = 0,
):
    if not datasets:
        raise ValueError("at least one dataset path is required")
    backend = ViewerBackend(
        datasets,
        urdf_path,
        deploy_async_steps=deploy_async_steps,
        deploy_boundary_context_steps=deploy_boundary_context_steps,
    )
    return _create_app(
        backend,
        Path(rv_web.__file__).parent / "static",
        title="RoboTwin 2 viewer",
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode()
    return value


def parse_dataset(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("dataset must use NAME=/absolute/path")
    return name, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the interactive RoboTwin action viewer"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument("--deploy-async-steps", type=int, default=0)
    parser.add_argument("--deploy-boundary-context-steps", type=int, default=0)
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset,
        required=True,
        help="NAME=/absolute/path; may be repeated",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    datasets = dict(args.dataset)
    app = create_app(
        datasets,
        args.urdf,
        args.deploy_async_steps,
        args.deploy_boundary_context_steps,
    )
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
