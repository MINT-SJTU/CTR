from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from robo_visualize.arms.piperx.model import JOINT_NAMES, STATE_NAMES

CAMERAS = ("left_wrist", "top")
TRAJECTORY_SIGNAL = "observation.state"
TRAJECTORY_SIGNALS = (TRAJECTORY_SIGNAL, "action")
EXPECTED_FEATURE_NAMES = tuple(f"{joint}.pos" for joint in STATE_NAMES)


@dataclass(frozen=True)
class PiperXTrajectory:
    state_values: np.ndarray
    fps: float
    timestamps_s: np.ndarray
    source: str = "array"

    def __post_init__(self) -> None:
        values = np.asarray(self.state_values)
        if values.ndim != 2 or values.shape[1] != len(STATE_NAMES):
            raise ValueError(
                f"state_values must have shape (frames, {len(STATE_NAMES)}), got {values.shape}"
            )
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if len(self.timestamps_s) != len(values):
            raise ValueError("timestamps and joint frames must have equal length")
        if not np.all(np.isfinite(values)):
            raise ValueError("trajectory joint values must be finite")

    @property
    def frame_count(self) -> int:
        return int(self.state_values.shape[0])

    @classmethod
    def from_lerobot_parquet(
        cls,
        parquet_path: str | Path,
        episode_index: int,
        signal: str = "action",
    ) -> "PiperXTrajectory":
        if signal not in TRAJECTORY_SIGNALS:
            raise ValueError(f"unsupported trajectory signal: {signal}")
        path = Path(parquet_path)
        root = dataset_root_for(path)
        info = json.loads((root / "meta" / "info.json").read_text())
        feature_names = tuple(info["features"][signal]["names"])
        if feature_names != EXPECTED_FEATURE_NAMES:
            raise ValueError(f"unexpected Piper X feature order: {feature_names}")
        table = pq.read_table(
            path,
            columns=["timestamp", "episode_index", signal],
            filters=[("episode_index", "=", episode_index)],
        )
        if len(table) == 0:
            raise ValueError(f"episode {episode_index} has no frames in {path}")
        vector = table[signal].combine_chunks()
        flat = vector.values.to_numpy(zero_copy_only=False)
        values = np.asarray(flat, dtype=np.float64).reshape(len(vector), len(EXPECTED_FEATURE_NAMES))
        timestamps = pc.cast(table["timestamp"].combine_chunks(), "double").to_numpy()
        return cls(values, float(info["fps"]), timestamps, f"{path}:episode={episode_index}:{signal}")


@dataclass(frozen=True)
class PiperXDatasetView:
    name: str
    root: Path
    fps: float
    episodes: tuple[dict[str, Any], ...]

    @classmethod
    def load(cls, name: str, root: str | Path) -> "PiperXDatasetView":
        path = Path(root).resolve()
        info_path = path / "meta" / "info.json"
        episodes_root = path / "meta" / "episodes"
        if not info_path.is_file():
            raise FileNotFoundError(f"dataset metadata missing: {info_path}")
        if not episodes_root.is_dir():
            raise FileNotFoundError(f"dataset episode metadata missing: {episodes_root}")
        info = json.loads(info_path.read_text())
        episode_files = sorted(
            parquet_file
            for chunk_dir in episodes_root.glob("chunk-*")
            if chunk_dir.is_dir()
            for parquet_file in chunk_dir.glob("file-*.parquet")
        )
        if not episode_files:
            raise FileNotFoundError(f"no episode parquet files found under {episodes_root}")
        tables = [pq.read_table(parquet_file) for parquet_file in episode_files]
        if len(tables) == 1:
            episodes_table = tables[0]
        else:
            episodes_table = pa.concat_tables(tables)
        episodes = tuple(episodes_table.to_pylist())
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


def dataset_root_for(parquet_path: Path) -> Path:
    root = parquet_path.resolve().parents[2]
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata missing: {info_path}")
    return root
