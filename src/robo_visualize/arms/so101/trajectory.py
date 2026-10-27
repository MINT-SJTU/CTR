from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from robo_visualize.arms.so101.model import JOINT_NAMES

EXPECTED_FEATURE_NAMES = tuple(
    f"{side}_{joint}.pos"
    for side in ("left", "right")
    for joint in JOINT_NAMES
)


@dataclass(frozen=True)
class SO101Trajectory:
    joint_degrees: np.ndarray
    fps: float
    timestamps_s: np.ndarray
    source: str = "array"

    def __post_init__(self) -> None:
        values = np.asarray(self.joint_degrees)
        if values.ndim != 3 or values.shape[1:] != (2, 6):
            raise ValueError(f"joint_degrees must have shape (frames, 2, 6), got {values.shape}")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if len(self.timestamps_s) != len(values):
            raise ValueError("timestamps and joint frames must have equal length")
        if not np.all(np.isfinite(values)):
            raise ValueError("trajectory joint values must be finite")

    @property
    def frame_count(self) -> int:
        return int(self.joint_degrees.shape[0])

    @property
    def duration_s(self) -> float:
        if self.frame_count < 2:
            return 0.0
        return float(self.timestamps_s[-1] - self.timestamps_s[0])

    def first_seconds(self, duration_s: float) -> "SO101Trajectory":
        frame_count = min(self.frame_count, int(round(duration_s * self.fps)))
        if frame_count < 1:
            raise ValueError("requested duration contains no frames")
        return SO101Trajectory(
            self.joint_degrees[:frame_count].copy(),
            self.fps,
            self.timestamps_s[:frame_count].copy(),
            self.source,
        )

    def interpolated(self, substeps: int) -> "SO101Trajectory":
        if substeps < 1:
            raise ValueError("interpolation substeps must be at least one")
        if substeps == 1 or self.frame_count < 2:
            return self
        fractions = np.arange(substeps, dtype=np.float64) / substeps
        values = []
        timestamps = []
        for index in range(self.frame_count - 1):
            current = self.joint_degrees[index]
            delta = self.joint_degrees[index + 1] - current
            values.extend(current + fraction * delta for fraction in fractions)
            start = self.timestamps_s[index]
            span = self.timestamps_s[index + 1] - start
            timestamps.extend(start + fraction * span for fraction in fractions)
        values.append(self.joint_degrees[-1])
        timestamps.append(self.timestamps_s[-1])
        return SO101Trajectory(
            np.asarray(values),
            self.fps * substeps,
            np.asarray(timestamps),
            f"{self.source}:interpolation_substeps={substeps}",
        )

    @classmethod
    def from_lerobot_parquet(
        cls,
        parquet_path: str | Path,
        episode_index: int,
        signal: str = "action",
    ) -> "SO101Trajectory":
        if signal not in ("action", "observation.state"):
            raise ValueError(f"unsupported trajectory signal: {signal}")
        path = Path(parquet_path)
        root = dataset_root_for(path)
        info = json.loads((root / "meta" / "info.json").read_text())
        feature_names = tuple(info["features"][signal]["names"])
        if feature_names != EXPECTED_FEATURE_NAMES:
            raise ValueError(f"unexpected SO101 feature order: {feature_names}")
        table = pq.read_table(
            path,
            columns=["timestamp", "episode_index", signal],
            filters=[("episode_index", "=", episode_index)],
        )
        if len(table) == 0:
            raise ValueError(f"episode {episode_index} has no frames in {path}")
        vector = table[signal].combine_chunks()
        flat = vector.values.to_numpy(zero_copy_only=False)
        values = np.asarray(flat, dtype=np.float64).reshape(len(vector), 2, 6)
        timestamps = pc.cast(table["timestamp"].combine_chunks(), "double").to_numpy()
        return cls(values, float(info["fps"]), timestamps, f"{path}:episode={episode_index}:{signal}")


def dataset_root_for(parquet_path: Path) -> Path:
    root = parquet_path.resolve().parents[2]
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata missing: {info_path}")
    return root
