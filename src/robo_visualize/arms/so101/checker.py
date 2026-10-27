from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import coal as hppfcl
import numpy as np

from robo_visualize.arms.so101.model import SO101Model
from robo_visualize.arms.so101.trajectory import SO101Trajectory


@dataclass(frozen=True)
class CollisionPair:
    left_geometry: str
    left_link: str
    right_geometry: str
    right_link: str
    distance_m: float


@dataclass(frozen=True)
class CollisionFrame:
    frame_index: int
    timestamp_s: float
    in_collision: bool
    minimum_distance_m: float | None
    pairs: tuple[CollisionPair, ...]


@dataclass(frozen=True)
class TrajectoryCollisionReport:
    source: str
    frame_count: int
    duration_s: float
    fps: float
    collision_frames: int
    collision_rate: float
    first_collision_frame: int | None
    minimum_distance_m: float | None
    setup_seconds: float
    analysis_seconds: float
    frames_per_second: float
    frames: tuple[CollisionFrame, ...]

    def to_dict(self) -> dict:
        return asdict(self)


class SO101CollisionChecker:
    """Exact inter-arm mesh collision checks with vectorized AABB broad phase."""

    def __init__(
        self,
        urdf_path: str | None = None,
        safety_margin_m: float = 0.0,
    ) -> None:
        if safety_margin_m < 0:
            raise ValueError("safety_margin_m must be non-negative")
        started = perf_counter()
        self.robot = SO101Model(urdf_path)
        self.left = self.robot.arm("left")
        self.right = self.robot.arm("right")
        self.safety_margin_m = float(safety_margin_m)
        self.setup_seconds = perf_counter() - started
        self._pair_mask = self._build_pair_mask()
        self._collision_request = hppfcl.CollisionRequest()
        self._distance_request = hppfcl.DistanceRequest()

    def check_frame(
        self,
        joint_degrees: np.ndarray,
        *,
        frame_index: int = 0,
        timestamp_s: float = 0.0,
        compute_minimum_distance: bool = False,
        collect_all_pairs: bool = False,
    ) -> CollisionFrame:
        values = np.asarray(joint_degrees, dtype=np.float64)
        if values.shape != (2, 6):
            raise ValueError(f"expected one dual-arm frame with shape (2, 6), got {values.shape}")
        self.left.update(values[0])
        self.right.update(values[1])
        lower_bounds = self._aabb_lower_bounds()
        pairs = self._colliding_pairs(lower_bounds, collect_all_pairs)
        minimum_distance = None
        if pairs:
            minimum_distance = min(pair.distance_m for pair in pairs)
        elif compute_minimum_distance:
            minimum_distance = self._minimum_distance(lower_bounds)
        return CollisionFrame(
            frame_index,
            float(timestamp_s),
            bool(pairs),
            minimum_distance,
            tuple(pairs),
        )

    def analyze_trajectory(
        self,
        trajectory: SO101Trajectory,
        *,
        compute_minimum_distance: bool = False,
        collect_all_pairs: bool = False,
    ) -> TrajectoryCollisionReport:
        started = perf_counter()
        frames = tuple(
            self.check_frame(
                values,
                frame_index=index,
                timestamp_s=float(trajectory.timestamps_s[index]),
                compute_minimum_distance=compute_minimum_distance,
                collect_all_pairs=collect_all_pairs,
            )
            for index, values in enumerate(trajectory.joint_degrees)
        )
        elapsed = perf_counter() - started
        collision_indices = [frame.frame_index for frame in frames if frame.in_collision]
        measured_distances = [
            frame.minimum_distance_m
            for frame in frames
            if frame.minimum_distance_m is not None
        ]
        return TrajectoryCollisionReport(
            source=trajectory.source,
            frame_count=trajectory.frame_count,
            duration_s=trajectory.duration_s,
            fps=trajectory.fps,
            collision_frames=len(collision_indices),
            collision_rate=len(collision_indices) / max(trajectory.frame_count, 1),
            first_collision_frame=collision_indices[0] if collision_indices else None,
            minimum_distance_m=min(measured_distances) if measured_distances else None,
            setup_seconds=self.setup_seconds,
            analysis_seconds=elapsed,
            frames_per_second=trajectory.frame_count / max(elapsed, np.finfo(float).eps),
            frames=frames,
        )

    def _build_pair_mask(self) -> np.ndarray:
        mask = np.ones(
            (len(self.left.collision_refs), len(self.right.collision_refs)),
            dtype=bool,
        )
        for left in self.left.collision_refs:
            for right in self.right.collision_refs:
                if left.link_name == "base_link" and right.link_name == "base_link":
                    mask[left.index, right.index] = False
        return mask

    def _aabb_lower_bounds(self) -> np.ndarray:
        left = self.left.collision_aabbs()
        right = self.right.collision_aabbs()
        gap = np.maximum(
            np.maximum(left[:, None, 0, :] - right[None, :, 1, :], right[None, :, 0, :] - left[:, None, 1, :]),
            0.0,
        )
        lower_bounds = np.linalg.norm(gap, axis=2)
        lower_bounds[~self._pair_mask] = np.inf
        return lower_bounds

    def _colliding_pairs(
        self,
        lower_bounds: np.ndarray,
        collect_all_pairs: bool,
    ) -> list[CollisionPair]:
        candidates = np.argwhere(lower_bounds <= self.safety_margin_m)
        pairs = []
        for left_index, right_index in candidates:
            pair = self._check_pair(int(left_index), int(right_index))
            if pair is None:
                continue
            pairs.append(pair)
            if not collect_all_pairs:
                break
        return pairs

    def _check_pair(self, left_index: int, right_index: int) -> CollisionPair | None:
        left_object = self.robot.collision_model.geometryObjects[left_index]
        right_object = self.robot.collision_model.geometryObjects[right_index]
        left_transform = self.left.collision_transform(left_index)
        right_transform = self.right.collision_transform(right_index)
        result = hppfcl.CollisionResult()
        hppfcl.collide(
            left_object.geometry,
            left_transform,
            right_object.geometry,
            right_transform,
            self._collision_request,
            result,
        )
        if result.isCollision():
            return self._pair_result(left_index, right_index, 0.0)
        if self.safety_margin_m == 0.0:
            return None
        distance = self._pair_distance(left_index, right_index)
        if distance <= self.safety_margin_m:
            return self._pair_result(left_index, right_index, distance)
        return None

    def _minimum_distance(self, lower_bounds: np.ndarray) -> float:
        best = np.inf
        order = np.argsort(lower_bounds, axis=None)
        for flat_index in order:
            lower_bound = float(lower_bounds.flat[flat_index])
            if not np.isfinite(lower_bound) or lower_bound >= best:
                break
            left_index, right_index = np.unravel_index(flat_index, lower_bounds.shape)
            best = min(best, self._pair_distance(int(left_index), int(right_index)))
            if best == 0.0:
                break
        return float(best)

    def _pair_distance(self, left_index: int, right_index: int) -> float:
        left_object = self.robot.collision_model.geometryObjects[left_index]
        right_object = self.robot.collision_model.geometryObjects[right_index]
        result = hppfcl.DistanceResult()
        hppfcl.distance(
            left_object.geometry,
            self.left.collision_transform(left_index),
            right_object.geometry,
            self.right.collision_transform(right_index),
            self._distance_request,
            result,
        )
        return max(float(result.min_distance), 0.0)

    def _pair_result(
        self,
        left_index: int,
        right_index: int,
        distance_m: float,
    ) -> CollisionPair:
        left_ref = self.left.collision_refs[left_index]
        right_ref = self.right.collision_refs[right_index]
        return CollisionPair(
            left_ref.geometry_name,
            left_ref.link_name,
            right_ref.geometry_name,
            right_ref.link_name,
            float(distance_m),
        )
