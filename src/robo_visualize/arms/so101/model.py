from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import coal as hppfcl
import numpy as np
import pinocchio as pin

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
LEFT_BASE_XYZ = np.array([0.0, 0.115, 0.0], dtype=np.float64)
RIGHT_BASE_XYZ = np.array([0.0, -0.115, 0.0], dtype=np.float64)
WRIST_ROLL_OFFSET_RAD = -np.pi / 2.0


@dataclass(frozen=True)
class GeometryRef:
    side: str
    index: int
    geometry_name: str
    link_name: str


class SO101Model:
    """One URDF model shared by two independently placed arm states."""

    def __init__(self, urdf_path: str | Path | None = None) -> None:
        self.urdf_path = Path(urdf_path) if urdf_path else default_urdf_path()
        built = pin.buildModelsFromUrdf(
            str(self.urdf_path),
            package_dirs=[str(self.urdf_path.parent)],
            geometry_types=[pin.GeometryType.COLLISION, pin.GeometryType.VISUAL],
        )
        self.model, self.collision_model, self.visual_model = built
        actual_names = tuple(self.model.names[1:])
        if actual_names != JOINT_NAMES:
            raise ValueError(f"unexpected SO101 joint order: {actual_names}")
        for geometry_object in self.collision_model.geometryObjects:
            geometry_object.geometry.computeLocalAABB()

    def arm(self, side: str) -> "SO101ArmState":
        if side == "left":
            return SO101ArmState(self, side, LEFT_BASE_XYZ)
        if side == "right":
            return SO101ArmState(self, side, RIGHT_BASE_XYZ)
        raise ValueError(f"unknown arm side: {side}")


class SO101ArmState:
    def __init__(self, source: SO101Model, side: str, base_xyz: np.ndarray) -> None:
        self.source = source
        self.side = side
        self.base_placement = pin.SE3(np.eye(3), base_xyz)
        self.model_data = source.model.createData()
        self.collision_data = pin.GeometryData(source.collision_model)
        self.visual_data = pin.GeometryData(source.visual_model)
        self.q_rad = pin.neutral(source.model)
        self.collision_refs = self._geometry_refs(source.collision_model)
        self.visual_refs = self._geometry_refs(source.visual_model)
        self.update(np.zeros(len(JOINT_NAMES), dtype=np.float64))

    def update(self, joint_degrees: np.ndarray) -> None:
        values = np.asarray(joint_degrees, dtype=np.float64)
        if values.shape != (len(JOINT_NAMES),):
            raise ValueError(f"expected six SO101 joints, got shape {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("SO101 joint values must be finite")
        self.q_rad[:] = np.deg2rad(values)
        self.q_rad[4] += WRIST_ROLL_OFFSET_RAD
        self.q_rad[5] = np.clip(
            self.q_rad[5],
            self.source.model.lowerPositionLimit[5],
            self.source.model.upperPositionLimit[5],
        )
        pin.forwardKinematics(self.source.model, self.model_data, self.q_rad)
        pin.updateGeometryPlacements(
            self.source.model,
            self.model_data,
            self.source.collision_model,
            self.collision_data,
        )
        pin.updateGeometryPlacements(
            self.source.model,
            self.model_data,
            self.source.visual_model,
            self.visual_data,
        )

    def collision_transform(self, index: int) -> hppfcl.Transform3s:
        placement = self.base_placement * self.collision_data.oMg[index]
        return hppfcl.Transform3s(placement.rotation, placement.translation)

    def visual_transform_matrix(self, index: int) -> np.ndarray:
        placement = self.base_placement * self.visual_data.oMg[index]
        return placement.homogeneous

    def collision_aabbs(self) -> np.ndarray:
        bounds = np.empty((len(self.collision_refs), 2, 3), dtype=np.float64)
        for ref in self.collision_refs:
            geometry = self.source.collision_model.geometryObjects[ref.index].geometry
            local = geometry.aabb_local
            bounds[ref.index] = transformed_aabb(
                np.asarray(local.min_),
                np.asarray(local.max_),
                self.collision_transform(ref.index),
            )
        return bounds

    def _geometry_refs(self, geometry_model: pin.GeometryModel) -> tuple[GeometryRef, ...]:
        refs = []
        for index, geometry_object in enumerate(geometry_model.geometryObjects):
            link_name = self.source.model.frames[geometry_object.parentFrame].name
            refs.append(GeometryRef(self.side, index, geometry_object.name, link_name))
        return tuple(refs)


def transformed_aabb(
    minimum: np.ndarray,
    maximum: np.ndarray,
    transform: hppfcl.Transform3s,
) -> np.ndarray:
    corners = np.array(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )
    world = corners @ np.asarray(transform.getRotation()).T + np.asarray(transform.getTranslation())
    return np.stack((world.min(axis=0), world.max(axis=0)))


def default_urdf_path() -> Path:
    return Path(str(files("robo_visualize.arms.so101") / "assets" / "so101_new_calib.urdf"))
