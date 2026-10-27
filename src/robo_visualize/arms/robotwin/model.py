from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


GRIPPER_MIN = -0.01
GRIPPER_MAX = 0.045
BASE_XYZ = np.array([0.0, -0.65, 0.0], dtype=np.float64)
BASE_QUATERNION_WXYZ = np.array([0.707, 0.0, 0.0, 0.707], dtype=np.float64)
ARM_JOINTS = {
    "left": tuple(f"fl_joint{index}" for index in range(1, 7)),
    "right": tuple(f"fr_joint{index}" for index in range(1, 7)),
}
GRIPPER_JOINTS = {
    "left": ("fl_joint7", "fl_joint8"),
    "right": ("fr_joint7", "fr_joint8"),
}
ACTION_SLICES = {
    "left": slice(0, 6),
    "right": slice(7, 13),
}
GRIPPER_ACTION_INDEX = {"left": 6, "right": 13}


def default_urdf_path() -> Path:
    root = Path(os.environ["ROBOTWIN_ROOT"])
    return (
        root
        / "assets"
        / "embodiments"
        / "aloha-agilex"
        / "urdf"
        / "arx5_description_isaac.urdf"
    )


@dataclass(frozen=True)
class VisualMesh:
    output_index: int
    geometry_index: int
    name: str
    link_name: str
    mesh_path: Path
    scale: tuple[float, float, float]


class AlohaActionModel:
    def __init__(self, urdf_path: str | Path | None = None) -> None:
        self.urdf_path = Path(urdf_path or default_urdf_path()).resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"RoboTwin Aloha URDF missing: {self.urdf_path}")
        import pinocchio as pin

        self.pin = pin
        package_dirs = [str(self.urdf_path.parent), str(self.urdf_path.parents[1])]
        self.model, self.visual_model = pin.buildModelsFromUrdf(
            str(self.urdf_path),
            package_dirs=package_dirs,
            geometry_types=[pin.GeometryType.VISUAL],
        )
        self.model_data = self.model.createData()
        self.visual_data = pin.GeometryData(self.visual_model)
        self.q = pin.neutral(self.model)
        base_quaternion = pin.Quaternion(
            float(BASE_QUATERNION_WXYZ[0]),
            float(BASE_QUATERNION_WXYZ[1]),
            float(BASE_QUATERNION_WXYZ[2]),
            float(BASE_QUATERNION_WXYZ[3]),
        )
        base_quaternion.normalize()
        self.base_placement = pin.SE3(base_quaternion.matrix(), BASE_XYZ)
        self.joint_q_indices = self._joint_q_indices()
        self.meshes = self._visual_meshes()
        self.update(np.zeros(14, dtype=np.float64))

    def update(self, action: np.ndarray) -> None:
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (14,):
            raise ValueError(f"expected RoboTwin action shape (14,), got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("RoboTwin action values must be finite")
        for side in ("left", "right"):
            for joint_name, value in zip(ARM_JOINTS[side], values[ACTION_SLICES[side]]):
                self.q[self.joint_q_indices[joint_name]] = value
            gripper = GRIPPER_MIN + values[GRIPPER_ACTION_INDEX[side]] * (
                GRIPPER_MAX - GRIPPER_MIN
            )
            for joint_name in GRIPPER_JOINTS[side]:
                joint_id = self.model.getJointId(joint_name)
                lower = self.model.lowerPositionLimit[self.model.joints[joint_id].idx_q]
                upper = self.model.upperPositionLimit[self.model.joints[joint_id].idx_q]
                self.q[self.joint_q_indices[joint_name]] = np.clip(
                    gripper, lower, upper
                )
        self.pin.forwardKinematics(self.model, self.model_data, self.q)
        self.pin.updateGeometryPlacements(
            self.model,
            self.model_data,
            self.visual_model,
            self.visual_data,
        )

    def transform_matrix(self, geometry_index: int) -> np.ndarray:
        placement = self.base_placement * self.visual_data.oMg[geometry_index]
        return placement.homogeneous

    def _joint_q_indices(self) -> dict[str, int]:
        names = (
            *ARM_JOINTS["left"],
            *GRIPPER_JOINTS["left"],
            *ARM_JOINTS["right"],
            *GRIPPER_JOINTS["right"],
        )
        indices = {}
        for name in names:
            joint_id = self.model.getJointId(name)
            if joint_id == 0:
                raise ValueError(f"Aloha URDF joint missing: {name}")
            joint = self.model.joints[joint_id]
            if joint.nq != 1:
                raise ValueError(f"Aloha action joint must have nq=1: {name}")
            indices[name] = joint.idx_q
        return indices

    def _visual_meshes(self) -> tuple[VisualMesh, ...]:
        meshes = []
        for geometry_index, geometry in enumerate(self.visual_model.geometryObjects):
            link_name = self.model.frames[geometry.parentFrame].name
            if not _include_visual_link(link_name):
                continue
            mesh_path = Path(geometry.meshPath)
            if mesh_path.suffix.lower() != ".dae":
                raise ValueError(f"Aloha visual mesh must be Collada: {mesh_path}")
            if not mesh_path.is_file():
                raise FileNotFoundError(
                    f"Aloha Collada visual mesh missing: {mesh_path}"
                )
            meshes.append(
                VisualMesh(
                    output_index=len(meshes),
                    geometry_index=geometry_index,
                    name=geometry.name,
                    link_name=link_name,
                    mesh_path=mesh_path,
                    scale=tuple(float(value) for value in geometry.meshScale),
                )
            )
        if not meshes:
            raise ValueError("Aloha URDF produced no selected visual meshes")
        return tuple(meshes)


def _include_visual_link(link_name: str) -> bool:
    arm_prefixes = ("fl_base_link", "fl_link", "fr_base_link", "fr_link")
    return link_name.startswith(arm_prefixes)


def pack_visual_transforms(model: AlohaActionModel, actions: np.ndarray) -> bytes:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 14:
        raise ValueError(
            f"expected RoboTwin actions shape (frames, 14), got {values.shape}"
        )
    transforms = np.empty((len(values), len(model.meshes), 16), dtype="<f4")
    for frame_index, action in enumerate(values):
        model.update(action)
        for mesh in model.meshes:
            transforms[frame_index, mesh.output_index] = model.transform_matrix(
                mesh.geometry_index
            ).T.reshape(16)
    return transforms.tobytes()
