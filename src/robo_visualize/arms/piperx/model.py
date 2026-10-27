from __future__ import annotations

from dataclasses import dataclass
from robo_visualize.assets import assets_root
from pathlib import Path

import numpy as np

JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")
URDF_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
GRIPPER_NAMES = ("gripper",)
STATE_NAMES = JOINT_NAMES + GRIPPER_NAMES
GRIPPER_STROKE_M = 0.035


@dataclass(frozen=True)
class VisualMesh:
    output_index: int
    geometry_index: int
    name: str
    link_name: str
    mesh_path: Path
    scale: tuple[float, float, float]


def default_urdf_path() -> Path:
    return Path(
        str(assets_root() / "urdf" / "piper_x_description.urdf")
    )


class PiperXModel:
    """Pinocchio-backed single-arm Piper X visual model."""

    def __init__(self, urdf_path: str | Path | None = None) -> None:
        import pinocchio as pin

        self.urdf_path = Path(urdf_path) if urdf_path else default_urdf_path()
        package_dirs = [
            str(self.urdf_path.parents[1]),
            str(self.urdf_path.parent),
        ]
        self.model, self.visual_model = pin.buildModelsFromUrdf(
            str(self.urdf_path),
            package_dirs=package_dirs,
            geometry_types=[pin.GeometryType.VISUAL],
        )
        actual_names = tuple(self.model.names[1:])
        if actual_names[: len(URDF_JOINT_NAMES)] != URDF_JOINT_NAMES:
            raise ValueError(f"unexpected Piper X joint order: {actual_names}")
        required_gripper_joints = ("joint7", "joint8")
        missing_gripper_joints = [name for name in required_gripper_joints if name not in actual_names]
        if missing_gripper_joints:
            raise ValueError(f"Piper X URDF missing gripper joints: {missing_gripper_joints}")
        self.model_data = self.model.createData()
        self.visual_data = pin.GeometryData(self.visual_model)
        self.pin = pin
        self.q = pin.neutral(self.model)
        self.joint_q_indices = self._joint_q_indices()
        self.meshes = self._visual_meshes()
        self.base_placement = pin.SE3.Identity()
        self.update(np.zeros(len(STATE_NAMES), dtype=np.float64))

    def update(self, state_values: np.ndarray) -> None:
        values = np.asarray(state_values, dtype=np.float64)
        if values.shape != (len(STATE_NAMES),):
            raise ValueError(f"expected {len(STATE_NAMES)} Piper X state values, got shape {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("Piper X state values must be finite")
        for joint_name, value in zip(URDF_JOINT_NAMES, values):
            q_index = self.joint_q_indices[joint_name]
            radians = np.deg2rad(value)
            self.q[q_index] = float(radians)
        gripper_metres = float(np.clip(values[6] / 1000.0, 0.0, GRIPPER_STROKE_M))
        for joint_name, gripper_value in (("joint7", gripper_metres), ("joint8", -gripper_metres)):
            q_index = self.joint_q_indices[joint_name]
            self.q[q_index] = float(
                np.clip(gripper_value, self.model.lowerPositionLimit[q_index], self.model.upperPositionLimit[q_index])
            )
        self.pin.forwardKinematics(self.model, self.model_data, self.q)
        self.pin.updateFramePlacements(self.model, self.model_data)
        self.pin.updateGeometryPlacements(
            self.model,
            self.model_data,
            self.visual_model,
            self.visual_data,
        )

    def transform_matrix(self, geometry_index: int) -> np.ndarray:
        placement = self.base_placement * self.visual_data.oMg[geometry_index]
        return placement.homogeneous

    def tcp_transform(self) -> np.ndarray:
        frame_id = self.model.getFrameId("tcp")
        if frame_id >= len(self.model.frames):
            raise ValueError("Piper X URDF tcp frame is missing")
        return self.base_placement * self.model_data.oMf[frame_id]

    def relative_frame_pose(
        self,
        parent_frame: str,
        child_frame: str,
    ) -> tuple[float, float, float, float, float, float, float]:
        parent_id = self.model.getFrameId(parent_frame)
        child_id = self.model.getFrameId(child_frame)
        if parent_id >= len(self.model.frames):
            raise ValueError(f"Piper X URDF frame is missing: {parent_frame}")
        if child_id >= len(self.model.frames):
            raise ValueError(f"Piper X URDF frame is missing: {child_frame}")
        placement = self.model_data.oMf[parent_id].inverse() * self.model_data.oMf[child_id]
        quaternion_xyzw = self.pin.Quaternion(placement.rotation).coeffs()
        return (
            *map(float, placement.translation),
            float(quaternion_xyzw[3]),
            *map(float, quaternion_xyzw[:3]),
        )

    def _joint_q_indices(self) -> dict[str, int]:
        indices = {}
        for name in (*URDF_JOINT_NAMES, "joint7", "joint8"):
            joint_id = self.model.getJointId(name)
            if joint_id == 0:
                raise ValueError(f"Piper X URDF joint missing: {name}")
            joint = self.model.joints[joint_id]
            if joint.nq != 1:
                raise ValueError(f"Piper X action joint must have nq=1: {name}")
            indices[name] = joint.idx_q
        return indices

    def _visual_meshes(self) -> tuple[VisualMesh, ...]:
        meshes = []
        for geometry_index, geometry in enumerate(self.visual_model.geometryObjects):
            link_name = self.model.frames[geometry.parentFrame].name
            mesh_path = self._mesh_path(geometry.meshPath)
            if not mesh_path.is_file():
                raise FileNotFoundError(f"Piper X visual mesh missing: {mesh_path}")
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
            raise ValueError("Piper X URDF produced no selected visual meshes")
        return tuple(meshes)

    def _mesh_path(self, raw: str) -> Path:
        candidate = Path(raw)
        if candidate.is_file():
            return candidate
        if raw.startswith("package://"):
            relative = Path(raw[len("package://") :])
            roots = (self.urdf_path.parents[1], self.urdf_path.parent)
            for root in roots:
                resolved = root / relative
                if resolved.is_file():
                    return resolved
        raise FileNotFoundError(f"cannot resolve Piper X URDF mesh: {raw}")


def pack_visual_transforms(model: PiperXModel, state_values: np.ndarray) -> bytes:
    values = np.asarray(state_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(STATE_NAMES):
        raise ValueError(
            f"expected Piper X state trajectory shape (frames, {len(STATE_NAMES)}), got {values.shape}"
        )
    transforms = np.empty((len(values), len(model.meshes), 16), dtype="<f4")
    for frame_index, state in enumerate(values):
        model.update(state)
        for mesh in model.meshes:
            transforms[frame_index, mesh.output_index] = model.transform_matrix(
                mesh.geometry_index
            ).T.reshape(16)
    return transforms.tobytes()
