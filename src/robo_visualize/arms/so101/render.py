from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from robo_visualize.arms.so101.checker import CollisionFrame, SO101CollisionChecker
from robo_visualize.arms.so101.model import SO101ArmState
from robo_visualize.arms.so101.trajectory import SO101Trajectory

ARM_COLORS = {
    "left": (0.20, 0.48, 0.76, 1.0),
    "right": (0.90, 0.48, 0.16, 1.0),
}
COLLISION_COLOR = (0.90, 0.08, 0.08, 1.0)


class SO101Renderer:
    def __init__(
        self,
        checker: SO101CollisionChecker,
        width: int = 960,
        height: int = 720,
    ) -> None:
        self.checker = checker
        self.width = width
        self.height = height
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        self.renderer.scene.set_background((0.96, 0.97, 0.98, 1.0))
        self.renderer.scene.scene.set_sun_light(
            np.array([-0.5, -0.5, -1.0]),
            np.array([1.0, 1.0, 1.0]),
            70000.0,
        )
        self.renderer.scene.scene.enable_sun_light(True)
        self._materials = self._build_materials()
        self._add_floor()
        self._add_arm(self.checker.left)
        self._add_arm(self.checker.right)
        self.renderer.setup_camera(
            43.0,
            np.array([0.0, 0.0, 0.12]),
            np.array([-0.72, -0.62, 0.48]),
            np.array([0.0, 0.0, 1.0]),
        )

    def render_frame(
        self,
        joint_degrees: np.ndarray,
        collision_frame: CollisionFrame | None = None,
    ) -> np.ndarray:
        frame = collision_frame or self.checker.check_frame(
            joint_degrees,
            collect_all_pairs=True,
        )
        if collision_frame is not None:
            self.checker.left.update(joint_degrees[0])
            self.checker.right.update(joint_degrees[1])
        colliding_links = {
            ("left", pair.left_link)
            for pair in frame.pairs
        } | {
            ("right", pair.right_link)
            for pair in frame.pairs
        }
        self._update_arm(self.checker.left, colliding_links)
        self._update_arm(self.checker.right, colliding_links)
        return np.asarray(self.renderer.render_to_image())

    def render_trajectory(
        self,
        trajectory: SO101Trajectory,
        output_path: str | Path,
        *,
        stride: int = 1,
    ) -> Path:
        if stride < 1:
            raise ValueError("render stride must be at least one")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            trajectory.fps / stride,
            (self.width, self.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"failed to create video writer: {output}")
        try:
            for joint_degrees in trajectory.joint_degrees[::stride]:
                image = self.render_frame(joint_degrees)
                writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        return output

    def _build_materials(self) -> dict[str, o3d.visualization.rendering.MaterialRecord]:
        materials = {}
        for name, color in (*ARM_COLORS.items(), ("collision", COLLISION_COLOR)):
            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = "defaultLit"
            material.base_color = color
            material.base_roughness = 0.78
            materials[name] = material
        return materials

    def _add_floor(self) -> None:
        floor = o3d.geometry.TriangleMesh.create_box(1.2, 1.0, 0.008)
        floor.translate((-0.6, -0.5, -0.012))
        floor.compute_vertex_normals()
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"
        material.base_color = (0.75, 0.78, 0.82, 1.0)
        material.base_roughness = 0.92
        self.renderer.scene.add_geometry("floor", floor, material)

    def _add_arm(self, arm: SO101ArmState) -> None:
        for ref in arm.visual_refs:
            geometry_object = arm.source.visual_model.geometryObjects[ref.index]
            mesh = o3d.io.read_triangle_mesh(geometry_object.meshPath)
            if mesh.is_empty():
                raise ValueError(f"failed to load visual mesh: {geometry_object.meshPath}")
            vertices = np.asarray(mesh.vertices)
            vertices[:] = vertices * np.asarray(geometry_object.meshScale)
            mesh.compute_vertex_normals()
            self.renderer.scene.add_geometry(
                self._geometry_key(arm.side, ref.geometry_name),
                mesh,
                self._materials[arm.side],
            )

    def _update_arm(
        self,
        arm: SO101ArmState,
        colliding_links: set[tuple[str, str]],
    ) -> None:
        for ref in arm.visual_refs:
            key = self._geometry_key(arm.side, ref.geometry_name)
            self.renderer.scene.set_geometry_transform(
                key,
                arm.visual_transform_matrix(ref.index),
            )
            material_key = "collision" if (arm.side, ref.link_name) in colliding_links else arm.side
            self.renderer.scene.modify_geometry_material(key, self._materials[material_key])

    @staticmethod
    def _geometry_key(side: str, geometry_name: str) -> str:
        return f"{side}:{geometry_name}"
