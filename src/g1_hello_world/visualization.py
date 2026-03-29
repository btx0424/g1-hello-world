from __future__ import annotations

import mujoco
import numpy as np
import viser

from .robot_model import RobotModelWrapper


class ViserRobotModelHandle:
    def __init__(self, scene: viser.SceneApi, robot_model: RobotModelWrapper) -> None:
        self.robot_model = robot_model
        self.mesh_handles: list[viser.GlbHandle | None] = []
        for body_name, mesh in zip(
            self.robot_model.body_names, self.robot_model.body_meshes, strict=True
        ):
            if mesh is None:
                self.mesh_handles.append(None)
                continue
            handle = scene.add_mesh_trimesh(
                name=f"/robot/{body_name}",
                mesh=mesh,
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
            )
            self.mesh_handles.append(handle)

    def update(self) -> None:
        data = self.robot_model.mj_data
        for body_id, mesh_handle in zip(
            self.robot_model.body_ids, self.mesh_handles, strict=True
        ):
            if mesh_handle is None:
                continue
            quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quat, data.xmat[body_id])
            mesh_handle.position = np.asarray(data.xpos[body_id], dtype=np.float64)
            mesh_handle.wxyz = quat


class ViserVisualizer:
    def __init__(self) -> None:
        self.server = viser.ViserServer()

    def add_camera_image(
        self,
        hwc: tuple[int, int, int],
        *,
        render_width: float,
        render_height: float,
    ) -> viser.ImageHandle:
        return self.server.scene.add_image(
            "/realsense/color",
            np.zeros(hwc, dtype=np.uint8),
            render_width=render_width,
            render_height=render_height,
            format="jpeg",
            jpeg_quality=85,
            position=(0.0, 0.0, 0.0),
        )

    def add_robot(self, robot_model: RobotModelWrapper) -> ViserRobotModelHandle:
        return ViserRobotModelHandle(self.server.scene, robot_model)

