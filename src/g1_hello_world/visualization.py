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
        name: str,
        hwc: tuple[int, int, int],
        *,
        fov_y: float,
        aspect: float,
        frustum_depth: float,
        line_width: float = 2.0,
        color: tuple[int, int, int] = (48, 48, 48),
    ) -> viser.CameraFrustumHandle:
        """Frustum in OpenCV camera convention (+Z forward); pose in world from caller."""
        image = np.zeros(hwc, dtype=np.uint8)
        handle = self.server.scene.add_camera_frustum(
            name=name,
            fov=fov_y,
            aspect=aspect,
            scale=1.0,
            line_width=line_width,
            color=color,
            image=image,
            format="jpeg",
            jpeg_quality=85,
            variant="wireframe",
        )
        _z = handle.compute_canonical_frustum_size()[2]
        handle.scale = frustum_depth / _z
        return handle

    def add_robot(self, robot_model: RobotModelWrapper) -> ViserRobotModelHandle:
        return ViserRobotModelHandle(self.server.scene, robot_model)

