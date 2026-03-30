from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Optional

import mujoco
import numpy as np
import viser

from .constants import R_SITE_FROM_OPENCV
from .realsense_device import RealSenseDeviceManager
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


class ViserCameraHandle:
    """
    RealSense RGB frustum in Viser; runs a background loop that grabs aligned
    RGB-D, pushes images, and (optionally) updates pose from a robot site frame.
    """

    def __init__(
        self,
        scene: viser.SceneApi,
        name: str,
        realsense_device: RealSenseDeviceManager,
        hwc: tuple[int, int, int],
        *,
        frustum_depth: float,
        robot_model: RobotModelWrapper | None = None,
        site_name: str = "d435",
        line_width: float = 2.0,
        color: tuple[int, int, int] = (48, 48, 48),
    ) -> None:
        self._realsense = realsense_device
        self._robot_model = robot_model
        self._site_name = site_name

        image = np.zeros(hwc, dtype=np.uint8)
        self.frustum = scene.add_camera_frustum(
            name=name,
            fov=self._realsense.fov_y,
            aspect=self._realsense.aspect,
            scale=1.0,
            line_width=line_width,
            color=color,
            image=image,
            format="jpeg",
            jpeg_quality=85,
            variant="wireframe",
        )
        _z = self.frustum.compute_canonical_frustum_size()[2]
        self.frustum.scale = frustum_depth / _z

    @property
    def image_frustum(self) -> viser.CameraFrustumHandle:
        return self.frustum

    def update(self) -> None:
        self.frustum.image = self._realsense.rgb
        if self._robot_model is not None:
            pos_link, world_from_link = self._robot_model.get_site_frame(
                self._site_name
            )
            world_from_cv = world_from_link @ R_SITE_FROM_OPENCV
            wxyz = np.zeros(4, dtype=np.float64)
            mujoco.mju_mat2Quat(wxyz, world_from_cv.flatten(order="C"))
            self.frustum.position = pos_link
            self.frustum.wxyz = wxyz


class ViserVisualizer:
    def __init__(self) -> None:
        self.server = viser.ViserServer()
        self.robot_model_handles: list[ViserRobotModelHandle] = []
        self.camera_handles: list[ViserCameraHandle] = []

    def update(self) -> None:
        for handle in self.robot_model_handles:
            handle.update()
        for handle in self.camera_handles:
            handle.update()

    def add_camera(
        self,
        name: str,
        realsense_device: RealSenseDeviceManager,
        hwc: tuple[int, int, int],
        *,
        frustum_depth: float,
        robot_model: RobotModelWrapper | None = None,
        site_name: str = "d435",
        line_width: float = 2.0,
        color: tuple[int, int, int] = (48, 48, 48),
    ) -> ViserCameraHandle:
        handle = ViserCameraHandle(
            self.server.scene,
            name,
            realsense_device,
            hwc,
            frustum_depth=frustum_depth,
            robot_model=robot_model,
            site_name=site_name,
            line_width=line_width,
            color=color,
        )
        self.camera_handles.append(handle)
        return handle

    def add_robot(self, robot_model: RobotModelWrapper) -> ViserRobotModelHandle:
        handle = ViserRobotModelHandle(self.server.scene, robot_model)
        self.robot_model_handles.append(handle)
        return handle
