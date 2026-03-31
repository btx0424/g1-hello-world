from __future__ import annotations

import threading
import time

import mujoco
import numpy as np
import viser

from .constants import R_SITE_FROM_OPENCV
from .realsense_device import RealSenseDeviceManager
from .robot_model import RobotModelWrapper
from .utils.string import resolve_matching_names


class ViserRobotModelHandle:
    def __init__(
        self,
        scene: viser.SceneApi,
        robot_model: RobotModelWrapper,
        body_names: str | list[str] = ".*", # all bodies by default
    ) -> None:
        self.robot_model = robot_model
        self.mesh_handles: list[viser.GlbHandle | None] = []
        
        body_ids, body_names = resolve_matching_names(body_names, self.robot_model.body_names)
        
        # visualized bodies' addresses and names
        self.body_adrs = [self.robot_model.body_adrs[i] for i in body_ids]
        self.body_names = [self.robot_model.body_names[i] for i in body_ids]
        for body_name in self.body_names:
            mesh = self.robot_model.body_meshes[body_name]
            if mesh is None:
                self.mesh_handles.append(None)
                continue
            print(f"Adding mesh for body {body_name}")
            handle = scene.add_mesh_trimesh(
                name=f"/robot/{body_name}",
                mesh=mesh,
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
            )
            self.mesh_handles.append(handle)

    def update(self) -> None:
        data = self.robot_model.mj_data
        for body_addr, mesh_handle in zip(
            self.body_adrs, self.mesh_handles, strict=True
        ):
            if mesh_handle is None:
                continue
            quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quat, data.xmat[body_addr])
            mesh_handle.position = np.asarray(data.xpos[body_addr], dtype=np.float64)
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
        self._tracker_points_handle: viser.PointCloudHandle | None = None
        self._async_stop = threading.Event()
        self._async_thread: threading.Thread | None = None

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

    def add_robot(self, robot_model: RobotModelWrapper, body_names: str | list[str] = ".*") -> ViserRobotModelHandle:
        handle = ViserRobotModelHandle(self.server.scene, robot_model, body_names)
        self.robot_model_handles.append(handle)
        return handle

    def set_tracker_points(
        self,
        points: np.ndarray | None,
        *,
        colors: np.ndarray | tuple[int, int, int] = (0, 255, 0),
        point_size: float = 0.04,
        name: str = "/tracker/points",
    ) -> None:
        if self._tracker_points_handle is not None:
            self._tracker_points_handle.remove()
            self._tracker_points_handle = None

        if points is None:
            return

        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")

        finite_mask = np.all(np.isfinite(pts), axis=1)
        pts = pts[finite_mask]
        if pts.shape[0] == 0:
            return

        rgb = np.asarray(colors)
        if rgb.ndim == 2:
            if rgb.shape != (len(points), 3):
                raise ValueError("colors must have shape (N, 3)")
            rgb = rgb[finite_mask]

        self._tracker_points_handle = self.server.scene.add_point_cloud(
            name=name,
            points=pts,
            colors=rgb,
            point_size=point_size,
            point_shape="circle",
        )

    def run_async(self, freq: float = 20.0) -> None:
        """
        Start a daemon thread that calls :meth:`update` at roughly ``freq`` Hz.

        Call :meth:`stop_async` before discarding the visualizer or exiting the process.
        Idempotent if a thread is already running.
        """
        if freq <= 0.0:
            raise ValueError("freq must be positive")
        if self._async_thread is not None and self._async_thread.is_alive():
            return
        period_s = 1.0 / float(freq)
        self._async_stop.clear()

        def _loop() -> None:
            while not self._async_stop.is_set():
                t0 = time.perf_counter()
                self.update()
                elapsed = time.perf_counter() - t0
                slack = period_s - elapsed
                if slack > 0.0:
                    time.sleep(slack)

        self._async_thread = threading.Thread(
            target=_loop,
            daemon=True,
            name="viser-visualizer",
        )
        self._async_thread.start()

    def stop_async(self, *, join_timeout_s: float = 2.0) -> None:
        """Signal the async update loop to stop and wait for the thread to exit."""
        self._async_stop.set()
        if self._async_thread is not None:
            self._async_thread.join(timeout=join_timeout_s)
            self._async_thread = None
