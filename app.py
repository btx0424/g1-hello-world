from __future__ import annotations

import itertools
import math
import threading
import time

import mujoco
import numpy as np
import pyrealsense2 as rs
import trimesh

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

from g1_hello_world.constants import R_SITE_FROM_OPENCV
from g1_hello_world.realsense_device import RealSenseDeviceManager
from g1_hello_world.robot_model import RobotModelWrapper
from g1_hello_world.timing import timer_decorator
from g1_hello_world.visualization import ViserVisualizer


class GroundPlaneEstimator:
    """
    Estimates a dominant plane from aligned RealSense depth (RGB-D), assuming the
    ground is visible in the lower image band. The fitted normal is flipped so it
    aligns with world +Z (MuJoCo / robot world up).
    """

    def __init__(self, *, world_up: np.ndarray | None = None) -> None:
        u = np.asarray(
            world_up if world_up is not None else (0.0, 0.0, 1.0),
            dtype=np.float64,
        )
        self._world_up = u / np.linalg.norm(u)

    def fit_and_visualize(
        self,
        *,
        scene,
        pipeline: rs.pipeline,
        K: np.ndarray,
        robot_model: RobotModelWrapper,
        image_width: int,
        image_height: int,
        site_name: str = "d435",
        half_size: float = 2.5,
        min_points: int = 250,
        max_attempts: int = 45,
    ) -> None:
        align = rs.align(rs.stream.color)
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        pos_link, world_from_link = robot_model.get_site_frame(site_name)
        world_from_cv = world_from_link @ R_SITE_FROM_OPENCV

        bottom_v0 = int(image_height * 0.62)
        stride = 4
        pts: list[np.ndarray] = []

        for _ in range(max_attempts):
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            if not depth_frame:
                continue

            pts.clear()
            for v in range(bottom_v0, image_height, stride):
                for u in range(0, image_width, stride):
                    d = depth_frame.get_distance(int(u), int(v))
                    if d <= 0.0 or d > 6.0 or not np.isfinite(d):
                        continue
                    x = (float(u) - cx) * d / fx
                    y = (float(v) - cy) * d / fy
                    z = d
                    p_cv = np.array([x, y, z], dtype=np.float64)
                    p_w = pos_link + world_from_cv @ p_cv
                    if np.all(np.isfinite(p_w)):
                        pts.append(p_w)
            if len(pts) >= min_points:
                break

        if len(pts) < min_points:
            print(
                f"GroundPlaneEstimator: too few depth points ({len(pts)} < {min_points}); "
                "skipping plane fit."
            )
            return

        P = np.stack(pts, axis=0)
        centroid = P.mean(axis=0)
        _, _, vh = np.linalg.svd(P - centroid, full_matrices=False)
        normal = vh[-1].astype(np.float64)
        if float(np.dot(normal, self._world_up)) < 0.0:
            normal = -normal
        normal /= np.linalg.norm(normal)

        # Plane n·x + d = 0 with n the upward-pointing normal; d = -n·centroid
        d = float(-np.dot(normal, centroid))
        print(
            "Ground plane (world): "
            f"n=[{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}], "
            f"d={d:.3f} ({len(pts)} points)"
        )

        mesh = _ground_plane_quad_trimesh(centroid, normal, half_size=half_size)
        scene.add_mesh_trimesh(
            "/ground_plane",
            mesh,
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
        )


def _ground_plane_quad_trimesh(
    point_on_plane: np.ndarray,
    normal: np.ndarray,
    *,
    half_size: float,
) -> trimesh.Trimesh:
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    aux = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(normal, aux))) > 0.92:
        aux = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    t1 = np.cross(normal, aux)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    corners = []
    for sx, sy in (-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0):
        corners.append(
            point_on_plane + half_size * (sx * t1 + sy * t2),
        )
    verts = np.asarray(corners, dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    rgba = np.tile(
        np.array([[90, 140, 95, 130]], dtype=np.uint8),
        (mesh.faces.shape[0], 1),
    )
    mesh.visual.face_colors = rgba
    return mesh


class Manager:
    def __init__(self, *, initial_pose_timeout_s: float = 10.0) -> None:
        self._initial_pose_timeout_s = initial_pose_timeout_s

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        print(status, result)

        for info in RealSenseDeviceManager.list_devices():
            print(f"Device found: {info.name} (serial={info.serial})")

        self._rs_width, self._rs_height, self._rs_fps = 640, 480, 30
        self._realsense = RealSenseDeviceManager(
            self._rs_width,
            self._rs_height,
            self._rs_fps,
            # serial="236422074588",
            serial="347622073775",
            enable_color=True,
            enable_depth=True,
        )

        # Distance from optical center to the frustum image plane (matches pinhole FOV).
        self._cam_image_depth = 0.4
        fy = float(self._realsense.K[1, 1])
        fov_y = 2.0 * math.atan(self._rs_height / (2.0 * fy))
        aspect = float(self._rs_width) / float(self._rs_height)

        self.visualizer = ViserVisualizer()
        self.ground_plane_estimator = GroundPlaneEstimator()
        self.camera_image_handle = self.visualizer.add_camera_image(
            "/realsense/color",
            (self._rs_height, self._rs_width, 3),
            fov_y=fov_y,
            aspect=aspect,
            frustum_depth=self._cam_image_depth,
        )
        self.robot_model = RobotModelWrapper("robot_model/g1_29dof_rev_1_0.xml")
        self.robot_model_handle = self.visualizer.add_robot(self.robot_model)
        self._qpos = np.zeros(self.robot_model.mj_model.nq, dtype=np.float64)
        self._qpos[3] = 1.0
        self.robot_model.update(self._qpos)
        self.robot_model_handle.update()

        self._initial_odom = threading.Event()
        self._initial_lowstate = threading.Event()

        self._camera_stop = threading.Event()
        self._camera_thread = threading.Thread(
            target=self._stream_camera_to_viser,
            daemon=True,
        )
        self._camera_thread.start()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 1)

        self.odom_subscriber = ChannelSubscriber("rt/odommodestate", SportModeState_)
        self.odom_subscriber.Init(self.SportModeStateHandler, 1)

        if self._wait_for_initial_pose(self._initial_pose_timeout_s):
            self.ground_plane_estimator.fit_and_visualize(
                scene=self.visualizer.server.scene,
                pipeline=self._realsense.pipeline,
                K=self._realsense.K,
                robot_model=self.robot_model,
                image_width=self._rs_width,
                image_height=self._rs_height,
            )

    def _wait_for_initial_pose(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._initial_odom.is_set() and self._initial_lowstate.is_set():
                self.robot_model.update(self._qpos)
                self.robot_model_handle.update()
                print(
                    "Manager: initial odometry and lowstate received; "
                    "synced robot model for ground plane."
                )
                return True
            time.sleep(0.01)
        print(
            "Manager: timeout waiting for initial pose ("
            f"odom={self._initial_odom.is_set()}, "
            f"lowstate={self._initial_lowstate.is_set()}); "
            "skipping ground plane fit."
        )
        return False

    def _stream_camera_to_viser(self) -> None:
        while not self._camera_stop.is_set():
            try:
                frames = self._realsense.pipeline.wait_for_frames()
            except RuntimeError:
                break

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            bgr = np.asanyarray(color_frame.get_data())
            rgb = bgr[:, :, ::-1]
            self.camera_image_handle.image = rgb

            pos_link, world_from_link = self.robot_model.get_site_frame("d435")
            world_from_cv = world_from_link @ R_SITE_FROM_OPENCV

            wxyz = np.zeros(4, dtype=np.float64)
            mujoco.mju_mat2Quat(wxyz, world_from_cv.flatten(order="C"))
            self.camera_image_handle.position = pos_link
            self.camera_image_handle.wxyz = wxyz

    def switch_mode(self) -> None:
        pass

    @timer_decorator
    def LowStateHandler(self, msg: LowState_) -> None:
        self.jpos = np.asarray([msg.motor_state[i].q for i in range(29)])
        self.jvel = np.asarray([msg.motor_state[i].dq for i in range(29)])
        self.quat_wxyz = np.asarray(msg.imu_state.quaternion)
        self._qpos[3:7] = self.quat_wxyz
        self._qpos[7 : 7 + len(self.jpos)] = self.jpos
        self._initial_lowstate.set()

    @timer_decorator
    def SportModeStateHandler(self, msg: SportModeState_) -> None:
        self._qpos[0:3] = msg.position
        self._initial_odom.set()

    def run(self) -> None:
        try:
            for step in itertools.count():
                self.robot_model.update(self._qpos)
                self.robot_model_handle.update()
                time.sleep(0.02)

                if step % 50 == 0:
                    print(f"LowStateHandler freq: {self.LowStateHandler.freq:.1f} Hz")
                    print(
                        f"SportModeStateHandler freq: {self.SportModeStateHandler.freq:.1f} Hz"
                    )
        except KeyboardInterrupt:
            pass
        finally:
            self._camera_stop.set()
            self._camera_thread.join(timeout=2.0)
            self._realsense.stop()


if __name__ == "__main__":

    iface = "eth0"
    ChannelFactoryInitialize(0, iface)

    manager = Manager()
    manager.run()
