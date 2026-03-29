from __future__ import annotations

import time
import argparse
import threading
import itertools
from functools import update_wrapper
from pathlib import Path
from weakref import WeakKeyDictionary

import trimesh
import numpy as np
import pyrealsense2 as rs
import viser
import mujoco
from typing import Any, Callable, Tuple
from scipy.spatial.transform import Rotation as sRot

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

MJ_MESH = mujoco.mjtGeom.mjGEOM_MESH

# MuJoCo `d435` site = ROS `camera_link` (+X forward, +Y left, +Z up). Viser / COLMAP / OpenCV
# camera: +X right, +Y down, +Z forward. Same static rotation as Intel RealSense ROS descriptions.
# p_link = R_SITE_FROM_OPENCV @ p_cv  =>  p_world = R_world_site @ R_SITE_FROM_OPENCV @ p_cv
R_SITE_FROM_OPENCV = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


class _BoundTimed:
    __slots__ = ("_fn", "_inst", "_last_t", "_ema_dt", "_alpha")

    def __init__(self, fn: Callable[..., Any], inst: Any, *, alpha: float) -> None:
        self._fn = fn
        self._inst = inst
        self._last_t: float | None = None
        self._ema_dt: float | None = None
        self._alpha = alpha

    @property
    def freq(self) -> float:
        if self._ema_dt is None or self._ema_dt <= 0.0:
            return 0.0
        return 1.0 / self._ema_dt

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        now = time.monotonic()
        if self._last_t is not None:
            dt = now - self._last_t
            if self._ema_dt is None:
                self._ema_dt = dt
            else:
                self._ema_dt = (1.0 - self._alpha) * self._ema_dt + self._alpha * dt
        self._last_t = now
        return self._fn(self._inst, *args, **kwargs)


class _TimedMethod:
    """Descriptor: instance access returns a callable with a `.freq` property (Hz)."""

    def __init__(self, fn: Callable[..., Any], *, alpha: float = 0.2) -> None:
        self._fn = fn
        self._alpha = alpha
        self._bound: WeakKeyDictionary[Any, _BoundTimed] = WeakKeyDictionary()
        update_wrapper(self, fn)

    def __get__(self, instance: Any, owner: type | None) -> _BoundTimed | _TimedMethod:
        if instance is None:
            return self
        if instance not in self._bound:
            self._bound[instance] = _BoundTimed(self._fn, instance, alpha=self._alpha)
        return self._bound[instance]


def timer_decorator(
    fn: Callable[..., Any] | None = None, *, alpha: float = 0.2
) -> _TimedMethod | Callable[[Callable[..., Any]], _TimedMethod]:
    """Track callback rate with an EMA of inter-arrival times. Use ``self.Handler.freq`` (Hz)."""

    if fn is None:
        return lambda f: _TimedMethod(f, alpha=alpha)
    return _TimedMethod(fn, alpha=alpha)


def _geom_mesh_trimesh(model: mujoco.MjModel, gid: int) -> trimesh.Trimesh:
    mesh_id = int(model.geom_dataid[gid])
    v0 = int(model.mesh_vertadr[mesh_id])
    nv = int(model.mesh_vertnum[mesh_id])
    f0 = int(model.mesh_faceadr[mesh_id])
    nf = int(model.mesh_facenum[mesh_id])
    verts = model.mesh_vert[v0 : v0 + nv].copy() * model.mesh_scale[mesh_id]
    R9 = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(R9, model.geom_quat[gid])
    R = R9.reshape(3, 3)
    verts = verts @ R.T + model.geom_pos[gid]
    faces = np.asarray(model.mesh_face[f0 : f0 + nf])
    return trimesh.Trimesh(vertices=verts, faces=faces)


class RobotModelWrapper:
    def __init__(self, xml_path: str | Path) -> None:
        self.mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.mj_data = mujoco.MjData(self.mj_model)

        self.joint_names = [
            mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(self.mj_model.njnt)
        ]
        self.body_ids = list(range(1, self.mj_model.nbody))
        self.body_names = [
            mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, bid)
            or f"body_{bid}"
            for bid in self.body_ids
        ]

        meshes_by_body: dict[int, list[trimesh.Trimesh]] = {}
        for gid in range(self.mj_model.ngeom):
            if self.mj_model.geom_type[gid] != MJ_MESH:
                continue
            bid = int(self.mj_model.geom_bodyid[gid])
            meshes_by_body.setdefault(bid, []).append(_geom_mesh_trimesh(self.mj_model, gid))

        self.body_meshes: list[trimesh.Trimesh | None] = []
        for bid in self.body_ids:
            parts = meshes_by_body.get(bid)
            if not parts:
                self.body_meshes.append(None)
            elif len(parts) > 1:
                self.body_meshes.append(trimesh.util.concatenate(parts))
            else:
                self.body_meshes.append(parts[0])
        
        self._site_ids = {}

    def update(self, q: np.ndarray):
        self.mj_data.qpos[:] = q
        mujoco.mj_forward(self.mj_model, self.mj_data)
    
    def get_site_pose(self, site_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """World pose of the site after the last `update()`: position (3,) and wxyz quaternion (4)."""
        if site_name not in self._site_ids:
            sid = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name
            )
            if sid < 0:
                raise ValueError(f"unknown site name: {site_name!r}")
            self._site_ids[site_name] = sid
        sid = self._site_ids[site_name]
        d = self.mj_data
        pos = np.asarray(d.site_xpos[sid], dtype=np.float64).copy()
        mat = d.site_xmat[sid].reshape(3, 3)
        quat_wxyz = sRot.from_matrix(mat).as_quat(scalar_first=True)
        return pos, quat_wxyz

    def get_site_frame(self, site_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Position (3,) and world-from-site rotation (3,3), same convention as `site_xmat`."""
        if site_name not in self._site_ids:
            sid = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name
            )
            if sid < 0:
                raise ValueError(f"unknown site name: {site_name!r}")
            self._site_ids[site_name] = sid
        sid = self._site_ids[site_name]
        d = self.mj_data
        pos = np.asarray(d.site_xpos[sid], dtype=np.float64).copy()
        R = np.asarray(d.site_xmat[sid], dtype=np.float64).reshape(3, 3).copy()
        return pos, R


class ViserRobotModelHandle:
    def __init__(self, scene: viser.SceneApi, robot_model: RobotModelWrapper) -> None:
        self.scene = scene
        self.robot_model = robot_model
        self.mesh_handles: list[viser.GlbHandle | None] = []
        for body_name, mesh in zip(
            self.robot_model.body_names, self.robot_model.body_meshes, strict=True
        ):
            if mesh is None:
                self.mesh_handles.append(None)
                continue
            handle = self.scene.add_mesh_trimesh(
                name=f"/robot/{body_name}",
                mesh=mesh,
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
            )
            self.mesh_handles.append(handle)

    def update(self):
        data = self.robot_model.mj_data
        for bid, mesh_handle in zip(self.robot_model.body_ids, self.mesh_handles, strict=True):
            if mesh_handle is None:
                continue
            quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quat, data.xmat[bid])
            mesh_handle.position = np.asarray(data.xpos[bid], dtype=np.float64)
            mesh_handle.wxyz = quat


class ViserVisualizer:
    def __init__(self) -> None:
        self.server = viser.ViserServer()
    
    def add_camera_image(
        self,
        hwc: Tuple[int, int, int],
        *,
        render_width: float,
        render_height: float,
    ) -> viser.ImageHandle:
        self.camera_image_handle = self.server.scene.add_image(
            "/realsense/color",
            np.zeros(hwc, dtype=np.uint8),
            render_width=render_width,
            render_height=render_height,
            format="jpeg",
            jpeg_quality=85,
            position=(0.0, 0.0, 0.0),
        )
        return self.camera_image_handle
    
    def add_robot(self, robot_model: RobotModelWrapper) -> ViserRobotModelHandle:
        return ViserRobotModelHandle(self.server.scene, robot_model)


class Manager:

    def __init__(self):
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        print(status, result)

        self._rs_width, self._rs_height, self._rs_fps = 640, 480, 30
        self.rs_pipeline = rs.pipeline()
        rs_cfg = rs.config()
        rs_cfg.enable_stream(
            rs.stream.color,
            self._rs_width,
            self._rs_height,
            rs.format.bgr8,
            self._rs_fps,
        )
        self.rs_pipeline.start(rs_cfg)
        color_stream = (
            self.rs_pipeline.get_active_profile()
            .get_stream(rs.stream.color)
            .as_video_stream_profile()
        )
        rs_intr = color_stream.get_intrinsics()
        self.rs_K = np.array(
            [
                [rs_intr.fx, 0.0, rs_intr.ppx],
                [0.0, rs_intr.fy, rs_intr.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        # Distance from optical center to the image quad; scales quad to match pinhole FOV.
        self._cam_image_depth = 0.4
        _rw = self._cam_image_depth * self._rs_width / rs_intr.fx
        _rh = self._cam_image_depth * self._rs_height / rs_intr.fy

        self.visualizer = ViserVisualizer()
        self.camera_image_handle = self.visualizer.add_camera_image(
            (self._rs_height, self._rs_width, 3),
            render_width=_rw,
            render_height=_rh,
        )
        self._d435_pose_lock = threading.Lock()
        self._d435_pose = (
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64),
        )
        self.robot_model = RobotModelWrapper("robot_model/g1_29dof_rev_1_0.xml")
        
        self.robot_model_handle = self.visualizer.add_robot(self.robot_model)
        self._qpos = np.zeros(self.robot_model.mj_model.nq, dtype=np.float64)
        self._qpos[3] = 1.0
        self.robot_model.update(self._qpos)
        self.robot_model_handle.update()

        self._camera_stop = threading.Event()
        self._camera_thread = threading.Thread(target=self._stream_camera_to_viser, daemon=True)
        self._camera_thread.start()

        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 1)

        self.odom_subscriber = ChannelSubscriber("rt/odommodestate", SportModeState_)
        self.odom_subscriber.Init(self.SportModeStateHandler, 1)

    def _stream_camera_to_viser(self):
        while not self._camera_stop.is_set():
            try:
                frames = self.rs_pipeline.wait_for_frames()
            except RuntimeError:
                break
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            bgr = np.asanyarray(color_frame.get_data())
            rgb = bgr[:, :, ::-1]
            self.camera_image_handle.image = rgb
            
            pos_link, R_world_from_link = self.robot_model.get_site_frame("d435")
            pos_link = np.asarray(pos_link, dtype=np.float64)

            R_world_from_link = np.asarray(R_world_from_link, dtype=np.float64).reshape(3, 3)
            d = self._cam_image_depth
            # OpenCV/COLMAP world rotation: columns are +X right, +Y down, +Z forward in world.
            R_world_from_cv = R_world_from_link @ R_SITE_FROM_OPENCV
            offset = np.array([0.0, 0.0, d], dtype=np.float64)
            center = pos_link + R_world_from_cv @ offset
            wxyz = np.zeros(4, dtype=np.float64)
            mujoco.mju_mat2Quat(wxyz, R_world_from_cv.flatten(order="C"))
            self.camera_image_handle.position = center
            self.camera_image_handle.wxyz = wxyz

    def switch_mode(self):
        pass
    
    @timer_decorator
    def LowStateHandler(self, msg: LowState_):
        jpos = []
        jvel = []
        for i in range(29):
            jpos.append(msg.motor_state[i].q)
            jvel.append(msg.motor_state[i].dq)
        self.jpos = np.asarray(jpos)
        self.jvel = np.asarray(jvel)
        self.quat_wxyz = np.asarray(msg.imu_state.quaternion)
        self._qpos[3 : 7] = self.quat_wxyz
        self._qpos[7 : 7 + len(self.jpos)] = self.jpos
    
    @timer_decorator
    def SportModeStateHandler(self, msg: SportModeState_):
        self._qpos[0:3] = msg.position
    
    def run(self):
        try:
            for i in itertools.count():
                self.robot_model.update(self._qpos)
                self.robot_model_handle.update()
                time.sleep(0.02)

                if i % 50 == 0:
                    print(f"LowStateHandler freq: {self.LowStateHandler.freq:.1f} Hz")
                    print(f"SportModeStateHandler freq: {self.SportModeStateHandler.freq:.1f} Hz")
        except KeyboardInterrupt:
            pass
        finally:
            self._camera_stop.set()
            self._camera_thread.join(timeout=2.0)
            self.rs_pipeline.stop()


def main():
    parser = argparse.ArgumentParser()

    iface = "eth0"
    ChannelFactoryInitialize(0, iface)
    manager = Manager()
    manager.run()


if __name__ == "__main__":
    main()
