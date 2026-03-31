from __future__ import annotations

import os

# Renderer + launch_passive both default to GLX; mixed GLX contexts often fail with
# BadAccess on X_GLXMakeCurrent. EGL for MuJoCo's GL lets the viewer use X while
# offscreen renders stay on EGL. Override with MUJOCO_GL if needed.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import math
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from g1_hello_world.cameras import MujocoCameraStreamer
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__SportModeState_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


def _quat_wxyz_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = [float(v) for v in q1]
    w2, x2, y2, z2 = [float(v) for v in q2]
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_wxyz_to_rpy(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat_wxyz]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _build_sim_xml(base_xml_path: Path) -> str:
    tree = ET.parse(base_xml_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str((base_xml_path.parent / "meshes").resolve()))

    rot_x_pi = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    torso_body = root.find(".//body[@name='torso_link']")
    if torso_body is not None and torso_body.find("./camera[@name='sim_head_camera']") is None:
        head_site = torso_body.find("./site[@name='d435_head']")
        if head_site is not None:
            site_quat = np.fromstring(
                head_site.get("quat", "1 0 0 0"),
                sep=" ",
                dtype=np.float64,
            )
            cam_quat = _quat_wxyz_mul(site_quat, rot_x_pi)
            ET.SubElement(
                torso_body,
                "camera",
                {
                    "name": "sim_head_camera",
                    "pos": head_site.get("pos", "0 0 0"),
                    "quat": " ".join(f"{v:.9f}" for v in cam_quat),
                    "fovy": "58.0",
                },
            )

    wrist_body = root.find(".//body[@name='wrist_camera_link']")
    if wrist_body is not None and wrist_body.find("./camera[@name='sim_wrist_camera']") is None:
        wrist_site = wrist_body.find("./site[@name='d435_wrist']")
        if wrist_site is not None:
            ET.SubElement(
                wrist_body,
                "camera",
                {
                    "name": "sim_wrist_camera",
                    "pos": wrist_site.get("pos", "0 0 0"),
                    "quat": "0 1 0 0",
                    "fovy": "58.0",
                },
            )

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    if asset.find("./texture[@name='sim_floor_texture']") is None:
        ET.SubElement(
            asset,
            "texture",
            {
                "type": "2d",
                "name": "sim_floor_texture",
                "builtin": "checker",
                "rgb1": "0.18 0.20 0.24",
                "rgb2": "0.10 0.12 0.15",
                "width": "512",
                "height": "512",
            },
        )
    if asset.find("./material[@name='sim_floor_material']") is None:
        ET.SubElement(
            asset,
            "material",
            {
                "name": "sim_floor_material",
                "texture": "sim_floor_texture",
                "texuniform": "true",
                "texrepeat": "8 8",
                "reflectance": "0.15",
            },
        )

    worldbodies = root.findall("worldbody")
    worldbody = worldbodies[-1] if worldbodies else ET.SubElement(root, "worldbody")
    if worldbody.find("./geom[@name='sim_floor']") is None:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "sim_floor",
                "type": "plane",
                "size": "0 0 0.05",
                "pos": "0 0 0",
                "material": "sim_floor_material",
                "friction": "0.9 0.1 0.1",
            },
        )

    return ET.tostring(root, encoding="unicode")


class Sim2Sim:
    """Minimal MuJoCo-to-DDS bridge for local app testing without hardware."""

    def __init__(
        self,
        *,
        xml_path: str = "robot_model/g1_29dof_rev_1_0.xml",
        sim_dt: float = 0.002,
        lowstate_hz: float = 500.0,
        odom_hz: float = 200.0,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
        head_camera_endpoint: str = "tcp://127.0.0.1:6001",
        wrist_camera_endpoint: str = "tcp://127.0.0.1:6002",
    ) -> None:
        self._xml_path = Path(xml_path)
        self._sim_dt = float(sim_dt)
        self._lowstate_period = 1.0 / float(lowstate_hz)
        self._odom_period = 1.0 / float(odom_hz)
        self.lock = threading.Lock()

        xml_text = _build_sim_xml(self._xml_path)
        self.model = mujoco.MjModel.from_xml_string(xml_text)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self._sim_dt

        self._init_pose()
        with self.lock:
            mujoco.mj_forward(self.model, self.data)

        self.lowstate_publisher = ChannelPublisher("rt/lowstate", LowState_)
        self.lowstate_publisher.Init()
        self.odom_publisher = ChannelPublisher("rt/odommodestate", SportModeState_)
        self.odom_publisher.Init()
        self.head_camera = MujocoCameraStreamer(
            self,
            camera_name="sim_head_camera",
            endpoint=head_camera_endpoint,
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
        )
        self.wrist_camera = MujocoCameraStreamer(
            self,
            camera_name="sim_wrist_camera",
            endpoint=wrist_camera_endpoint,
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
        )

        self._camera_fps = int(camera_fps)
        self._camera_stop = threading.Event()
        self._camera_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._viewer = None

    def _init_pose(self) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[2] = 0.793
        self.data.qpos[3] = 1.0

    def _publish_lowstate(self) -> None:
        msg = unitree_hg_msg_dds__LowState_()
        with self.lock:
            q = np.asarray(self.data.qpos[7 : 7 + 29], dtype=np.float64).copy()
            dq = np.asarray(self.data.qvel[6 : 6 + 29], dtype=np.float64).copy()
            quat_wxyz = np.asarray(self.data.qpos[3:7], dtype=np.float64).copy()
        rpy = _quat_wxyz_to_rpy(quat_wxyz)

        for i in range(29):
            msg.motor_state[i].q = float(q[i])
            msg.motor_state[i].dq = float(dq[i])
        msg.imu_state.quaternion = quat_wxyz.tolist()
        msg.imu_state.rpy = rpy.tolist()
        self.lowstate_publisher.Write(msg)

    def _publish_odom(self) -> None:
        msg = unitree_go_msg_dds__SportModeState_()
        with self.lock:
            pos = np.asarray(self.data.qpos[0:3], dtype=np.float64).copy()
        msg.position = pos.tolist()
        self.odom_publisher.Write(msg)

    def _run(self) -> None:
        next_lowstate_t = time.monotonic()
        next_odom_t = next_lowstate_t
        next_step_t = next_lowstate_t

        while not self._stop_event.is_set():
            now = time.monotonic()
            if now < next_step_t:
                time.sleep(min(0.001, next_step_t - now))
                continue

            with self.lock:
                mujoco.mj_step(self.model, self.data)
            next_step_t += self._sim_dt

            if now >= next_lowstate_t:
                self._publish_lowstate()
                next_lowstate_t += self._lowstate_period

            if now >= next_odom_t:
                self._publish_odom()
                next_odom_t += self._odom_period

    def _camera_loop(self) -> None:
        period = 1.0 / float(max(1, self._camera_fps))
        while not self._camera_stop.is_set():
            t0 = time.monotonic()
            self.head_camera.render_and_publish()
            self.wrist_camera.render_and_publish()
            slack = period - (time.monotonic() - t0)
            if slack > 0.0:
                time.sleep(slack)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._camera_stop.clear()
        self._camera_thread = threading.Thread(
            target=self._camera_loop,
            name="sim2sim-cameras",
            daemon=True,
        )
        self._camera_thread.start()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="sim2sim",
            daemon=True,
        )
        self._thread.start()

    def sync_viewer(self) -> None:
        """Call from the thread that owns the GLFW/GLX context (the main thread)."""
        if self._viewer is None:
            return
        with self.lock:
            self._viewer.sync()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._camera_stop.set()
        if self._camera_thread is not None and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=2.0)
        self._camera_thread = None
        self.head_camera.close()
        self.wrist_camera.close()
        if self._viewer is not None:
            self._viewer.close()
        self._viewer = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo DDS emulator for g1-hello-world")
    parser.add_argument("--xml", default="robot_model/g1_29dof_rev_1_0.xml", help="Base robot XML")
    parser.add_argument("--head-camera-endpoint", default="tcp://127.0.0.1:6001")
    parser.add_argument("--wrist-camera-endpoint", default="tcp://127.0.0.1:6002")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ChannelFactoryInitialize(0, "lo")
    sim = Sim2Sim(
        xml_path=args.xml,
        head_camera_endpoint=args.head_camera_endpoint,
        wrist_camera_endpoint=args.wrist_camera_endpoint,
    )
    sim.start()
    print("sim2sim running. Publishing rt/lowstate and rt/odommodestate.")
    try:
        while True:
            sim.sync_viewer()
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        pass
    finally:
        sim.stop()
