from __future__ import annotations

import itertools
import threading
import time

import numpy as np
import pyrealsense2 as rs

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

from g1_hello_world.constants import R_SITE_FROM_OPENCV
from g1_hello_world.robot_model import RobotModelWrapper
from g1_hello_world.timing import timer_decorator
from g1_hello_world.visualization import ViserVisualizer


class Manager:
    def __init__(self) -> None:

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
        render_width = self._cam_image_depth * self._rs_width / rs_intr.fx
        render_height = self._cam_image_depth * self._rs_height / rs_intr.fy

        self.visualizer = ViserVisualizer()
        self.camera_image_handle = self.visualizer.add_camera_image(
            (self._rs_height, self._rs_width, 3),
            render_width=render_width,
            render_height=render_height,
        )
        self.robot_model = RobotModelWrapper("robot_model/g1_29dof_rev_1_0.xml")
        self.robot_model_handle = self.visualizer.add_robot(self.robot_model)
        self._qpos = np.zeros(self.robot_model.mj_model.nq, dtype=np.float64)
        self._qpos[3] = 1.0
        self.robot_model.update(self._qpos)
        self.robot_model_handle.update()

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

    def _stream_camera_to_viser(self) -> None:
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

            pos_link, world_from_link = self.robot_model.get_site_frame("d435")
            world_from_cv = world_from_link @ R_SITE_FROM_OPENCV
            center = pos_link + world_from_cv @ np.array(
                [0.0, 0.0, self._cam_image_depth],
                dtype=np.float64,
            )

            wxyz = np.zeros(4, dtype=np.float64)
            import mujoco

            mujoco.mju_mat2Quat(wxyz, world_from_cv.flatten(order="C"))
            self.camera_image_handle.position = center
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

    @timer_decorator
    def SportModeStateHandler(self, msg: SportModeState_) -> None:
        self._qpos[0:3] = msg.position

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
            self.rs_pipeline.stop()


if __name__ == "__main__":

    iface = "eth0"
    ChannelFactoryInitialize(0, iface)

    manager = Manager()
    manager.run()
