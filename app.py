from __future__ import annotations

import argparse
import itertools
import threading
import time

import numpy as np
import trimesh

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

from g1_hello_world.realsense_device import RealSenseDeviceManager
from g1_hello_world.robot_model import RobotModelWrapper
from g1_hello_world.timing import timer_decorator
from g1_hello_world.visualization import ViserVisualizer
from g1_hello_world.estimators import GroundPlaneEstimator, PointTrackerRemote


class Manager:
    def __init__(
        self,
        *,
        initial_pose_timeout_s: float = 10.0,
        track_server_endpoint: str = "tcp://127.0.0.1:5555",
        point_tracker_port: int = 0,
    ) -> None:
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

        self.visualizer = ViserVisualizer()
        self.ground_plane_estimator = GroundPlaneEstimator()

        self.point_tracker_remote: PointTrackerRemote | None = None
        on_frame = None
        if point_tracker_port > 0:
            self.point_tracker_remote = PointTrackerRemote(
                server_endpoint=track_server_endpoint,
                realsense_device=self._realsense,
                use_internal_frame_loop=False,
            )

        self.robot_model = RobotModelWrapper("robot_model/g1_29dof_rev_1_0.xml")
        self.robot_model_handle = self.visualizer.add_robot(self.robot_model)
        self._qpos = np.zeros(self.robot_model.mj_model.nq, dtype=np.float64)
        self._qpos[3] = 1.0
        self.robot_model.update(self._qpos)
        self.robot_model_handle.update()

        self._camera_handle = self.visualizer.add_camera(
            "/realsense/color",
            self._realsense,
            (self._rs_height, self._rs_width, 3),
            frustum_depth=self._cam_image_depth,
            robot_model=self.robot_model,
            on_frame=on_frame,
        )

        self._initial_odom = threading.Event()
        self._initial_lowstate = threading.Event()

        self._camera_stop = threading.Event()
        self._camera_handle.start_stream(self._camera_stop)

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 1)

        self.odom_subscriber = ChannelSubscriber("rt/odommodestate", SportModeState_)
        self.odom_subscriber.Init(self.SportModeStateHandler, 1)

        if self._wait_for_initial_pose(self._initial_pose_timeout_s):
            self.ground_plane_estimator.fit_and_visualize(
                scene=self.visualizer.server.scene,
                realsense=self._realsense,
                K=self._realsense.K,
                robot_model=self.robot_model,
                image_width=self._rs_width,
                image_height=self._rs_height,
            )

        # if point_tracker_port > 0 and self.point_tracker_remote is not None:
        #     _tracker = self.point_tracker_remote

        #     def _run_point_tracker_ui() -> None:
        #         demo = _tracker.build_ui()
        #         demo.queue()
        #         demo.launch(
        #             server_name="0.0.0.0",
        #             server_port=point_tracker_port,
        #             share=False,
        #         )

        #     threading.Thread(target=_run_point_tracker_ui, daemon=True).start()
        #     print(
        #         "Point tracker UI (Gradio): open "
        #         f"http://127.0.0.1:{point_tracker_port} — click “Start remote tracker”, "
        #         "then capture / query / submit. Run the Track-On ZMQ server on "
        #         f"{track_server_endpoint!r}."
        #     )

    def _forward_point_tracker_frame(
        self, rgb: np.ndarray, depth: np.ndarray, capture_ms: float
    ) -> None:
        if self.point_tracker_remote is not None:
            self.point_tracker_remote.on_aligned_frame(
                rgb, depth, capture_ms=capture_ms
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
                self._realsense.read_aligned_rgb_depth()
                self.visualizer.update()
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
            self.visualizer.join_camera_streams(timeout=2.0)
            if self.point_tracker_remote is not None:
                self.point_tracker_remote.stop()
            self._realsense.stop()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G1 hello-world Viser + RealSense viewer")
    parser.add_argument("--iface", default="eth0", help="Network interface for DDS")
    parser.add_argument(
        "--point-tracker-port",
        type=int,
        default=0,
        help="Gradio port for remote point tracker UI; 0 disables (example: 7861)",
    )
    parser.add_argument(
        "--track-server",
        default="tcp://127.0.0.1:5555",
        help="ZMQ endpoint for Track-On server (same as track_on client)",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = _parse_args()
    ChannelFactoryInitialize(0, args.iface)

    manager = Manager(
        point_tracker_port=args.point_tracker_port,
        track_server_endpoint=args.track_server,
    )
    manager.run()
