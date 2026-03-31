from __future__ import annotations

import cv2 # type: ignore
import argparse
import itertools
import threading
import time

import numpy as np
import trimesh
import logging

logging.basicConfig(level=logging.INFO)

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_
from unitree_sdk2py.utils.crc import CRC

from g1_hello_world.constants import (
    R_SITE_FROM_OPENCV,
    T_LEFT_WRIST_LINK_END_TO_RGB_PLACEHOLDER,
    T_LEFT_WRIST_YAW_TO_LINK_END,
)
from g1_hello_world.cameras import RealSenseDeviceManager, ZmqSimCameraDevice
from g1_hello_world.robot_model import RobotModelWrapper
from g1_hello_world.timing import timer_decorator
from g1_hello_world.visualization import ViserVisualizer
from g1_hello_world.estimators import GroundPlaneEstimator, PointTrackerRemote
from g1_hello_world.utils.timerfd import Timer

HEAD_SERIAL = "347622073775"
WRIST_SERIAL = "236422074588"

class Manager:
    def __init__(
        self,
        visualization: bool = True,
        arm_sdk: bool = True,
        sim2sim: bool = False,
        *,
        initial_pose_timeout_s: float = 10.0,
        track_server_endpoint: str = "tcp://127.0.0.1:5555",
        point_tracker_port: int = 0,
        sim_head_camera_endpoint: str = "tcp://127.0.0.1:6001",
        sim_wrist_camera_endpoint: str = "tcp://127.0.0.1:6002",
    ) -> None:
        self._visualization = visualization
        self._initial_pose_timeout_s = initial_pose_timeout_s
        self._sim2sim = sim2sim

        self._rs_width, self._rs_height, self._rs_fps = 640, 480, 30
        self.msc = None
        self.realsense_head = None
        self.realsense_wrist = None
        self.ground_plane_estimator = None
        if not self._sim2sim:
            self.msc = MotionSwitcherClient()
            self.msc.SetTimeout(5.0)
            self.msc.Init()

            status, result = self.msc.CheckMode()
            print(status, result)

            for info in RealSenseDeviceManager.list_devices():
                print(f"Device found: {info.name} (serial={info.serial})")

            try:
                self.realsense_head = RealSenseDeviceManager(
                    self._rs_width,
                    self._rs_height,
                    self._rs_fps,
                    serial=HEAD_SERIAL,
                    enable_color=True,
                    enable_depth=True,
                )
                self.realsense_head.start()
                self.ground_plane_estimator = GroundPlaneEstimator()
            except Exception as e:
                print(f"Error starting realsense_head: {e}")
                self.realsense_head = None
                self.ground_plane_estimator = None
            
            try:
                self.realsense_wrist = RealSenseDeviceManager(
                    self._rs_width,
                    self._rs_height,
                    self._rs_fps,
                    serial=WRIST_SERIAL,
                    enable_color=True,
                    enable_depth=True,
                )
                self.realsense_wrist.start()
            except Exception as e:
                print(f"Error starting realsense_wrist: {e}")
                self.realsense_wrist = None
        else:
            self.realsense_head = ZmqSimCameraDevice(sim_head_camera_endpoint)
            self.realsense_wrist = ZmqSimCameraDevice(sim_wrist_camera_endpoint)
            self.realsense_head.start()
            self.realsense_wrist.start()
            self.ground_plane_estimator = GroundPlaneEstimator()

        self.robot_model = RobotModelWrapper("robot_model/g1_29dof_rev_1_0.xml")
        self._qpos = np.zeros(self.robot_model.mj_model.nq, dtype=np.float64)
        self._qpos[3] = 1.0  # floating base quaternion w (identity), MuJoCo qpos[3:7] wxyz
        self.robot_model.update(self._qpos)
        self.imu_rpy = np.zeros(3, dtype=np.float64)

        self._initial_odom = threading.Event()
        self._initial_lowstate = threading.Event()

        if arm_sdk:
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

            self.arm_sdk_publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
            self.arm_sdk_publisher.Init()
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.crc = CRC()
        else:
            self.arm_sdk_publisher = None

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 0)

        self.odom_subscriber = ChannelSubscriber("rt/odommodestate", SportModeState_)
        self.odom_subscriber.Init(self.SportModeStateHandler, 0)

        self.point_tracker_remote: PointTrackerRemote | None = None
        if point_tracker_port > 0 and self.realsense_wrist is not None:
            self.point_tracker_remote = PointTrackerRemote(
                server_endpoint=track_server_endpoint,
                realsense_device=self.realsense_wrist,
            )
            self.point_tracker_remote.start()
        if self._visualization:
            self.setup_visualization()
        else:
            self.visualizer = None

        if point_tracker_port > 0 and self.point_tracker_remote is not None:
            port = point_tracker_port
            tracker = self.point_tracker_remote

            def _serve_gradio() -> None:
                demo = tracker.build_ui()
                demo.queue()
                demo.launch(
                    server_name="0.0.0.0",
                    server_port=port,
                    share=False,
                )

            threading.Thread(
                target=_serve_gradio,
                daemon=True,
                name="gradio-point-tracker",
            ).start()
            logging.info(
                "Point tracker Gradio UI — open http://127.0.0.1:%s "
                "(Start remote tracker, then run the ZMQ track server at %r).",
                port,
                track_server_endpoint,
            )

    def setup_visualization(self) -> None:
        self.visualizer = ViserVisualizer()
        self.visualizer.add_robot(
            self.robot_model,
            body_names=["torso_link", ".*wrist_yaw_link", ".*ankle_roll_link"],
        )
        if self.realsense_head is not None:
            self.visualizer.add_camera(
                "/realsense_head/color",
                self.realsense_head,
                (self._rs_height, self._rs_width, 3),
                frustum_depth=0.4,
                robot_model=self.robot_model,
                site_name="d435_head",
            )
        if self.realsense_wrist is not None:
            self.visualizer.add_camera(
                "/realsense_wrist/color",
                self.realsense_wrist,
                (self._rs_height, self._rs_width, 3),
                frustum_depth=0.4,
                robot_model=self.robot_model,
                site_name="d435_wrist",
            )
        self.visualizer.add_body_frame(
            "/frames/left_wrist_rgb_mount",
            self.robot_model,
            body_name="left_wrist_yaw_link",
            body_from_frame=(
                T_LEFT_WRIST_YAW_TO_LINK_END
                @ T_LEFT_WRIST_LINK_END_TO_RGB_PLACEHOLDER
            ),
        )
        if self._wait_for_initial_pose(self._initial_pose_timeout_s):
            if self.ground_plane_estimator is not None:
                self.ground_plane_estimator.fit_and_visualize(
                    scene=self.visualizer.server.scene,
                    realsense=self.realsense_head,
                    K=self.realsense_head.K,
                    robot_model=self.robot_model,
                    image_width=self._rs_width,
                    image_height=self._rs_height,
                    site_name="d435_head",
                )
        self.visualizer.run_async(freq=20)

    def _update_point_tracker_visualization(
        self,
        tracked_points_link: np.ndarray | None,
        tracked_visibility: np.ndarray | None,
    ) -> None:
        if tracked_points_link is None or tracked_visibility is None:
            self.visualizer.set_tracker_points(None)
            return

        pos_link, world_from_link = self.robot_model.get_site_frame("d435_wrist")
        tracked_points_world = (
            tracked_points_link @ world_from_link.T
        ) + pos_link[None, :]

        colors = np.tile(
            np.array([[255, 0, 0]], dtype=np.uint8),
            (tracked_points_world.shape[0], 1),
        )
        colors[np.asarray(tracked_visibility, dtype=bool)] = np.array(
            [0, 255, 0], dtype=np.uint8
        )
        self.visualizer.set_tracker_points(
            tracked_points_world,
            colors=colors,
        )

    def _wait_for_initial_pose(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._initial_odom.is_set() and self._initial_lowstate.is_set():
                self.robot_model.update(self._qpos)
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
        self.imu_rpy = np.asarray(msg.imu_state.rpy, dtype=np.float64).copy()
        # qpos: [0:3] free-joint pos from odom; [3:7] pelvis quat wxyz (IMU);
        # [7:36] hinge angles in SDK order (same as joint_names in XML after free joint).
        self._qpos[3:7] = self.quat_wxyz
        self._qpos[7:] = self.jpos
        self._initial_lowstate.set()

    @timer_decorator
    def SportModeStateHandler(self, msg: SportModeState_) -> None:
        self._qpos[0:3] = msg.position
        self._initial_odom.set()

    def compute_arm_control(self, points: np.ndarray | None) -> None:
        # Left arm only; leave the right arm as reported by LowState (no separate right-arm goals).
        #
        # Resolving the chain:
        #   arm_joint_indices, _ = self.robot_model.find_joints(
        #       ["left_(shoulder|elbow|wrist).*"]
        #   )
        #   typ. SDK indices [15, 16, 17, 18, 19, 20, 21] for
        #   ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
        #    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint',
        #    'left_wrist_yaw_joint']
        #   MuJoCo hinge joint ids are i+1 (joint 0 is the free joint); map to dof columns via
        #   mj_model.jnt_dofadr[joint_id].
        #
        # Use self.jpos / self.jvel and Jacobians from self.robot_model.mj_data after update().
        #
        # Task specification:
        #   1. points is None — gaze: desired wrist “look” direction follows pelvis forward in body
        #      (_pelvis_forward_body, e.g. [1,0,0] in pelvis frame), rotated by IMU like _qpos[3:7],
        #      then projected to the horizontal plane (normal _world_horizontal_normal).
        #   2. points is not None — tracking: steer left_wrist_yaw_link (camera bore via
        #      get_site_frame("d435_wrist")) so the optical axis aims at the centroid of points
        #      in world (transform points from link frame to world first).
        #
        # Shared secondary objectives:
        #   - Align link +Z with world “right” [0, -1, 0] (_world_up).
        #   - Regulate wrist height: keep wrist origin at _wrist_height_in_root (m) along pelvis +Z.
        #
        # Implementation hints (removed here): differential IK / task-space errors → Δq for the
        # left-arm dof columns; smooth and integrate to q_cmd; fill self.low_cmd (kp/kd, q, dq for
        # left arm; mirror measured q for other joints); set CRC; self.arm_sdk_publisher.Write(...).
        # Floating-base quat in LowState is wxyz; if gaze heading is mirrored, try conjugating
        # before body→world. For moving targets, EMA + short lead time on the 3D point helps during gait.
        if self.arm_sdk_publisher is None:
            return

    def run(self) -> None:
        timer = Timer(0.02)
        try:
            for step in itertools.count():
                t0 = time.perf_counter()
                points: np.ndarray | None = None
                if self.point_tracker_remote is not None:
                    points, visibility = (
                        self.point_tracker_remote.get_tracked_points_snapshot()
                    )
                    self._update_point_tracker_visualization(points, visibility)
                self.robot_model.update(self._qpos, jacobian=False)
                if self.arm_sdk_publisher is not None:
                    self.compute_arm_control(points)
                if step % 50 == 0:
                    msg = f"LowStateHandler freq: {self.LowStateHandler.freq:.1f} Hz, SportModeStateHandler freq: {self.SportModeStateHandler.freq:.1f} Hz"
                    print(msg)
                timer.sleep()
        except KeyboardInterrupt:
            pass
        finally:
            if self.point_tracker_remote is not None:
                self.point_tracker_remote.stop()
            if self.visualizer is not None:
                self.visualizer.stop_async()
            if self.realsense_head is not None:
                print("Stopping RealSense head...")
                self.realsense_head.stop()
            if self.realsense_wrist is not None:
                print("Stopping RealSense wrist...")
                self.realsense_wrist.stop()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G1 hello-world Viser + RealSense viewer")
    parser.add_argument("--iface", default="eth0", help="Network interface for DDS")
    parser.add_argument(
        "--sim2sim",
        action="store_true",
        help="Connect to a separately running sim2sim.py process. Use with --iface lo.",
    )
    parser.add_argument("--sim-head-camera-endpoint", default="tcp://127.0.0.1:6001")
    parser.add_argument("--sim-wrist-camera-endpoint", default="tcp://127.0.0.1:6002")
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

    # Real robot:
    #   uv run app.py --iface eth0
    #
    # Local sim2sim smoke test without hardware:
    #   terminal 1: uv run python sim2sim.py
    #   terminal 2: uv run app.py --iface lo --sim2sim
    #
    # In sim2sim mode, sim2sim.py runs as a separate process. It publishes DDS state
    # locally and streams rendered head/wrist RGB-D frames over ZMQ. Manager skips
    # MotionSwitcher / RealSense setup and connects receiver-side camera devices.

    manager = Manager(
        point_tracker_port=args.point_tracker_port,
        track_server_endpoint=args.track_server,
        sim2sim=args.sim2sim,
        sim_head_camera_endpoint=args.sim_head_camera_endpoint,
        sim_wrist_camera_endpoint=args.sim_wrist_camera_endpoint,
    )
    manager.run()
