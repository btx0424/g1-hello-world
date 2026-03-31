from __future__ import annotations

import cv2 # type: ignore
import argparse
import itertools
import threading
import time

import mujoco
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
    G1JointIndex,
    R_SITE_FROM_OPENCV,
    T_LEFT_WRIST_LINK_END_TO_RGB_PLACEHOLDER,
    T_LEFT_WRIST_YAW_TO_LINK_END,
)
from g1_hello_world.cameras import RealSenseDeviceManager
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
        simulator = None,
        *,
        initial_pose_timeout_s: float = 10.0,
        track_server_endpoint: str = "tcp://127.0.0.1:5555",
        point_tracker_port: int = 0,
    ) -> None:
        self._visualization = visualization
        self._initial_pose_timeout_s = initial_pose_timeout_s
        self._sim2sim = sim2sim

        self._rs_width, self._rs_height, self._rs_fps = 640, 480, 30
        self.msc = None
        self.realsense_head = None
        self.realsense_wrist = None
        self.ground_plane_estimator = None
        self._simulator = simulator
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
            if self._simulator is None:
                raise ValueError("sim2sim=True requires a simulator instance.")
            self.realsense_head = self._simulator.head_camera
            self.realsense_wrist = self._simulator.wrist_camera
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
            self.arm_joint_indices, self.arm_joint_names = self.robot_model.find_joints(["left_(shoulder|elbow|wrist).*"])
            # this should give us:
            # [15, 16, 17, 18, 19, 20, 21]
            # ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint']
            self._arm_kp = 42.0
            self._arm_kd = 2.8
            self._control_dt = 0.02
            self._look_gain = 2.5
            self._look_dlam = 0.06
            self._look_dq_max = 0.08
            # IK smoothing: scale DLS Δq, EMA filter, task error clamps (reduces jerk at 50 Hz).
            self._ik_step_scale = 0.5
            self._ik_dq_ema_beta = 0.4
            self._omega_task_max = 0.45
            self._height_err_clip = 0.12
            # Filter/predict the target in world space so tracking stays responsive during gait.
            self._target_pos_ema_beta = 0.4
            self._target_lead_s = 0.02
            self._target_speed_clip = 2.5
            self._target_pos_filt_w: np.ndarray | None = None
            self._target_vel_filt_w = np.zeros(3, dtype=np.float64)
            self._target_t_prev: float | None = None
            self._task_ang_damping = 0.35
            # Floating-base quat from LowState is wxyz; MuJoCo uses it as body→world. If gaze heading
            # is mirrored / wrong, try True (conjugate quaternion before mju_quat2Mat).
            self._imu_quat_wxyz_conjugate = False
            # Unit direction in pelvis body frame treated as “forward” (XML / SDK pelvis +X).
            self._pelvis_forward_body = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            # Wrist body axis steered toward fwd_des (camera bore may differ from link +X).
            self._wrist_look_axis_body = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            # World “up” for horizontal projection of pelvis forward (same as flat ground normal).
            self._world_horizontal_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            self._quat_mat9 = np.zeros(9, dtype=np.float64)
            # Pelvis is the floating-base / root body; keep wrist origin at this z in pelvis frame.
            self._wrist_height_in_root = 0.1
            self._height_gain = 12.0
            # World direction for link +Z (hardware frame): right = -world Y.
            self._world_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
            self._up_align_gain = 2.0
            m = self.robot_model.mj_model
            mj_jids = [i + 1 for i in self.arm_joint_indices]
            self._arm_dof_cols = np.array(
                [int(m.jnt_dofadr[j]) for j in mj_jids], dtype=np.int32
            )
            self._left_wrist_body_id = mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"
            )
            self._jacp_buf = np.zeros((3, m.nv), dtype=np.float64)
            self._jacr_buf = np.zeros((3, m.nv), dtype=np.float64)
            # Stacked task: 3 orientation rows + 1 height row (pelvis-frame z).
            self._task_dlam_I4 = self._look_dlam * np.eye(4, dtype=np.float64)
            self._arm_jnt_lo = np.array(
                [m.jnt_range[j, 0] for j in mj_jids], dtype=np.float64
            )
            self._arm_jnt_hi = np.array(
                [m.jnt_range[j, 1] for j in mj_jids], dtype=np.float64
            )
            self._dq_arm_ema = np.zeros(len(self.arm_joint_indices), dtype=np.float64)
            self._dq_cmd_max = 1.2
            self._q_arm_cmd = np.zeros(len(self.arm_joint_indices), dtype=np.float64)
            self._arm_cmd_initialized = False
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
        # KEEP THIS COMMENT
        # compute arm (only arm joints) control for the following cases:
        # 1. points is None: gaze along pelvis forward in body (_pelvis_forward_body) rotated by
        #    the same IMU quat as _qpos[3:7], projected to horizontal (normal _world_horizontal_normal).
        # 2. points is not None: let left_wrist_yaw_link face the center of the points
        # for both cases: align link +Z with world right [0,-1,0] (_world_up); keep wrist origin at _wrist_height_in_root (m) along pelvis +Z
        # leave the right arm where it is
        # get the joint states from self.jpos, self.jvel,
        # and jacobian from self.robot_model.mj_data.

        if self.arm_sdk_publisher is None:
            return

        m = self.robot_model.mj_model
        d = self.robot_model.mj_data
        # arm_joint_indices = SDK motor indices 15..21; MuJoCo hinge joint ids are +1 (joint 0 is free).
        arm_sdk = self.arm_joint_indices
        dof_cols = self._arm_dof_cols
        if not self._arm_cmd_initialized:
            self._q_arm_cmd[:] = self.jpos[arm_sdk]
            self._arm_cmd_initialized = True

        pos_w, R_wrist = self.robot_model.get_body_frame("left_wrist_yaw_link")
        pos_cam, world_from_link = self.robot_model.get_site_frame("d435_wrist")
        pos_root, R_root = self.robot_model.get_body_frame("pelvis")
        wl = self._wrist_look_axis_body
        forward = R_wrist @ wl
        forward = forward / (np.linalg.norm(forward) + 1e-9)

        if points is None:
            q = self._qpos[3:7].astype(np.float64).copy()
            if self._imu_quat_wxyz_conjugate:
                q[1:4] *= -1.0
            mujoco.mju_quat2Mat(self._quat_mat9, q)
            R_bw = self._quat_mat9.reshape(3, 3)
            fwd_w = R_bw @ self._pelvis_forward_body
            n_h = self._world_horizontal_normal
            fwd_des = fwd_w - float(fwd_w @ n_h) * n_h
            n = np.linalg.norm(fwd_des)
            if n < 1e-9:
                # Pelvis forward parallel to “up”; fallback along horizontal basis ex.
                ex = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                fwd_des = ex - float(ex @ n_h) * n_h
                n = np.linalg.norm(fwd_des)
                if n < 1e-9:
                    fwd_des = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                else:
                    fwd_des /= n
            else:
                fwd_des /= n
        else:
            points_link = np.asarray(points, dtype=np.float64)
            points_world = (points_link @ world_from_link.T) + pos_cam[None, :]
            valid = np.all(np.isfinite(points_world), axis=1)
            if np.any(valid):
                point_center = np.mean(points_world[valid], axis=0)
            else:
                point_center = pos_cam + forward
            t_now = time.perf_counter()
            if self._target_pos_filt_w is None:
                self._target_pos_filt_w = point_center.copy()
                self._target_vel_filt_w.fill(0.0)
                self._target_t_prev = t_now
            else:
                dt = max(1e-3, t_now - (self._target_t_prev or t_now))
                pos_prev = self._target_pos_filt_w.copy()
                beta = self._target_pos_ema_beta
                self._target_pos_filt_w = (
                    (1.0 - beta) * self._target_pos_filt_w + beta * point_center
                )
                vel_meas = (self._target_pos_filt_w - pos_prev) / dt
                speed = np.linalg.norm(vel_meas)
                if speed > self._target_speed_clip:
                    vel_meas *= self._target_speed_clip / (speed + 1e-12)
                self._target_vel_filt_w = (
                    (1.0 - beta) * self._target_vel_filt_w + beta * vel_meas
                )
                self._target_t_prev = t_now
            point_center = self._target_pos_filt_w + self._target_lead_s * self._target_vel_filt_w
            # Aim the wrist camera optical center at the tracked point, not the wrist-body origin.
            fwd_des = point_center - pos_cam
            n = np.linalg.norm(fwd_des)
            if n < 1e-6:
                fwd_des = forward.copy()
            else:
                fwd_des = fwd_des / n

        right_ref = self._world_up.copy()
        right_ref -= float(right_ref @ fwd_des) * fwd_des
        n_right = np.linalg.norm(right_ref)
        if n_right < 1e-6:
            right_ref = R_wrist[:, 2].copy()
            right_ref -= float(right_ref @ fwd_des) * fwd_des
            n_right = np.linalg.norm(right_ref)
        if n_right < 1e-6:
            fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(float(fallback @ fwd_des)) > 0.95:
                fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            right_ref = fallback - float(fallback @ fwd_des) * fwd_des
            n_right = np.linalg.norm(right_ref)
        z_des = right_ref / (n_right + 1e-9)
        y_des = np.cross(z_des, fwd_des)
        y_des /= np.linalg.norm(y_des) + 1e-9
        z_des = np.cross(fwd_des, y_des)
        z_des /= np.linalg.norm(z_des) + 1e-9

        R_des = np.column_stack((fwd_des, y_des, z_des))
        omega = 0.5 * (
            np.cross(R_wrist[:, 0], R_des[:, 0])
            + np.cross(R_wrist[:, 1], R_des[:, 1])
            + np.cross(R_wrist[:, 2], R_des[:, 2])
        )

        r = pos_w - pos_root
        p_local = R_root.T @ r
        err_h = self._wrist_height_in_root - float(p_local[2])
        err_h = float(np.clip(err_h, -self._height_err_clip, self._height_err_clip))

        jacp = self._jacp_buf
        jacr = self._jacr_buf
        mujoco.mj_jacBody(m, d, jacp, jacr, self._left_wrist_body_id)
        J_r = jacr[:, dof_cols]
        z_w = R_root[:, 2]
        J_h = (z_w @ jacp)[dof_cols].reshape(1, -1)
        omega_wrist = J_r @ self.jvel[arm_sdk]

        e = np.empty(4, dtype=np.float64)
        e[:3] = self._look_gain * omega - self._task_ang_damping * omega_wrist
        n_omega = float(np.linalg.norm(e[:3]))
        if n_omega > self._omega_task_max:
            e[:3] *= self._omega_task_max / (n_omega + 1e-12)
        e[3] = self._height_gain * err_h
        J_stack = np.vstack([J_r, J_h])
        M = J_stack @ J_stack.T + self._task_dlam_I4
        dq_raw = J_stack.T @ np.linalg.solve(M, e)
        dq_raw *= self._ik_step_scale
        dq_raw = np.clip(dq_raw, -self._look_dq_max, self._look_dq_max)
        self._dq_arm_ema = (1.0 - self._ik_dq_ema_beta) * self._dq_arm_ema
        self._dq_arm_ema += self._ik_dq_ema_beta * dq_raw

        dq_arm_cmd = self._dq_arm_ema / self._control_dt
        dq_arm_cmd = np.clip(dq_arm_cmd, -self._dq_cmd_max, self._dq_cmd_max)
        self._q_arm_cmd += dq_arm_cmd * self._control_dt
        self._q_arm_cmd = np.clip(self._q_arm_cmd, self._arm_jnt_lo, self._arm_jnt_hi)
        q_arm = self._q_arm_cmd.copy()

        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0

        for i in range(29):
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = self._arm_kp
            self.low_cmd.motor_cmd[i].kd = self._arm_kd
            self.low_cmd.motor_cmd[i].q = float(self.jpos[i])

        for k, sdk_i in enumerate(arm_sdk):
            self.low_cmd.motor_cmd[sdk_i].q = q_arm[k]
            self.low_cmd.motor_cmd[sdk_i].dq = float(dq_arm_cmd[k])

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_sdk_publisher.Write(self.low_cmd)

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
                self.robot_model.update(
                    self._qpos, jacobian=self.arm_sdk_publisher is not None
                )
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
        help="Start a local MuJoCo-based DDS emulator before running the app. Use with --iface lo.",
    )
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
    #   uv run app.py --iface lo --sim2sim
    #
    # In sim2sim mode we start a MuJoCo publisher thread from sim2sim.py that emits
    # rt/lowstate and rt/odommodestate on the local DDS transport. Manager then skips
    # MotionSwitcher / RealSense setup and subscribes to those simulated channels.
    simulator = None
    if args.sim2sim:
        from sim2sim import Sim2Sim

        simulator = Sim2Sim()
        simulator.start()

    manager = Manager(
        point_tracker_port=args.point_tracker_port,
        track_server_endpoint=args.track_server,
        sim2sim=args.sim2sim,
        simulator=simulator,
    )
    try:
        manager.run()
    finally:
        if simulator is not None:
            simulator.stop()
