from __future__ import annotations

import cv2 # type: ignore
import argparse
import itertools
import threading
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import trimesh
import logging
import requests
import zmq

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
from g1_hello_world.cameras import RealSenseDeviceManager, ZmqSimCameraDevice
from g1_hello_world.robot_model import RobotModelWrapper
from g1_hello_world.timing import timer_decorator
from g1_hello_world.visualization import ViserVisualizer
from g1_hello_world.estimators import GroundPlaneEstimator, PointTrackerRemote
from g1_hello_world.utils.timerfd import Timer
from query import DEFAULT_PORT as DEFAULT_SEGMENTATION_PORT
from query import DEFAULT_SERVER_ADDRESS as DEFAULT_SEGMENTATION_SERVER

try:
    import pinocchio as pin
except ModuleNotFoundError:
    pin = None

HEAD_SERIAL = "347622073775"
WRIST_SERIAL = "236422074588"
LEFT_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)


class HeadSegmentationProxyServer:
    """ZMQ REP server that forwards head-camera RGB frames to the HTTP segmentation service."""

    def __init__(
        self,
        *,
        camera_device: RealSenseDeviceManager | ZmqSimCameraDevice,
        point_tracker_remote: PointTrackerRemote | None,
        bind_endpoint: str,
        segmentation_server: str,
        segmentation_port: int,
        request_timeout_s: float = 15.0,
    ) -> None:
        self._camera_device = camera_device
        self._point_tracker_remote = point_tracker_remote
        self._bind_endpoint = bind_endpoint
        self._segmentation_server = segmentation_server
        self._segmentation_port = int(segmentation_port)
        self._request_timeout_s = float(request_timeout_s)
        self._context = zmq.Context.instance()
        self._socket: zmq.Socket | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _read_head_rgb(self) -> np.ndarray:
        rgb, _ = self._camera_device.read_aligned_rgb_depth(timeout_s=1.0)
        return rgb

    def _post_segmentation_request(
        self,
        *,
        rgb: np.ndarray,
        target_type: str,
        predict_endpoint: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )
        if not ok:
            raise RuntimeError("Failed to encode head RGB frame as JPEG.")
        url = (
            f"http://{self._segmentation_server}:{self._segmentation_port}"
            f"{predict_endpoint}"
        )
        response = requests.post(
            url,
            data={"target_type": target_type},
            files={
                "file": (
                    "head_camera.jpg",
                    encoded.tobytes(),
                    "image/jpeg",
                )
            },
            timeout=timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Segmentation server did not return a JSON object.")
        return payload

    @staticmethod
    def _extract_bboxes(payload: dict[str, Any]) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for key in ("handle", "local_white_region", "global_white_door_region"):
            value = payload.get(key)
            if isinstance(value, dict) and isinstance(value.get("bbox_xyxy"), list):
                bbox = [int(v) for v in value["bbox_xyxy"][:4]]
                if len(bbox) == 4:
                    out[key] = bbox
        return out

    @staticmethod
    def _bbox_corner_queries(bbox_xyxy: list[int]) -> list[tuple[int, int]]:
        x_min, y_min, x_max, y_max = [int(v) for v in bbox_xyxy[:4]]
        x_mid = int(round(0.5 * (x_min + x_max)))
        y_mid = int(round(0.5 * (y_min + y_max)))
        return [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
            (x_mid, y_mid),
        ]

    def _handle_request(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            return {"ok": False, "error": "request must be a dict"}
        op = str(request.get("op", ""))
        if op == "ping":
            return {
                "ok": True,
                "status": "head-segmentation-proxy-ready",
                "bind_endpoint": self._bind_endpoint,
                "segmentation_server": self._segmentation_server,
                "segmentation_port": self._segmentation_port,
            }
        if op != "segment_head":
            return {"ok": False, "error": f"unknown op: {op!r}"}

        target_type = str(request.get("target_type", "handle"))
        predict_endpoint = str(request.get("predict_endpoint", "/predict"))
        timeout_s = float(request.get("timeout_s", self._request_timeout_s))
        t0 = time.perf_counter()
        rgb = self._read_head_rgb()
        capture_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        payload = self._post_segmentation_request(
            rgb=rgb,
            target_type=target_type,
            predict_endpoint=predict_endpoint,
            timeout_s=timeout_s,
        )
        request_ms = (time.perf_counter() - t1) * 1000.0
        bboxes = self._extract_bboxes(payload)
        tracker_submit_ok = False
        tracker_submit_status: str | None = None
        if self._point_tracker_remote is not None and "handle" in bboxes:
            tracker_submit_status = self._point_tracker_remote.submit_query_points(
                self._bbox_corner_queries(bboxes["handle"]),
                capture_latest=True,
            )
            tracker_submit_ok = "Tracking started" in tracker_submit_status
        return {
            "ok": True,
            "image_width": int(rgb.shape[1]),
            "image_height": int(rgb.shape[0]),
            "capture_ms": capture_ms,
            "request_ms": request_ms,
            "predict": payload,
            "bboxes": bboxes,
            "tracker_submit_ok": tracker_submit_ok,
            "tracker_submit_status": tracker_submit_status,
        }

    def _serve(self) -> None:
        sock = self._context.socket(zmq.REP)
        sock.setsockopt(zmq.RCVTIMEO, 200)
        sock.setsockopt(zmq.SNDTIMEO, 200)
        sock.bind(self._bind_endpoint)
        self._socket = sock
        while not self._stop_event.is_set():
            try:
                request = sock.recv_pyobj()
            except zmq.Again:
                continue
            except Exception as exc:
                logging.warning("Head segmentation proxy receive failed: %s", exc)
                continue
            try:
                response = self._handle_request(request)
            except Exception as exc:
                response = {"ok": False, "error": str(exc)}
            try:
                sock.send_pyobj(response)
            except Exception as exc:
                logging.warning("Head segmentation proxy send failed: %s", exc)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._serve,
            name="head-segmentation-proxy",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = None


class _BaseArmController:
    def __init__(
        self,
        *,
        robot_model: RobotModelWrapper,
        arm_sdk_publisher: ChannelPublisher,
        low_cmd: LowCmd_,
        crc: CRC,
    ) -> None:
        self._robot_model = robot_model
        self._arm_sdk_publisher = arm_sdk_publisher
        self._low_cmd = low_cmd
        self._crc = crc
        self._arm_joint_indices, self._arm_joint_names = self._robot_model.find_joints(
            list(LEFT_ARM_JOINT_NAMES)
        )
        self._arm_kp = 28.0
        self._arm_kd = 1.8
        self._look_gain = 1.2
        self._task_damping = 0.20
        self._joint_damping = 0.45
        self._null_gain = 0.12
        self._step_size = 0.12
        self._dq_limit = 0.018
        self._cmd_alpha = 0.10
        self._world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self._nominal_pose = np.array(
            [0.30, 0.55, 0.0, 1.05, 0.0, -0.35, 0.0],
            dtype=np.float64,
        )
        m = self._robot_model.mj_model
        mj_joint_ids = [sdk_i + 1 for sdk_i in self._arm_joint_indices]
        self._joint_lo = np.array([m.jnt_range[jid, 0] for jid in mj_joint_ids], dtype=np.float64)
        self._joint_hi = np.array([m.jnt_range[jid, 1] for jid in mj_joint_ids], dtype=np.float64)
        self._q_cmd = self._nominal_pose.copy()
        self._initialized = False

    def _initialize_command(self, jpos: np.ndarray) -> None:
        if self._initialized:
            return
        self._q_cmd = np.asarray(jpos[self._arm_joint_indices], dtype=np.float64).copy()
        self._initialized = True

    def _publish_command(self, *, jpos: np.ndarray) -> None:
        self._low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0
        for i in range(29):
            self._low_cmd.motor_cmd[i].tau = 0.0
            self._low_cmd.motor_cmd[i].dq = 0.0
            self._low_cmd.motor_cmd[i].kp = 0.0
            self._low_cmd.motor_cmd[i].kd = 0.0
            self._low_cmd.motor_cmd[i].q = float(jpos[i])

        for k, sdk_i in enumerate(self._arm_joint_indices):
            self._low_cmd.motor_cmd[sdk_i].kp = self._arm_kp
            self._low_cmd.motor_cmd[sdk_i].kd = self._arm_kd
            self._low_cmd.motor_cmd[sdk_i].q = float(self._q_cmd[k])

        self._low_cmd.crc = self._crc.Crc(self._low_cmd)
        self._arm_sdk_publisher.Write(self._low_cmd)


class ArmController(_BaseArmController):
    """MuJoCo-Jacobian left-arm orientation IK."""

    def __init__(
        self,
        *,
        robot_model: RobotModelWrapper,
        arm_sdk_publisher: ChannelPublisher,
        low_cmd: LowCmd_,
        crc: CRC,
    ) -> None:
        super().__init__(
            robot_model=robot_model,
            arm_sdk_publisher=arm_sdk_publisher,
            low_cmd=low_cmd,
            crc=crc,
        )
        m = self._robot_model.mj_model
        mj_joint_ids = [sdk_i + 1 for sdk_i in self._arm_joint_indices]
        self._arm_dof_cols = np.array(
            [int(m.jnt_dofadr[jid]) for jid in mj_joint_ids],
            dtype=np.int32,
        )
        self._wrist_site_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_SITE, "d435_wrist"
        )
        self._jacp_buf = np.zeros((3, m.nv), dtype=np.float64)
        self._jacr_buf = np.zeros((3, m.nv), dtype=np.float64)

    def _compute_orientation_dq(
        self,
        *,
        forward_des: np.ndarray,
        jpos: np.ndarray,
        jvel: np.ndarray,
    ) -> np.ndarray:
        q_arm = np.asarray(jpos[self._arm_joint_indices], dtype=np.float64)
        dq_arm = np.asarray(jvel[self._arm_joint_indices], dtype=np.float64)
        _, world_from_wrist = self._robot_model.get_site_frame("d435_wrist")
        norm_forward = np.linalg.norm(forward_des)
        if norm_forward < 1e-6:
            forward_des = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            forward_des /= norm_forward

        up_des = self._world_up - float(self._world_up @ forward_des) * forward_des
        norm_up = np.linalg.norm(up_des)
        if norm_up < 1e-6:
            up_des = world_from_wrist[:, 2].copy()
            up_des -= float(up_des @ forward_des) * forward_des
            norm_up = np.linalg.norm(up_des)
        if norm_up < 1e-6:
            up_des = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            up_des -= float(up_des @ forward_des) * forward_des
            up_des /= np.linalg.norm(up_des) + 1e-9
        else:
            up_des /= norm_up

        left_des = np.cross(up_des, forward_des)
        left_des /= np.linalg.norm(left_des) + 1e-9
        up_des = np.cross(forward_des, left_des)
        up_des /= np.linalg.norm(up_des) + 1e-9

        rot_des = np.column_stack((forward_des, left_des, up_des))
        omega = 0.5 * (
            np.cross(world_from_wrist[:, 0], rot_des[:, 0])
            + np.cross(world_from_wrist[:, 1], rot_des[:, 1])
            + np.cross(world_from_wrist[:, 2], rot_des[:, 2])
        )

        mujoco.mj_jacSite(
            self._robot_model.mj_model,
            self._robot_model.mj_data,
            self._jacp_buf,
            self._jacr_buf,
            self._wrist_site_id,
        )
        j_rot = self._jacr_buf[:, self._arm_dof_cols]
        task = self._look_gain * omega - self._joint_damping * (j_rot @ dq_arm)
        lhs = j_rot @ j_rot.T + self._task_damping * np.eye(3, dtype=np.float64)
        dq_task = j_rot.T @ np.linalg.solve(lhs, task)

        eye = np.eye(len(self._arm_joint_indices), dtype=np.float64)
        null_proj = eye - j_rot.T @ np.linalg.solve(lhs, j_rot)
        dq_null = null_proj @ (self._null_gain * (self._nominal_pose - q_arm))
        dq = self._step_size * (dq_task + dq_null)
        return np.clip(dq, -self._dq_limit, self._dq_limit)

    def _compute_no_target_dq(
        self,
        *,
        jpos: np.ndarray,
        jvel: np.ndarray,
    ) -> np.ndarray:
        _, world_from_torso = self._robot_model.get_body_frame("torso_link")
        forward_des = world_from_torso[:, 0].copy()
        forward_des -= float(forward_des @ self._world_up) * self._world_up
        return self._compute_orientation_dq(
            forward_des=forward_des,
            jpos=jpos,
            jvel=jvel,
        )

    def step(
        self,
        *,
        points: np.ndarray | None,
        jpos: np.ndarray,
        jvel: np.ndarray,
    ) -> None:
        self._initialize_command(jpos)
        if points is None:
            dq_cmd = self._compute_no_target_dq(jpos=jpos, jvel=jvel)
        else:
            points_world = np.asarray(points, dtype=np.float64)
            valid = np.all(np.isfinite(points_world), axis=1)
            pos_wrist, world_from_wrist = self._robot_model.get_site_frame("d435_wrist")
            if np.any(valid):
                center_world = np.mean(points_world[valid], axis=0)
                forward_des = center_world - pos_wrist
            else:
                forward_des = world_from_wrist[:, 0].copy()
            dq_cmd = self._compute_orientation_dq(
                forward_des=forward_des,
                jpos=jpos,
                jvel=jvel,
            )
        q_des = self._q_cmd + dq_cmd
        q_des = np.clip(q_des, self._joint_lo, self._joint_hi)
        self._q_cmd = (1.0 - self._cmd_alpha) * self._q_cmd + self._cmd_alpha * q_des
        self._q_cmd = np.clip(self._q_cmd, self._joint_lo, self._joint_hi)
        self._publish_command(jpos=jpos)


class PinocchioArmController(_BaseArmController):
    """Pinocchio-based left-arm orientation IK alternative."""

    def __init__(
        self,
        *,
        robot_model: RobotModelWrapper,
        arm_sdk_publisher: ChannelPublisher,
        low_cmd: LowCmd_,
        crc: CRC,
        urdf_path: str | Path,
    ) -> None:
        if pin is None:
            raise RuntimeError(
                "Pinocchio is not available in this Python environment. "
                "Run the app in the project environment, e.g. with `uv run`."
            )
        super().__init__(
            robot_model=robot_model,
            arm_sdk_publisher=arm_sdk_publisher,
            low_cmd=low_cmd,
            crc=crc,
        )
        self._pin_model = pin.buildModelFromUrdf(str(urdf_path))
        self._pin_data = self._pin_model.createData()
        self._pin_q_neutral = pin.neutral(self._pin_model)
        self._pin_q_indices = np.array(
            [
                int(self._pin_model.joints[self._pin_model.getJointId(joint_name)].idx_q)
                for joint_name in self._robot_model.joint_names
            ],
            dtype=np.int32,
        )
        self._pin_v_indices = np.array(
            [
                int(self._pin_model.joints[self._pin_model.getJointId(joint_name)].idx_v)
                for joint_name in self._robot_model.joint_names
            ],
            dtype=np.int32,
        )
        self._arm_pin_q_indices = np.array(
            [
                int(self._pin_model.joints[self._pin_model.getJointId(joint_name)].idx_q)
                for joint_name in self._arm_joint_names
            ],
            dtype=np.int32,
        )
        self._arm_pin_v_indices = np.array(
            [
                int(self._pin_model.joints[self._pin_model.getJointId(joint_name)].idx_v)
                for joint_name in self._arm_joint_names
            ],
            dtype=np.int32,
        )
        self._torso_frame_id = int(self._pin_model.getFrameId("torso_link"))
        self._wrist_frame_id = int(self._pin_model.getFrameId("left_wrist_yaw_link"))

        pos_body, world_from_body = self._robot_model.get_body_frame("left_wrist_yaw_link")
        pos_site, world_from_site = self._robot_model.get_site_frame("d435_wrist")
        self._wrist_from_site_rot = world_from_body.T @ world_from_site
        self._wrist_from_site_pos = world_from_body.T @ (pos_site - pos_body)

    def _build_pin_state(
        self,
        *,
        jpos: np.ndarray,
        jvel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        q_pin = self._pin_q_neutral.copy()
        v_pin = np.zeros(self._pin_model.nv, dtype=np.float64)
        q_pin[self._pin_q_indices] = np.asarray(jpos, dtype=np.float64)
        v_pin[self._pin_v_indices] = np.asarray(jvel, dtype=np.float64)
        return q_pin, v_pin

    def _compute_orientation_dq(
        self,
        *,
        forward_des_world: np.ndarray,
        jpos: np.ndarray,
        jvel: np.ndarray,
    ) -> np.ndarray:
        q_pin, v_pin = self._build_pin_state(jpos=jpos, jvel=jvel)
        pin.forwardKinematics(self._pin_model, self._pin_data, q_pin, v_pin)
        pin.updateFramePlacements(self._pin_model, self._pin_data)
        pin.computeJointJacobians(self._pin_model, self._pin_data, q_pin)

        world_from_torso_actual = self._robot_model.get_body_frame("torso_link")[1]
        torso_from_world_actual = world_from_torso_actual.T
        forward_des_torso = torso_from_world_actual @ np.asarray(
            forward_des_world,
            dtype=np.float64,
        )
        norm_forward = np.linalg.norm(forward_des_torso)
        if norm_forward < 1e-6:
            forward_des_torso = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            forward_des_torso /= norm_forward

        world_up_torso = torso_from_world_actual @ self._world_up
        torso_from_wrist = (
            self._pin_data.oMf[self._torso_frame_id].inverse()
            * self._pin_data.oMf[self._wrist_frame_id]
        )
        torso_from_site_rot = torso_from_wrist.rotation @ self._wrist_from_site_rot
        current_up_torso = torso_from_site_rot[:, 2].copy()
        up_des_torso = (
            world_up_torso
            - float(world_up_torso @ forward_des_torso) * forward_des_torso
        )
        norm_up = np.linalg.norm(up_des_torso)
        if norm_up < 1e-6:
            up_des_torso = (
                current_up_torso
                - float(current_up_torso @ forward_des_torso) * forward_des_torso
            )
            norm_up = np.linalg.norm(up_des_torso)
        if norm_up < 1e-6:
            up_des_torso = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            up_des_torso -= float(up_des_torso @ forward_des_torso) * forward_des_torso
            up_des_torso /= np.linalg.norm(up_des_torso) + 1e-9
        else:
            up_des_torso /= norm_up

        left_des_torso = np.cross(up_des_torso, forward_des_torso)
        left_des_torso /= np.linalg.norm(left_des_torso) + 1e-9
        up_des_torso = np.cross(forward_des_torso, left_des_torso)
        up_des_torso /= np.linalg.norm(up_des_torso) + 1e-9
        rot_des_torso = np.column_stack(
            (forward_des_torso, left_des_torso, up_des_torso)
        )
        omega_torso = 0.5 * (
            np.cross(torso_from_site_rot[:, 0], rot_des_torso[:, 0])
            + np.cross(torso_from_site_rot[:, 1], rot_des_torso[:, 1])
            + np.cross(torso_from_site_rot[:, 2], rot_des_torso[:, 2])
        )

        jac_world = pin.computeFrameJacobian(
            self._pin_model,
            self._pin_data,
            q_pin,
            self._wrist_frame_id,
            pin.ReferenceFrame.WORLD,
        )
        torso_from_world_pin = self._pin_data.oMf[self._torso_frame_id].rotation.T
        j_rot_torso = (
            torso_from_world_pin @ jac_world[3:, :]
        )[:, self._arm_pin_v_indices]

        q_arm = q_pin[self._arm_pin_q_indices]
        dq_arm = np.asarray(jvel[self._arm_joint_indices], dtype=np.float64)
        task = self._look_gain * omega_torso - self._joint_damping * (j_rot_torso @ dq_arm)
        lhs = j_rot_torso @ j_rot_torso.T + self._task_damping * np.eye(3, dtype=np.float64)
        dq_task = j_rot_torso.T @ np.linalg.solve(lhs, task)

        eye = np.eye(len(self._arm_joint_indices), dtype=np.float64)
        null_proj = eye - j_rot_torso.T @ np.linalg.solve(lhs, j_rot_torso)
        dq_null = null_proj @ (self._null_gain * (self._nominal_pose - q_arm))
        dq = self._step_size * (dq_task + dq_null)
        return np.clip(dq, -self._dq_limit, self._dq_limit)

    def _compute_no_target_dq(
        self,
        *,
        jpos: np.ndarray,
        jvel: np.ndarray,
    ) -> np.ndarray:
        _, world_from_torso = self._robot_model.get_body_frame("torso_link")
        forward_des_world = world_from_torso[:, 0].copy()
        forward_des_world -= float(forward_des_world @ self._world_up) * self._world_up
        return self._compute_orientation_dq(
            forward_des_world=forward_des_world,
            jpos=jpos,
            jvel=jvel,
        )

    def step(
        self,
        *,
        points: np.ndarray | None,
        jpos: np.ndarray,
        jvel: np.ndarray,
    ) -> None:
        self._initialize_command(jpos)
        if points is None:
            dq_cmd = self._compute_no_target_dq(jpos=jpos, jvel=jvel)
        else:
            points_world = np.asarray(points, dtype=np.float64)
            valid = np.all(np.isfinite(points_world), axis=1)
            pos_wrist, world_from_wrist = self._robot_model.get_site_frame("d435_wrist")
            if np.any(valid):
                center_world = np.mean(points_world[valid], axis=0)
                forward_des_world = center_world - pos_wrist
            else:
                forward_des_world = world_from_wrist[:, 0].copy()
            dq_cmd = self._compute_orientation_dq(
                forward_des_world=forward_des_world,
                jpos=jpos,
                jvel=jvel,
            )

        q_des = self._q_cmd + dq_cmd
        q_des = np.clip(q_des, self._joint_lo, self._joint_hi)
        self._q_cmd = (1.0 - self._cmd_alpha) * self._q_cmd + self._cmd_alpha * q_des
        self._q_cmd = np.clip(self._q_cmd, self._joint_lo, self._joint_hi)
        self._publish_command(jpos=jpos)

class Manager:
    def __init__(
        self,
        visualization: bool = True,
        arm_sdk: bool = True,
        sim2sim: bool = False,
        arm_controller_backend: str = "mujoco",
        head_segmentation_proxy_endpoint: str | None = None,
        segmentation_server: str = DEFAULT_SEGMENTATION_SERVER,
        segmentation_port: int = DEFAULT_SEGMENTATION_PORT,
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
        self._head_segmentation_proxy_endpoint = head_segmentation_proxy_endpoint
        self._point_tracker_site_name = "d435_head"

        self._rs_width, self._rs_height, self._rs_fps = 640, 480, 30
        self.msc = None
        self.realsense_head = None
        self.realsense_wrist = None
        self.ground_plane_estimator = None
        self.head_segmentation_proxy: HeadSegmentationProxyServer | None = None
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
        self.jpos = np.zeros(29, dtype=np.float64)
        self.jvel = np.zeros(29, dtype=np.float64)
        self.quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.imu_rpy = np.zeros(3, dtype=np.float64)

        self._initial_odom = threading.Event()
        self._initial_lowstate = threading.Event()

        if arm_sdk:
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

            self.arm_sdk_publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
            self.arm_sdk_publisher.Init()
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.crc = CRC()
            if arm_controller_backend == "pinocchio":
                self.arm_controller = PinocchioArmController(
                    robot_model=self.robot_model,
                    arm_sdk_publisher=self.arm_sdk_publisher,
                    low_cmd=self.low_cmd,
                    crc=self.crc,
                    urdf_path=Path(__file__).resolve().parent / "robot_model" / "g1_29dof_rev_1_0.urdf",
                )
            else:
                self.arm_controller = ArmController(
                    robot_model=self.robot_model,
                    arm_sdk_publisher=self.arm_sdk_publisher,
                    low_cmd=self.low_cmd,
                    crc=self.crc,
                )
        else:
            self.arm_sdk_publisher = None
            self.arm_controller = None

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 0)

        self.odom_subscriber = ChannelSubscriber("rt/odommodestate", SportModeState_)
        self.odom_subscriber.Init(self.SportModeStateHandler, 0)

        self.point_tracker_remote: PointTrackerRemote | None = None
        need_point_tracker = (
            point_tracker_port > 0
            or self._head_segmentation_proxy_endpoint is not None
        )
        if need_point_tracker and self.realsense_head is not None:
            self.point_tracker_remote = PointTrackerRemote(
                server_endpoint=track_server_endpoint,
                realsense_device=self.realsense_head,
            )
            self.point_tracker_remote.start()
        if (
            self._head_segmentation_proxy_endpoint
            and self.realsense_head is not None
        ):
            self.head_segmentation_proxy = HeadSegmentationProxyServer(
                camera_device=self.realsense_head,
                point_tracker_remote=self.point_tracker_remote,
                bind_endpoint=self._head_segmentation_proxy_endpoint,
                segmentation_server=segmentation_server,
                segmentation_port=segmentation_port,
            )
            self.head_segmentation_proxy.start()
            logging.info(
                "Head segmentation proxy listening at %s -> http://%s:%s",
                self._head_segmentation_proxy_endpoint,
                segmentation_server,
                segmentation_port,
            )
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

        pos_link, world_from_link = self.robot_model.get_site_frame(
            self._point_tracker_site_name
        )
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
        if self.arm_controller is None:
            return
        # now = time.perf_counter() - self.start_time
        # r = 0.4
        # points = np.array([[1.2, 0.0 + r * np.sin(now * np.pi), 0.8 + r * np.cos(now * np.pi)]], dtype=np.float64) # for testing
        self.arm_controller.step(points=points, jpos=self.jpos, jvel=self.jvel)

    def run(self) -> None:
        self.start_time = time.perf_counter()
        timer = Timer(0.02)
        try:
            for step in itertools.count():
                tracked_points: np.ndarray | None = None
                control_points: np.ndarray | None = None
                if self.point_tracker_remote is not None:
                    tracked_points, visibility = (
                        self.point_tracker_remote.get_tracked_points_snapshot()
                    )
                    self._update_point_tracker_visualization(
                        tracked_points, visibility
                    )
                    if self._point_tracker_site_name == "d435_wrist":
                        control_points = tracked_points
                self.robot_model.update(self._qpos, jacobian=False)
                if self.arm_sdk_publisher is not None:
                    self.compute_arm_control(control_points)
                if step % 50 == 0:
                    msg = f"LowStateHandler freq: {self.LowStateHandler.freq:.1f} Hz, SportModeStateHandler freq: {self.SportModeStateHandler.freq:.1f} Hz"
                    print(msg)
                timer.sleep()
        except KeyboardInterrupt:
            pass
        finally:
            if self.head_segmentation_proxy is not None:
                self.head_segmentation_proxy.stop()
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
        "--arm-controller-backend",
        choices=("mujoco", "pinocchio"),
        default="pinocchio",
        help="Left-arm controller backend.",
    )
    parser.add_argument(
        "--head-segmentation-proxy-endpoint",
        default="",
        help="If non-empty, bind a ZMQ REP server here for `segment_head` requests.",
    )
    parser.add_argument(
        "--segmentation-server",
        default=DEFAULT_SEGMENTATION_SERVER,
        help=f"HTTP segmentation server hostname or IP (default: {DEFAULT_SEGMENTATION_SERVER}).",
    )
    parser.add_argument(
        "--segmentation-port",
        type=int,
        default=DEFAULT_SEGMENTATION_PORT,
        help=f"HTTP segmentation server port (default: {DEFAULT_SEGMENTATION_PORT}).",
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
    if args.sim2sim:
        iface = "lo"
    else:
        iface = args.iface
    ChannelFactoryInitialize(0, iface)

    # Real robot:
    #   uv run app.py --iface eth0
    #
    # Local sim2sim smoke test without hardware:
    #   terminal 1: uv run python sim2sim.py
    #   terminal 2: uv run app.py --iface lo --sim2sim
    #
    # Optional head segmentation proxy:
    #   uv run app.py --iface eth0 --head-segmentation-proxy-endpoint tcp://127.0.0.1:5562
    #   Request payload: {"op": "segment_head", "target_type": "handle"}
    #
    # In sim2sim mode, sim2sim.py runs as a separate process. It publishes DDS state
    # locally and streams rendered head/wrist RGB-D frames over ZMQ. Manager skips
    # MotionSwitcher / RealSense setup and connects receiver-side camera devices.

    manager = Manager(
        point_tracker_port=args.point_tracker_port,
        track_server_endpoint=args.track_server,
        sim2sim=args.sim2sim,
        arm_controller_backend=args.arm_controller_backend,
        head_segmentation_proxy_endpoint=(
            args.head_segmentation_proxy_endpoint or None
        ),
        segmentation_server=args.segmentation_server,
        segmentation_port=args.segmentation_port,
        sim_head_camera_endpoint=args.sim_head_camera_endpoint,
        sim_wrist_camera_endpoint=args.sim_wrist_camera_endpoint,
    )
    manager.run()
