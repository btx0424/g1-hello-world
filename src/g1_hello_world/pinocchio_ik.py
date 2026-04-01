from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares


RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class IKResult:
    success: bool
    iterations: int
    position_error_norm: float
    orientation_error_norm: float


class RightArmPinocchioIK:
    def __init__(
        self,
        urdf_path: str | Path,
        *,
        base_frame: str = "pelvis",
        end_effector_frame: str = "right_rubber_hand",
    ) -> None:
        self.urdf_path = Path(urdf_path)
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()

        if not self.model.existFrame(base_frame):
            raise ValueError(f"unknown base frame: {base_frame!r}")
        if not self.model.existFrame(end_effector_frame):
            raise ValueError(f"unknown end effector frame: {end_effector_frame!r}")

        self.base_frame = base_frame
        self.end_effector_frame = end_effector_frame
        self.base_frame_id = self.model.getFrameId(base_frame)
        self.end_effector_frame_id = self.model.getFrameId(end_effector_frame)

        self.arm_joint_ids = tuple(
            int(self.model.getJointId(joint_name)) for joint_name in RIGHT_ARM_JOINT_NAMES
        )
        self.arm_q_indices = np.array(
            [int(self.model.joints[jid].idx_q) for jid in self.arm_joint_ids],
            dtype=np.int32,
        )
        self.arm_lower_limits = self.model.lowerPositionLimit[self.arm_q_indices].copy()
        self.arm_upper_limits = self.model.upperPositionLimit[self.arm_q_indices].copy()

        self.pin_joint_names = tuple(
            self.model.names[joint_id] for joint_id in range(1, self.model.njoints)
        )

    def neutral_configuration(self) -> np.ndarray:
        return pin.neutral(self.model)

    def solve_in_base_frame(
        self,
        target_xyz: np.ndarray,
        target_rpy_xyz: np.ndarray,
        *,
        q_init: np.ndarray | None = None,
        max_iters: int = 200,
        position_tol: float = 1e-4,
        orientation_tol: float = 1e-4,
        damping: float = 1e-4,
        step_size: float = 0.6,
        max_delta: float = 0.12,
        orientation_weight: float = 0.35,
    ) -> tuple[np.ndarray, IKResult]:
        target_xyz = np.asarray(target_xyz, dtype=np.float64)
        target_rpy_xyz = np.asarray(target_rpy_xyz, dtype=np.float64)
        if target_xyz.shape != (3,):
            raise ValueError("target_xyz must have shape (3,)")
        if target_rpy_xyz.shape != (3,):
            raise ValueError("target_rpy_xyz must have shape (3,)")

        q = self.neutral_configuration() if q_init is None else np.asarray(q_init, dtype=np.float64).copy()
        if q.shape != (self.model.nq,):
            raise ValueError(f"q_init must have shape ({self.model.nq},)")
        q[self.arm_q_indices] = np.clip(
            q[self.arm_q_indices],
            self.arm_lower_limits,
            self.arm_upper_limits,
        )

        base_M_target = pin.SE3(pin.rpy.rpyToMatrix(*target_rpy_xyz), target_xyz)

        def _residual(arm_q: np.ndarray) -> np.ndarray:
            q_trial = q.copy()
            q_trial[self.arm_q_indices] = arm_q
            pin.forwardKinematics(self.model, self.data, q_trial)
            pin.updateFramePlacements(self.model, self.data)

            base_M_ee = self.data.oMf[self.base_frame_id].inverse() * self.data.oMf[
                self.end_effector_frame_id
            ]
            rotation_error = pin.log3(base_M_ee.rotation.T @ base_M_target.rotation)
            position_error = base_M_ee.translation - target_xyz
            return np.concatenate(
                [
                    orientation_weight * rotation_error,
                    position_error,
                ]
            )

        opt = least_squares(
            _residual,
            x0=q[self.arm_q_indices],
            bounds=(self.arm_lower_limits, self.arm_upper_limits),
            max_nfev=max_iters,
            ftol=damping,
            xtol=damping,
            gtol=damping,
        )
        q[self.arm_q_indices] = np.clip(
            opt.x,
            self.arm_lower_limits,
            self.arm_upper_limits,
        )

        residual = _residual(q[self.arm_q_indices])
        rot_err_norm = float(np.linalg.norm(residual[:3]) / max(orientation_weight, 1e-12))
        pos_err_norm = float(np.linalg.norm(residual[3:]))
        success = pos_err_norm < position_tol and rot_err_norm < orientation_tol

        result = IKResult(
            success=success,
            iterations=int(opt.nfev),
            position_error_norm=pos_err_norm,
            orientation_error_norm=rot_err_norm,
        )
        return q, result

    def get_frame_pose_in_base(self, q: np.ndarray, frame_name: str) -> pin.SE3:
        if not self.model.existFrame(frame_name):
            raise ValueError(f"unknown frame: {frame_name!r}")
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        base_frame_id = self.base_frame_id
        frame_id = self.model.getFrameId(frame_name)
        return self.data.oMf[base_frame_id].inverse() * self.data.oMf[frame_id]

    def configuration_to_joint_map(self, q: np.ndarray) -> dict[str, float]:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (self.model.nq,):
            raise ValueError(f"q must have shape ({self.model.nq},)")
        return {
            joint_name: float(q[self.model.joints[joint_id].idx_q])
            for joint_name, joint_id in zip(RIGHT_ARM_JOINT_NAMES, self.arm_joint_ids, strict=True)
        }
