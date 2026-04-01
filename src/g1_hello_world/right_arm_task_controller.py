from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as sRot

from .constants import G1JointIndex
from .pinocchio_ik import IKResult, RightArmPinocchioIK
from .sim_command_interface import SimCommandBuffer, SimCommandInterface


RIGHT_ARM_INDICES = np.arange(G1JointIndex.RightShoulderPitch, G1JointIndex.RightWristYaw + 1)


@dataclass(frozen=True)
class RightArmTarget:
    xyz: np.ndarray
    rpy_xyz: np.ndarray


@dataclass(frozen=True)
class RightArmSolveOutput:
    joint_targets: np.ndarray
    ik_result: IKResult
    solved_xyz: np.ndarray
    solved_rpy_xyz: np.ndarray


class RightArmTaskController:
    def __init__(self, urdf_path: str | Path, sim_interface: SimCommandInterface) -> None:
        self.sim_interface = sim_interface
        self.solver = RightArmPinocchioIK(urdf_path)
        self.target = RightArmTarget(
            xyz=np.array([0.30, -0.25, 0.18], dtype=np.float64),
            rpy_xyz=np.zeros(3, dtype=np.float64),
        )
        self.arm_kp = np.full(len(RIGHT_ARM_INDICES), 80.0, dtype=np.float64)
        self.arm_kd = np.full(len(RIGHT_ARM_INDICES), 6.0, dtype=np.float64)
        self.position_tol = 1e-3
        self.orientation_tol = 2e-3
        self._last_solution = self.solver.neutral_configuration()

    def set_target(self, xyz: np.ndarray, rpy_xyz: np.ndarray) -> None:
        self.target = RightArmTarget(
            xyz=np.asarray(xyz, dtype=np.float64).copy(),
            rpy_xyz=np.asarray(rpy_xyz, dtype=np.float64).copy(),
        )

    def solve(self, current_joint_positions: np.ndarray, *, max_iters: int = 200) -> RightArmSolveOutput:
        q_seed = np.asarray(current_joint_positions, dtype=np.float64).copy()
        q_solution, ik_result = self.solver.solve_in_base_frame(
            self.target.xyz,
            self.target.rpy_xyz,
            q_init=q_seed,
            max_iters=max_iters,
            position_tol=self.position_tol,
            orientation_tol=self.orientation_tol,
        )
        self._last_solution = q_solution.copy()
        pose = self.solver.get_frame_pose_in_base(q_solution, "right_rubber_hand")
        solved_rpy = sRot.from_matrix(pose.rotation).as_euler("xyz")
        return RightArmSolveOutput(
            joint_targets=q_solution[RIGHT_ARM_INDICES].copy(),
            ik_result=ik_result,
            solved_xyz=pose.translation.copy(),
            solved_rpy_xyz=solved_rpy,
        )

    def last_solution(self) -> np.ndarray:
        return self._last_solution.copy()

    def apply_targets(self, buffer: SimCommandBuffer, joint_targets: np.ndarray) -> None:
        self.sim_interface.set_targets(
            buffer,
            RIGHT_ARM_INDICES,
            np.asarray(joint_targets, dtype=np.float64),
            kp=self.arm_kp,
            kd=self.arm_kd,
        )
