from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from g1_hello_world.constants import T_RIGHT_WRIST_YAW_TO_LINK_END
from g1_hello_world.pinocchio_ik import RIGHT_ARM_JOINT_NAMES, RightArmPinocchioIK
from g1_hello_world.robot_model import RobotModelWrapper
from g1_hello_world.visualization import ViserVisualizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Kinematic simulation: move G1 pelvis forward, then solve right-arm IK "
            "with Pinocchio and visualize the sequence in Viser."
        )
    )
    parser.add_argument(
        "--forward-distance",
        type=float,
        default=1.5,
        help="Pelvis forward motion along world +X [m]",
    )
    parser.add_argument(
        "--walk-duration",
        type=float,
        default=5.0,
        help="Duration of the forward motion animation [s]",
    )
    parser.add_argument(
        "--arm-duration",
        type=float,
        default=1.8,
        help="Duration of the right-arm reach animation [s]",
    )
    parser.add_argument(
        "--hold-after",
        type=float,
        default=0.5,
        help="Pause between forward motion and arm reach [s]",
    )
    parser.add_argument("--x", type=float, default=0.30, help="Right EE target x in pelvis frame [m]")
    parser.add_argument("--y", type=float, default=-0.25, help="Right EE target y in pelvis frame [m]")
    parser.add_argument("--z", type=float, default=0.18, help="Right EE target z in pelvis frame [m]")
    parser.add_argument("--rx", type=float, default=0.0, help="Right EE target roll in pelvis frame [rad]")
    parser.add_argument("--ry", type=float, default=0.0, help="Right EE target pitch in pelvis frame [rad]")
    parser.add_argument("--rz", type=float, default=0.0, help="Right EE target yaw in pelvis frame [rad]")
    parser.add_argument(
        "--max-iters",
        type=int,
        default=200,
        help="Maximum Pinocchio IK iterations",
    )
    return parser.parse_args()


def _pin_configuration_to_mujoco_qpos(
    robot_model: RobotModelWrapper,
    solver: RightArmPinocchioIK,
    q_pin: np.ndarray,
) -> np.ndarray:
    qpos = np.zeros(robot_model.mj_model.nq, dtype=np.float64)
    qpos[3] = 1.0
    for idx, joint_name in enumerate(robot_model.joint_names):
        joint_id = solver.model.getJointId(joint_name)
        qpos[7 + idx] = q_pin[solver.model.joints[joint_id].idx_q]
    return qpos


def _world_target_from_pelvis_target(
    pelvis_pos: np.ndarray,
    pelvis_quat_wxyz: np.ndarray,
    target_xyz_pelvis: np.ndarray,
    target_rpy_pelvis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pelvis_rot = sRot.from_quat(
        [pelvis_quat_wxyz[1], pelvis_quat_wxyz[2], pelvis_quat_wxyz[3], pelvis_quat_wxyz[0]]
    )
    target_rot_pelvis = sRot.from_euler("xyz", target_rpy_pelvis)
    world_target_pos = pelvis_pos + pelvis_rot.apply(target_xyz_pelvis)
    world_target_quat_xyzw = (pelvis_rot * target_rot_pelvis).as_quat()
    world_target_quat_wxyz = np.array(
        [
            world_target_quat_xyzw[3],
            world_target_quat_xyzw[0],
            world_target_quat_xyzw[1],
            world_target_quat_xyzw[2],
        ],
        dtype=np.float64,
    )
    return world_target_pos, world_target_quat_wxyz


def _status_markdown(
    *,
    phase: str,
    forward_distance: float,
    target_xyz_pelvis: np.ndarray,
    target_rpy_pelvis: np.ndarray,
    solved_xyz_pelvis: np.ndarray,
    solved_rpy_pelvis: np.ndarray,
    success: bool,
    iterations: int,
    pos_err: float,
    rot_err: float,
) -> str:
    return (
        "# Walk Then Reach\n"
        f"- phase: `{phase}`\n"
        f"- pelvis forward distance: `{forward_distance:.3f}` m\n"
        f"- IK success: `{success}`\n"
        f"- IK iterations: `{iterations}`\n"
        f"- position error: `{pos_err:.6f}` m\n"
        f"- orientation error: `{rot_err:.6f}` rad\n\n"
        "## Right EE Target In Pelvis Frame\n"
        f"- xyz: `[{target_xyz_pelvis[0]:.4f}, {target_xyz_pelvis[1]:.4f}, {target_xyz_pelvis[2]:.4f}]`\n"
        f"- rpy xyz: `[{target_rpy_pelvis[0]:.4f}, {target_rpy_pelvis[1]:.4f}, {target_rpy_pelvis[2]:.4f}]`\n\n"
        "## Solved Right EE In Pelvis Frame\n"
        f"- xyz: `[{solved_xyz_pelvis[0]:.4f}, {solved_xyz_pelvis[1]:.4f}, {solved_xyz_pelvis[2]:.4f}]`\n"
        f"- rpy xyz: `[{solved_rpy_pelvis[0]:.4f}, {solved_rpy_pelvis[1]:.4f}, {solved_rpy_pelvis[2]:.4f}]`\n"
    )


def _animate_qpos(
    robot_model: RobotModelWrapper,
    qpos_start: np.ndarray,
    qpos_end: np.ndarray,
    duration: float,
    *,
    fps: float = 30.0,
) -> None:
    steps = max(1, int(duration * fps))
    for alpha in np.linspace(0.0, 1.0, steps):
        qpos = (1.0 - alpha) * qpos_start + alpha * qpos_end
        qpos[3:7] = qpos_end[3:7]
        robot_model.update(qpos)
        time.sleep(duration / steps)


def main() -> None:
    args = _parse_args()

    project_root = Path(__file__).resolve().parent
    urdf_path = project_root / "robot_model" / "g1_29dof_rev_1_0.urdf"
    xml_path = project_root / "robot_model" / "g1_29dof_rev_1_0.xml"

    solver = RightArmPinocchioIK(urdf_path)
    q_pin_init = solver.neutral_configuration()
    q_pin_goal, ik_result = solver.solve_in_base_frame(
        np.array([args.x, args.y, args.z], dtype=np.float64),
        np.array([args.rx, args.ry, args.rz], dtype=np.float64),
        q_init=q_pin_init,
        max_iters=args.max_iters,
    )
    ee_pose_pelvis = solver.get_frame_pose_in_base(q_pin_goal, "right_rubber_hand")
    ee_rpy_pelvis = sRot.from_matrix(ee_pose_pelvis.rotation).as_euler("xyz")

    print(
        "IK result:",
        f"success={ik_result.success}",
        f"iterations={ik_result.iterations}",
        f"pos_err={ik_result.position_error_norm:.6f}",
        f"rot_err={ik_result.orientation_error_norm:.6f}",
    )
    print("Right-arm joint targets [rad]:")
    joint_map = solver.configuration_to_joint_map(q_pin_goal)
    for joint_name in RIGHT_ARM_JOINT_NAMES:
        print(f"  {joint_name}: {joint_map[joint_name]: .6f}")

    robot_model = RobotModelWrapper(xml_path)
    qpos_start = _pin_configuration_to_mujoco_qpos(robot_model, solver, q_pin_init)
    qpos_walk_end = qpos_start.copy()
    qpos_walk_end[0] = args.forward_distance
    qpos_arm_end = _pin_configuration_to_mujoco_qpos(robot_model, solver, q_pin_goal)
    qpos_arm_end[0] = args.forward_distance

    robot_model.update(qpos_start)

    visualizer = ViserVisualizer()
    visualizer.add_robot(
        robot_model,
        body_names=["pelvis", "waist_.*", "torso_link", "right_.*", "left_.*"],
    )
    visualizer.add_body_frame(
        "/frames/pelvis",
        robot_model,
        body_name="pelvis",
        axes_length=0.10,
        axes_radius=0.005,
    )
    visualizer.add_body_frame(
        "/frames/right_ee",
        robot_model,
        body_name="right_wrist_yaw_link",
        body_from_frame=T_RIGHT_WRIST_YAW_TO_LINK_END,
        axes_length=0.12,
        axes_radius=0.006,
    )
    target_frame = visualizer.server.scene.add_frame(
        "/frames/right_ee_target_world",
        axes_length=0.12,
        axes_radius=0.006,
        origin_radius=0.014,
        origin_color=(0, 200, 255),
    )
    path_handle = visualizer.server.scene.add_line_segments(
        "/debug/pelvis_path",
        points=np.array(
            [
                [[0.0, 0.0, qpos_start[2]], [args.forward_distance, 0.0, qpos_start[2]]],
            ],
            dtype=np.float32,
        ),
        colors=np.array([[[255, 200, 0], [255, 200, 0]]], dtype=np.uint8),
        line_width=4.0,
    )
    del path_handle

    visualizer.run_async(freq=30.0)
    gui = visualizer.server.gui
    gui.add_markdown(
        "这是一段**运动学仿真**：先把 `pelvis` 沿世界 `+X` 平移 1.5 米，"
        "然后再让右臂到达一个相对于 `pelvis` 的目标位姿。"
    )
    status = gui.add_markdown("Preparing sequence...")
    replay = gui.add_button("Replay Sequence")

    target_xyz_pelvis = np.array([args.x, args.y, args.z], dtype=np.float64)
    target_rpy_pelvis = np.array([args.rx, args.ry, args.rz], dtype=np.float64)

    def _refresh_target_frame() -> None:
        pelvis_pos = robot_model.mj_data.qpos[:3].copy()
        pelvis_quat = robot_model.mj_data.qpos[3:7].copy()
        world_target_pos, world_target_quat = _world_target_from_pelvis_target(
            pelvis_pos,
            pelvis_quat,
            target_xyz_pelvis,
            target_rpy_pelvis,
        )
        target_frame.position = world_target_pos
        target_frame.wxyz = world_target_quat

    def _run_sequence() -> None:
        robot_model.update(qpos_start)
        _refresh_target_frame()
        status.content = _status_markdown(
            phase="walking forward",
            forward_distance=args.forward_distance,
            target_xyz_pelvis=target_xyz_pelvis,
            target_rpy_pelvis=target_rpy_pelvis,
            solved_xyz_pelvis=ee_pose_pelvis.translation,
            solved_rpy_pelvis=ee_rpy_pelvis,
            success=ik_result.success,
            iterations=ik_result.iterations,
            pos_err=ik_result.position_error_norm,
            rot_err=ik_result.orientation_error_norm,
        )
        _animate_qpos(robot_model, qpos_start, qpos_walk_end, args.walk_duration)
        _refresh_target_frame()
        time.sleep(args.hold_after)

        status.content = _status_markdown(
            phase="reaching with right arm",
            forward_distance=args.forward_distance,
            target_xyz_pelvis=target_xyz_pelvis,
            target_rpy_pelvis=target_rpy_pelvis,
            solved_xyz_pelvis=ee_pose_pelvis.translation,
            solved_rpy_pelvis=ee_rpy_pelvis,
            success=ik_result.success,
            iterations=ik_result.iterations,
            pos_err=ik_result.position_error_norm,
            rot_err=ik_result.orientation_error_norm,
        )
        _animate_qpos(robot_model, qpos_walk_end, qpos_arm_end, args.arm_duration)
        _refresh_target_frame()

        status.content = _status_markdown(
            phase="done",
            forward_distance=args.forward_distance,
            target_xyz_pelvis=target_xyz_pelvis,
            target_rpy_pelvis=target_rpy_pelvis,
            solved_xyz_pelvis=ee_pose_pelvis.translation,
            solved_rpy_pelvis=ee_rpy_pelvis,
            success=ik_result.success,
            iterations=ik_result.iterations,
            pos_err=ik_result.position_error_norm,
            rot_err=ik_result.orientation_error_norm,
        )

    @replay.on_click
    def _(_event: object) -> None:
        _run_sequence()

    _refresh_target_frame()
    _run_sequence()

    print(f"Viser running at http://localhost:{visualizer.server.get_port()}")
    print("Sequence: pelvis moves forward 1.5 m, then right arm reaches target in pelvis frame.")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        visualizer.stop_async()


if __name__ == "__main__":
    main()
