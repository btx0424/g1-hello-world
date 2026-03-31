from __future__ import annotations

import argparse
import threading
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
        description="Solve G1 right-arm IK with Pinocchio and visualize the result in Viser."
    )
    parser.add_argument("--x", type=float, default=0.30, help="Target x in pelvis frame [m]")
    parser.add_argument("--y", type=float, default=-0.25, help="Target y in pelvis frame [m]")
    parser.add_argument("--z", type=float, default=0.18, help="Target z in pelvis frame [m]")
    parser.add_argument("--rx", type=float, default=0.0, help="Target roll in pelvis frame [rad]")
    parser.add_argument("--ry", type=float, default=0.0, help="Target pitch in pelvis frame [rad]")
    parser.add_argument("--rz", type=float, default=0.0, help="Target yaw in pelvis frame [rad]")
    parser.add_argument(
        "--duration",
        type=float,
        default=1.5,
        help="Animation duration from the neutral pose to the IK result [s]",
    )
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


def _format_pose_markdown(
    target_xyz: np.ndarray,
    target_rpy: np.ndarray,
    solved_xyz: np.ndarray,
    solved_rpy: np.ndarray,
    *,
    success: bool,
    iterations: int,
    pos_err: float,
    rot_err: float,
) -> str:
    status = "success" if success else "best effort"
    return (
        "## IK Status\n"
        f"- result: `{status}`\n"
        f"- iterations: `{iterations}`\n"
        f"- position error: `{pos_err:.6f}` m\n"
        f"- orientation error: `{rot_err:.6f}` rad\n\n"
        "## Target Pose\n"
        "- frame: `pelvis`\n"
        f"- xyz: `[{target_xyz[0]:.4f}, {target_xyz[1]:.4f}, {target_xyz[2]:.4f}]` m\n"
        f"- rpy xyz: `[{target_rpy[0]:.4f}, {target_rpy[1]:.4f}, {target_rpy[2]:.4f}]` rad\n\n"
        "## Solved EE Pose\n"
        "- end effector: `right_rubber_hand`\n"
        f"- xyz: `[{solved_xyz[0]:.4f}, {solved_xyz[1]:.4f}, {solved_xyz[2]:.4f}]` m\n"
        f"- rpy xyz: `[{solved_rpy[0]:.4f}, {solved_rpy[1]:.4f}, {solved_rpy[2]:.4f}]` rad\n"
    )


def main() -> None:
    args = _parse_args()

    project_root = Path(__file__).resolve().parent
    urdf_path = project_root / "robot_model" / "g1_29dof_rev_1_0.urdf"
    xml_path = project_root / "robot_model" / "g1_29dof_rev_1_0.xml"

    solver = RightArmPinocchioIK(urdf_path)
    q_init = solver.neutral_configuration()
    q_solution = q_init.copy()

    robot_model = RobotModelWrapper(xml_path)
    qpos_init = _pin_configuration_to_mujoco_qpos(robot_model, solver, q_init)
    robot_model.update(qpos_init)

    visualizer = ViserVisualizer()
    visualizer.add_robot(
        robot_model,
        body_names=["pelvis", "waist_.*", "torso_link", "right_.*"],
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
        "/frames/right_ee_target",
        axes_length=0.12,
        axes_radius=0.006,
        origin_radius=0.014,
        origin_color=(0, 200, 255),
    )
    visualizer.run_async(freq=30.0)

    gui = visualizer.server.gui
    gui.add_markdown(
        "# G1 Right Arm IK\n"
        "这些滑块定义的是 `right_rubber_hand` 相对于 `pelvis` 的目标位姿。"
    )
    with gui.add_folder("Target Pose"):
        slider_x = gui.add_slider("x [m]", min=-0.20, max=0.70, step=0.005, initial_value=args.x)
        slider_y = gui.add_slider("y [m]", min=-0.60, max=0.20, step=0.005, initial_value=args.y)
        slider_z = gui.add_slider("z [m]", min=-0.30, max=0.60, step=0.005, initial_value=args.z)
        slider_rx = gui.add_slider("rx [rad]", min=-np.pi, max=np.pi, step=0.01, initial_value=args.rx)
        slider_ry = gui.add_slider("ry [rad]", min=-np.pi, max=np.pi, step=0.01, initial_value=args.ry)
        slider_rz = gui.add_slider("rz [rad]", min=-np.pi, max=np.pi, step=0.01, initial_value=args.rz)
        reset_button = gui.add_button("Reset To CLI Args")

    status_markdown = gui.add_markdown("Solving...")

    state_lock = threading.Lock()
    current_q_solution = q_solution.copy()
    suppress_updates = False

    def _read_target() -> tuple[np.ndarray, np.ndarray]:
        target_xyz = np.array(
            [slider_x.value, slider_y.value, slider_z.value],
            dtype=np.float64,
        )
        target_rpy = np.array(
            [slider_rx.value, slider_ry.value, slider_rz.value],
            dtype=np.float64,
        )
        return target_xyz, target_rpy

    def _solve_and_render(*, animate: bool) -> None:
        nonlocal current_q_solution
        with state_lock:
            target_xyz, target_rpy = _read_target()
            q_seed = current_q_solution.copy()
            q_new, ik_result = solver.solve_in_base_frame(
                target_xyz,
                target_rpy,
                q_init=q_seed,
                max_iters=args.max_iters,
            )
            qpos_prev = _pin_configuration_to_mujoco_qpos(robot_model, solver, current_q_solution)
            qpos_new = _pin_configuration_to_mujoco_qpos(robot_model, solver, q_new)

            target_frame.position = target_xyz
            target_frame.wxyz = sRot.from_euler(
                "xyz",
                target_rpy,
            ).as_quat(scalar_first=True)

            if animate:
                steps = max(1, int(args.duration * 30.0))
                for alpha in np.linspace(0.0, 1.0, steps):
                    qpos = (1.0 - alpha) * qpos_prev + alpha * qpos_new
                    qpos[3:7] = qpos_new[3:7]
                    robot_model.update(qpos)
                    time.sleep(args.duration / steps)
            else:
                robot_model.update(qpos_new)

            current_q_solution = q_new
            ee_pose = solver.get_frame_pose_in_base(q_new, "right_rubber_hand")
            ee_rpy = sRot.from_matrix(ee_pose.rotation).as_euler("xyz")
            status_markdown.content = _format_pose_markdown(
                target_xyz,
                target_rpy,
                ee_pose.translation,
                ee_rpy,
                success=ik_result.success,
                iterations=ik_result.iterations,
                pos_err=ik_result.position_error_norm,
                rot_err=ik_result.orientation_error_norm,
            )

            print(
                "Updated target pose:",
                f"xyz={target_xyz.round(4).tolist()}",
                f"rpy={target_rpy.round(4).tolist()}",
                f"success={ik_result.success}",
                f"pos_err={ik_result.position_error_norm:.6f}",
                f"rot_err={ik_result.orientation_error_norm:.6f}",
            )
            joint_map = solver.configuration_to_joint_map(q_new)
            print("Right-arm joint targets [rad]:")
            for joint_name in RIGHT_ARM_JOINT_NAMES:
                print(f"  {joint_name}: {joint_map[joint_name]: .6f}")

    def _on_slider_update(_event: object) -> None:
        if suppress_updates:
            return
        _solve_and_render(animate=False)

    for handle in (slider_x, slider_y, slider_z, slider_rx, slider_ry, slider_rz):
        handle.on_update(_on_slider_update)

    @reset_button.on_click
    def _(_event: object) -> None:
        nonlocal suppress_updates
        suppress_updates = True
        slider_x.value = args.x
        slider_y.value = args.y
        slider_z.value = args.z
        slider_rx.value = args.rx
        slider_ry.value = args.ry
        slider_rz.value = args.rz
        suppress_updates = False
        _solve_and_render(animate=True)

    _solve_and_render(animate=True)

    print(f"Viser running at http://localhost:{visualizer.server.get_port()}")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        visualizer.stop_async()


if __name__ == "__main__":
    main()
