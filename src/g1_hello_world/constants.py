from __future__ import annotations

import numpy as np

MJ_MESH = 7  # mujoco.mjtGeom.mjGEOM_MESH

# MuJoCo `d435` site = ROS `camera_link` (+X forward, +Y left, +Z up). Viser / COLMAP / OpenCV
# camera: +X right, +Y down, +Z forward. Same static rotation as Intel RealSense ROS descriptions.
# p_link = R_SITE_FROM_OPENCV @ p_cv  =>  p_world = R_world_site @ R_SITE_FROM_OPENCV @ p_cv
R_SITE_FROM_OPENCV = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)

# Fixed transform from the `left_wrist_yaw_link` body origin to the end of the
# last wrist link, matching the `left_hand_palm_joint` mount location in the
# robot model.
T_LEFT_WRIST_YAW_TO_LINK_END = np.eye(4, dtype=np.float64)
T_LEFT_WRIST_YAW_TO_LINK_END[:3, 3] = np.array([0.0415, 0.0030, 0.0], dtype=np.float64)

T_RIGHT_WRIST_YAW_TO_LINK_END = np.eye(4, dtype=np.float64)
T_RIGHT_WRIST_YAW_TO_LINK_END[:3, 3] = np.array([0.0415, -0.0030, 0.0], dtype=np.float64)

# Placeholder homogeneous transform from the end of the last wrist link to the
# mounted RGB camera frame on the custom printed bracket. Replace this with the
# measured extrinsics once the printed connector is calibrated.
T_LEFT_WRIST_LINK_END_TO_RGB_PLACEHOLDER = np.eye(4, dtype=np.float64)

class G1JointIndex:
    # Left leg
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5

    # Right leg
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11

    WaistYaw = 12
    WaistRoll = 13        # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13           # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14       # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14           # NOTE: INVALID for g1 23dof/29dof with waist locked

    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20   # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21     # NOTE: INVALID for g1 23dof

    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28    # NOTE: INVALID for g1 23dof

    kNotUsedJoint = 29 # NOTE: Weight
