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

