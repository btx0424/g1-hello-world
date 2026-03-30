import numpy as np
import trimesh

from g1_hello_world.constants import R_SITE_FROM_OPENCV
from g1_hello_world.realsense_device import RealSenseDeviceManager
from g1_hello_world.robot_model import RobotModelWrapper


class GroundPlaneEstimator:
    """
    Estimates a dominant plane from aligned RealSense depth (RGB-D), assuming the
    ground is visible in the lower image band. The fitted normal is flipped so it
    aligns with world +Z (MuJoCo / robot world up).
    """

    def __init__(self, *, world_up: np.ndarray | None = None) -> None:
        u = np.asarray(
            world_up if world_up is not None else (0.0, 0.0, 1.0),
            dtype=np.float64,
        )
        self._world_up = u / np.linalg.norm(u)

    def fit_and_visualize(
        self,
        *,
        scene,
        realsense: RealSenseDeviceManager,
        K: np.ndarray,
        robot_model: RobotModelWrapper,
        image_width: int,
        image_height: int,
        site_name: str = "d435",
        half_size: float = 2.5,
        min_points: int = 250,
        max_attempts: int = 45,
    ) -> None:
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        ds = float(realsense.depth_scale)
        pos_link, world_from_link = robot_model.get_site_frame(site_name)
        world_from_cv = world_from_link @ R_SITE_FROM_OPENCV

        bottom_v0 = int(image_height * 0.62)
        stride = 4
        pts: list[np.ndarray] = []

        for _ in range(max_attempts):
            _, depth_u16 = realsense.read_aligned_rgb_depth()

            pts.clear()
            for v in range(bottom_v0, image_height, stride):
                for u in range(0, image_width, stride):
                    raw = int(depth_u16[v, u])
                    if raw <= 0:
                        continue
                    d = float(raw) * ds
                    if d <= 0.0 or d > 6.0 or not np.isfinite(d):
                        continue
                    x = (float(u) - cx) * d / fx
                    y = (float(v) - cy) * d / fy
                    z = d
                    p_cv = np.array([x, y, z], dtype=np.float64)
                    p_w = pos_link + world_from_cv @ p_cv
                    if np.all(np.isfinite(p_w)):
                        pts.append(p_w)
            if len(pts) >= min_points:
                break

        if len(pts) < min_points:
            print(
                f"GroundPlaneEstimator: too few depth points ({len(pts)} < {min_points}); "
                "skipping plane fit."
            )
            return

        P = np.stack(pts, axis=0)
        centroid = P.mean(axis=0)
        _, _, vh = np.linalg.svd(P - centroid, full_matrices=False)
        normal = vh[-1].astype(np.float64)
        if float(np.dot(normal, self._world_up)) < 0.0:
            normal = -normal
        normal /= np.linalg.norm(normal)

        # Plane n·x + d = 0 with n the upward-pointing normal; d = -n·centroid
        d = float(-np.dot(normal, centroid))
        print(
            "Ground plane (world): "
            f"n=[{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}], "
            f"d={d:.3f} ({len(pts)} points)"
        )

        mesh = _ground_plane_quad_trimesh(centroid, normal, half_size=half_size)
        scene.add_mesh_trimesh(
            "/ground_plane",
            mesh,
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
        )


def _ground_plane_quad_trimesh(
    point_on_plane: np.ndarray,
    normal: np.ndarray,
    *,
    half_size: float,
) -> trimesh.Trimesh:
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    aux = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(normal, aux))) > 0.92:
        aux = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    t1 = np.cross(normal, aux)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    corners = []
    for sx, sy in (-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0):
        corners.append(
            point_on_plane + half_size * (sx * t1 + sy * t2),
        )
    verts = np.asarray(corners, dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    rgba = np.tile(
        np.array([[90, 140, 95, 130]], dtype=np.uint8),
        (mesh.faces.shape[0], 1),
    )
    mesh.visual.face_colors = rgba
    return mesh