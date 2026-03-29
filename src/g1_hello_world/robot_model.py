from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as sRot

from .constants import MJ_MESH


def geom_mesh_trimesh(model: mujoco.MjModel, gid: int) -> trimesh.Trimesh:
    mesh_id = int(model.geom_dataid[gid])
    v0 = int(model.mesh_vertadr[mesh_id])
    nv = int(model.mesh_vertnum[mesh_id])
    f0 = int(model.mesh_faceadr[mesh_id])
    nf = int(model.mesh_facenum[mesh_id])
    verts = model.mesh_vert[v0 : v0 + nv].copy() * model.mesh_scale[mesh_id]
    rotation_flat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation_flat, model.geom_quat[gid])
    rotation = rotation_flat.reshape(3, 3)
    verts = verts @ rotation.T + model.geom_pos[gid]
    faces = np.asarray(model.mesh_face[f0 : f0 + nf])
    return trimesh.Trimesh(vertices=verts, faces=faces)


class RobotModelWrapper:
    def __init__(self, xml_path: str | Path) -> None:
        self.mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.mj_data = mujoco.MjData(self.mj_model)

        self.joint_names = [
            mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            for joint_id in range(self.mj_model.njnt)
        ]
        self.body_ids = list(range(1, self.mj_model.nbody))
        self.body_names = [
            mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            or f"body_{body_id}"
            for body_id in self.body_ids
        ]

        meshes_by_body: dict[int, list[trimesh.Trimesh]] = {}
        for geom_id in range(self.mj_model.ngeom):
            if self.mj_model.geom_type[geom_id] != MJ_MESH:
                continue
            body_id = int(self.mj_model.geom_bodyid[geom_id])
            meshes_by_body.setdefault(body_id, []).append(
                geom_mesh_trimesh(self.mj_model, geom_id)
            )

        self.body_meshes: list[trimesh.Trimesh | None] = []
        for body_id in self.body_ids:
            parts = meshes_by_body.get(body_id)
            if not parts:
                self.body_meshes.append(None)
            elif len(parts) > 1:
                self.body_meshes.append(trimesh.util.concatenate(parts))
            else:
                self.body_meshes.append(parts[0])

        self._site_ids: dict[str, int] = {}

    def update(self, qpos: np.ndarray) -> None:
        self.mj_data.qpos[:] = qpos
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def get_site_pose(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        """World pose of the site after the last `update()`: position (3,) and wxyz quaternion (4)."""
        site_id = self._get_site_id(site_name)
        pos = np.asarray(self.mj_data.site_xpos[site_id], dtype=np.float64).copy()
        mat = self.mj_data.site_xmat[site_id].reshape(3, 3)
        quat_wxyz = sRot.from_matrix(mat).as_quat(scalar_first=True)
        return pos, quat_wxyz

    def get_site_frame(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Position (3,) and world-from-site rotation (3,3), same convention as `site_xmat`."""
        site_id = self._get_site_id(site_name)
        pos = np.asarray(self.mj_data.site_xpos[site_id], dtype=np.float64).copy()
        rotation = (
            np.asarray(self.mj_data.site_xmat[site_id], dtype=np.float64)
            .reshape(3, 3)
            .copy()
        )
        return pos, rotation

    def _get_site_id(self, site_name: str) -> int:
        if site_name not in self._site_ids:
            site_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name
            )
            if site_id < 0:
                raise ValueError(f"unknown site name: {site_name!r}")
            self._site_ids[site_name] = site_id
        return self._site_ids[site_name]

