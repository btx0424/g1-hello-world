from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as sRot

from .constants import MJ_MESH
from collections import defaultdict


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
        self.body_adrs = list(range(1, self.mj_model.nbody)) # exclude world body
        self.body_names = [
            mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_addr)
            or f"body_{body_addr}"
            for body_addr in self.body_adrs
        ]

        meshes_by_body: dict[str, list[trimesh.Trimesh]] = defaultdict(list)
        for geom_id in range(self.mj_model.ngeom):
            geom = self.mj_model.geom(geom_id)
            if geom.type != MJ_MESH:
                continue
            # Skip collision duplicates (contype/conaffinity > 0); keep visual-only meshes.
            if (
                int(self.mj_model.geom_contype[geom_id]) != 0
                or int(self.mj_model.geom_conaffinity[geom_id]) != 0
            ):
                continue
            body_name = self.mj_model.body(int(self.mj_model.geom_bodyid[geom_id])).name
            meshes_by_body[body_name].append(
                geom_mesh_trimesh(self.mj_model, geom_id)
            )

        self.body_meshes: dict[str, trimesh.Trimesh | None] = {}
        for body_name in self.body_names:
            parts = meshes_by_body.get(body_name)
            if not parts:
                continue
            body_mesh = trimesh.util.concatenate(parts)
            body_mesh.merge_vertices()
            body_mesh = body_mesh.simplify_quadric_decimation(0.9)
            self.body_meshes[body_name] = body_mesh

        self._site_ids: dict[str, int] = {}
        self._body_ids: dict[str, int] = {}

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

    def get_body_frame(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Position (3,) and world-from-body rotation (3,3), same convention as `xmat`."""
        body_id = self._get_body_id(body_name)
        pos = np.asarray(self.mj_data.xpos[body_id], dtype=np.float64).copy()
        rotation = (
            np.asarray(self.mj_data.xmat[body_id], dtype=np.float64)
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

    def _get_body_id(self, body_name: str) -> int:
        if body_name not in self._body_ids:
            body_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id < 0:
                raise ValueError(f"unknown body name: {body_name!r}")
            self._body_ids[body_name] = body_id
        return self._body_ids[body_name]

