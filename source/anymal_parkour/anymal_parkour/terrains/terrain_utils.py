from __future__ import annotations

import numpy as np
import scipy.spatial.transform as tf
import trimesh
from typing import TYPE_CHECKING

from functools import wraps

from isaaclab.terrains.trimesh.utils import *  # noqa: F401, F403

from . import terrain_utils

if TYPE_CHECKING:
    from .sub_terrain_cfg import ParkourSubTerrainBaseCfg


def allow_surface_roughness(func):
    """Decorator to add random noise to the y-vertices of generated meshes.
    It overwrites the box function as `rough_box` by default. To prevent this
    set `cfg.surface_roughness` to `False`.

    This decorator assumes the wrapped function returns a tuple of (meshes_list, origin).
    """
    @wraps(func)
    def wrapper(difficulty: float, cfg: ParkourSubTerrainBaseCfg, *args, **kwargs):
        # Add surface roughness parameters to the function call

        original_box = terrain_utils.box

        def new_box(dim, transform=None):
            
            if cfg.surface_roughness:
                amp = (cfg.noise_range[0] + difficulty
                       * (cfg.noise_range[1] - cfg.noise_range[0]))
                return rough_box(dim,
                                 transform=transform, 
                                 res=cfg.resolution,
                                 v_step=cfg.v_step,
                                 noise_amplitude=amp)
            return original_box(dim, transform=transform)

        # Replace the box function with the rough_box function
        terrain_utils.box = new_box

        try:
            result = func(difficulty, cfg, *args, **kwargs)

        finally:
            # Restore the original box function
            terrain_utils.box = original_box

        return result
    return wrapper


def box(dim, transform=None):
    """Simple box function for efficient mesh generation."""
    return trimesh.creation.box(extents=dim, transform=transform)


def rough_box(extents=(1, 1, 1), transform=None, res=0.05, v_step=0.005, noise_amplitude=0.0):
    """
    Create a box mesh with vertical surface roughness.

    Parameters
    ----------
    extents : (3,) float
        Size of the box in X, Y, Z.
    transform : (4, 4) float, optional
        Homogeneous transform applied to the box.
    res : float
        Horizontal spacing of points.
    v_step : float
        Vertical discretization of noise.
    noise_amplitude : float
        Standard deviation of random vertical displacement (Z-axis only).
    """

    ex, ey, ez = np.array(extents) / 2.0  # half extents
    verts_all = []
    faces_all = []
    vert_offset = 0

    def add_face(origin, u_dir, v_dir, add_noise=False):
        nonlocal verts_all, faces_all, vert_offset
        u_len = np.linalg.norm(u_dir)
        v_len = np.linalg.norm(v_dir)
        u_dir /= u_len
        v_dir /= v_len

        # Decide resolution from spacing
        if add_noise:
            nu = max(2, int(np.ceil(u_len / (res if abs(u_dir[0]) > 0 else res))) + 1)
            nv = max(2, int(np.ceil(v_len / (res if abs(v_dir[1]) > 0 else res))) + 1)
        else:
            nu = 2
            nv = 2

        u = np.linspace(0, u_len, nu)
        v = np.linspace(0, v_len, nv)
        U, V = np.meshgrid(u, v)
        points = origin + np.outer(U.flatten(), u_dir) + np.outer(V.flatten(), v_dir)

        # Add vertical (Z-axis) noise if requested
        if add_noise:
            min_height = 0
            max_height = noise_amplitude * 2  # Always add so that horizontal faces aren't revealed
            heights_range = np.arange(min_height, max_height + v_step, v_step)
            points[:, 2] += np.random.choice(heights_range, points.shape[0])

        verts_all.append(points)

        # Faces
        for i in range(nu - 1):
            for j in range(nv - 1):
                v0 = vert_offset + j * nu + i
                v1 = vert_offset + j * nu + (i + 1)
                v2 = vert_offset + (j + 1) * nu + (i + 1)
                v3 = vert_offset + (j + 1) * nu + i
                faces_all.append([v0, v1, v2])
                faces_all.append([v0, v2, v3])

        vert_offset += points.shape[0]

    # Build faces (bottom, top, left, right, front, back)
    add_face(np.array([-ex, -ey, -ez]), np.array([2*ex, 0, 0]), np.array([0, 2*ey, 0]), add_noise=False) # bottom
    add_face(np.array([-ex, -ey,  ez]), np.array([2*ex, 0, 0]), np.array([0, 2*ey, 0]), add_noise=True) # top
    add_face(np.array([-ex, -ey, -ez]), np.array([0, 0, 2*ez]), np.array([0, 2*ey, 0]), add_noise=False)  # left
    add_face(np.array([ ex, -ey, -ez]), np.array([0, 0, 2*ez]), np.array([0, 2*ey, 0]), add_noise=False)  # right
    add_face(np.array([-ex, -ey, -ez]), np.array([2*ex, 0, 0]), np.array([0, 0, 2*ez]), add_noise=False)  # front
    add_face(np.array([-ex,  ey, -ez]), np.array([2*ex, 0, 0]), np.array([0, 0, 2*ez]), add_noise=False)  # back

    # Combine vertices and faces
    vertices = np.vstack(verts_all)
    faces = np.array(faces_all)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.merge_vertices()

    # Apply transform if given
    if transform is not None:
        mesh.apply_transform(transform)

    return mesh