"""Different terrain configurations for parkour tasks."""

from dataclasses import MISSING
from typing import Literal

import isaaclab.terrains.trimesh.utils as mesh_utils_terrains
from isaaclab.utils import configclass

from . import parkour_mesh_terrains
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg


@configclass
class MeshBoxTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with boxes (similar to a pyramid)."""

    function = parkour_mesh_terrains.box_terrain

    box_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the box (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    double_box: bool = False
    """If True, the pit contains two levels of stairs/boxes. Defaults to False."""