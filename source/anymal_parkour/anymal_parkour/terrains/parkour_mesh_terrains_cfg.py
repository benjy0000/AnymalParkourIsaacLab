"""Different terrain configurations for parkour tasks."""

from dataclasses import MISSING
from typing import Literal

import isaaclab.terrains.trimesh.utils as mesh_utils_terrains
from isaaclab.utils import configclass

from . import parkour_mesh_terrains
from .sub_terrain_cfg import ParkourSubTerrainCfg
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg

@configclass
class MeshBoxTerrainCfg(ParkourSubTerrainCfg):
    """Configuration for a terrain with boxes (similar to a pyramid)."""

    function = parkour_mesh_terrains.box_terrain

    box_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the box (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    double_box: bool = False
    """If True, the pit contains two levels of stairs/boxes. Defaults to False."""


@configclass
class MeshLargeStepsTerrainCfg(ParkourSubTerrainCfg):
    """Configuration for a terrain with large steps."""

    function = parkour_mesh_terrains.large_steps_terrain

    num_steps: int = MISSING
    step_height_range: tuple[float, float] = MISSING
    platform_length: float = 1.0
    step_length_range: tuple[float, float] = MISSING
    step_width_range: tuple[float, float] = MISSING
    step_mismatch_range: tuple[float, float] = MISSING


@configclass
class MeshBoxesTerrainCfg(ParkourSubTerrainCfg):
    """Configuration for a terrain with boxes."""

    function = parkour_mesh_terrains.boxes_terrain

    num_boxes: int = MISSING
    box_height_range: tuple[float, float] = MISSING
    platform_length: float = 1.0
    box_length: float = MISSING
    box_width: float = MISSING
    box_x_offset_range: tuple[float, float] = MISSING


@configclass
class MeshStairsTerrainCfg(ParkourSubTerrainCfg):
    """Configuration for a terrain with stairs."""

    function = parkour_mesh_terrains.stairs_terrain

    num_steps: int = MISSING
    tread_range: tuple[float, float] = MISSING
    riser_range: tuple[float, float] = MISSING
    platform_length: float = 1.0


@configclass
class MeshGapsTerrainCfg(ParkourSubTerrainCfg):
    """Configuration for a terrain with gaps."""

    function = parkour_mesh_terrains.gaps_terrain

    num_gaps: int = MISSING
    gap_length_range: tuple[float, float] = MISSING
    platform_length: float = 1.0
    stone_x_offset_range: tuple[float, float] = MISSING
    stone_length: float = 1.2


@configclass
class MeshInclinedBoxesTerrainCfg(ParkourSubTerrainCfg):
    """Configuration for a terrain with inclined boxes."""

    function = parkour_mesh_terrains.inclined_boxes_terrain

    num_stones: int = MISSING
    platform_length: float = 1.0
    pit_depth: tuple[float, float] = MISSING


@configclass
class MeshWeavePoleTerrainCfg(ParkourSubTerrainCfg):
    """Configuration for a terrain with weave poles."""

    function = parkour_mesh_terrains.weave_pole_terrain

    num_poles: int = MISSING
    pole_radius: float = MISSING
    platform_length: float = 1.0
    pole_x_range: tuple[float, float] = MISSING
    pole_x_noise: float = 0.0
    pole_y_range: tuple[float, float] = MISSING
    pole_height_range: tuple[float, float] = MISSING


@configclass
class MeshFlatTerrainCfg(ParkourSubTerrainCfg):
    """Configuration for a flat terrain."""

    function = parkour_mesh_terrains.flat_terrain

    platform_length: float = 1.0