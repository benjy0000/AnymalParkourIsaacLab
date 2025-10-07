"""Different terrain configurations for parkour tasks."""

from dataclasses import MISSING
from typing import Literal

import isaaclab.terrains.trimesh.utils as mesh_utils_terrains
from isaaclab.utils import configclass

from . import parkour_mesh_terrains
from .sub_terrain_cfg import ParkourSubTerrainCfg
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg


@configclass
class MeshLargeStepsTerrainCfg(ParkourSubTerrainCfg):
    """Configuration for a terrain with large steps."""

    function = parkour_mesh_terrains.large_steps_terrain

    step_height_range: tuple[float, float] = MISSING
    step_length_range: tuple[float, float] = MISSING
    step_width_range: tuple[float, float] = MISSING
    step_mismatch_range: tuple[float, float] = MISSING


@configclass
class MeshBoxesTerrainCfg(ParkourSubTerrainCfg):
    """Configuration for a terrain with boxes."""

    function = parkour_mesh_terrains.boxes_terrain

    box_height_range: tuple[float, float] = MISSING
    box_length: float = MISSING
    box_width: float = MISSING
    box_x_offset_range: tuple[float, float] = MISSING


@configclass
class MeshStairsTerrainCfg(ParkourSubTerrainCfg):
    """Configuration for a terrain with stairs."""

    function = parkour_mesh_terrains.stairs_terrain

    num_steps: int = 6
    tread_range: tuple[float, float] = MISSING
    riser_range: tuple[float, float] = MISSING


@configclass
class MeshGapsTerrainCfg(ParkourSubTerrainCfg):
    """Configuration for a terrain with gaps."""

    function = parkour_mesh_terrains.gaps_terrain

    gap_length_range: tuple[float, float] = MISSING
    stone_x_offset_range: tuple[float, float] = MISSING
    stone_length: float = 1.2


@configclass
class MeshInclinedBoxesTerrainCfg(ParkourSubTerrainCfg):
    """Configuration for a terrain with inclined boxes."""

    function = parkour_mesh_terrains.inclined_boxes_terrain

    pit_depth: tuple[float, float] = MISSING


@configclass
class MeshWeavePoleTerrainCfg(ParkourSubTerrainCfg):
    """Configuration for a terrain with weave poles."""

    function = parkour_mesh_terrains.weave_pole_terrain

    pole_radius: float = MISSING,
    robot_x_clearance_range: tuple[float, float] = (0.2, 0.4),
    pole_x_range: tuple[float, float] = MISSING
    pole_x_noise: float = 0.0
    pole_y_range: tuple[float, float] = MISSING
    pole_height_range: tuple[float, float] = MISSING


@configclass
class MeshFlatTerrainCfg(ParkourSubTerrainCfg):
    """Configuration for a flat terrain."""

    function = parkour_mesh_terrains.flat_terrain

    x_range: float = 0.5


@configclass
class MeshPyramidStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a pyramid stair mesh terrain."""

    function = parkour_mesh_terrains.pyramid_stairs_terrain

    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0.

    The border is a flat terrain with the same height as the terrain.
    """
    step_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the steps (in m)."""
    step_width: float = MISSING
    """The width of the steps (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    holes: bool = False
    """If True, the terrain will have holes in the steps. Defaults to False.

    If :obj:`holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.
    """


@configclass
class MeshInvertedPyramidStairsTerrainCfg(MeshPyramidStairsTerrainCfg):
    """Configuration for an inverted pyramid stair mesh terrain.

    Note:
        This is the same as :class:`MeshPyramidStairsTerrainCfg` except that the steps are inverted.
    """

    function = parkour_mesh_terrains.inverted_pyramid_stairs_terrain
