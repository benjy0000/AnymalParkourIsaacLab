
from isaaclab.utils import configclass
import numpy as np

from isaaclab.terrains import HfTerrainBaseCfg
from . import hf_terrains


@configclass
class HfTerrainCfg(HfTerrainBaseCfg):
    
    pad_width: float = 0.1
    pad_height: float = 0.0
    num_goals: int = 8
    horizontal_scale: float = 0.075
    vertical_scale: float = 0.005
    slope_threshold: float = 0.75


@configclass
class HfBarkourTerrainCfg(HfTerrainCfg):

    function = hf_terrains.barkour_terrain
    
    platform_len: float = 2
    platform_height: float = 0.6
    weave_poles_y_range: tuple[float, float] = (0.35, 0.45)
    A_frame_height_range: tuple[float, float] = (1.13, 1.17)
    gap_size: float = 0.72
    box_height: float = 0.6

    # Friction parameters
    height = [0.02, 0.02]


@configclass
class HfWavesTerrainCfg(HfTerrainCfg):

    function = hf_terrains.waves_terrain

    wave_len_range: tuple[float, float] = (3.0, 7.0)
    wave_height_range: tuple[float, float] = (0.3, 0.8)
    platform_len: float = 3.0

    slope_threshold: float = 6.0  # Waves are steep, so increase threshold

    # Friction parameters
    height = [0.02, 0.02]


@configclass
class HfValleyTerrainCfg(HfTerrainCfg):

    function = hf_terrains.valley_terrain

    platform_len: float = 3.0
    valley_width_range: tuple[float, float] = (1.0, 1.5)
    bend_x_range: tuple[float, float] = (0.35, 0.45)
    bend_y_range: tuple[float, float] = (0.8, 1.2)
    depth_range: tuple[float, float] = (0.10, 0.8)
    exit_segment_len: float = 0.8

    # Friction/roughness (kept for compatibility, not used to add roughness here)
    height = [0.02, 0.02]


@configclass
class HfSlopeUpTerrainCfg(HfTerrainCfg):

    function = hf_terrains.slope_up_terrain

    # Geometry
    platform_len: float = 2.0
    platform_height: float = 0.0
    slope_len: float = 2.0
    slope_width: float = 1.0
    height_range: tuple[float, float] = (0.35, 1.15)
    pit_depth: float = 1.0

    # Friction parameters (used by add_roughness)
    height = [0.02, 0.02]


@configclass
class HfSlopeDownTerrainCfg(HfTerrainCfg):

    function = hf_terrains.slope_down_terrain

    # Geometry
    platform_len: float = 2.0
    platform_height: float = 0.0
    slope_len: float = 2.0
    slope_width: float = 1.0
    height_range: tuple[float, float] = (-0.35, -1.15)
    pit_depth: float = 3.0

    # Friction parameters (used by add_roughness)
    height = [0.02, 0.02]


@configclass
class HfDiscreteObstaclesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a discrete obstacles height field terrain."""

    function = hf_terrains.discrete_obstacles_terrain

    obstacle_height_mode: str = "choice"
    obstacle_width_range: tuple[float, float] = (1.0, 2.0)
    """The minimum and maximum width of the obstacles (in m)."""
    obstacle_height_range: tuple[float, float] = (0.05, 0.30)
    """The minimum and maximum height of the obstacles (in m)."""
    num_obstacles: int = 20
    """The number of obstacles to generate."""
    platform_width: float = 2.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@configclass
class HfPyramidSlopedTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a pyramid sloped height field terrain."""

    function = hf_terrains.pyramid_sloped_terrain

    slope_range: tuple[float, float] = (np.pi / 36, np.pi / 4)
    """The slope of the terrain (in radians)."""
    platform_width: float = 2.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    inverted: bool = False
    """Whether the pyramid is inverted. Defaults to False.

    If True, the terrain is inverted such that the platform is at the bottom and the slopes are upwards.
    """

    slope_threshold: float = 2.0  # Slopes are steep, so increase threshold


@configclass
class HfInvertedPyramidSlopedTerrainCfg(HfPyramidSlopedTerrainCfg):
    """Configuration for an inverted pyramid sloped height field terrain.

    Note:
        This is a subclass of :class:`HfPyramidSlopedTerrainCfg` with :obj:`inverted` set to True.
        We make it as a separate class to make it easier to distinguish between the two and match
        the naming convention of the other terrains.
    """

    inverted: bool = True