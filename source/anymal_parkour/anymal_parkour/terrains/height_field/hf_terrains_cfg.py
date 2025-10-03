
from isaaclab.utils import configclass

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
    pad_width: float = 0.1
    pad_height: float = 0.0
    horizontal_scale: float = 0.075
    vertical_scale: float = 0.005

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
