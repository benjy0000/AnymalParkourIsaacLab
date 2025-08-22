
from isaaclab.utils import configclass

from isaaclab.terrains import HfTerrainBaseCfg
from . import hf_terrains

@configclass
class HfBarkourTerrainCfg(HfTerrainBaseCfg):

    function = hf_terrains.barkour_terrain
    
    platform_len: float = 2
    platform_height: float = 0.6
    weave_poles_y_range: tuple[float, float] = (0.35, 0.45)
    A_frame_height_range: tuple[float, float] = (1.13, 1.17)
    gap_size: float = 0.72
    box_height: float = 0.6
    pad_width: float = 0.1
    pad_height: float = 0.0

    slope_threshold: float = 0.75

