
"""Configuration for parkour sub-terrain."""

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

@configclass
class ParkourSubTerrainCfg(SubTerrainBaseCfg):
    """Configuration for parkour sub-terrain. Allows for surface noise to be
    configured in the wrapper allow_surface_roughness."""
    surface_roughness: bool = True
    noise_range: tuple[float, float] = (0.02, 0.06)
    v_step: float = 0.005
    resolution: float = 0.075