
"""Configuration for parkour sub-terrain."""

import numpy as np
import trimesh
from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass


@configclass
class ParkourSubTerrainCfg(SubTerrainBaseCfg):
    """Configuration for parkour sub-terrain. Allows for surface noise to be
    configured in the wrapper allow_surface_roughness."""
    surface_roughness: bool = True
    noise_range: tuple[float, float] = (0.02, 0.04)
    v_step: float = 0.005
    resolution: float = 0.075

    platform_length: float = 5.0

    num_goals: int = MISSING

    function: Callable[[float, SubTerrainBaseCfg], tuple[list[trimesh.Trimesh], np.ndarray, int]] = MISSING
    