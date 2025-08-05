from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import anymal_parkour.tasks.locomotion.velocity.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation 


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_velocity_out_of_bounds(
    env: ManagerBasedRLEnv, limit_vel: float, limit_ang_vel: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's base velocity exceeds the specified limits."""
    # extract the used quantities
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_vel_w[:, :3]
    ang_vel = asset.data.root_vel_w[:, 3:6]
    return torch.any((lin_vel.abs() > limit_vel) | (ang_vel.abs() > limit_ang_vel))