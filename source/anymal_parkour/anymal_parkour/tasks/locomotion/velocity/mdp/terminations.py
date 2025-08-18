from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import anymal_parkour.tasks.locomotion.velocity.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation 


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def orientation_out_of_bounds(
    env: ManagerBasedRLEnv, limit_roll: float, limit_pitch: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's base velocity exceeds the specified limits."""
    # extract the used quantities
    asset: Articulation = env.scene[asset_cfg.name]
    roll = asset.data.root_state_w[:, 3]
    pitch = asset.data.root_state_w[:, 4]
    return torch.any((roll.abs() > limit_roll) | (pitch.abs() > limit_pitch))


def reached_final_goal(
        env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset reaches the final goal."""
    # extract the used quantities
    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
    locations = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    finished = mdp.reached_last_goal(env, env_ids, locations)
    if torch.any(finished):
        return finished
    return finished
