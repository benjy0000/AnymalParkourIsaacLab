"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

import anymal_parkour.tasks.locomotion.velocity.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import RLTaskEnv


def terrain_levels_vel(
    env: RLTaskEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    speed_command = env.command_manager.get_command("target_speed")[env_ids]
    # compute the distance the robot walked
    locations = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    distance = torch.norm(locations[env_ids, :], dim=1)
    # compute thise that have rached final goal
    env_ids_tensor = torch.tensor(env_ids, device=env.device, dtype=torch.int64)
    finished = mdp.reached_last_goal(env, env_ids_tensor, locations)
    # robots that walked far enough progress to harder terrains
    # 0.7 factor is to account for the fact that the root is not fully straight
    move_up = torch.logical_or(distance > speed_command * env.max_episode_length_s * 0.8 * 0.7, finished)

    # robots that walked less than half of their required distance go to simpler terrains
    move_down = torch.norm(distance) < speed_command * env.max_episode_length_s * 0.4 * 0.7
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())
