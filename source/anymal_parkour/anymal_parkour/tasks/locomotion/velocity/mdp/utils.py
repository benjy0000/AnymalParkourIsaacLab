
"""Helper functions for the MDP commands."""

from __future__ import annotations

import torch

from typing import cast, TYPE_CHECKING
from collections.abc import Sequence
from anymal_parkour.terrains import ParkourTerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reached_last_goal(
        env: ManagerBasedRLEnv, env_ids: torch.Tensor, locations_from_origin: torch.Tensor
) -> torch.Tensor:
    
    terrain: ParkourTerrainImporter = cast(ParkourTerrainImporter, env.scene.terrain)

    last_goals = terrain.fetch_goals_from_env(
        torch.tensor(env_ids,
                     dtype=torch.int64,
                     device=env.device
                     )
    )[:, -1, :2].squeeze()
    return torch.linalg.vector_norm(locations_from_origin[env_ids] - last_goals, dim=-1) < 0.1
