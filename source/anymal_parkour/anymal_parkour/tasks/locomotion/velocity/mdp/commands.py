from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from dataclasses import MISSING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from collections.abc import Sequence

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class RandomPathCommandCfg(CommandTermCfg):
    """Configuration for the random path command."""
    # The asset to command
    asset_name: str = MISSING
    # The range [min, max] of the radius from the origin to sample the target point.
    target_radius_range: tuple[float, float] = (5.0, 10.0)
    target_speed_range: tuple[float, float] = (0.0, 1.0)
    # The proportion of commands that should have speed = 0.0
    rel_standing_env: float = 0.0


class RandomPathCommand(CommandTerm):
    """
    A command generator that produces a fixed, randomly generated target point in the world frame
    along with a target speed. It returns the command vector followed by the next command.

    This command is sampled once per resampling period. The output command vector will
    be the [[x1, y1, s1], [x2, y2, s2]] relative to the origin of the environment.
    """
    cfg: RandomPathCommandCfg

    def __init__(self, cfg: RandomPathCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.total_command = torch.zeros(self.num_envs, 2, 3, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Returns the command vector."""
        return self.total_command
        
    def _update_metrics(self):
        pass  # No metrics to update for this command
   
    def _resample_command(self, env_ids: Sequence[int] | None = None):
        if env_ids is not None:
            self.total_command[env_ids, 0] = self.total_command[env_ids, 1]
            self.total_command[env_ids, -1] += self._sample_random_target_(len(env_ids))

    def _update_command(self):
        """
        Checks if the robot has reached the current target. If so, it resamples a new command
        for that environment.
        """
        # Get the robot's position relative to its environment's origin
        robot_pos_local = self.robot.data.root_pos_w - self._env.scene.env_origins
        # Calculate the distance to the current target point
        current_goal_pos = self.total_command[:, 0, :2]
        distance_to_goal = torch.linalg.vector_norm(robot_pos_local[:, :2] - current_goal_pos, dim=1)
        # Find the environments where the robot is within the 0.1m threshold
        reached_goal_env_ids = torch.where(distance_to_goal < 0.1)[0]
        # If any environments have reached their goal, resample their command
        if reached_goal_env_ids.numel() > 0:
            reached_goal_env_ids_list = reached_goal_env_ids.tolist()
            self._resample_command(reached_goal_env_ids_list)

    def _sample_random_target_(self, size) -> torch.Tensor:
        """
        Generates a new random target point for each environment.
        This is called by the command manager at every resampling interval.
        """
        # Generate a new command vector for each environment
        new_command = torch.zeros((size, 3), device=self.device)

        # Generate random radii, angles and speedsfor each environment
        radii = torch.rand(size, device=self.device) \
            * (self.cfg.target_radius_range[1] - self.cfg.target_radius_range[0]) \
            + self.cfg.target_radius_range[0]

        angles = torch.rand(size, device=self.device) * 2 * torch.pi

        speeds = torch.rand(size, device=self.device) \
            * (self.cfg.target_speed_range[1] - self.cfg.target_speed_range[0]) \
            + self.cfg.target_speed_range[0]

        # Convert polar coordinates to Cartesian to get the target points in the world frame
        target_x = radii * torch.cos(angles)
        target_y = radii * torch.sin(angles)

        # Store the target point in the command vector
        new_command[:, 0] = target_x
        new_command[:, 1] = target_y
        new_command[:, 2] = speeds

        return new_command

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Resets the command for the specified environments."""
        # Reset the command vector to zeros for the specified environments
        # Flushes out the zeros by resampling one extra time
        if env_ids is not None:
            self.total_command[env_ids] = torch.zeros(len(env_ids), 2, 3, device=self.device)
            self._resample_command(env_ids)
        return super().reset(env_ids)
