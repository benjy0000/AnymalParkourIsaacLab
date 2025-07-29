from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.envs.mdp.observations as mdp_observations
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation 
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


previous_torques = None


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def tracking_goal_reward(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Reward for tracking the goal velocity."""
    # Get the commanded speed and direction
    commanded_speed = env.command_manager.get_command(command_name)[:, 0]
    commanded_yaw = env.command_manager.get_command(command_name)[:, 2].unsqueeze(1)
    commanded_direction = torch.cat(
        (torch.cos(commanded_yaw),
         torch.sin(commanded_yaw),
         torch.zeros(commanded_yaw.shape[0], 1, device=commanded_yaw.device)),
        dim=1
    )
    # Get the current base linear velocity
    current_velocity = mdp_observations.base_lin_vel(env)
    # Compute the dot product between the commanded direction and the current velocity
    directional_speed = torch.sum(current_velocity * commanded_direction, dim=1)
    # Reward is the minimum of the commanded speed and the directional speed
    reward = torch.min(commanded_speed, directional_speed)
    return reward


def tracking_yaw_reward(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Reward for tracking the goal yaw."""
    # -- Get current robot yaw
    # Get root state orientation quaternion in world frame
    robot_quat_w_xyz = env.scene.articulations["robot"].data.root_state_w[:, 3:7]
    # Convert to RPY euler angles
    _, _, current_yaw = euler_xyz_from_quat(robot_quat_w_xyz)

    # -- Get commanded heading
    # The heading command is the 3rd element (index 2) in the velocity command vector
    commanded_heading = env.command_manager.get_command(command_name)[:, 2]

    # -- Compute the smallest angle difference
    # The wrap_to_pi function handles the circular nature of angles (e.g., -pi vs pi)
    heading_error = wrap_to_pi(commanded_heading - current_yaw)
    # Reward is the negative exponential of the absolute heading error
    reward = torch.exp(-torch.abs(heading_error))
    return reward


def action_rate_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for the rate of change of actions."""
    # Get change in actions over the last step
    action_change = env.action_manager.action - env.action_manager.prev_action
    # Reward is the magnitude of the action change
    reward = torch.linalg.vector_norm(action_change, dim=-1)
    return reward


def torque_variation_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for the variation in torques applied to the robot."""
    global previous_torques
    # Access the current torques
    current_torques = env.scene.articulations["robot"].data.applied_torque
    # Initialize the previous torques if this is the first timestep
    if previous_torques is None:
        previous_torques = torch.zeros_like(current_torques)
    # Identify environments that are being reset
    if hasattr(env, "termination_manager"):
        reset_ids = env.termination_manager.dones.nonzero(as_tuple=False).squeeze(-1)
        # Reset the previous torques for these environments
        previous_torques[reset_ids] = current_torques[reset_ids]
    # Compute the change in torques
    torque_change = current_torques - previous_torques
    # Update the previous torques for the next timestep
    previous_torques = current_torques.clone()
    # Compute the reward as the sum of squared differences in torques
    reward = torch.sum(torch.square(torque_change), dim=1)
    return reward


def feet_stumble_reward(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward for feet stumbles based on contact forces."""
    # Extract the contact sensor data for the feet
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces_w_history = contact_sensor.data.net_forces_w_history
    if net_forces_w_history is None:
        # Return zero reward if no contact force history is available
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    contact_forces = net_forces_w_history[:, 0, :, :].squeeze(1)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(torch.norm(contact_forces[:, :, :2], dim=2) > 
                       4 * torch.abs(contact_forces[:, :, 2]), dim=1)
    return reward.float()


def calc_feet_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Calculate the feet contact based on the contact sensor data."""
    # Extract the contact sensor data for the feet
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces_w_history = contact_sensor.data.net_forces_w_history
    if net_forces_w_history is None:
        # Return zero reward if no contact force history is available
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    feet_contact_history = torch.linalg.vector_norm(net_forces_w_history, dim=-1).squeeze() > 2.0
    filtered_feet_contact_history = torch.logical_or(feet_contact_history[:, 0, :],
                                                     feet_contact_history[:, 1, :])
    feet_contact = filtered_feet_contact_history.squeeze()
    # Return the feet contact data
    return feet_contact


def feet_trotting_reward(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward for feet trotting based on contact forces."""
    # Extract the contact sensor data for the feet
    feet_contact = calc_feet_contact(env, sensor_cfg)
    # Detect trotting
    trotting_1 = torch.logical_and(torch.logical_and(feet_contact[:, 0], feet_contact[:, 3]),
                                   torch.logical_and(~feet_contact[:, 1], ~feet_contact[:, 2]))
    trotting_2 = torch.logical_and(torch.logical_and(~feet_contact[:, 0], ~feet_contact[:, 3]),
                                   torch.logical_and(feet_contact[:, 1], feet_contact[:, 2]))
    trotting = torch.logical_or(trotting_1, trotting_2)
    # Reward for trotting
    reward = trotting.float()
    return reward


def feet_in_air_reward(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward for feet being in the air based on contact forces."""
    # Extract the contact sensor data for the feet
    feet_contact = calc_feet_contact(env, sensor_cfg)
    # Detect feet in the air
    in_air = ~feet_contact.any(dim=1)
    # Reward for feet in the air
    reward = in_air.float()
    return reward


def error_dof_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(angle), dim=1)
