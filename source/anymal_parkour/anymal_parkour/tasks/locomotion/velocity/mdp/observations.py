from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import anymal_parkour.tasks.locomotion.velocity.mdp as mdp
import isaaclab.envs.mdp.observations as mdp_observations
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# buffer to store the history
g_proprio_history_buffer: torch.Tensor | None = None


class obs_scales:
    base_lin_vel: float = 2.0
    base_ang_vel: float = 0.25
    joint_vel: float = 0.05


def proprioception_history(
    env: ManagerBasedRLEnv,
    history_length: int,
    contact_sensor_cfg: SceneEntityCfg,
    imu_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """A buffer of the last `history_length` proprioceptive observations."""
    global g_proprio_history_buffer

    # --- Compute the current proprioceptive state directly from the environment ---
    # This is the correct pattern, as observation functions should not depend on each other's outputs.
    current_proprio_obs = torch.cat(
        (
            mdp.base_lin_vel(env) * obs_scales.base_lin_vel,
            mdp.base_ang_vel(env) * obs_scales.base_ang_vel,
            imu_observations(env, imu_sensor_cfg),
            delta_yaw(env, "target_points"),
            speed_command(env, "target_speed"),
            mdp.joint_pos_rel(env),
            mdp.joint_vel_rel(env) * obs_scales.joint_vel,
            mdp.last_action(env),
            contact_detector(env, contact_sensor_cfg)
        ),
        dim=1,
    )

    # --- Buffer initialization and update logic (this part remains the same) ---

    # Initialize buffer on the first call
    if g_proprio_history_buffer is None or g_proprio_history_buffer.shape[0] != env.num_envs:
        proprio_dim = current_proprio_obs.shape[1]
        g_proprio_history_buffer = torch.zeros(
            env.num_envs, history_length + 1, proprio_dim, dtype=torch.float, device=env.device
        )
        # On the very first step, fill the buffer with the initial state
        initial_history = current_proprio_obs.unsqueeze(1).repeat(1, history_length + 1, 1)
        g_proprio_history_buffer[:] = initial_history

    # Update the buffer: shift old history and add the new observation
    g_proprio_history_buffer = torch.roll(g_proprio_history_buffer, shifts=-1, dims=1)
    g_proprio_history_buffer[:, -1, :] = current_proprio_obs

    # On environment resets, fill the history with the new initial state
    # Note: we check if the attribute exists since this function is called during initialization
    # before all managers are created.
    if hasattr(env, "termination_manager"):
        if env.termination_manager.dones.any():
            reset_ids = env.termination_manager.dones.nonzero(as_tuple=False).squeeze(-1)
            # Get the initial proprioceptive state for the resetting environments
            initial_proprio_for_resets = current_proprio_obs[reset_ids]
            # Create the initial history by repeating the state
            history_for_resets = initial_proprio_for_resets.unsqueeze(1).repeat(1, history_length + 1, 1)
            # Assign this repeated initial state to the buffer for the reset environments
            g_proprio_history_buffer[reset_ids] = history_for_resets

    # Return the flattened buffer
    return g_proprio_history_buffer[:, :-1, :].flatten(start_dim=1)


def contact_detector(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns the contact sensor data for the feet."""
    # Extract the contact sensor data for the feet
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_ids = sensor_cfg.body_ids
    feet_contact_data = contact_sensor.data.net_forces_w_history[:, :, feet_ids, :]
    feet_contact = torch.linalg.vector_norm(feet_contact_data, dim=-1).squeeze(-1) > 2.0

    # Filter the current contact data with the previous measurement
    filtered_feet_contact_data = torch.logical_or(feet_contact[:, 0, :], feet_contact[:, 1, :])

    # Return the contact data shifted about zero
    return filtered_feet_contact_data.to(torch.float) - 0.5


def delta_yaw(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    Calculates the difference between the robot's current yaw and the yaw required
    to face the current target point from a Path Command, along with the next command.

    The target point is given relative to the environment's origin.
    """
    # Get the robot's current yaw from its world-frame orientation
    robot_quat_w = env.scene.articulations["robot"].data.root_state_w[:, 3:7]
    _, _, current_yaw = euler_xyz_from_quat(robot_quat_w)
    # Get the robot's current XY position relative to its environment origin
    robot_pos_local = env.scene.articulations["robot"].data.root_pos_w - env.scene.env_origins
    robot_xy_local = robot_pos_local[:, :2].unsqueeze(1)
    # Get the current target point from the command manager
    # The command has shape (num_envs, 2, 3)
    command = env.command_manager.get_command(command_name)
    target_xy_local = command[:, :, :2]
    # Calculate the desired yaw to face the target point
    direction_vector = target_xy_local - robot_xy_local
    desired_yaw = torch.atan2(direction_vector[..., 1], direction_vector[..., 0])
    # Compute the smallest angle difference and return it
    return wrap_to_pi(desired_yaw - current_yaw.unsqueeze(1))


def speed_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Returns the velocity command for the robot."""
    # Get the velocity command from the command manager
    command = env.command_manager.get_command(command_name)
    # Return the command velocity component (x-axis velocity)
    speed_command = command.unsqueeze(1)
    return speed_command


def imu_observations(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns the IMU sensor observations for the robot."""
    # Get the IMU orientation quaternion from the asset configuration
    imu_quat = mdp_observations.imu_orientation(env, asset_cfg)
    roll, pitch, _ = euler_xyz_from_quat(imu_quat)
    # Concatenate roll and pitch to form the IMU observation tensor
    imu_obs = torch.cat(
        (
            roll.unsqueeze(1),
            pitch.unsqueeze(1),
        ),
        dim=1,
    )
    # Return the IMU observations
    return imu_obs