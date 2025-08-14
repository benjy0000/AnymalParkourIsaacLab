"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

import numpy as np
import scipy.spatial.transform as tf
import torch
import trimesh
from typing import TYPE_CHECKING

from isaaclab.terrains.trimesh.utils import *  # noqa: F401, F403
from . import terrain_utils

if TYPE_CHECKING:
    from . import parkour_mesh_terrains_cfg


@terrain_utils.allow_surface_roughness
def large_steps_terrain(
    difficulty: float, cfg: parkour_mesh_terrains_cfg.MeshLargeStepsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray]:
    """Generate a terrain with large steps.

    The terrain consists of a series of large steps that the robot must navigate.
    The size and height of the steps are determined by the configuration.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    num_steps = cfg.num_goals - 2
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])

    # initialize list of meshes
    meshes_list = list()
    goals = np.zeros((cfg.num_goals, 3), dtype=np.float32)
    goals[0, :] = (cfg.size[0] / 2, cfg.platform_length - 0.3, 0.3)

    # constants for terrain generation
    terrain_height = 1.0

    # Generate the steps
    x = cfg.size[0] / 2
    y = cfg.platform_length
    cumulative_height = 0.0
    
    for i in range(num_steps):

        step_width = np.random.uniform(cfg.step_width_range[0], cfg.step_width_range[1])
        step_length = np.random.uniform(cfg.step_length_range[0], cfg.step_length_range[1])     
        step_mismatch = np.random.uniform(cfg.step_mismatch_range[0], cfg.step_mismatch_range[1])

        if i < num_steps / 2:
            cumulative_height += step_height
        if i > num_steps / 2:
            cumulative_height -= step_height

        pos = (x + step_mismatch, y + step_length / 2, cumulative_height / 2)
        dim = (step_width, step_length, cumulative_height)

        goals[i + 1, :] = (x + step_mismatch, y + step_length / 2, cumulative_height + 0.3)

        step_mesh = terrain_utils.box(dim, trimesh.transformations.translation_matrix(pos))
        meshes_list.append(step_mesh)

        y += step_length

    goals[-1, :] = (x, y + step_length / 2 + 1, cumulative_height + 0.3)

    # Generate the ground
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground_mesh = terrain_utils.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground_mesh)

    # specify the origin of the terrain
    origin = np.array([cfg.size[0] / 2, cfg.platform_length / 2, terrain_height])

    # shift the goals relative to the origin
    goals -= origin

    return meshes_list, origin, goals


@terrain_utils.allow_surface_roughness
def boxes_terrain(
    difficulty: float, cfg: parkour_mesh_terrains_cfg.MeshBoxesTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray]:
    """Generate a terrain with boxes."""
    
    # resolve the terrain configuration
    num_boxes = 4
    box_height = (cfg.box_height_range[0] + difficulty
                  * (cfg.box_height_range[1] - cfg.box_height_range[0]))
    max_box_offset = (cfg.box_x_offset_range[0] + difficulty
                      * (cfg.box_x_offset_range[1] - cfg.box_x_offset_range[0]))

    # initialise mesh list
    boxes_mesh = list()
    goals = np.zeros((cfg.num_goals, 3), dtype=np.float32)

    # constants for terrain generation
    terrain_height = 1.0
    box_spacing = 2.5 * cfg.box_length
    noise = 0.1 * difficulty

    # Generate the boxes
    x = cfg.size[0] / 2
    y = cfg.platform_length

    for i in range(num_boxes):

        box_x_offset = np.random.uniform(-max_box_offset, max_box_offset)
        box_noise = np.random.uniform(-noise, noise)

        box_x = x + box_x_offset
        box_y = y + cfg.box_length / 2
        pos = (box_x, box_y, (box_height + box_noise) / 2)
        size = (cfg.box_width, cfg.box_length, box_height + box_noise)

        goals[i + 1, :] = (box_x, box_y, box_height + box_noise + 0.3)

        box_mesh = terrain_utils.box(size, trimesh.transformations.translation_matrix(pos))
        boxes_mesh.append(box_mesh)

        y += cfg.box_length + box_spacing

    # Generate the ground
    ground_pos = (cfg.size[0] / 2, cfg.size[1] / 2, -terrain_height / 2)
    ground_dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground_mesh = terrain_utils.box(ground_dim, trimesh.transformations.translation_matrix(ground_pos))
    boxes_mesh.append(ground_mesh)

    origin = np.array([cfg.size[0] / 2, cfg.platform_length / 2, terrain_height])

    # define the goals (height doesn't matter!)
    goals[:, (0, 2)] = (cfg.size[0] / 2, 1.0)
    goals[:, 1] = np.linspace(cfg.platform_length - 0.3, y + 1, cfg.num_goals)
    # shift the goals relative to the origin
    goals -= origin

    return boxes_mesh, origin, goals


@terrain_utils.allow_surface_roughness
def stairs_terrain(
    difficulty: float, cfg: parkour_mesh_terrains_cfg.MeshStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray]:
    """Generate a terrain with stairs."""

    def generate_stairs_trimesh(
        tread: float,
        riser: float,
        noise: float,
        width: float,
        position: tuple[float, float, float],
        num_steps: int
    ) -> list[trimesh.Trimesh]:

        mesh_list = list()

        z = position[2]
        length = (2 * num_steps + 1) * tread

        for i in range(num_steps):
            pos = (position[0], position[1], z + riser / 2)
            noise_tread = 2 * np.random.uniform(-noise, noise)
            noise_riser = np.random.uniform(-noise, noise)
            size = (width, length + noise_tread, riser + noise_riser)
            step_mesh = terrain_utils.box(size, trimesh.transformations.translation_matrix(pos))
            mesh_list.append(step_mesh)
            z += riser + noise_riser
            length -= 2 * tread + noise_tread

        return mesh_list
    
    # resolve the terrain configuration
    tread = cfg.tread_range[0] + difficulty * (cfg.tread_range[1] - cfg.tread_range[0])
    riser = cfg.riser_range[0] + difficulty * (cfg.riser_range[1] - cfg.riser_range[0])
    noise = 0.025 * difficulty
    terrain_height = 1.0

    # initialise mesh list
    stairs_mesh = list()
    goals = np.zeros((cfg.num_goals, 3), dtype=np.float32)

    # Generate the stairs
    offset = (2 * cfg.num_steps + 1) * tread / 2
    pos_1 = (cfg.size[0] / 2, cfg.platform_length + offset, 0)
    pos_2 = (cfg.size[0] / 2, pos_1[1] + 2 * offset + 1, 0)

    stairs_mesh_1 = generate_stairs_trimesh(
        tread=tread,
        riser=riser,
        noise=noise,
        width=cfg.size[0],
        position=pos_1, 
        num_steps=cfg.num_steps
    )

    stairs_mesh_2 = generate_stairs_trimesh(
        tread=tread,
        riser=riser,
        noise=noise,
        width=cfg.size[0],
        position=pos_2,
        num_steps=cfg.num_steps
    )

    stairs_mesh.extend(stairs_mesh_1)
    stairs_mesh.extend(stairs_mesh_2)

    # Generate the ground
    ground_pos = (cfg.size[0] / 2, cfg.size[1] / 2, -terrain_height / 2)
    ground_dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground_mesh = terrain_utils.box(ground_dim, trimesh.transformations.translation_matrix(ground_pos))
    stairs_mesh.append(ground_mesh)

    origin = np.array([cfg.size[0] / 2, cfg.platform_length / 2, terrain_height])

    # Distribute goals (note that z does not affect anything!)
    goals[:, (0, 2)] = (cfg.size[0] / 2, 1)
    goals[:, 1] = np.linspace(cfg.platform_length - 0.3, pos_2[1] + offset + 1, num=cfg.num_goals)
    goals -= origin

    return stairs_mesh, origin, goals


@terrain_utils.allow_surface_roughness
def gaps_terrain(
    difficulty: float, cfg: parkour_mesh_terrains_cfg.MeshGapsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray]:
    """Generate a terrain with gaps."""

    # resolve the terrain configuration
    gap_length = cfg.gap_length_range[0] + difficulty * (cfg.gap_length_range[1] - cfg.gap_length_range[0])
    max_stone_offset = (cfg.stone_x_offset_range[1] + difficulty
                        * (cfg.stone_x_offset_range[1] - cfg.stone_x_offset_range[0]))

    # initialise mesh list
    meshes_list = list()
    goals = np.zeros((cfg.num_goals, 3), dtype=np.float32)
    goals[0, :] = (cfg.size[0] / 2, cfg.platform_length - 0.3, 0.3)

    # constants for terrain generation  
    terrain_height = 1.0
    pit_depth = 0.95
    num_stones = cfg.num_goals - 2
    gap_length_noise = 0.1 * difficulty

    # Create the starting platform
    pos = (0.5 * cfg.size[0], 0.5 * cfg.platform_length, -terrain_height / 2)
    dim = (cfg.size[0], cfg.platform_length, terrain_height)
    platform_mesh = terrain_utils.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(platform_mesh)

    # Generate the gaps
    y = cfg.platform_length
    x = 0.5 * cfg.size[0]

    for i in range(num_stones):

        stone_x_offset = np.random.uniform(-max_stone_offset, max_stone_offset)
        gap_noise = np.random.uniform(-gap_length_noise, gap_length_noise)
        
        stone_x = x + stone_x_offset
        stone_y = y + gap_length + gap_noise + cfg.stone_length / 2
        pos = (stone_x, stone_y, -terrain_height / 2)
        size = (cfg.stone_length, cfg.stone_length, terrain_height)

        goals[i + 1, :] = (stone_x, stone_y, 0.3)

        stone_mesh = terrain_utils.box(size, trimesh.transformations.translation_matrix(pos))
        meshes_list.append(stone_mesh)

        y += gap_length + gap_noise + cfg.stone_length

    y += gap_length

    goals[-1, :] = (x, y + 1, 0.3)

    # Generate bottom of the pit and the end platform
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height + pit_depth / 2)
    dim = (cfg.size[0], cfg.size[1], terrain_height - pit_depth)
    ground_mesh = terrain_utils.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground_mesh)
    
    pos = (x, (cfg.size[1] - y) / 2 + y, -terrain_height / 2)
    dim = (cfg.size[0], (cfg.size[1] - y), terrain_height)
    platform_mesh = terrain_utils.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(platform_mesh)

    origin = np.array([cfg.size[0] / 2, cfg.platform_length / 2, terrain_height])

    goals -= origin

    return meshes_list, origin, goals


@terrain_utils.allow_surface_roughness
def inclined_boxes_terrain(
    difficulty: float, cfg: parkour_mesh_terrains_cfg.MeshInclinedBoxesTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray]:
    """Generate a terrain with inclined boxes."""

    def generate_inclined_boxes_trimesh(
        width: float, length: float, height: float, incline_height: float
    ) -> trimesh.Trimesh:
        """Generate a single inclined box mesh."""
        dim = (width, length, height)
        box_mesh = terrain_utils.box(dim)
        angle = np.arctan2(incline_height, width)
        rotation_matrix = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])

        # Apply rotation so that the positive x, z edge remains invariant
        translation_to_origin = trimesh.transformations.translation_matrix([-width / 2, 0, -height / 2])
        translation_back = trimesh.transformations.translation_matrix([width / 2, 0, height / 2])
        combined_transform = translation_back @ rotation_matrix @ translation_to_origin
        box_mesh.apply_transform(combined_transform)

        # Return the inclined box mesh.
        return box_mesh
        
    # resolve the terrain configuration
    num_stones = cfg.num_goals - 2
    x_gap = [0.2, 0.3 + 0.1 * difficulty]
    y_gap = [-0.1, 0.1 + 0.3 * difficulty]
    stone_len_range = [1.4 - 0.5 * difficulty, 1.8 - 0.5 * difficulty]
    incline_height = 0.25 * difficulty
    last_incline_height = incline_height + 0.1 - 0.1 * difficulty

    # initialize list of meshes
    meshes_list = list()
    goals = np.zeros((cfg.num_goals, 3), dtype=np.float32)
    goals[0, :] = (cfg.size[0] / 2, cfg.platform_length - 0.3, 0.3)

    # constants for terrain generation
    terrain_height = 1.0
    pit_depth = np.random.uniform(cfg.pit_depth[0], cfg.pit_depth[1])
    stone_width = 1.0
    last_stone_len = 1.6
    left_right_flag = np.random.choice([-1, 1])  # Randomly choose position of the first stone

    # Create the starting platform
    pos = (0.5 * cfg.size[0], 0.5 * cfg.platform_length, -terrain_height / 2)
    dim = (cfg.size[0], cfg.platform_length, terrain_height)
    platform_mesh = terrain_utils.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(platform_mesh)

    # Generate the inclined boxes
    x = cfg.size[0] / 2
    y = cfg.platform_length

    for i in range(num_stones):
        
        y += np.random.uniform(y_gap[0], y_gap[1])
        if i == num_stones - 1:
            stone_len = last_stone_len
            y += last_stone_len / 4
            incline_height = last_incline_height
        else:
            stone_len = np.random.uniform(stone_len_range[0], stone_len_range[1])
        
        stone_y = y + stone_len / 2
        if i % 2 == 0:
            x_stone = x + left_right_flag * np.random.uniform(x_gap[0], x_gap[1])
        else:
            x_stone = x - left_right_flag * np.random.uniform(x_gap[0], x_gap[1])

        pos = (x_stone, stone_y, -terrain_height / 2)
        dim = (stone_width, stone_len, terrain_height)

        goals[i + 1, :] = (x_stone, stone_y, 0.5)

        # Generate the inclined box mesh
        inclined_box_mesh = generate_inclined_boxes_trimesh(*dim, incline_height)
        # Reflect if necessary and apply translation
        reflection_matrix = np.eye(4)
        if i % 2 == 0:
            reflection_matrix[0, 0] = -left_right_flag
        else:
            reflection_matrix[0, 0] = left_right_flag
        inclined_box_mesh.apply_transform(reflection_matrix)
        inclined_box_mesh.apply_translation(pos)
        meshes_list.append(inclined_box_mesh)

        y += stone_len

    # Generate bottom of the pit and the end platform
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height + pit_depth / 2)
    dim = (cfg.size[0], cfg.size[1], terrain_height - pit_depth)
    ground_mesh = terrain_utils.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground_mesh)
    
    y += np.random.uniform(y_gap[0], y_gap[1])
    pos = (x, (cfg.size[1] - y) / 2 + y, -terrain_height / 2)
    dim = (cfg.size[0], (cfg.size[1] - y), terrain_height)
    platform_mesh = terrain_utils.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(platform_mesh)

    goals[-1, :] = (x, y + 1, 0.3)

    # Specify the origin of the terrain
    origin = np.array([cfg.size[0] / 2, cfg.platform_length / 2, terrain_height])

    # shift the goals down by the origin
    goals -= origin

    return meshes_list, origin, goals


@terrain_utils.allow_surface_roughness
def weave_pole_terrain(
    difficulty: float, cfg: parkour_mesh_terrains_cfg.MeshWeavePoleTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray]:
    """Generate a terrain with weave poles."""

    # resolve the terrain configuration
    pole_x_separation = cfg.pole_x_range[1] - difficulty * (cfg.pole_x_range[1] - cfg.pole_x_range[0])

    # initialize list of meshes
    num_poles = cfg.num_goals
    meshes_list = list()
    goals = np.zeros((cfg.num_goals, 3), dtype=np.float32)
    goals[0, :] = (cfg.size[0] / 2, cfg.platform_length - 0.3, 0)

    # constants for terrain generation
    terrain_height = 1.0
    left_right_flag = np.random.choice([-1, 1])
    robot_width = 0.53

    # Generate the poles
    x = cfg.size[0] / 2
    y = cfg.platform_length

    for i in range(num_poles):

        y += np.random.uniform(cfg.pole_y_range[0], cfg.pole_y_range[1])

        pole_height = np.random.uniform(cfg.pole_height_range[0], cfg.pole_height_range[1])        
        pole_x_noise = np.random.uniform(-cfg.pole_x_noise, cfg.pole_x_noise)

        clearance = np.random.uniform(cfg.robot_x_clearance_range[0], cfg.robot_x_clearance_range[1])

        pole_y = y
        if i % 2 == 0:
            pole_x = x - left_right_flag * (pole_x_separation + pole_x_noise) / 2
            goal_x = pole_x + left_right_flag * (robot_width / 2 + clearance + cfg.pole_radius)
        else:
            pole_x = x + left_right_flag * (pole_x_separation + pole_x_noise) / 2
            goal_x = pole_x - left_right_flag * (robot_width / 2 + clearance + cfg.pole_radius)

        pos = (pole_x, pole_y, pole_height / 2)
        dim = (cfg.pole_radius, pole_height)

        goals[i, :] = (goal_x, pole_y, 0.3)

        pole_mesh = trimesh.creation.cylinder(cfg.pole_radius,
                                              pole_height,
                                              transform=trimesh.transformations.translation_matrix(pos)
                                              )
        meshes_list.append(pole_mesh)    

    # Generate the ground
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground_mesh = terrain_utils.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground_mesh)

    # specify the origin of the terrain
    origin = np.array([cfg.size[0] / 2, cfg.platform_length / 2, terrain_height])

    # shift the goals down by the origin
    goals -= origin

    return meshes_list, origin, goals


@terrain_utils.allow_surface_roughness
def flat_terrain(
    difficulty: float, cfg: parkour_mesh_terrains_cfg.MeshFlatTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray]:
    """Generate a flat terrain."""

    # Constants for terrain generation
    terrain_height = 1.0

    # initialize list of meshes
    meshes_list = list()
    goals = np.zeros((cfg.num_goals, 3), dtype=np.float32)
    goals[:, 2] = 0.3
    goals[:, 0] = cfg.size[0] / 2 + np.random.uniform(-cfg.x_range, cfg.x_range, cfg.num_goals)
    goals[:, 1] = np.linspace(cfg.platform_length, cfg.size[1], cfg.num_goals)

    # Generate the ground
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground_mesh = terrain_utils.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground_mesh)

    # specify the origin of the terrain
    origin = np.array([cfg.size[0] / 2, cfg.platform_length / 2, terrain_height])

    # shift the goals up by the origin
    goals -= origin

    return meshes_list, origin, goals
