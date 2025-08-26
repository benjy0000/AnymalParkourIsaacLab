
from __future__ import annotations

import numpy as np
import scipy.interpolate as interpolate
from typing import TYPE_CHECKING

from .utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import hf_terrains_cfg


"""All copied from previous project and modified to make compatible with IsaacLab
   so rather messy but does the job."""


def random_uniform_terrain(terrain, cfg, min_height, max_height, step, downsampled_scale):
    """
    Generate a uniform noise terrain

    Parameters
        terrain (SubTerrain): the terrain
        min_height (float): the minimum height of the terrain [meters]
        max_height (float): the maximum height of the terrain [meters]
        step (float): minimum height change between two points [meters]
        downsampled_scale (float): distance between two randomly sampled points ( musty be larger or equal to terrain.horizontal_scale)

    """
    if downsampled_scale is None:
        downsampled_scale = cfg.horizontal_scale

    # switch parameters to discrete units
    min_height = int(min_height / cfg.vertical_scale)
    max_height = int(max_height / cfg.vertical_scale)
    step = int(step / cfg.vertical_scale)

    size = terrain.shape
    width = int(cfg.size[1] / cfg.horizontal_scale)
    length = int(cfg.size[0] / cfg.horizontal_scale)

    heights_range = np.arange(min_height, max_height + step, step)
    height_field_downsampled = np.random.choice(heights_range, size)

    x = np.linspace(0, width * cfg.horizontal_scale, height_field_downsampled.shape[0])
    y = np.linspace(0, length * cfg.horizontal_scale, height_field_downsampled.shape[1])

    f = interpolate.RectBivariateSpline(y, x, height_field_downsampled, kx=1, ky=1)

    x_upsampled = np.linspace(0, width * cfg.horizontal_scale, width)
    y_upsampled = np.linspace(0, length * cfg.horizontal_scale, length)
    z_upsampled = np.rint(f(y_upsampled, x_upsampled))

    terrain += z_upsampled.astype(np.int16)


def add_roughness(terrain, cfg, difficulty: float):
    max_height = (cfg.height[1] - cfg.height[0]) * difficulty + cfg.height[0]
    height = np.random.uniform(cfg.height[0], max_height)
    random_uniform_terrain(terrain, cfg, min_height=-height, max_height=height, step=0.005, downsampled_scale=0.075)


@height_field_to_mesh
def barkour_terrain(difficulty: float, cfg: hf_terrains_cfg.HfBarkourTerrainCfg):
    """This is copied over from IsaacGym project with minimal modifications to make this compatible with IsaacLab."""

    # Get config values which are defined in paper
    platform_len = cfg.platform_len
    platform_height = cfg.platform_height
    weave_poles_y_range = cfg.weave_poles_y_range
    A_frame_height_range = [0.0, 0.02]  #cfg.A_frame_height_range
    gap_size = cfg.gap_size
    box_height = cfg.box_height
    pad_width = cfg.pad_width
    pad_height = cfg.pad_height

    # Get config values for environment
    horizontal_scale = cfg.horizontal_scale
    vertical_scale = cfg.vertical_scale

    # Define the height field
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_field_raw = np.zeros((width_pixels, length_pixels))

    # define goals
    goals = np.zeros((8, 3))
    goals[:, 2] = platform_height

    # Other parameters which are permanent
    num_poles = 5
    pole_depth = 0.2
    pole_width = 0.2
    x_range = [1.1, 1.4]  # distance between poles (should depend on the difficulty: the closest the toughest)
    pole_height = [0.75, 1.5]   # height of a pole: random to not associate a specific height value with a pole
    dist_to_pole = [0.2, 0.4]  # distance the robot should leave between it and the pole

    dis_y_min = round(x_range[0] / horizontal_scale)
    dis_y_max = round(x_range[1] / horizontal_scale)
    dis_x_min = round(weave_poles_y_range[0] / horizontal_scale)
    dis_x_max = round(weave_poles_y_range[1] / horizontal_scale)

    robot_width = round(0.530 / 2. / horizontal_scale)

    length = int(cfg.size[0] / horizontal_scale)
    mid_x = length // 2  # length is actually y width, sligthly off center

    width = int(cfg.size[1] / horizontal_scale)

    # half_valid_width = round(np.random.uniform(y_range[1]+0.2, y_range[1]+1) / horizontal_scale)
    pole_width = round(pole_width / horizontal_scale)

    platform_len = round(platform_len / horizontal_scale)
    platform_height = round(platform_height / vertical_scale)
    height_field_raw[0:platform_len, :] = platform_height

    pole_depth = round(pole_depth / horizontal_scale)
    pole_height_min = round(pole_height[0] / vertical_scale)
    pole_height_max = round(pole_height[1] / vertical_scale)

    ##################### WEAVE POLES ########################
    dis_y = platform_len
    last_dis_y = dis_y
    pole_left = 1  # -1 represent a right pole: the goals are placed on the right side of a left pole and vice-versa
    for i in range(num_poles):
        gap = round(np.random.uniform(dist_to_pole[0], dist_to_pole[1]) / horizontal_scale)

        rand_y = np.random.randint(dis_y_min, dis_y_max)
        rand_x = pole_left * np.random.randint(dis_x_min, dis_x_max)
        dis_y += rand_y

        height_field_raw[dis_y - pole_depth // 2 : dis_y + pole_depth // 2,] = np.random.randint(pole_height_min, pole_height_max)
        height_field_raw[dis_y - pole_depth // 2 : dis_y + pole_depth // 2, :mid_x + rand_x - pole_width // 2] = 0
        height_field_raw[dis_y - pole_depth // 2 : dis_y + pole_depth // 2, mid_x + rand_x + pole_width // 2:] = 0
        last_dis_y = dis_y
        
        goals[i, :2] = [dis_y, mid_x + rand_x - pole_left * (pole_width // 2 + robot_width + gap)]
        
        pole_left *= -1
        goal_idx = i
        #print(dis_y * horizontal_scale)

    goal_idx += 1


    ############## A-FRAME ##################
    slope_len = 2.0
    slope_width = 1.0

    # 1st dimension: x, 2nd dimension: y
    #goals = np.zeros((8, 2))
    #height_field_raw[dis_y:] = pit_depth
    
    mid_x = length // 2  # length is actually y width
    
    slope_len = round(slope_len / horizontal_scale)
    height_min = round(A_frame_height_range[0] / vertical_scale)
    height_max = round(A_frame_height_range[1] / vertical_scale)
    dis_between_slopes = round(2.0 / horizontal_scale)

    platform_len = platform_len

    dis_y += platform_len // 2
    mid_x = length // 2  #+ round(np.random.uniform(-y_range, y_range) / horizontal_scale)

    #print(dis_y * horizontal_scale)

    slope_width = round(slope_width / horizontal_scale)
    
    goals[goal_idx, :2] = [dis_y, mid_x]
    goal_idx += 1

    final_height = np.random.randint(height_min, height_max)
            
    slope_up = np.tile(np.linspace(0, final_height, slope_len)[:, np.newaxis], slope_width)
    height_field_raw[dis_y:dis_y+slope_len, mid_x-slope_width//2: mid_x + slope_width//2] = slope_up

    dis_y += slope_len

    slope_down = np.tile((final_height - np.linspace(0, final_height, slope_len))[:, np.newaxis], slope_width)
    height_field_raw[dis_y:dis_y+slope_len, mid_x - slope_width//2: mid_x + slope_width//2] = slope_down

    dis_y += slope_len
    goals[goal_idx, :2] = [dis_y, mid_x]
    goal_idx += 1

    #print("L.1426")
    #print(dis_y * horizontal_scale)

    #################### GAP #########################
    x_range=[1.6, 2.4]
    y_range=[-1.2, 1.2]
    half_valid_width=[0.6, 1.2]
    gap_depth=[1.0, 1.5]

    mid_x = length // 2  # length is actually y width

    dis_x_min = round(y_range[0] / horizontal_scale)
    dis_x_max = round(y_range[1] / horizontal_scale)

    gap_depth = -round(np.random.uniform(gap_depth[0], gap_depth[1]) / vertical_scale)
    
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / horizontal_scale)
    
    gap_size = round(gap_size / horizontal_scale)
    dis_y_min = round(x_range[0] / horizontal_scale) + gap_size
    dis_y_max = round(x_range[1] / horizontal_scale) + gap_size

    dis_y += round(platform_len/1.5)
    last_dis_y = dis_y

    #print(dis_y * horizontal_scale)
        
    height_field_raw[dis_y-gap_size//2 : dis_y+gap_size//2, :] = gap_depth

    height_field_raw[last_dis_y:dis_y, :mid_x+rand_x-half_valid_width] = gap_depth
    height_field_raw[last_dis_y:dis_y, mid_x+rand_x+half_valid_width:] = gap_depth

    #print(dis_y * horizontal_scale)

    ################ FINAL STEP #####################
    dis_y += platform_len

    y_range=[-0.1, 0.1]  # NEW : distance in y between stones (zig zag)
    half_valid_width=[0.9, 1.1]      # NEW : width of the steps
    
    mid_x = length // 2  # length is actually y width

    dis_x_min = round(y_range[0] / horizontal_scale)
    dis_x_max = round(y_range[1] / horizontal_scale)

    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / horizontal_scale)
    box_height_max = round((box_height + 0.05)/ vertical_scale) 
    box_height_min = round((box_height - 0.05)/ vertical_scale)

    stone_len = 1.3
    stone_len = round(stone_len / horizontal_scale)
    
    dis_y += platform_len//4 # middle of the box: (0 to 2) - (stone_len / 2) m between end of platform and beginning of box
    rand_y = np.random.randint(-stone_len // 2, stone_len // 2)    # next goal is anywhere on the box
    rand_x = np.random.randint(dis_x_min, dis_x_max)
    height_field_raw[dis_y-stone_len//2:dis_y+stone_len//2, ] = np.random.randint(box_height_min, box_height_max)
    height_field_raw[dis_y-stone_len//2:dis_y+stone_len//2, :mid_x+rand_x-half_valid_width] = 0
    height_field_raw[dis_y-stone_len//2:dis_y+stone_len//2, mid_x+rand_x+half_valid_width:] = 0
    rand_x = np.random.randint(dis_x_min, dis_x_max)

    final_dis_y = dis_y
    if final_dis_y > width:
        final_dis_y = width - 0.5 // horizontal_scale
    goals[-1] = [final_dis_y, mid_x, platform_height]
    goals[:, :2] = goals[:, :2] * cfg.horizontal_scale
    goals[:, 2] = goals[:, 2] * cfg.vertical_scale

    # pad edges
    pad_width = int(pad_width // horizontal_scale)
    pad_height = int(pad_height // vertical_scale)
    height_field_raw[:, :pad_width] = pad_height
    height_field_raw[:, -pad_width:] = pad_height
    height_field_raw[:pad_width, :] = pad_height
    height_field_raw[-pad_width:, :] = pad_height

    add_roughness(height_field_raw, cfg, difficulty)

    return np.rint(height_field_raw).astype(np.int16), goals
