
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
    width = int(cfg.size[0] / cfg.horizontal_scale)
    length = int(cfg.size[1] / cfg.horizontal_scale)

    heights_range = np.arange(min_height, max_height + step, step)
    height_field_downsampled = np.random.choice(heights_range, size)

    x = np.linspace(0, width * cfg.horizontal_scale, height_field_downsampled.shape[0])
    y = np.linspace(0, length * cfg.horizontal_scale, height_field_downsampled.shape[1])

    f = interpolate.RectBivariateSpline(x, y, height_field_downsampled, kx=1, ky=1)

    x_upsampled = np.linspace(0, width * cfg.horizontal_scale, width)
    y_upsampled = np.linspace(0, length * cfg.horizontal_scale, length)
    z_upsampled = np.rint(f(x_upsampled, y_upsampled))

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

    # compute origin
    origin = np.array([1, 0.5 * cfg.size[0], 0.6])
    goals -= origin

    return np.rint(height_field_raw).astype(np.int16), origin, goals
@height_field_to_mesh
def waves_terrain(difficulty: float, cfg: hf_terrains_cfg.HfWavesTerrainCfg):
    """This is copied over from IsaacGym project with minimal modifications to make this compatible with IsaacLab."""

    # Get config values
    wave_len_range = cfg.wave_len_range
    wave_height_range = cfg.wave_height_range
    platform_len = cfg.platform_len
    pad_width = cfg.pad_width
    pad_height = cfg.pad_height

    # Calculate wave parameters based on difficulty
    wave_len = wave_len_range[1] - (wave_len_range[1] - wave_len_range[0]) * difficulty
    wave_height = wave_height_range[0] + (wave_height_range[1] - wave_height_range[0]) * difficulty

    # Get config values for environment
    horizontal_scale = cfg.horizontal_scale
    vertical_scale = cfg.vertical_scale

    # Define the height field
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_field_raw = np.zeros((width_pixels, length_pixels))

    # define goals
    goals = np.zeros((cfg.num_goals, 3))
    goals[:, 2] = 1.0

    # Convert to discrete units
    obstacle_length = int((cfg.size[1] - platform_len - 0.5) / horizontal_scale)
    wave_len = round(wave_len / horizontal_scale)
    wave_height = round(wave_height / vertical_scale)
    platform_len = round(platform_len / horizontal_scale)

    obstacle_array = np.arange(0, obstacle_length)
    height_field_raw[:, platform_len: platform_len + obstacle_length] = (1 - np.cos(2 * np.pi * obstacle_array / wave_len)) * wave_height / 2

    goals[:, 0] = cfg.size[0] / 2
    goals[:, 1] = np.linspace(cfg.platform_len, cfg.size[1] - 1.5, cfg.num_goals)

    # pad edges
    pad_width = int(cfg.pad_width // horizontal_scale)
    pad_height = int(cfg.pad_height // vertical_scale)
    height_field_raw[:, :pad_width] = pad_height
    height_field_raw[:, -pad_width:] = pad_height
    height_field_raw[:pad_width, :] = pad_height
    height_field_raw[-pad_width:, :] = pad_height

    add_roughness(height_field_raw, cfg, difficulty)

    # compute origin
    origin = np.array([0.5 * cfg.size[0], cfg.platform_len / 2, 0])
    goals -= origin

    return np.rint(height_field_raw).astype(np.int16), origin, goals


@height_field_to_mesh
def slope_up_terrain(difficulty: float, cfg: hf_terrains_cfg.HfSlopeUpTerrainCfg):
    """Slope-up evaluation terrain in the standardized HF format."""

    # Scales and sizes
    h_scale = cfg.horizontal_scale
    v_scale = cfg.vertical_scale
    width_px = int(cfg.size[0] / h_scale)
    length_px = int(cfg.size[1] / h_scale)

    # Initialize entire field to pit depth (negative)
    pit_units = int(round(-cfg.pit_depth / v_scale))
    height_field_raw = np.full((width_px, length_px), pit_units, dtype=np.int32)

    # Platform region (set to platform height)
    platform_len_px = int(round(cfg.platform_len / h_scale))
    platform_h_units = int(round(cfg.platform_height / v_scale))
    if platform_len_px > 0:
        height_field_raw[:, :platform_len_px] = platform_h_units

    # Central band parameters
    mid_x_px = width_px // 2
    slope_width_px = max(1, int(round(cfg.slope_width / h_scale)))
    x0 = max(0, mid_x_px - slope_width_px // 2)
    x1 = min(width_px, x0 + slope_width_px)

    # Slope length and target height
    slope_len_px = max(1, int(round(cfg.slope_len / h_scale)))
    h_min, h_max = cfg.height_range
    final_h_units = int(round(h_min + (h_max - h_min) * difficulty / v_scale))

    # Ramp up after the platform
    y_start = platform_len_px
    y_end = min(length_px, y_start + slope_len_px)
    if x1 > x0 and y_end > y_start:
        ramp = np.linspace(platform_h_units, final_h_units, y_end - y_start, dtype=np.float64)
        for k, y in enumerate(range(y_start, y_end)):
            height_field_raw[x0:x1, y] = int(round(ramp[k]))

    # Keep central band at final height for the remainder
    if x1 > x0 and y_end < length_px:
        height_field_raw[x0:x1, y_end:] = final_h_units

    # Goals: evenly spaced along Y after the platform, centered in X
    goals = np.zeros((cfg.num_goals, 3), dtype=np.float32)
    goals[:, 2] = 1.5
    goals[:, 0] = 0.5 * cfg.size[0]
    goals[:, 1] = np.linspace(cfg.platform_len, cfg.size[1] - 1.5, cfg.num_goals)

    # Pad edges
    pad_w = int(cfg.pad_width // h_scale)
    pad_h = int(cfg.pad_height // v_scale)
    if pad_w > 0:
        height_field_raw[:, :pad_w] = pad_h
        height_field_raw[:, -pad_w:] = pad_h
        height_field_raw[:pad_w, :] = pad_h
        height_field_raw[-pad_w:, :] = pad_h

    # Optional roughness
    add_roughness(height_field_raw, cfg, difficulty)

    # Origin at center of platform
    origin = np.array([0.5 * cfg.size[0], cfg.platform_len / 2.0, 0.0], dtype=np.float32)
    goals -= origin

    return np.rint(height_field_raw).astype(np.int16), origin, goals


@height_field_to_mesh
def slope_up_terrain(difficulty: float, cfg: hf_terrains_cfg.HfSlopeUpTerrainCfg):
    """Slope-up evaluation terrain in the standardized HF format."""

    # Scales and sizes
    h_scale = cfg.horizontal_scale
    v_scale = cfg.vertical_scale
    width_px = int(cfg.size[0] / h_scale)
    length_px = int(cfg.size[1] / h_scale)

    # Initialize entire field to pit depth (negative)
    pit_units = int(round(-cfg.pit_depth / v_scale))
    height_field_raw = np.full((width_px, length_px), pit_units, dtype=np.int32)

    # Platform region (set to platform height)
    platform_len_px = int(round(cfg.platform_len / h_scale))
    platform_h_units = int(round(cfg.platform_height / v_scale))
    if platform_len_px > 0:
        height_field_raw[:, :platform_len_px] = platform_h_units

    # Central band parameters
    mid_x_px = width_px // 2
    slope_width_px = max(1, int(round(cfg.slope_width / h_scale)))
    x0 = max(0, mid_x_px - slope_width_px // 2)
    x1 = min(width_px, x0 + slope_width_px)

    # Slope length and target height
    slope_len_px = max(1, int(round(cfg.slope_len / h_scale)))
    h_min, h_max = cfg.height_range
    final_h_units = int(round(h_min + (h_max - h_min) * difficulty / v_scale))

    # Ramp up after the platform
    y_start = platform_len_px
    y_end = min(length_px, y_start + slope_len_px)
    if x1 > x0 and y_end > y_start:
        ramp = np.linspace(platform_h_units, final_h_units, y_end - y_start, dtype=np.float64)
        for k, y in enumerate(range(y_start, y_end)):
            height_field_raw[x0:x1, y] = int(round(ramp[k]))

    # Keep central band at final height for the remainder
    if x1 > x0 and y_end < length_px:
        height_field_raw[x0:x1, y_end:] = final_h_units

    # Goals: evenly spaced along Y after the platform, centered in X
    goals = np.zeros((cfg.num_goals, 3), dtype=np.float32)
    goals[:, 2] = 1.5
    goals[:, 0] = 0.5 * cfg.size[0]
    goals[:, 1] = np.linspace(cfg.platform_len, cfg.size[1] - 1.5, cfg.num_goals)

    # Pad edges
    pad_w = int(cfg.pad_width // h_scale)
    pad_h = int(cfg.pad_height // v_scale)
    if pad_w > 0:
        height_field_raw[:, :pad_w] = pad_h
        height_field_raw[:, -pad_w:] = pad_h
        height_field_raw[:pad_w, :] = pad_h
        height_field_raw[-pad_w:, :] = pad_h

    # Optional roughness
    add_roughness(height_field_raw, cfg, difficulty)

    # Origin at center of platform
    origin = np.array([0.5 * cfg.size[0], cfg.platform_len / 2.0, 0.0], dtype=np.float32)
    goals -= origin

    return np.rint(height_field_raw).astype(np.int16), origin, goals


@height_field_to_mesh
def slope_down_terrain(difficulty: float, cfg: hf_terrains_cfg.HfSlopeDownTerrainCfg):
    """Slope-down evaluation terrain in the standardized HF format."""

    # Scales and sizes
    h_scale = cfg.horizontal_scale
    v_scale = cfg.vertical_scale
    width_px = int(cfg.size[0] / h_scale)
    length_px = int(cfg.size[1] / h_scale)

    # Initialize entire field to pit depth (negative)
    pit_units = int(round(-cfg.pit_depth / v_scale))
    height_field_raw = np.full((width_px, length_px), pit_units, dtype=np.int32)

    # Platform region (set to platform height)
    platform_len_px = int(round(cfg.platform_len / h_scale))
    platform_h_units = int(round(cfg.platform_height / v_scale))
    if platform_len_px > 0:
        height_field_raw[:, :platform_len_px] = platform_h_units

    # Central band parameters
    mid_x_px = width_px // 2
    slope_width_px = max(1, int(round(cfg.slope_width / h_scale)))
    x0 = max(0, mid_x_px - slope_width_px // 2)
    x1 = min(width_px, x0 + slope_width_px)

    # Slope length and target height
    slope_len_px = max(1, int(round(cfg.slope_len / h_scale)))
    h_min, h_max = cfg.height_range
    final_h_units = int(round(h_min + (h_max - h_min) * difficulty / v_scale))

    # Ramp down after the platform
    y_start = platform_len_px
    y_end = min(length_px, y_start + slope_len_px)
    if x1 > x0 and y_end > y_start:
        ramp = np.linspace(platform_h_units, final_h_units, y_end - y_start, dtype=np.float64)
        for k, y in enumerate(range(y_start, y_end)):
            height_field_raw[x0:x1, y] = int(round(ramp[k]))

    # Keep central band at final height for the remainder
    if x1 > x0 and y_end < length_px:
        height_field_raw[x0:x1, y_end:] = final_h_units

    # Goals: evenly spaced along Y after the platform, centered in X
    goals = np.zeros((cfg.num_goals, 3), dtype=np.float32)
    goals[:, 2] = 1.1
    goals[:, 0] = 0.5 * cfg.size[0]
    goals[:, 1] = np.linspace(cfg.platform_len, cfg.size[1] - 1.5, cfg.num_goals)

    # Pad edges
    pad_w = int(cfg.pad_width // h_scale)
    pad_h = int(cfg.pad_height // v_scale)
    if pad_w > 0:
        height_field_raw[:, :pad_w] = pad_h
        height_field_raw[:, -pad_w:] = pad_h
        height_field_raw[:pad_w, :] = pad_h
        height_field_raw[-pad_w:, :] = pad_h

    # Optional roughness
    add_roughness(height_field_raw, cfg, difficulty)

    # Origin at center of platform
    origin = np.array([0.5 * cfg.size[0], cfg.platform_len / 2.0, 1.0], dtype=np.float32)
    goals -= origin

    height_field_raw = height_field_raw + int(1.0 / v_scale)  # raise entirety of the terrain to prevent root height termination

    return np.rint(height_field_raw).astype(np.int16), origin, goals


@height_field_to_mesh
def valley_terrain(difficulty: float, cfg: hf_terrains_cfg.HfValleyTerrainCfg):

    # Scales and sizes
    h_scale = cfg.horizontal_scale
    v_scale = cfg.vertical_scale
    width_px = int(cfg.size[0] / h_scale)     # X axis in pixels
    length_px = int(cfg.size[1] / h_scale)    # Y axis in pixels

    # Base height = 1.0 m everywhere
    base_height_units = int(round(1.0 / v_scale))
    height_field_raw = np.full((width_px, length_px), base_height_units, dtype=np.int32)

    # Parameters
    valley_width_m = cfg.valley_width_range[1] - (cfg.valley_width_range[1] - cfg.valley_width_range[0]) * difficulty
    valley_half_w_px = max(1, int(round(0.5 * valley_width_m / h_scale)))
    wall_height = 2

    # Lateral separation like weave poles (harder → smaller gap)
    sep_min, sep_max = cfg.bend_x_range
    x_separation_m = sep_max - difficulty * (sep_max - sep_min)
    x_separation_m = float(np.clip(x_separation_m, sep_min, sep_max))

    # Convert to pixels
    start_y_px = int(round(cfg.platform_len / h_scale))
    mid_x_px = width_px // 2

    # Depth like boxes
    dmin, dmax = cfg.depth_range
    depth_m = dmin + difficulty * (dmax - dmin)
    depth_units = int(round(depth_m / v_scale))
    valley_floor_units = int(max(0, base_height_units - depth_units))  # carve down from base
    # Valley walls target height value (raw units)
    wall_height_units = wall_height / v_scale

    # Zig-zag definition: 6 bends (apexes) with random Y segment lengths
    num_bends = 6
    seg_y_min_m, seg_y_max_m = cfg.bend_y_range
    rng = np.random.default_rng()
    # Build target centers for each bend (alternate left/right)
    left_right = rng.choice([-1, 1])
    bend_targets_x_px: list[int] = []
    for i in range(num_bends):
        offset_m = left_right * (x_separation_m / 2.0)
        x_c_px = int(round(mid_x_px + offset_m / h_scale))
        x_c_px = int(np.clip(x_c_px, valley_half_w_px, width_px - valley_half_w_px - 1))
        bend_targets_x_px.append(x_c_px)
        left_right *= -1

    # Compute valley Y extents and path
    y_cursor = start_y_px
    bend_y_indices: list[int] = []
    for i in range(num_bends):
        seg_len_px = max(1, int(round(rng.uniform(seg_y_min_m, seg_y_max_m) / h_scale)))
        y_end = min(length_px - 1, y_cursor + seg_len_px)
        bend_y_indices.append(y_end)
        # Linearly move center from previous center to target over this segment
        x_start = mid_x_px if i == 0 else bend_targets_x_px[i - 1]
        x_end = bend_targets_x_px[i]
        if y_end > y_cursor:
            for j, t in zip(range(y_cursor, y_end + 1), np.linspace(0.0, 1.0, y_end - y_cursor + 1)):
                # Set walls to constant small height outside the valley past the platform
                height_field_raw[:, j] = wall_height_units
                xc = int(round((1 - t) * x_start + t * x_end))
                x0 = max(0, xc - valley_half_w_px)
                x1 = min(width_px, xc + valley_half_w_px + 1)
                height_field_raw[x0:x1, j] = valley_floor_units
        y_cursor = y_end

    # After last bend, extend a short straight to exit, then step-up (back to base height)
    exit_seg_len_px = max(1, int(round(cfg.exit_segment_len / h_scale)))
    last_xc = bend_targets_x_px[-1]
    for j in range(y_cursor + 1, min(length_px, y_cursor + 1 + exit_seg_len_px)):
        # Set walls to constant small height outside the valley until step-up
        height_field_raw[:, j] = wall_height_units
        x0 = max(0, last_xc - valley_half_w_px)
        x1 = min(width_px, last_xc + valley_half_w_px + 1)
        height_field_raw[x0:x1, j] = valley_floor_units
    valley_end_y_px = min(length_px - 1, y_cursor + exit_seg_len_px)

    # Goals: 8 total → [step-down, 6 apexes, step-up]
    goals = np.zeros((cfg.num_goals, 3), dtype=np.float32)
    goals[:, 2] = 1.0  # marker z

    # Step-down goal at valley entry (centered)
    goals[0, 0] = mid_x_px * h_scale
    goals[0, 1] = start_y_px * h_scale

    # Apex goals at each bend end
    for i, y_b in enumerate(bend_y_indices):
        goals[i + 1, 0] = bend_targets_x_px[i] * h_scale
        goals[i + 1, 1] = y_b * h_scale

    # Step-up goal at valley exit
    goals[-1, 0] = last_xc * h_scale
    goals[-1, 1] = min(cfg.size[1] - 1.0, (valley_end_y_px + int(round(0.5 / h_scale))) * h_scale)

    # Pad edges
    pad_w = int(cfg.pad_width // h_scale)
    pad_h = int(cfg.pad_height // v_scale)
    if pad_w > 0:
        height_field_raw[:, :pad_w] = pad_h
        height_field_raw[:, -pad_w:] = pad_h
        height_field_raw[:pad_w, :] = pad_h
        height_field_raw[-pad_w:, :] = pad_h

    # Origin: 1 m above ground, centered in X and at half of platform in Y
    origin = np.array([0.5 * cfg.size[0], cfg.platform_len / 2.0, 1.0], dtype=np.float32)
    goals -= origin

    return np.rint(height_field_raw).astype(np.int16), origin, goals


####################################################################################
# Below are terrains for robustness testing and do not have goals associated with them
####################################################################################

@height_field_to_mesh
def discrete_obstacles_terrain(difficulty: float, cfg: hf_terrains_cfg.HfDiscreteObstaclesTerrainCfg):
    """Generate a terrain with randomly generated obstacles as pillars with positive and negative heights.

    The terrain is a flat platform at the center of the terrain with randomly generated obstacles as pillars
    with positive and negative height. The obstacles are randomly generated cuboids with a random width and
    height. They are placed randomly on the terrain with a minimum distance of :obj:`cfg.platform_width`
    from the center of the terrain.

    .. image:: ../../_static/terrains/height_field/discrete_obstacles_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    obs_height = cfg.obstacle_height_range[0] + difficulty * (
        cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0]
    )

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- obstacles
    obs_height = int(obs_height / cfg.vertical_scale)
    obs_width_min = int(cfg.obstacle_width_range[0] / cfg.horizontal_scale)
    obs_width_max = int(cfg.obstacle_width_range[1] / cfg.horizontal_scale)
    # -- center of the terrain
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    # create discrete ranges for the obstacles
    # -- shape
    obs_width_range = np.arange(obs_width_min, obs_width_max, 4)
    obs_length_range = np.arange(obs_width_min, obs_width_max, 4)
    # -- position
    obs_x_range = np.arange(0, width_pixels, 4)
    obs_y_range = np.arange(0, length_pixels, 4)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    # generate the obstacles
    for _ in range(cfg.num_obstacles):
        # sample size
        if cfg.obstacle_height_mode == "choice":
            height = np.random.choice([-obs_height, -obs_height // 2, obs_height // 2, obs_height])
        elif cfg.obstacle_height_mode == "fixed":
            height = obs_height
        else:
            raise ValueError(f"Unknown obstacle height mode '{cfg.obstacle_height_mode}'. Must be 'choice' or 'fixed'.")
        width = int(np.random.choice(obs_width_range))
        length = int(np.random.choice(obs_length_range))
        # sample position
        x_start = int(np.random.choice(obs_x_range))
        y_start = int(np.random.choice(obs_y_range))
        # clip start position to the terrain
        if x_start + width > width_pixels:
            x_start = width_pixels - width
        if y_start + length > length_pixels:
            y_start = length_pixels - length
        # add to terrain
        hf_raw[x_start : x_start + width, y_start : y_start + length] = height
    # clip the terrain to the platform
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    # round off the heights to the nearest vertical step

    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0], dtype=np.float32)
    
    return np.rint(hf_raw).astype(np.int16), origin, None


@height_field_to_mesh
def pyramid_sloped_terrain(difficulty: float, cfg: hf_terrains_cfg.HfPyramidSlopedTerrainCfg):
    """Generate a terrain with a truncated pyramid structure.

    The terrain is a pyramid-shaped sloped surface with a slope of :obj:`slope` that trims into a flat platform
    at the center. The slope is defined as the ratio of the height change along the x axis to the width along the
    x axis. For example, a slope of 1.0 means that the height changes by 1 unit for every 1 unit of width.

    If the :obj:`cfg.inverted` flag is set to :obj:`True`, the terrain is inverted such that
    the platform is at the bottom.

    .. image:: ../../_static/terrains/height_field/pyramid_sloped_terrain.jpg
       :width: 40%

    .. image:: ../../_static/terrains/height_field/inverted_pyramid_sloped_terrain.jpg
       :width: 40%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    if cfg.inverted:
        slope = -cfg.slope_range[0] - difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    else:
        slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])

    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- height
    # we want the height to be 1/2 of the width since the terrain is a pyramid
    height_max = int(slope * cfg.size[0] / 2 / cfg.vertical_scale)
    # -- center of the terrain
    center_x = int(width_pixels / 2)
    center_y = int(length_pixels / 2)

    # create a meshgrid of the terrain
    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)
    # offset the meshgrid to the center of the terrain
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y
    # reshape the meshgrid to be 2D
    xx = xx.reshape(width_pixels, 1)
    yy = yy.reshape(1, length_pixels)
    # create a sloped surface
    hf_raw = np.zeros((width_pixels, length_pixels))
    hf_raw = height_max * xx * yy

    # create a flat platform at the center of the terrain
    platform_width = int(cfg.platform_width / cfg.horizontal_scale / 2)
    # get the height of the platform at the corner of the platform
    x_pf = width_pixels // 2 - platform_width
    y_pf = length_pixels // 2 - platform_width
    z_pf = hf_raw[x_pf, y_pf]
    hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))

    origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, z_pf * cfg.vertical_scale], dtype=np.float32)

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16), origin, None


