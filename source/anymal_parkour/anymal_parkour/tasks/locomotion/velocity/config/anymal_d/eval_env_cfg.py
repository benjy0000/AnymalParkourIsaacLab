from isaaclab.utils import configclass
from anymal_parkour.terrains.parkour_terrain_generator import ParkourTerrainGeneratorCfg
from anymal_parkour.terrains import terrain_gen
from anymal_parkour.terrains import hf_terrain_gen
from isaaclab.managers import EventTermCfg as EventTerm
from anymal_parkour.tasks.locomotion.velocity import mdp

from .rough_env_cfg import AnymalDRoughEnvCfg

NUM_GOALS = 8
NOISE_RANGE = (0.02, 0.02)

EVAL_TERRAIN_CFG = ParkourTerrainGeneratorCfg(
    size=(4.0, 20.0),
    difficulty_range=(0.2, 1.0),
    border_width=20.0,
    num_rows=20,
    num_cols=15,  # Lots of rows and columns so that representative distribution of difficulties is captured
    num_goals=NUM_GOALS,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # NOTE: noise range should be = NOISE_RANGE for eval
        "large_steps": terrain_gen.MeshLargeStepsTerrainCfg(
            proportion=1.0,
            num_goals=NUM_GOALS,
            step_height_range=(0.1, 0.8),
            step_length_range=(0.5, 1.5),
            step_width_range=(1.4, 2.0),
            step_mismatch_range=(-0.4, 0.4),
            noise_range=NOISE_RANGE
        ),
    },
)

# NOTE: noise range is set correctly by default as not using this for training
# Difficulty does not affect this
EVAL_BARKOUR_TERRAIN_CFG = ParkourTerrainGeneratorCfg(
    size=(20.0, 20.0),
    difficulty_range=(0.7, 0.7),
    border_width=20.0,
    num_rows=3,
    num_cols=3,
    num_goals=NUM_GOALS,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "barkour": hf_terrain_gen.HfBarkourTerrainCfg()
    },
)

@configclass
class AnymalDRoughEvalCfg(AnymalDRoughEnvCfg):
    """Evaluation configuration for the AnymalD rough terrain environment.
       Use EVAL_TERRAIN_CFG to select terrain to be evaluated and difficulty
       by copying in terrain cfg from PARKOUR_TERRAIN_CFG and adjusting
       difficulty range appropriately."""

    def __post_init__(self):
  
        # post init of parent
        super().__post_init__()

        # make a smaller scene for evaluation
        self.scene.num_envs = 1000
        self.scene.env_spacing = 2.5

        # Constrain speeds of robots
        self.commands.target_speed.target_speed_range = (0.4, 0.8)

        # make the episode length long enough to ensure the robot can traverse the terrain
        self.episode_length_s = 60

        # create a custom terrain generator config for eval
        self.scene.terrain.terrain_generator = EVAL_TERRAIN_CFG
      
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        self.scene.terrain.terrain_generator.curriculum = False

        # set initial orientation and velocity
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot.init_state.lin_vel = (0.0, 0.0, 0.0)
        self.scene.robot.init_state.ang_vel = (0.0, 0.0, 0.0)
        self.events.reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (0, 0), "y": (0, 0), "yaw": (0, 0)},
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            },
        )
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None


class AnymalDRoughEvalBarkourCfg(AnymalDRoughEvalCfg):
    """Specific evaluation configuration for the AnymalD rough terrain environment with barkour.
       Computes the barkour score for a model."""
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Initialise the barkour terrain
        self.scene.terrain.terrain_generator = EVAL_BARKOUR_TERRAIN_CFG
