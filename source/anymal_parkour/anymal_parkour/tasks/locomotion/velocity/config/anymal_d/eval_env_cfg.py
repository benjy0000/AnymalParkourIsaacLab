from isaaclab.utils import configclass
from anymal_parkour.terrains.parkour_terrain_generator import ParkourTerrainGeneratorCfg
from anymal_parkour.terrains import terrain_gen

from .rough_env_cfg import AnymalDRoughEnvCfg


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
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        # create a custom terrain generator config for eval
        NUM_GOALS = 8

        EVAL_TERRAIN_CFG = ParkourTerrainGeneratorCfg(
            size=(4.0, 20.0),
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
                "gaps": terrain_gen.MeshGapsTerrainCfg(
                    proportion=1.0,
                    num_goals=NUM_GOALS,
                    gap_length_range=(0.1, 1.0),
                    stone_x_offset_range=(0.0, 0.15),
                    stone_length=1.2
                )
            }
        )
        self.scene.terrain.terrain_generator = EVAL_TERRAIN_CFG
      

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None