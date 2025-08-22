from isaaclab.utils import configclass

from .rough_env_cfg import AnymalDRoughEnvCfg
from ...mdp import RandomPathCommandCfg, RandomPathCommand


@configclass
class AnymalDFlatEnvCfg(AnymalDRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.base_orientation.weight = -1.0
        self.rewards.trotting.weight = 0.1
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # adjust starting point
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.6)
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        self.commands.target_points = RandomPathCommandCfg(
            class_type=RandomPathCommand,
            asset_name="robot",
            resampling_time_range=(30.0, 30.0),
            target_radius_range=(5.0, 10.0),
            rel_standing_env=0.0,
        )

        self.terminations.reached_finish = None


class AnymalDFlatEnvCfg_PLAY(AnymalDFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
