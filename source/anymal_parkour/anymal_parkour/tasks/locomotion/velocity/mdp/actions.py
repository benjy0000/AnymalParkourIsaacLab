from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions import JointEffortAction, JointEffortActionCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ParkourJointEffortAction(JointEffortAction):
    """
    Class that exactly recreates the action implementation from anymal_parkour
    Clipping occurs before the action is scaled and default joint positions are
    added. This gives:

    control_action = scale * clip(action) + default_pos
    lstm_input = [control_action - joint_pos, joint_vel]
    """

    def __init__(self, cfg: JointEffortActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def process_actions(self, actions):
        # store the raw actions
        self._raw_actions[:] = actions
        # clip the actions
        if self.cfg.clip is not None:
            self._raw_actions = torch.clamp(
                self._raw_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
