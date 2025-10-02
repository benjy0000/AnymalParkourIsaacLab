# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class ElevationMapEncoderCNN(nn.Module):
    
    def __init__(self, input_dims, output_dim):
        super().__init__()

        self.input_dims = input_dims
        self.cnn_output_dim = (input_dims[0] // 8) * (input_dims[1] // 8) * 32

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(self.cnn_output_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, 1, *self.input_dims)

        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
   
        x = x.view(-1, self.cnn_output_dim)
   
        x = self.fc(x)
        return x


class StateHistoryEncoder(nn.Module):

    def __init__(self, input_dims, output_dim):
        super().__init__()

        self.history_buffer_dim = input_dims  # (history_length, n_proprio)

        self.lin1 = nn.Linear(input_dims[1], 30)
        self.conv1 = nn.Conv1d(in_channels=30, out_channels=20, kernel_size=4, stride=2)
        self.conv2 = nn.Conv1d(in_channels=20, out_channels=10, kernel_size=2, stride=1)
        self.lin2 = nn.Linear(10 * 3, output_dim)

    def forward(self, x):
        
        x = x.view(-1, *self.history_buffer_dim)  # (nd, td, n_proprio)
        x = nn.functional.elu(self.lin1(x))  # (nd, td, 30)
        x = x.permute(0, 2, 1)  # (nd, 30, td)
        x = nn.functional.elu(self.conv1(x))  # (nd, 20, 4)
        x = nn.functional.elu(self.conv2(x))  # (nd, 10, 3)
        x = nn.Flatten()(x)  # (nd, 10 * 3)
        x = self.lin2(x)  # (nd, output_size)
        return x
    

class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        elevation_input_dims=(31, 34),
        elevation_output_dim=32,
        history_buffer_dim=(10, 51),
        history_output_dim=20,
        **kwargs,
    ):  
        elevation_input_dim = elevation_input_dims[0] * elevation_input_dims[1]
        history_input_dim = history_buffer_dim[0] * history_buffer_dim[1]

        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)
      
        class Policy(nn.Module):

            def __init__(self):
                super().__init__()

                self.elevation_encoder = ElevationMapEncoderCNN(input_dims=elevation_input_dims,
                                                                output_dim=elevation_output_dim)
                self.history_encoder = StateHistoryEncoder(input_dims=history_buffer_dim,
                                                           output_dim=history_output_dim)
                mlp_input_dim_a = (num_actor_obs
                                   - elevation_input_dim - history_input_dim
                                   + elevation_output_dim + history_output_dim)

                actor_layers = []
                actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
                actor_layers.append(activation)
                for layer_index in range(len(actor_hidden_dims) - 1):
                    actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                    actor_layers.append(activation)
                actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
                self.mlp = nn.Sequential(*actor_layers)

            def forward(self, x):
                elevation_features = self.elevation_encoder(x[:, -(elevation_input_dim + history_input_dim):-history_input_dim])
                history_features = self.history_encoder(x[:, -history_input_dim:])
                x = torch.cat([x[:, :-(elevation_input_dim + history_input_dim)],
                               elevation_features,
                               history_features], dim=1)
                x = self.mlp(x)
                return x

        self.actor: Policy = Policy()

        class ValueFunction(nn.Module):

            def __init__(self):
                super().__init__()

                self.elevation_encoder = ElevationMapEncoderCNN(input_dims=elevation_input_dims,
                                                                output_dim=elevation_output_dim)
                self.history_encoder = StateHistoryEncoder(input_dims=history_buffer_dim,
                                                           output_dim=history_output_dim)
                mlp_input_dim_c = (num_critic_obs
                                   - elevation_input_dim - history_input_dim
                                   + elevation_output_dim + history_output_dim)

                critic_layers = []
                critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
                critic_layers.append(activation)
                for layer_index in range(len(critic_hidden_dims) - 1):
                    critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                    critic_layers.append(activation)
                critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
                self.mlp = nn.Sequential(*critic_layers)

            def forward(self, x):
                elevation_features = self.elevation_encoder(x[:, -(elevation_input_dim + history_input_dim):-history_input_dim])
                history_features = self.history_encoder(x[:, -history_input_dim:])
                x = torch.cat([x[:, :-(elevation_input_dim + history_input_dim)],
                               elevation_features,
                               history_features], dim=1)
                x = self.mlp(x)
                return x
            
        self.critic: ValueFunction = ValueFunction()

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            # prevent exploding std
            self.std.data = torch.clamp(self.std.data, min=0, max=1.0)
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
