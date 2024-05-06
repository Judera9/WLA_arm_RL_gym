# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn


class Encoder(nn.Module):
    is_recurrent = False
    is_sequence = True

    def __init__(
        self,
        num_critic_obs,
        num_obs_history,
        encoder_output_dim,
        encoder_type="teacher",
        encoder_hidden_dims=[256, 128],
        activation="elu",
        encoder_detach=False,
        orthogonal_init=False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticSequence.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(Encoder, self).__init__()
        if num_critic_obs == num_obs_history:
            raise ValueError(
                "In the encoder, num_critic_obs == num_obs_history, which is ambiguous"
            )
        self.encoder_type = encoder_type
        self.encoder_detach = encoder_detach
        self.orthogonal_init = orthogonal_init

        activation = get_activation(activation)

        # Encoder
        encoder_layers = []
        if self.encoder_type == "teacher":
            encoder_layers.append(nn.Linear(num_critic_obs, encoder_hidden_dims[0]))
        else:
            encoder_layers.append(nn.Linear(num_obs_history, encoder_hidden_dims[0]))
        if self.orthogonal_init:
            torch.nn.init.orthogonal_(encoder_layers[-1].weight, np.sqrt(2))
        encoder_layers.append(activation)
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1:
                encoder_layers.append(
                    nn.Linear(encoder_hidden_dims[l], encoder_output_dim)
                )
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(encoder_layers[-1].weight, 0.01)
                    torch.nn.init.constant_(encoder_layers[-1].bias, 0.0)
            else:
                encoder_layers.append(
                    nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1])
                )
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(encoder_layers[-1].weight, np.sqrt(2))
                    torch.nn.init.constant_(encoder_layers[-1].bias, 0.0)
                encoder_layers.append(activation)
                # actor_layers.append(torch.nn.LayerNorm(actor_hidden_dims[l + 1]))
        self.encoder = nn.Sequential(*encoder_layers)

        if self.encoder_type == "teacher":
            print(f"Encoder teacher: {self.encoder}")
        else:
            print(f"Encoder student: {self.encoder}")

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def get_encoder_output(self, **kwargs):
        if self.encoder_out is None:
            raise ValueError("Encoder output is None. Run encode first.")

        return self.encoder_out

    # def act_inference(self, observations, observation_history):
    #     self.encoder_out = self.encoder(observation_history)
    #     actions_mean = self.actor(torch.cat((observations, self.encoder_out), dim=-1))
    #     return actions_mean, self.encoder_out

    def encode(self, observation, **kwargs):
        # print(observation.device)
        self.encoder_out = self.encoder(observation)

        return self.encoder_out


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
