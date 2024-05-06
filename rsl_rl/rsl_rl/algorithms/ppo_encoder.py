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

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic, Encoder
from rsl_rl.storage import RolloutStorage, RolloutStorage_encoder


class PPO_encoder:
    actor_critic: ActorCritic
    encoder: Encoder

    def __init__(
        self,
        actor_critic,
        encoder,
        teacher_init_encoder,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        extra_loss_coef=0.0,
        entropy_coef=0.01,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        early_stop=False,
        anneal_lr=False,
        device="cpu",
    ):
        self.device = device
        self.desired_kl = desired_kl
        self.early_stop = early_stop
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr
        self.teacher = True if encoder.encoder_type == "teacher" else False

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        if self.teacher:
            self.teacher_encoder: Encoder = encoder
            self.student_encoder: Encoder = None
            self.teacher_encoder.to(self.device)
            # self.student_encoder.to(self.device)
        else:
            self.teacher_encoder: Encoder = teacher_init_encoder
            self.student_encoder: Encoder = encoder
            self.teacher_encoder.to(self.device)
            self.student_encoder.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(
            [
                {"params": self.actor_critic.parameters()},
                {"params": self.teacher_encoder.parameters()},
                # {"params": self.student_encoder.parameters()},
            ],
            lr=learning_rate,
        )
        if self.student_encoder is not None:
            self.student_encoder_optimizer = optim.Adam(
                self.student_encoder.parameters(), lr=1e-4
            )

        self.transition = RolloutStorage_encoder.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.extra_loss_coef = extra_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        obs_history_shape,
        encoder_out_shape,
        action_shape,
    ):
        self.storage = RolloutStorage_encoder(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            obs_history_shape,
            encoder_out_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, obs_history, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        else:
            if self.student_encoder is not None:
                student_encoder_out = self.student_encoder.encode(obs_history)
                teacher_encoder_out = self.teacher_encoder.encode(critic_obs)
                tot_obs = torch.cat((obs, student_encoder_out), dim=-1)
                self.transition.teacher_encoder_out = teacher_encoder_out.detach()
                self.transition.student_encoder_out = student_encoder_out.detach()
            else:
                teacher_encoder_out = self.teacher_encoder.encode(critic_obs)
                tot_obs = torch.cat((obs, teacher_encoder_out), dim=-1)
                self.transition.teacher_encoder_out = teacher_encoder_out.detach()

        self.transition.actions = self.actor_critic.act(tot_obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.observation_history = obs_history
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        num_updates = 0
        mean_value_loss = 0
        mean_extra_loss = 0
        mean_surrogate_loss = 0
        mean_kl = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        for (
            obs_batch,
            critic_obs_batch,
            obs_history_batch,
            student_encoder_out_batch,
            teacher_encoder_out_batch,
            tot_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            if self.teacher:
                teacher_encoder_out_batch = self.teacher_encoder.encode(
                    critic_obs_batch
                )
                tot_obs_batch = torch.cat(
                    (obs_batch, teacher_encoder_out_batch), dim=-1
                )
                self.actor_critic.act(tot_obs_batch)
            else:
                self.actor_critic.act(tot_obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            kl_mean = torch.tensor(0, device=self.device, requires_grad=False)
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (
                        torch.square(old_sigma_batch)
                        + torch.square(old_mu_batch - mu_batch)
                    )
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                # the approximation provides less variance than more standard log(q/p)
                # logratio = actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
                # ratio = logratio.exp()
                # approx_kl = ((ratio - 1) - logratio).mean()
                # kl_mean = approx_kl
                # print("kl_mean:",kl_mean.item()," approx_kl:",approx_kl.item())

            # KL
            if self.desired_kl != None and self.schedule == "adaptive":
                with torch.inference_mode():
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            if self.desired_kl != None and self.early_stop:
                if kl_mean > self.desired_kl * 1.5:
                    print("early stop, num_updates =", num_updates)
                    break

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            if self.student_encoder is not None:
                student_encoder_out_batch = self.student_encoder.encode(
                    obs_history_batch
                )
                extra_loss = (
                    (
                        student_encoder_out_batch[:, 0:3]
                        - teacher_encoder_out_batch[:, 0:3].detach()
                    )
                    .pow(2)
                    .mean()
                )
            else:
                extra_loss = torch.zeros_like(value_loss)

            entropy_batch_mean = entropy_batch.mean()
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                + self.extra_loss_coef * extra_loss
                - self.entropy_coef * entropy_batch_mean
            )

            if self.anneal_lr:
                frac = 1.0 - num_updates / (
                    self.num_learning_epochs * self.num_mini_batches
                )
                self.optimizer.param_groups[0]["lr"] = frac * self.learning_rate

            # Gradient step
            # params_before_training = [
            #     param.clone() for param in self.teacher_encoder.parameters()
            # ]

            self.optimizer.zero_grad()
            if self.student_encoder is not None:
                self.student_encoder_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(
                self.teacher_encoder.parameters(), self.max_grad_norm
            )
            if self.student_encoder is not None:
                nn.utils.clip_grad_norm_(
                    self.student_encoder.parameters(), self.max_grad_norm
                )

            self.optimizer.step()

            if self.student_encoder is not None:
                self.student_encoder_optimizer.step()

            # params_after_training = [
            #     param.clone() for param in self.teacher_encoder.parameters()
            # ]

            # for i in range(len(params_before_training)):
            #     if torch.equal(params_before_training[i], params_after_training[i]):
            #         continue
            #     else:
            #         print("Encoder parameters have changed during training. ")

            num_updates += 1
            mean_value_loss += value_loss.item()
            mean_extra_loss += extra_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_kl += kl_mean.item()

        # num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_extra_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_kl /= num_updates
        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_extra_loss,
            mean_kl,
        )
