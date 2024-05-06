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

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import *
from legged_gym.utils.helpers import class_to_dict

# from ..base.legged_robot_config import LeggedRobotCfg
from .wheel_flat_config import WheelFlatCfg


class WheelRobot(BaseTask):
    def __init__(
        self, cfg: WheelFlatCfg, sim_params, physics_engine, sim_device, headless
    ):
        """Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.headless = headless
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # self.actions = torch.zeros_like(self.actions)
        if self.cfg.env.use_lp_filter:
            self.lp_now_actions = (
                self.lp_alpha * self.actions
                + (1 - self.lp_alpha) * self.lp_last_actions
            )
            self.lp_last_actions = self.lp_now_actions
        else:
            self.lp_now_actions = self.actions

        # step physics and render each frame
        self.render()
        self.pre_physics_step()
        # print(f"action is {self.actions}")
        for _ in range(self.cfg.control.decimation):
            self.action_fifo = torch.cat(
                (self.lp_now_actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1
            )
            self.envs_steps_buf += 1
            # self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs), self.action_delay_idx, :]
            ).view(self.torques.shape)
            # print(f"self.torques: {self.torques}")

            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            if self.cfg.domain_rand.push_Ee:
                self._push_Ee()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
            # self.compute_dof_acc()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.obs_history_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    # use difference to compute dof_vel
    def compute_dof_vel(self):
        self.diff_pos = self.dof_pos - self.last_dof_pos
        self.dof_pos_dot = self.diff_pos / self.sim_params.dt

        if self.cfg.env.dof_vel_use_pos_diff:
            self.dof_vel = self.dof_pos_dot

        self.last_dof_pos[:] = self.dof_pos[:]

    # use difference to compute dof_acc
    def compute_dof_acc(self):
        self.diff_vel = self.dof_vel - self.last_dof_vel
        self.dof_vel_dot = self.diff_vel / self.sim_params.dt

        self.dof_acc = self.dof_vel_dot

        self.last_dof_vel[:] = self.dof_vel[:]

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf += 1
        self.now_time += self.dt
        self.error_desire -= self.delta_error
        # self.common_step_counter += 1
        # self.curriculum_step_counter += 1
        # print(f"dof_vel is {self.dof_vel}")
        # prepare quantities
        self.base_quat[:] = standardize_quaternion(
            change_quat_scalar_first(self.root_states[:, 3:7])
        )
        self.base_quat_first = self.base_quat[:]
        self.base_position = self.root_states[:, :3]
        if self.cfg.env.base_lin_vel_use_pos_diff:
            self.base_lin_vel = (self.base_position - self.last_base_position) / self.dt
            self.base_lin_vel[:] = quaternion_apply_inverse(
                self.base_quat, self.base_lin_vel
            )
        else:
            self.base_lin_vel[:] = quaternion_apply_inverse(
                self.base_quat, self.root_states[:, 7:10]
            )
        self.base_ang_vel[:] = quaternion_apply_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        # compute heading
        self.heading = quaternion_apply(self.base_quat_first, self.forward_vec)
        # self.heading_xy[:, :2] = self.heading[:, :2]
        # self.heading_xy = self.heading_xy / (
        #     torch.norm(self.heading_xy, p=2, dim=1).unsqueeze(1) + 1e-6
        # )
        # print(f"self.forward_vec is {self.forward_vec}")
        self.projected_gravity[:] = quaternion_apply_inverse(
            self.base_quat, self.gravity_vec
        )
        self.Ee_state = self.rigid_body_state[:, self.Ee_index, :]
        self.Ee_state = self.Ee_state.squeeze(1)
        self.Ee_pos[:] = self.Ee_state[:, 0:3]
        self.Ee_orient[:] = standardize_quaternion(
            change_quat_scalar_first(self.Ee_state[:, 3:7])
        )
        # if (self.Ee_state[:, 6] < -0.2).any():
        #     print("real part can be negative!")
        # if (self.Ee_state[:, 3] < -0.2).any():
        #     print("certified! real part can be negative!")
        #     print(torch.norm(self.Ee_state[:, 3:7], dim=1).mean())

        # self.compute_Ee_vel()
        self.Ee_lin_vel[:] = quaternion_apply_inverse(
            self.base_quat, (self.Ee_pos - self.last_Ee_pos) / self.dt
        )
        self.last_Ee_pos[:] = self.Ee_pos[:]

        self.wheel_vel = (
            self.dof_pos[:, self.wheel_j_index] - self.last_wheel_pos[:]
        ) / self.dt
        self.last_wheel_pos[:] = self.dof_pos[:, self.wheel_j_index]

        # self.bt_distance = torch.norm(
        #     (self.base_target_pos_xy[:, :2] - self.base_position[:, :2]), p=2, dim=1
        # )

        # self.loco_phase &= self.bt_distance > self.cfg.commands.ok_radius
        # self.generate_Ee_lin_vel_target()  # generate Ee_lin_vel_target vectors

        self._post_physics_step_callback()  # resampling commands

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # print(f"env_ids is {env_ids}")
        self.reset_idx(env_ids)

        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:, :, 1] = self.last_actions[:, :, 0]
        self.last_actions[:, :, 0] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_base_position[:] = self.base_position[:]
        # self.last_root_vel[:] = self.root_states[:, 7:13]

        # draw Ee_target_point
        if not self.headless and self.enable_viewer_sync:
            self._draw_Ee_target_point()

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    # def generate_Ee_lin_vel_target(self):
    #     self.Ee_pos_error[self.loco_phase] = self.bt_distance[self.loco_phase]
    #     self.Ee_pos_error[~self.loco_phase] = torch.norm(
    #         self.commands[~self.loco_phase, :3] - self.Ee_pos[~self.loco_phase],
    #         p=2,
    #         dim=1,
    #     )
    #     ratio = (5 * torch.square(self.Ee_pos_error.unsqueeze(1))).clip(max=1.0)
    #     temp = self.base_target_pos_xy - self.base_position
    #     temp[:, 2] = 0
    #     self.Ee_lin_vel_target[self.loco_phase] = (
    #         temp[self.loco_phase]
    #         * ratio[self.loco_phase]
    #         / self.Ee_pos_error[self.loco_phase].unsqueeze(1)
    #     )
    #     self.Ee_lin_vel_target[~self.loco_phase] = (
    #         (self.commands[~self.loco_phase, :3] - self.Ee_pos[~self.loco_phase])
    #         / self.Ee_pos_error[~self.loco_phase].unsqueeze(1)
    #         * ratio[~self.loco_phase]
    #     )
    #     self.Ee_lin_vel_target = quaternion_apply_inverse(
    #         self.base_quat, self.Ee_lin_vel_target
    #     )
    #     # print(f"self.Ee_lin_vel_target is {self.Ee_lin_vel_target}")

    # def compute_Ee_vel(self):
    #     if self.common_step_counter % 4 == 0:
    #         self.Ee_ang_vel[:] = quaternion_apply_inverse(
    #             self.base_quat, self.Ee_ang_vel_sum / 3
    #         ).clip(max=10)
    #         self.Ee_ang_vel_sum[:] = 0
    #         self.Ee_ang_vel_sum += self.Ee_state[:, 10:13]
    #         self.common_step_counter = 1
    #     else:
    #         self.Ee_ang_vel_sum += self.Ee_state[:, 10:13]

    def check_termination(self):
        """Check if environments need to be reset"""
        # for i in range(self.num_envs):
        #     contacts_info = torch.tensor(self.gym.get_env_rigid_contacts(self.envs[i]))
        #     print(f"contacts_info is {contacts_info}")

        self.reset_buf = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            )
            > 10.0,
            dim=1,
        )

        # self.reset_buf |= (self.base_position[:, 2] < 0.15)

        self.time_out_buf = (
            self.episode_length_buf > self.max_episode_length
        )  # no terminal reward for time-outs

        # check fall_down by comparing the projected gravity with the actual gravity, calculate their cosine similarity
        self.fall_down_buf = torch.any(
            torch.acos(
                torch.cosine_similarity(self.projected_gravity, self.gravity_vec, dim=1)
            ).unsqueeze(1)
            > 75.0 / 180.0 * 3.14159,  # i.e. 70 degrees tilted, denoted it as fall_down
            dim=1,
        )
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.fall_down_buf

    def reset_idx(self, env_ids):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (
            self.curriculum_step_counter % self.max_episode_length == 0
        ):
            self.curriculum_step_counter = 1
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_net_contact_force_tensor(self.sim)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_dof_state_tensor(self.sim)

        # self.base_quat[:] = change_quat_scalar_last(
        #     standardize_quaternion(change_quat_scalar_first(self.root_states[:, 3:7]))
        # )
        # self.base_quat_first = change_quat_scalar_first(self.base_quat)
        # self.base_position[env_ids] = self.root_states[env_ids, :3]
        # self.bt_distance = torch.norm(
        #     (self.base_target_pos_xy[:, :2] - self.base_position[:, :2]), p=2, dim=1
        # )
        # # self.loco_phase &= self.bt_distance > self.cfg.commands.ok_radius

        # self.heading = quaternion_apply(self.base_quat_first, self.forward_vec)
        # self.heading_xy[:, :2] = self.heading[:, :2]
        # self.heading_xy = self.heading_xy / (
        #     torch.norm(self.heading_xy, p=2, dim=1).unsqueeze(1) + 1e-6
        # )
        # # print(f"self.forward_vec is {self.forward_vec}")
        # self.projected_gravity[env_ids] = quat_rotate_inverse(
        #     self.base_quat[env_ids], self.gravity_vec[env_ids]
        # )
        # self.Ee_state = self.rigid_body_state[:, self.Ee_index, :]
        # self.Ee_state = self.Ee_state.squeeze(1)

        # self.Ee_pos[env_ids] = self.Ee_state[env_ids, 0:3]
        # self.Ee_orient[env_ids] = standardize_quaternion(
        #     change_quat_scalar_first(self.Ee_state[env_ids, 3:7])
        # )

        # self.base_lin_vel[env_ids] = 0.0
        # self.base_ang_vel = quat_rotate_inverse(
        #     self.base_quat, self.root_states[:, 10:13]
        # )
        # self.Ee_lin_vel = torch.zeros_like(self.Ee_lin_vel)
        # self.Ee_ang_vel = torch.zeros_like(self.Ee_ang_vel)

        # reset buffers
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
        self.last_Ee_pos[env_ids] = self.Ee_state[env_ids, 0:3]
        self.last_wheel_pos[:] = self.dof_pos[:, self.wheel_j_index]
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0
        # self.reset_buf[env_ids] = 1
        self.action_fifo[env_ids, ...] = 0
        self.obs_history_fifo[env_ids, ...] = 0
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.terrain_levels.float()
            )
        if self.cfg.commands.curriculum:
            self.extras["episode"]["curriculum_level"] = self.command_ranges[
                "distance"
            ][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.compute_phase_scaling()
        self.rew_buf[:] = 0.0
        # compute basic rewards
        for i in range(len(self.basic_reward_functions)):
            name = self.basic_reward_names[i]
            rew = self.basic_reward_functions[i]() * self.basic_reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute locomotion rewards

        if self.cfg.basic_rewards.reward_type == "loco":
            for i in range(len(self.loco_reward_functions)):
                name = self.loco_reward_names[i]
                rew = (
                    self.loco_reward_functions[i]()
                    * self.loco_reward_scales[name]
                    # * self.locomotion_phase_scaling
                    # * (self.loco_phase)
                )
                self.rew_buf += rew
                self.episode_sums[name] += rew

        # compute manipulation rewards

        elif self.cfg.basic_rewards.reward_type == "mani":
            for i in range(len(self.man_reward_functions)):
                name = self.man_reward_names[i]
                if "tracking" in name and "cb" not in name:
                    rew = (
                        self.man_reward_functions[i]()
                        * self.man_reward_scales[name]
                        # * self.manipulation_phase_scaling
                        * self.Ee_tracking_scaling
                        # * (~self.loco_phase)
                    )
                else:
                    rew = (
                        self.man_reward_functions[i]()
                        * self.man_reward_scales[name]
                        # * self.manipulation_phase_scaling
                        # * self.heading_scaling
                    )

                self.rew_buf += rew
                self.episode_sums[name] += rew

        if self.cfg.basic_rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

        # add termination reward after clipping
        if "termination" in self.basic_reward_scales:
            rew = self._reward_termination() * self.basic_reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_phase_scaling(self):
        """Compute phase scaling"""
        # distance = torch.norm((self.commands[:, :3] - self.Ee_pos), dim=1)
        # sigmoid_z = self.cfg.sigmoid_phase_change.omega * (
        #     distance - self.cfg.sigmoid_phase_change.change_point
        # )
        # # print(f"sigmoid_z: {sigmoid_z}")
        # self.locomotion_phase_scaling = torch.sigmoid(sigmoid_z)
        # # print(f"self.locomotion_phase_scaling: {self.locomotion_phase_scaling}")
        # self.manipulation_phase_scaling = 1 - self.locomotion_phase_scaling

        # compute heading scaling
        ee_x = quaternion_apply(self.commands[:, 3:7], self.forward_vec)
        ee_x[:, 2] = 0.0
        ee_x = ee_x / (torch.norm(ee_x, p=2, dim=1).unsqueeze(1) + 1e-6)

        ee_y = find_orthogonal_vector(ee_x)
        coordinates = project_vector(
            (self.commands[:, :3] - self.base_position),
            ee_x,
            ee_y,
        )
        heading_x = self.heading.clone()
        heading_x[:, 2] = 0
        heading_x = heading_x / (torch.norm(heading_x, p=2, dim=1).unsqueeze(1) + 1e-6)

        self.Ee_base_error = (
            (torch.norm((heading_x - ee_x), p=2, dim=1) - 0.8).clip(min=0.0)
            - (coordinates[:, 0] - 0.3).clip(max=0.0)
            + (coordinates[:, 1].abs() - 0.4).clip(min=0.0)
        )

        Ee_base_error = 2 * self.Ee_base_error

        nominal_leg_error = 0.25 * torch.sum(
            torch.square(
                torch.index_select(
                    self.dof_pos,
                    1,
                    torch.tensor(self.leg_index, device=self.device),
                )
                - self.nominal_dof_pos[self.leg_index]
            ),
            dim=1,
        )
        base_height_error = (
            torch.abs(
                self.base_position[:, 2] - self.cfg.basic_rewards.base_height_target
            )
            / 0.25
        )
        orient_error = 0.5 * (
            torch.abs(self.projected_gravity[:, 0]) / 0.707
            + torch.abs(self.projected_gravity[:, 1]) / 0.34
        )

        self.tot_error = (
            Ee_base_error + nominal_leg_error + base_height_error + orient_error
        )

        self.Ee_tracking_scaling = torch.exp(-self.tot_error / 2) + 0.3

        # self.loco_phase &= self.Ee_base_error > self.cfg.commands.ok_radius

        # heading_sigmoid_z = 10 * (self.Ee_base_error - 0.34)
        # self.heading_scaling = torch.sigmoid(heading_sigmoid_z)
        # self.Ee_tracking_scaling = 1 - self.heading_scaling

        # print(f"Ee_tracking_scaling_mean is: {self.Ee_tracking_scaling.mean()}")

        # print(f"manipulation_phase_scaling: {self.manipulation_phase_scaling}")

    def compute_observations(self):
        """Computes observations and privileged observations"""

        # convert some terms into base frame
        Ee_pos_b = quaternion_apply_inverse(
            self.base_quat, (self.Ee_pos - self.base_position)
        )

        Ee_target_R = quaternion_to_matrix(self.commands[:, 3:7])

        Ee_R = torch.matmul(
            quaternion_to_matrix(self.Ee_orient), self.change_coordinates
        )

        Ee_orient_x_b = quaternion_apply_inverse(self.base_quat, Ee_R[:, :, 0])

        Ee_orient_y_b = quaternion_apply_inverse(self.base_quat, Ee_R[:, :, 1])

        Ee_target_x_b = quaternion_apply_inverse(self.base_quat, Ee_target_R[:, :, 0])

        Ee_target_y_b = quaternion_apply_inverse(self.base_quat, Ee_target_R[:, :, 1])

        Ee_command_pos_b = quaternion_apply_inverse(
            self.base_quat, (self.commands[:, :3] - self.base_position)
        )

        magnitude = torch.norm(Ee_command_pos_b, p=2, dim=1).unsqueeze(1)
        too_large = torch.any(magnitude > 5.0, dim=1)
        too_large = too_large.nonzero(as_tuple=False).flatten()
        Ee_command_pos_b[too_large, :] *= 5.0 / magnitude[too_large, :]

        whole_command = torch.cat(
            (
                Ee_command_pos_b,
                Ee_target_x_b,
                Ee_target_y_b,
            ),
            dim=-1,
        )

        self.obs_buf = torch.cat(
            (
                # self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                Ee_pos_b * self.obs_scales.Ee_pos,
                Ee_orient_x_b * self.obs_scales.base_orient,
                Ee_orient_y_b * self.obs_scales.base_orient,
                self.projected_gravity,
                whole_command * self.commands_scale,
                (
                    self.dof_pos[:, self.not_wheel]
                    - self.default_dof_pos[:, self.not_wheel]
                )
                * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                self.Ee_x_error_cumu.view(self.num_envs, 1)
                * self.obs_scales.cumu_error,
                self.Ee_y_error_cumu.view(self.num_envs, 1)
                * self.obs_scales.cumu_error,
                self.Ee_pos_error_cumu.view(self.num_envs, 1)
                * self.obs_scales.cumu_error,
                self.now_time.unsqueeze(1) * self.obs_scales.time,
            ),
            dim=-1,
        )
        # print(f"self.obs_buf.shape: {self.obs_buf.shape}")

        # print(self.contact_forces[:, self.feet_indices, :].flatten(start_dim=1))
        dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt
        if self.cfg.env.num_privileged_obs is not None:
            # if (self.contact_forces[:, self.feet_indices, :] > 500).any():
            # print("too big contact force!")
            # print(f"feet_indices is {self.feet_indices}")
            # print(
            #     f"the biggest contact force is {torch.max(self.contact_forces[:, self.feet_indices, :])}"
            # )
            self.privileged_obs_buf = torch.cat(
                (
                    self.base_lin_vel * self.obs_scales.lin_vel,
                    (
                        self.base_position[:, 2]
                        - self.cfg.basic_rewards.base_height_target
                    ).unsqueeze(1)
                    * self.obs_scales.height_measurements,
                    self.Ee_lin_vel * self.obs_scales.lin_vel,
                    self.Ee_ang_vel * self.obs_scales.ang_vel,
                    # self.distance_proj_xy * self.obs_scales.distance_proj_xy,
                    self.torques * self.obs_scales.torque,
                    dof_acc * self.obs_scales.dof_acc,
                    self.contact_forces[:, self.feet_indices, :].flatten(start_dim=1)
                    * self.obs_scales.contact_forces,
                    self.contact_forces[:, self.termination_contact_indices, :].flatten(
                        start_dim=1
                    )
                    * self.obs_scales.contact_forces,
                    self.obs_buf,
                    (self.base_mass - self.base_mass.mean()).view(self.num_envs, 1),
                    self.base_com,
                    self.default_dof_pos - self.raw_default_dof_pos,
                    (self.last_actions[:, :, 0]).flatten(start_dim=1),
                    self.contact_forces[:, self.penalised_contact_indices, :].flatten(
                        start_dim=1
                    )
                    * self.obs_scales.contact_forces,
                    self.friction_coeffs.view(self.num_envs, 1),
                    self.restitution_coeffs.view(self.num_envs, 1),
                    self.rigid_body_external_forces_Ee * self.obs_scales.ex_torque,
                ),
                dim=-1,
            )
            # print(f"self.privileged_obs_buf.shape: {self.privileged_obs_buf.shape}")

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

        self.obs_history_fifo = torch.cat(
            (self.obs_buf.unsqueeze(1), self.obs_history_fifo[:, :-1, :]), dim=1
        )
        self.obs_history_buf = self.obs_history_fifo.flatten(start_dim=1)

    # def chagne_to_base_frame(self, ee_quat, base_quat_first, change_coordinates):
    #     R_bs = quaternion_to_matrix(quaternion_invert(base_quat_first))
    #     R_se = quaternion_to_matrix(ee_quat)
    #     R_ee_star = change_coordinates
    #     return torch.matmul(R_bs, torch.matmul(R_se, R_ee_star))

    def create_sim(self):
        """Creates simulation, terrain and evironments"""
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ["heightfield", "trimesh"]:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]"
            )
        self._create_envs()

    def set_camera(self, position, lookat):
        """Set camera position and direction"""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(
                    friction_range[0],
                    friction_range[1],
                    (num_buckets, 1),
                    device=self.device,
                )
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id == 0:
                (
                    min_restitution,
                    max_restitution,
                ) = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (max_restitution - min_restitution)
                    + min_restitution
                )
            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dof,
                2,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.dof_vel_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.torque_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
            )
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = (
                    m - 0.5 * r * self.cfg.basic_rewards.soft_dof_pos_limit
                )
                self.dof_pos_limits[i, 1] = (
                    m + 0.5 * r * self.cfg.basic_rewards.soft_dof_pos_limit
                )
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            if env_id == 0:
                min_add_mass, max_add_mass = self.cfg.domain_rand.added_mass_range
                self.base_add_mass = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (max_add_mass - min_add_mass)
                    + min_add_mass
                )
                self.base_mass = props[0].mass + self.base_add_mass
            props[0].mass += self.base_add_mass[env_id]
        else:
            self.base_mass[:] = props[0].mass
        if self.cfg.domain_rand.randomize_base_com:
            if env_id == 0:
                com_x, com_y, com_z = self.cfg.domain_rand.rand_com_vec
                self.base_com[:, 0] = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (com_x * 2)
                    - com_x
                )
                self.base_com[:, 1] = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (com_y * 2)
                    - com_y
                )
                self.base_com[:, 2] = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (com_z * 2)
                    - com_z
                )
            props[0].com.x += self.base_com[env_id, 0]
            props[0].com.y += self.base_com[env_id, 1]
            props[0].com.z += self.base_com[env_id, 2]
        if self.cfg.domain_rand.randomize_inertia:
            for i in range(len(props)):
                low_bound, high_bound = self.cfg.domain_rand.randomize_inertia_range
                inertia_scale = np.random.uniform(low_bound, high_bound)
                props[i].mass *= inertia_scale
                props[i].inertia.x.x *= inertia_scale
                props[i].inertia.y.y *= inertia_scale
                props[i].inertia.z.z *= inertia_scale
        return props

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #

        env_ids = (
            (
                self.episode_length_buf
                % int(self.cfg.commands.resampling_time / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )

        if len(env_ids) == 0:
            return
        # print(f"self.dt is {self.dt}")  self.dt is 0.02
        self._resample_commands(env_ids)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        # print(f"env_ids.shape is : {env_ids.shape}")
        # print(f"self.Ee_state.shape is {self.Ee_state.shape}")
        angle = torch_rand_float(
            self.command_ranges["Ee_pos_angle"][0],
            self.command_ranges["Ee_pos_angle"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        ditance = torch_rand_float(
            self.command_ranges["distance"][0],
            self.command_ranges["distance"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.commands[env_ids, 0] = self.Ee_pos[env_ids, 0] + torch.cos(angle) * ditance
        self.commands[env_ids, 1] = self.Ee_pos[env_ids, 1] + torch.sin(angle) * ditance

        # print(
        #     f"torch.sin(angle) * self.cfg.commands.distance is {torch.sin(angle) * self.cfg.commands.distance}"
        # )
        # print(
        #     f"torch.cos(angle) * self.cfg.commands.distance is {torch.cos(angle) * self.cfg.commands.distance}"
        # )
        self.commands[env_ids, 2] = torch_rand_float(
            self.command_ranges["Ee_pos_lim_z"][0],
            self.command_ranges["Ee_pos_lim_z"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.commands[env_ids, 3:7] = random_quaternions(
            len(env_ids), device=self.device, dtype=torch.float
        )

        # self.commands[env_ids, 7] = torch_rand_float(
        #     self.command_ranges["lin_vel"][0],
        #     self.command_ranges["lin_vel"][1],
        #     (len(env_ids), 1),
        #     device=self.device,
        # ).squeeze(1)

        # self.commands[env_ids, 8] = torch_rand_float(
        #     self.command_ranges["ang_vel"][self.curriculum_level][0],
        #     self.command_ranges["ang_vel"][self.curriculum_level][1],
        #     (len(env_ids), 1),
        #     device=self.device,
        # ).squeeze(1)

        # self.Ee_x_target[env_ids, :2] = quaternion_to_matrix(self.commands[:, 3:7])[
        #     env_ids, 0:2, 0
        # ]
        # self.Ee_x_target = self.Ee_x_target / (
        #     torch.norm(self.Ee_x_target, p=2, dim=1).unsqueeze(1) + 1e-6
        # )

        # self.Ee_y_target = find_orthogonal_vector(self.Ee_x_target)
        # command_base = self.commands[:, :3].clone()
        # command_base[:, 2] = 0.0
        # self.base_target_pos_xy = (
        #     command_base
        #     + self.Ee_x_target * self.cfg.commands.target_x
        #     + self.Ee_y_target * self.cfg.commands.target_y
        # )

        self.distance_proj_xy[env_ids, :2] = (
            self.commands[env_ids, :2] - self.base_position[env_ids, :2]
        ) / (
            torch.norm(
                (self.commands[env_ids, :2] - self.base_position[env_ids, :2]),
                p=2,
                dim=1,
            ).unsqueeze(1)
            + 1e-6
        )

        Ee_target_R = quaternion_to_matrix(self.commands[:, 3:7])

        Ee_R = torch.matmul(
            quaternion_to_matrix(self.Ee_orient), self.change_coordinates
        )

        Ee_command_x = Ee_target_R[:, :, 0]
        Ee_x = Ee_R[:, :, 0]
        Ee_orient_error_x = torch.norm(Ee_command_x - Ee_x, p=2, dim=-1)

        Ee_command_y = Ee_target_R[:, :, 1]
        Ee_y = Ee_R[:, :, 1]
        Ee_orient_error_y = torch.norm(Ee_command_y - Ee_y, p=2, dim=-1)

        Ee_pos_error = torch.norm((self.commands[:, 0:3] - self.Ee_pos), p=2, dim=1)

        ee_x = quaternion_apply(self.commands[:, 3:7], self.forward_vec)
        ee_x[:, 2] = 0.0
        ee_x = ee_x / (torch.norm(ee_x, p=2, dim=1).unsqueeze(1) + 1e-6)

        ee_y = find_orthogonal_vector(ee_x)
        coordinates = project_vector(
            (self.commands[:, :3] - self.base_position),
            ee_x,
            ee_y,
        )
        heading_x = self.heading.clone()
        heading_x[:, 2] = 0
        heading_x = heading_x / (torch.norm(heading_x, p=2, dim=1).unsqueeze(1) + 1e-6)

        base_height_error = (
            torch.abs(
                self.base_position[:, 2] - self.cfg.basic_rewards.base_height_target
            )
            / 0.25
        )
        orient_error = 0.5 * (
            torch.abs(self.projected_gravity[:, 0]) / 0.707
            + torch.abs(self.projected_gravity[:, 1]) / 0.34
        )

        t_pos = Ee_pos_error / self.cfg.basic_rewards.displace_pos

        t_x = Ee_orient_error_x / self.cfg.basic_rewards.displace_xy

        t_y = Ee_orient_error_y / self.cfg.basic_rewards.displace_xy

        t_base_pos = (
            -(coordinates[:, 0] - 0.3).clip(max=0.0)
            + (coordinates[:, 1].abs() - 0.4).clip(min=0.0)
            + base_height_error
        ) / self.cfg.basic_rewards.displace_pos

        t_base_orient = (
            orient_error
            + (torch.norm((heading_x - ee_x), p=2, dim=1) - 0.8).clip(min=0.0)
        ) / self.cfg.basic_rewards.displace_xy

        max_value = torch.max(
            torch.max(torch.max(torch.max(t_pos, t_x), t_y), t_base_pos), t_base_orient
        )

        # print(f"max_value.mean() is {max_value[env_ids].mean()}")

        # self.reference_time[env_ids] = max_value[env_ids]
        error_desire = (
            # Ee_base_error
            # + base_height_error
            # + orient_error
            +Ee_orient_error_y
            + Ee_orient_error_x
            + Ee_pos_error
        )

        delta_error = error_desire / max_value * self.dt
        self.delta_error[env_ids] = delta_error[env_ids]
        self.error_desire[env_ids] = error_desire[env_ids]
        self.now_time[env_ids] = 0.0

        # self.loco_phase[env_ids] = True
        self.Ee_pos_error_cumu[env_ids] = 0.0
        self.Ee_x_error_cumu[env_ids] = 0.0
        self.Ee_y_error_cumu[env_ids] = 0.0
        # self.loco_phase[:] = False

        # print(self.commands[env_ids, :])

        # # set small commands to zero
        # self.commands[env_ids, :7] *= (
        #     torch.norm(self.commands[env_ids, :6], dim=1) > 0.6
        # ).unsqueeze(1)

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions
        actions_scaled[:, self.wheel_j_index] *= self.cfg.control.action_scale_vel
        actions_scaled[:, self.not_wheel] *= self.cfg.control.action_scale_pos

        # clip actions to be within the limits
        actions_scaled[:, self.not_wheel] = torch.clip(
            actions_scaled[:, self.not_wheel],
            self.dof_pos[:, self.not_wheel]
            - self.default_dof_pos[:, self.not_wheel]
            + (
                self.d_gains[:, self.not_wheel] * self.dof_vel[:, self.not_wheel]
                - self.cfg.basic_rewards.soft_torque_limit
                * self.torque_limits[self.not_wheel]
            )
            / self.p_gains[:, self.not_wheel],
            self.dof_pos[:, self.not_wheel]
            - self.default_dof_pos[:, self.not_wheel]
            + (
                self.d_gains[:, self.not_wheel] * self.dof_vel[:, self.not_wheel]
                + self.cfg.basic_rewards.soft_torque_limit
                * self.torque_limits[self.not_wheel]
            )
            / self.p_gains[:, self.not_wheel],
        )

        actions_scaled[:, self.wheel_j_index] = torch.clip(
            actions_scaled[:, self.wheel_j_index],
            -self.cfg.basic_rewards.soft_torque_limit
            * self.torque_limits[self.wheel_j_index]
            / self.d_gains[:, self.wheel_j_index]
            + self.dof_vel[:, self.wheel_j_index],
            self.cfg.basic_rewards.soft_torque_limit
            * self.torque_limits[self.wheel_j_index]
            / self.d_gains[:, self.wheel_j_index]
            + self.dof_vel[:, self.wheel_j_index],
        )

        # compute torque
        torques = torch.zeros_like(actions_scaled)
        torques[:, self.not_wheel] = (
            self.p_gains[:, self.not_wheel]
            * (
                actions_scaled[:, self.not_wheel]
                + self.default_dof_pos[:, self.not_wheel]
                - self.dof_pos[:, self.not_wheel]
            )
            - self.d_gains[:, self.not_wheel] * self.dof_vel[:, self.not_wheel]
        )
        torques[:, self.wheel_j_index] = self.d_gains[:, self.wheel_j_index] * (
            actions_scaled[:, self.wheel_j_index] - self.dof_vel[:, self.wheel_j_index]
        )

        # if (torques > 1.2 * self.torque_limits).any():
        #     print("User torque limit not working!")

        # torques[:, self.wheel_j_index] = (
        #     torch.ones_like(torques[:, self.wheel_j_index]) * 5
        # )

        # print(f"for NOT WHL  self.p_gains[:, i] is {self.p_gains[:, i]}")

        return torch.clip(
            torques * self.torques_scale, -self.torque_limits, self.torque_limits
        )
        # return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        # dof_arm_index = [i for i in range(len(self.dof_names)) if "arm" in self.dof_names[i]]
        self.dof_pos[env_ids] = self.init_dof_pos
        # self.dof_pos[env_ids,12:] = self.raw_default_dof_pos[env_ids,12:]
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        """Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1.0, 1.0, (len(env_ids), 2), device=self.device
            )  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device
        )  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    #  original version
    # def _push_robots(self):
    #     """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
    #     max_vel = self.cfg.domain_rand.max_push_vel_xy
    #     self.root_states[:, 7:9] = torch_rand_float(
    #         -max_vel, max_vel, (self.num_envs, 2), device=self.device
    #     )  # lin vel x/y
    #     self.gym.set_actor_root_state_tensor(
    #         self.sim, gymtorch.unwrap_tensor(self.root_states)
    #     )

    def _push_robots(self):
        """Random pushes the robots."""

        env_ids = (
            (
                self.envs_steps_buf
                % int(self.cfg.domain_rand.push_interval_s / self.sim_params.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )

        if len(env_ids) == 0:
            self.rigid_body_external_forces[:] = 0
            return

        max_push_force = (
            self.base_mass.mean().item()
            * self.cfg.domain_rand.max_push_vel_xy
            / self.sim_params.dt
        )
        self.rigid_body_external_forces[:] = 0

        self.rigid_body_external_forces_base[:] = torch_rand_float(
            -max_push_force, max_push_force, (self.num_envs, 3), device=self.device
        )

        self.rigid_body_external_forces[env_ids, 0, 0:3] = (
            self.rigid_body_external_forces_base[env_ids]
        )

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.rigid_body_external_forces),
            gymtorch.unwrap_tensor(self.rigid_body_external_torques),
            gymapi.ENV_SPACE,
        )

    def _push_Ee(self):
        """Random pushes the robots."""
        env_ids_Ee = (
            (
                self.envs_steps_buf
                % int(self.cfg.domain_rand.push_Ee_interval_s / self.sim_params.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )

        if len(env_ids_Ee) == 0:
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(self.rigid_body_external_forces),
                gymtorch.unwrap_tensor(self.rigid_body_external_torques),
                gymapi.ENV_SPACE,
            )
            return

        self.rigid_body_external_forces_Ee[env_ids_Ee] = torch_rand_float(
            -self.cfg.domain_rand.max_push_force_Ee,
            self.cfg.domain_rand.max_push_force_Ee,
            (len(env_ids_Ee), 3),
            device=self.device,
        )

        self.rigid_body_external_forces[env_ids_Ee, self.Ee_index, 0:3] = (
            self.rigid_body_external_forces_Ee[env_ids_Ee]
        )

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.rigid_body_external_forces),
            gymtorch.unwrap_tensor(self.rigid_body_external_torques),
            gymapi.ENV_SPACE,
        )

    def _update_terrain_curriculum(self, env_ids):
        """Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(
            self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1
        )
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (
            distance
            < torch.norm(self.commands[env_ids, :2], dim=1)
            * self.max_episode_length_s
            * 0.5
        ) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]

    def update_command_curriculum(self, env_ids):
        """Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        print(f"i am in!! but i shouldn't!")
        if (
            torch.mean(self.episode_sums["tracking_Ee_pos"][env_ids])
            / self.max_episode_length
            > 0.4 * self.man_reward_scales["tracking_Ee_pos"]
            and torch.mean(self.episode_sums["tracking_Ee_orient_x"][env_ids])
            / self.max_episode_length
            > 0.4 * self.man_reward_scales["tracking_Ee_orient_x"]
        ):
            self.command_ranges["distance"][1] = (
                self.command_ranges["distance"][1] + 0.5
            )

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel

        # noise_vec[6:9] = (
        #     noise_scales.Ee_lin_vel * noise_level * self.obs_scales.Ee_lin_vel
        # )
        # noise_vec[9:12] = (
        #     noise_scales.Ee_ang_vel * noise_level * self.obs_scales.Ee_ang_vel
        # )

        # noise_vec[3:6] = noise_scales.Ee_pos * noise_level * self.obs_scales.Ee_pos
        # noise_vec[6:12] = (
        #     noise_scales.Ee_orient * noise_level * self.obs_scales.Ee_orient
        # )

        noise_vec[3:6] = noise_scales.Ee_pos * noise_level * self.obs_scales.Ee_pos
        noise_vec[6:12] = (
            noise_scales.Ee_orient * noise_level * self.obs_scales.Ee_orient
        )

        noise_vec[12:15] = noise_scales.gravity * noise_level
        noise_vec[15:24] = 0.0  # commands

        noise_vec[24:42] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )  # add 6 for arm
        noise_vec[42:64] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )  # add 6 for arm
        noise_vec[64:86] = 0.0  # actions
        noise_vec[86:89] = 0.0  # cumu error
        noise_vec[89:90] = 0.0  # time

        # noise_vec[86:87] = 0.0  # locomotion_phase_scaling

        if self.cfg.terrain.measure_heights:
            noise_vec[89:276] = (
                noise_scales.height_measurements
                * noise_level
                * self.obs_scales.height_measurements
            )
        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        # get gym GPU state tensors
        # body_contacts = self.gym.get_env_rigid_contacts(self.envs)
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[
            ..., 0
        ]  # equal [:,:, 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.dof_acc = torch.zeros_like(self.dof_vel)
        self.diff_pos = torch.zeros_like(self.dof_pos)
        self.diff_vel = torch.zeros_like(self.dof_vel)
        self.base_quat = standardize_quaternion(
            change_quat_scalar_first(self.root_states[:, 3:7])
        )
        self.base_quat_first = self.base_quat

        # """this is used for change Ee_orient to proper coordinates,
        # #the proper form is that when arm's angels of joints are all zero, the end effector's orientation is align with base"""
        self.change_coordinates = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )

        self.curriculum_level = 0
        self.curriculum_count = 0
        self.lp_alpha = self.cfg.env.lp_alpha
        # self.loco_phase = torch.ones(
        #     self.num_envs, dtype=torch.bool, device=self.device
        # )

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies, -1
        )
        self.Ee_state = self.rigid_body_state[:, self.Ee_index, :]
        self.Ee_state = self.Ee_state.squeeze(1)
        self.Ee_pos = self.Ee_state[:, :3]
        self.last_Ee_pos = torch.zeros_like(self.Ee_pos)
        self.Ee_orient = torch.zeros(
            self.num_envs,
            4,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.Ee_orient[:] = standardize_quaternion(
            change_quat_scalar_first(self.Ee_state[:, 3:7])
        )

        self.Ee_lin_vel = quaternion_apply_inverse(
            self.base_quat, self.Ee_state[:, 7:10]
        )
        self.Ee_ang_vel = quaternion_apply_inverse(
            self.base_quat, self.Ee_state[:, 10:13]
        )
        # self.Ee_lin_vel_target = torch.zeros_like(self.Ee_lin_vel)
        # self.Ee_pos_error = torch.zeros(
        #     self.num_envs, dtype=torch.float, device=self.device
        # )
        self.Ee_pos_error_cumu = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.Ee_x_error_cumu = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.Ee_y_error_cumu = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        # self.reference_time = torch.zeros(
        #     self.num_envs, dtype=torch.float, device=self.device
        # ) + 10.0
        self.delta_error = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.error_desire = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.now_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.Ee_ang_vel_sum = torch.zeros_like(self.Ee_ang_vel)
        self.last_wheel_pos = torch.zeros_like(self.dof_pos[:, self.wheel_j_index])
        self.wheel_vel = torch.zeros_like(self.dof_vel[:, self.wheel_j_index])
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies, -1
        )
        # self.foot_positions = self.rigid_body_state.view(
        #     self.num_envs, self.num_bodies, 13
        # )[:, self.feet_indices, 0:3]
        # self.last_foot_positions = torch.zeros_like(self.foot_positions)
        # self.foot_velocities = torch.zeros_like(self.foot_positions)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 1
        self.curriculum_step_counter = 1
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.torques = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.torques_scale = torch.ones(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.p_gains = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.d_gains = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.lp_last_actions = torch.zeros_like(self.actions)

        self.last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        # self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.action_delay_idx = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )

        self.Ee_base_error = torch.zeros_like(self.action_delay_idx)
        self.tot_error = torch.zeros_like(self.action_delay_idx)

        delay_max = np.int64(
            np.ceil(self.cfg.domain_rand.delay_ms_range[1] / 1000 / self.sim_params.dt)
        )
        self.action_fifo = torch.zeros(
            (self.num_envs, delay_max, self.cfg.env.num_actions),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.obs_history_fifo = torch.zeros(
            (self.num_envs, self.cfg.env.history_len, self.num_obs),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # x vel, y vel, yaw vel, heading

        self.distance_proj_xy = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.Ee_x_target = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.heading = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.heading_xy = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        # self.commands_scale = torch.tensor(
        #     [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
        #     device=self.device,
        #     requires_grad=False,
        # )  # TODO change this

        self.commands_scale = torch.tensor(
            ([self.obs_scales.Ee_pos] * 3 + [self.obs_scales.Ee_orient] * 6),
            device=self.device,
            requires_grad=False,
        )  # TODO change this
        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.base_position = self.root_states[:, :3]
        self.last_base_position = self.base_position.clone()
        self.base_lin_vel = quaternion_apply_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel = quaternion_apply_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.rigid_body_external_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False
        )
        self.rigid_body_external_forces_Ee = torch.zeros(
            (self.num_envs, 3), device=self.device, requires_grad=False
        )

        self.rigid_body_external_forces_base = torch.zeros(
            (self.num_envs, 3), device=self.device, requires_grad=False
        )

        self.rigid_body_external_torques = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False
        )
        self.projected_gravity = quaternion_apply_inverse(
            self.base_quat, self.gravity_vec
        )
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.raw_default_dof_pos = torch.zeros(
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.init_dof_pos = torch.zeros(
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.nominal_dof_pos = torch.zeros(
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.default_dof_pos = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.static_arm_pos = torch.tensor(
            self.cfg.init_state.static_arm_pos,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        # print("---self.dof_names", self.dof_names)
        # print("---default_joint_angles", self.cfg.init_state.default_joint_angles)
        # print("---stiffness", self.cfg.control.stiffness)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            # print(f"self.dof_names is {self.dof_names}")
            angle = self.cfg.init_state.default_joint_angles[name]
            init_angle = self.cfg.init_state.init_joint_angles[name]
            nominal_angle = self.cfg.init_state.nominal_joint_angles[name]
            self.raw_default_dof_pos[i] = angle
            self.default_dof_pos[:, i] = angle
            self.nominal_dof_pos[i] = nominal_angle
            self.init_dof_pos[i] = init_angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[:, i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[:, i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[:, i] = 0.0
                self.d_gains[:, i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(
                        f"PD gain of joint {name} were not defined, setting them to zero"
                    )

        if self.cfg.domain_rand.randomize_Kp:
            (
                p_gains_scale_min,
                p_gains_scale_max,
            ) = self.cfg.domain_rand.randomize_Kp_range

            self.p_gains *= torch_rand_float(
                p_gains_scale_min,
                p_gains_scale_max,
                # (self.p_gains.shape[0],self.p_gains.shape[1]),
                self.p_gains.shape,
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_Kd:
            (
                d_gains_scale_min,
                d_gains_scale_max,
            ) = self.cfg.domain_rand.randomize_Kd_range
            self.d_gains *= torch_rand_float(
                d_gains_scale_min,
                d_gains_scale_max,
                self.d_gains.shape,
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_motor_torque:
            (
                torque_scale_min,
                torque_scale_max,
            ) = self.cfg.domain_rand.randomize_motor_torque_range
            self.torques_scale *= torch_rand_float(
                torque_scale_min,
                torque_scale_max,
                self.torques_scale.shape,
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_default_dof_pos:
            self.default_dof_pos += torch_rand_float(
                self.cfg.domain_rand.randomize_default_dof_pos_range[0],
                self.cfg.domain_rand.randomize_default_dof_pos_range[1],
                (self.num_envs, self.num_dof),
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_action_delay:
            action_delay_idx = torch.round(
                torch_rand_float(
                    self.cfg.domain_rand.delay_ms_range[0] / 1000 / self.sim_params.dt,
                    self.cfg.domain_rand.delay_ms_range[1] / 1000 / self.sim_params.dt,
                    (self.num_envs, 1),
                    device=self.device,
                )
            ).squeeze(-1)
            self.action_delay_idx = action_delay_idx.long()

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.basic_reward_scales.keys()):
            scale = self.basic_reward_scales[key]
            if scale == 0:
                self.basic_reward_scales.pop(key)
            else:
                self.basic_reward_scales[key] *= self.dt

        # prepare list of functions
        self.basic_reward_functions = []
        self.basic_reward_names = []
        self.man_reward_functions = []
        self.man_reward_names = []
        self.loco_reward_functions = []
        self.loco_reward_names = []
        have_termination = False

        for name, scale in self.basic_reward_scales.items():
            if name == "termination":
                have_termination = True
                continue
            self.basic_reward_names.append(name)
            name = "_reward_" + name
            self.basic_reward_functions.append(getattr(self, name))
        # remove zero scales + multiply non-zero ones by dt
        if self.cfg.basic_rewards.reward_type == "loco":
            for key in list(self.loco_reward_scales.keys()):
                scale = self.loco_reward_scales[key]
                if scale == 0:
                    self.loco_reward_scales.pop(key)
                else:
                    self.loco_reward_scales[key] *= self.dt

            for name, scale in self.loco_reward_scales.items():
                self.loco_reward_names.append(name)
                name = "_reward_" + name
                self.loco_reward_functions.append(getattr(self, name))
        # remove zero scales + multiply non-zero ones by dt

        # prepare list of functions

        elif self.cfg.basic_rewards.reward_type == "mani":
            for key in list(self.man_reward_scales.keys()):
                scale = self.man_reward_scales[key]
                if scale == 0:
                    self.man_reward_scales.pop(key)
                else:
                    self.man_reward_scales[key] *= self.dt

            for name, scale in self.man_reward_scales.items():
                self.man_reward_names.append(name)
                name = "_reward_" + name
                self.man_reward_functions.append(getattr(self, name))
        else:
            raise ValueError(
                f"reward_type {self.cfg.basic_rewards.reward_type} not recognized"
            )

        if have_termination:
            self.reward_names = (
                self.man_reward_names
                + self.loco_reward_names
                + self.basic_reward_names
                + ["termination"]
            )
        else:
            self.reward_names = (
                self.man_reward_names + self.loco_reward_names + self.basic_reward_names
            )
        # print(f"reward_names: {self.reward_names}")
        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for name in self.reward_names
        }

    def _create_ground_plane(self):
        """Adds a ground plane to the simulation, sets friction and restitution based on the cfg."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """Adds a heightfield terrain to the simulation, sets parameters based on the cfg."""
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
        )

    def _create_trimesh(self):
        """Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        #"""
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(
            self.sim,
            self.terrain.vertices.flatten(order="C"),
            self.terrain.triangles.flatten(order="C"),
            tm_params,
        )
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
        )

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = (
            self.cfg.asset.replace_cylinder_with_capsule
        )
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        Ee_name = [s for s in body_names if self.cfg.asset.Ee_name_partial in s]
        self.wheel_j_index = [
            i for i in range(self.num_dofs) if "WHL" in self.dof_names[i]
        ]
        self.not_wheel = [
            i for i in range(self.num_dofs) if i not in self.wheel_j_index
        ]

        self.arm_index = [i for i in range(self.num_dofs) if "J" in self.dof_names[i]]
        self.not_arm = [i for i in range(self.num_dofs) if "J" not in self.dof_names[i]]

        self.leg_index = list(set(self.not_wheel) & set(self.not_arm))

        self.torque_weights = torch.ones(
            self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.torque_weights[self.wheel_j_index] = self.cfg.control.torque_weight_wheel
        self.torque_weights[self.arm_index] = torch.tensor(
            self.cfg.control.torque_weight_arm,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.torque_weights[self.leg_index] = self.cfg.control.torque_weight_leg

        if len(Ee_name) != 1:
            raise ValueError(f"Error: Ee_name is not unique, Ee_name is {Ee_name}")

        # print(f"Ee_name is {Ee_name}")
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # # omit the contact with arm's root, avoid continuously reboot
        # omit_contact_names = []
        # for name in self.cfg.asset.omit_contacts_on:
        #     omit_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.envs = []
        self.friction_coeffs = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.restitution_coeffs = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.base_mass = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.base_com = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(
                1
            )
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(
                rigid_shape_props_asset, i
            )
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i,
                self.cfg.asset.self_collisions,
                0,
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle
            )
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props, recomputeInertia=True
            )
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        self.Ee_index = torch.zeros(
            len(Ee_name), dtype=torch.long, device=self.device, requires_grad=False
        )

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )

        for i in range(len(Ee_name)):
            self.Ee_index[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], Ee_name[i]
            )

        # print(f"Ee_index is {self.Ee_index}")
        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

    def _get_env_origins(self):
        """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False
            )
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(
                0, max_init_level + 1, (self.num_envs,), device=self.device
            )
            self.terrain_types = torch.div(
                torch.arange(self.num_envs, device=self.device),
                (self.num_envs / self.cfg.terrain.num_cols),
                rounding_mode="floor",
            ).to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = (
                torch.from_numpy(self.terrain.env_origins)
                .to(self.device)
                .to(torch.float)
            )
            self.env_origins[:] = self.terrain_origins[
                self.terrain_levels, self.terrain_types
            ]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False
            )
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        # print(f"self.dt is {self.dt}")
        # print(f"self.sim_params.dt is {self.sim_params.dt}")
        self.obs_scales = self.cfg.normalization.obs_scales
        self.basic_reward_scales = class_to_dict(self.cfg.basic_rewards.scales)
        self.loco_reward_scales = class_to_dict(self.cfg.loco_rewards)
        self.man_reward_scales = class_to_dict(self.cfg.man_rewards)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

    def _draw_Ee_target_point(self):
        """Draws visualizations for Ee tartet position (slows down simulation a lot)."""
        self.gym.clear_lines(self.viewer)
        Ee_target_R = quaternion_to_matrix(self.commands[:, 3:7])

        Ee_R = torch.matmul(
            quaternion_to_matrix(self.Ee_orient), self.change_coordinates
        )

        base_R = quaternion_to_matrix(self.base_quat_first)
        sphere_geom_Ee = gymutil.WireframeSphereGeometry(
            0.05, 10, 10, None, color=(1, 0, 0)
        )
        sphere_geom_target = gymutil.WireframeSphereGeometry(
            0.05, 10, 10, None, color=(0, 0, 1)
        )
        for i in range(self.num_envs):
            Ee_target_pos = (self.commands[i, :3]).cpu().numpy()
            Ee_pos = (self.Ee_pos[i, :]).cpu().numpy()
            # base_pos = (self.base_position[i, :]).cpu().numpy()
            axis_geom_Ee = gymutil.WireframeOrientationGeometry(Ee_R[i, ...])
            axis_geom_target = gymutil.WireframeOrientationGeometry(Ee_target_R[i, ...])
            # axis_geom_base = gymutil.WireframeOrientationGeometry(base_R[i, ...])
            arrow_geom_Ee = gymutil.WireframeArrowGeometry(
                self.rigid_body_external_forces_Ee[i]
            )

            axis_origin_target = gymapi.Transform(
                gymapi.Vec3(Ee_target_pos[0], Ee_target_pos[1], Ee_target_pos[2]),
                r=None,
            )
            axis_origin_Ee = gymapi.Transform(
                gymapi.Vec3(Ee_pos[0], Ee_pos[1], Ee_pos[2]),
                r=None,
            )
            # axis_origin_base = gymapi.Transform(
            #     gymapi.Vec3(base_pos[0], base_pos[1], base_pos[2]),
            #     r=None,
            # )
            arrow_origin_Ee = gymapi.Transform(
                gymapi.Vec3(Ee_pos[0], Ee_pos[1], Ee_pos[2]),
                r=None,
            )
            # gymutil.draw_lines(
            #     axis_geom_base,
            #     self.gym,
            #     self.viewer,
            #     self.envs[i],
            #     axis_origin_base,
            # )
            gymutil.draw_lines(
                axis_geom_target,
                self.gym,
                self.viewer,
                self.envs[i],
                axis_origin_target,
            )
            gymutil.draw_lines(
                sphere_geom_target,
                self.gym,
                self.viewer,
                self.envs[i],
                axis_origin_target,
            )
            gymutil.draw_lines(
                axis_geom_Ee, self.gym, self.viewer, self.envs[i], axis_origin_Ee
            )

            gymutil.draw_lines(
                sphere_geom_Ee, self.gym, self.viewer, self.envs[i], axis_origin_Ee
            )
            # draw the force
            gymutil.draw_lines(
                arrow_geom_Ee,
                self.gym,
                self.viewer,
                self.envs[i],
                arrow_origin_Ee,
            )

    def _draw_debug_vis(self):
        """Draws visualizations for dubugging (slows down simulation a lot).
        Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.cfg.terrain.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = (
                quat_apply_yaw(
                    self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]
                )
                .cpu()
                .numpy()
            )
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(
                    sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose
                )

    def _init_height_points(self):
        """Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(
            self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False
        )
        x = torch.tensor(
            self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False
        )
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(
            self.num_envs,
            self.num_height_points,
            3,
            device=self.device,
            requires_grad=False,
        )
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(
                self.num_envs,
                self.num_height_points,
                device=self.device,
                requires_grad=False,
            )
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_height_points),
                self.height_points[env_ids],
            ) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(
                self.base_quat.repeat(1, self.num_height_points), self.height_points
            ) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    # ------------ reward functions----------------
    def pre_physics_step(self):
        self._rwd_heading_align_prev = self._reward_heading_align()
        self._rwd_good_looking_prev = self._reward_good_looking()
        # self._rwd_Ee_base_align_prev = self._reward_Ee_base_align()
        self._rwd_tracking_Ee_orient_x_prev = self._reward_tracking_Ee_orient_x()
        self._rwd_tracking_Ee_orient_y_prev = self._reward_tracking_Ee_orient_y()
        self._rwd_tracking_Ee_pos_prev = self._reward_tracking_Ee_pos()

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.abs(
            self.base_position[:, 2] - self.cfg.basic_rewards.base_height_target
        )

    def _reward_loco_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_man_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_loco_base_height(self):
        # Penalize base height away from target
        base_height_error = torch.abs(
            self.base_position[:, 2] - self.cfg.basic_rewards.base_height_target
        )
        if self.cfg.loco_rewards.loco_base_height >= 0:
            return torch.exp(
                -base_height_error / self.cfg.basic_rewards.smooth_tracking_sigma
            )
        else:
            return base_height_error

    def _reward_man_base_height(self):
        # Penalize base height away from target
        base_height_error = torch.abs(
            self.base_position[:, 2] - self.cfg.basic_rewards.base_height_target
        )
        if self.cfg.man_rewards.man_base_height >= 0:
            return torch.exp(
                -base_height_error / self.cfg.basic_rewards.smooth_tracking_sigma
            )
        else:
            return base_height_error

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques) * self.torque_weights, dim=1)

    def _reward_power(self):
        # Penalize power
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        # merge_list = list(set(self.not_wheel) & set(self.not_arm))
        return torch.sum(torch.square(self.dof_vel[:, self.not_wheel]), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations:
        # not_arm = [i for i in range(self.num_dofs) if "J" not in self.dof_names[i]]
        # merge_list = list(set(self.not_wheel) & set(self.not_arm))
        dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt
        return torch.sum(torch.square(dof_acc[:, self.not_wheel]), dim=1)
        # return torch.sum(torch.square(self.dof_acc[:, not_arm]), dim=1)

    def _reward_fall_down(self):
        # Penalize base falling down
        return self.fall_down_buf

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:, :, 0] - self.actions), dim=1)

    def _reward_action_smooth(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(
                self.actions
                - 2 * self.last_actions[:, :, 0]
                + self.last_actions[:, :, 1]
            ),
            dim=1,
        )

    def _reward_action_magn(self):
        # Penalize changes in actions

        actions_scaled = self.actions.clone()
        actions_scaled[:, self.wheel_j_index] *= self.cfg.control.action_scale_vel
        actions_scaled[:, self.not_wheel] *= self.cfg.control.action_scale_pos

        error = torch.zeros_like(self.dof_pos)

        error[:, self.not_wheel] = (
            torch.abs(
                actions_scaled[:, self.not_wheel]
                + self.default_dof_pos[:, self.not_wheel]
                - self.dof_pos[:, self.not_wheel]
            )
            / self.cfg.control.action_scale_pos
        )

        error[:, self.wheel_j_index] = (
            torch.abs(
                actions_scaled[:, self.wheel_j_index]
                - self.dof_vel[:, self.wheel_j_index]
            )
            / self.cfg.control.action_scale_vel
        )

        return torch.sum(error, dim=1)

    def _reward_alive(self):
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

    def _reward_man_phase(self):
        return (~self.loco_phase).to(torch.float)

    def _reward_static_arm(self):
        return torch.sum(
            torch.abs(
                torch.index_select(
                    self.dof_pos, 1, torch.tensor(self.arm_index, device=self.device)
                )
                - self.static_arm_pos[self.arm_index]
            ),
            dim=1,
        )

    def _reward_loco_nominal_state(self):
        wheel_vel_error = 0.01 * torch.sum(torch.abs(self.wheel_vel), dim=1)

        arm_error = 0.1 * torch.sum(
            torch.abs(
                torch.index_select(
                    self.dof_pos, 1, torch.tensor(self.arm_index, device=self.device)
                )
                - self.static_arm_pos[self.arm_index]
            ),
            dim=1,
        )

        leg_error = 0.2 * torch.sum(
            torch.square(
                torch.index_select(
                    self.dof_pos,
                    1,
                    torch.tensor(self.leg_index, device=self.device),
                )
                - self.nominal_dof_pos[self.leg_index]
            ),
            dim=1,
        )

        return leg_error + wheel_vel_error + arm_error

    def _reward_grounded(self):
        scaling = (
            torch.exp(
                -8 * torch.norm((self.commands[:, 0:3] - self.Ee_pos), p=2, dim=1)
            )
        ).clip(max=1)
        return (
            torch.any(
                torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) < 50.0,
                dim=1,
            )
            * scaling
        )

    def _reward_man_nominal_state(self):
        # print(f"merge_list is {merge_list}")
        wheel_vel_error = 0.01 * torch.sum(torch.abs(self.wheel_vel), dim=1)

        arm_error = 0.1 * torch.sum(
            torch.abs(
                torch.index_select(
                    self.dof_pos, 1, torch.tensor(self.arm_index, device=self.device)
                )
                - self.nominal_dof_pos[self.arm_index]
            ),
            dim=1,
        )

        # scaling = (
        #     torch.exp(
        #         -2 * torch.norm((self.commands[:, 0:3] - self.Ee_pos), p=2, dim=1)
        #     )
        #     + 0.3
        # ).clip(max=1)
        leg_error = 0.2 * torch.sum(
            torch.square(
                torch.index_select(
                    self.dof_pos,
                    1,
                    torch.tensor(self.leg_index, device=self.device),
                )
                - self.nominal_dof_pos[self.leg_index]
            ),
            dim=1,
        )
        return leg_error + wheel_vel_error + arm_error

    def _reward_good_looking(self):
        return torch.exp(-self.tot_error / 2)

    def _reward_good_looking_pb(self):
        return self._reward_good_looking() - self._rwd_good_looking_prev

    def _reward_heading_align(self):
        # reward heading_align
        heading_error = torch.norm((self.heading - self.distance_proj_xy), p=2, dim=1)
        return torch.exp(-heading_error / self.cfg.basic_rewards.tracking_sigma)

    def _reward_heading_align_pb(self):
        return (self._reward_heading_align() - self._rwd_heading_align_prev).clip(
            min=0.0
        )

    def _reward_Ee_base_align(self):
        # reward Ee_base_align
        # ee_xy = quaternion_apply(self.commands[:, 3:7], self.forward_vec)
        # ee_xy[:, 2] = 0.0
        # ee_xy = ee_xy / (torch.norm(ee_xy, p=2, dim=1).unsqueeze(1) + 1e-6)

        # ee_y = find_orthogonal_vector(ee_xy)
        # coordinates = project_vector(
        #     (self.commands[:, :3] - self.base_position),
        #     ee_xy,
        #     ee_y,
        # )

        # heading_xy = self.heading.clone()
        # heading_xy[:, 2] = 0
        # heading_xy = heading_xy / (
        #     torch.norm(heading_xy, p=2, dim=1).unsqueeze(1) + 1e-6
        # )

        # self.Ee_base_error = (
        #     (torch.norm((heading_xy - ee_xy), p=2, dim=1) - 0.5).clip(min=0.0)
        #     - (coordinates[:, 0] + 0.4).clip(max=0.0)
        #     + (coordinates[:, 1].abs() - 0.2).clip(min=0.0)
        # )

        # Ee_pos_error = torch.norm((self.commands[:, 0:3] - self.Ee_pos), p=2, dim=1)

        # scaling = torch.exp(
        #     -Ee_pos_error / self.cfg.basic_rewards.smooth_tracking_sigma
        # )

        return (
            torch.exp(-self.Ee_base_error / self.cfg.basic_rewards.tracking_sigma)
            - 0.1 * self.Ee_base_error
        )

    def _reward_Ee_base_align_pb(self):
        return (self._reward_Ee_base_align() - self._rwd_Ee_base_align_prev).clip(
            min=0.0
        )

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(
            1.0
            * (
                torch.norm(
                    self.contact_forces[:, self.penalised_contact_indices, :],
                    p=2,
                    dim=-1,
                )
                > 0.1
            ),
            dim=1,
        )

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(
            (self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        )  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (
                torch.abs(self.dof_vel)
                - self.dof_vel_limits * self.cfg.basic_rewards.soft_dof_vel_limit
            ).clip(min=0.0, max=3.0),
            dim=1,
        )

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (
                torch.abs(self.torques)
                - self.torque_limits * self.cfg.basic_rewards.soft_torque_limit
            ).clip(min=0.0, max=5.0),
            dim=1,
        )

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        base_lin_vel = quaternion_apply(self.base_quat_first, self.base_lin_vel)
        lin_vel_error = torch.sum(
            torch.square(
                self.commands[:, 7].unsqueeze(1) * self.distance_proj_xy - base_lin_vel
            ),
            dim=1,
        )
        return torch.exp(-lin_vel_error / self.cfg.basic_rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 8] - self.base_ang_vel[:, 2])

        if self.cfg.loco_rewards.tracking_ang_vel >= 0:
            return torch.exp(-ang_vel_error / self.cfg.basic_rewards.tracking_sigma)
        else:
            return ang_vel_error

    # ---------------Tracking Ee velocity----------------
    #############################################################################################################
    def _reward_tracking_Ee_lin_vel(self):
        # Tracking of linear Ee velocity commands
        Ee_lin_vel_error = torch.norm(
            (self.Ee_lin_vel - self.Ee_lin_vel_target), p=2, dim=1
        )
        return torch.exp(
            -Ee_lin_vel_error / self.cfg.basic_rewards.smooth_tracking_sigma
        )

    def _reward_tracking_Ee_ang_vel(self):
        # Tracking of angular Ee velocity commands
        Ee_ang_vel_error = torch.sum(
            torch.square(self.commands[:, 3:] - self.Ee_ang_vel), dim=-1
        )
        return torch.exp(-Ee_ang_vel_error / self.cfg.basic_rewards.tracking_sigma)

    # ------------------Tracking Ee configuration----------------
    #############################################################################################################
    def _reward_tracking_Ee_pos(self):
        # Tracking of Ee position commands
        # Ee_pos_error = torch.sum(
        #     torch.square(self.commands[:, 0:3] - self.Ee_pos), dim=1
        # )
        Ee_pos_error = torch.norm((self.commands[:, 0:3] - self.Ee_pos), p=2, dim=1)
        if self.cfg.man_rewards.tracking_Ee_pos >= 0:
            return (
                torch.exp(-Ee_pos_error / self.cfg.basic_rewards.tracking_sigma)
                - 0.1 * Ee_pos_error
            )
        else:
            return Ee_pos_error

    def _reward_displace_vel(self):
        Ee_target_R = quaternion_to_matrix(self.commands[:, 3:7])

        Ee_R = torch.matmul(
            quaternion_to_matrix(self.Ee_orient), self.change_coordinates
        )

        Ee_command_x = Ee_target_R[:, :, 0]
        Ee_x = Ee_R[:, :, 0]
        Ee_orient_error_x = torch.norm(Ee_command_x - Ee_x, p=2, dim=-1)

        Ee_command_y = Ee_target_R[:, :, 1]
        Ee_y = Ee_R[:, :, 1]
        Ee_orient_error_y = torch.norm(Ee_command_y - Ee_y, p=2, dim=-1)

        Ee_pos_error = torch.norm((self.commands[:, 0:3] - self.Ee_pos), p=2, dim=1)

        # self.reference_time[env_ids] = max_value[env_ids]
        error_now = Ee_orient_error_x + Ee_orient_error_y + Ee_pos_error

        error_error = torch.square((error_now - self.error_desire.clip(min=0)))

        return torch.exp(-error_error / self.cfg.basic_rewards.smooth_tracking_sigma)

    def _reward_tracking_Ee_pos_enhance(self):
        # Tracking of Ee position commands
        # Ee_pos_error = torch.sum(
        #     torch.square(self.commands[:, 0:3] - self.Ee_pos), dim=1
        # )
        Ee_pos_error = 20 * torch.norm(
            (self.commands[:, 0:3] - self.Ee_pos), p=2, dim=1
        )

        if self.cfg.man_rewards.tracking_Ee_pos >= 0:
            return torch.exp(-Ee_pos_error)
        else:
            return Ee_pos_error

    def _reward_tracking_Ee_pb(self):
        y_pb = (
            self._reward_tracking_Ee_orient_y() - self._rwd_tracking_Ee_orient_y_prev
        ).clip(min=0.0)

        x_pb = (
            self._reward_tracking_Ee_orient_x() - self._rwd_tracking_Ee_orient_x_prev
        ).clip(min=0.0)

        pos_pb = (self._reward_tracking_Ee_pos() - self._rwd_tracking_Ee_pos_prev).clip(
            min=0.0
        )
        return y_pb + x_pb + pos_pb

    def _reward_tracking_Ee_pos_cb(self):
        Ee_pos_error = torch.norm((self.commands[:, 0:3] - self.Ee_pos), p=2, dim=1)
        small_enough = Ee_pos_error < self.cfg.basic_rewards.small_pos
        self.Ee_pos_error_cumu[small_enough] += Ee_pos_error[small_enough].clip(max=0.2)
        return self.Ee_pos_error_cumu

    def _reward_tracking_Ee_orient_x(self):
        # Tracking of Ee orientation commands
        Ee_target_R = quaternion_to_matrix(self.commands[:, 3:7])

        Ee_R = torch.matmul(
            quaternion_to_matrix(self.Ee_orient), self.change_coordinates
        )

        Ee_command_x = Ee_target_R[:, :, 0]
        Ee_x = Ee_R[:, :, 0]
        Ee_orient_error = torch.norm(Ee_command_x - Ee_x, p=2, dim=-1)
        Ee_pos_error = torch.norm((self.commands[:, 0:3] - self.Ee_pos), p=2, dim=1)

        scaling = torch.exp(-Ee_pos_error / self.cfg.basic_rewards.tracking_sigma)
        return (
            torch.exp(-Ee_orient_error / self.cfg.basic_rewards.smooth_tracking_sigma)
            - 0.1 * Ee_orient_error
        ) * scaling

    # def _reward_tracking_Ee_orient_x_pb(self):
    #     return (
    #         self._reward_tracking_Ee_orient_x() - self._rwd_tracking_Ee_orient_x_prev
    #     ).clip(min=0.0)

    def _reward_tracking_Ee_orient_x_cb(self):
        """only X axis tracking"""
        Ee_target_R = quaternion_to_matrix(self.commands[:, 3:7])

        Ee_R = torch.matmul(
            quaternion_to_matrix(self.Ee_orient), self.change_coordinates
        )
        Ee_command_x = Ee_target_R[:, :, 0]
        Ee_x = Ee_R[:, :, 0]
        # Ee_x = quaternion_apply(self.Ee_orient, self.forward_vec)
        Ee_pos_error = torch.norm((self.commands[:, 0:3] - self.Ee_pos), p=2, dim=1)
        small_enough = Ee_pos_error < self.cfg.basic_rewards.small_xy
        Ee_orient_error = torch.norm(Ee_command_x - Ee_x, p=2, dim=-1)
        self.Ee_x_error_cumu[small_enough] += Ee_orient_error[small_enough].clip(
            max=0.8
        )
        return self.Ee_x_error_cumu

    def _reward_tracking_Ee_orient_y(self):
        # Tracking of Ee orientation commands

        Ee_target_R = quaternion_to_matrix(self.commands[:, 3:7])

        Ee_R = torch.matmul(
            quaternion_to_matrix(self.Ee_orient), self.change_coordinates
        )

        Ee_command_y = Ee_target_R[:, :, 1]
        Ee_y = Ee_R[:, :, 1]
        Ee_orient_error = torch.norm(Ee_command_y - Ee_y, p=2, dim=-1)

        Ee_pos_error = torch.norm((self.commands[:, 0:3] - self.Ee_pos), p=2, dim=1)

        scaling = torch.exp(-Ee_pos_error / self.cfg.basic_rewards.tracking_sigma)

        return (
            torch.exp(-Ee_orient_error / self.cfg.basic_rewards.smooth_tracking_sigma)
            - 0.1 * Ee_orient_error
        ) * scaling

    # def _reward_tracking_Ee_orient_y_pb(self):
    #     return (
    #         self._reward_tracking_Ee_orient_y() - self._rwd_tracking_Ee_orient_y_prev
    #     ).clip(min=0.0)

    def _reward_tracking_Ee_orient_y_cb(self):
        """only Y axis tracking"""
        Ee_target_R = quaternion_to_matrix(self.commands[:, 3:7])

        Ee_R = torch.matmul(
            quaternion_to_matrix(self.Ee_orient), self.change_coordinates
        )

        Ee_command_y = Ee_target_R[:, :, 1]
        Ee_y = Ee_R[:, :, 1]
        Ee_pos_error = torch.norm((self.commands[:, 0:3] - self.Ee_pos), p=2, dim=1)
        small_enough = Ee_pos_error < self.cfg.basic_rewards.small_xy
        Ee_orient_error = torch.norm(Ee_command_y - Ee_y, p=2, dim=1)
        self.Ee_y_error_cumu[small_enough] += Ee_orient_error[small_enough].clip(
            max=0.8
        )
        return self.Ee_y_error_cumu

    def _reward_tracking_Ee_config(self):
        # Tracking of Ee orientation commands
        Ee_orient_error = torch.norm(
            quaternion_to_matrix(self.commands[:, 3:7]).flatten(1)
            - torch.matmul(
                quaternion_to_matrix(self.Ee_orient), self.change_coordinates
            ).flatten(1),
            p=1,
            dim=1,
        )

        Ee_pos_error = torch.norm(self.commands[:, :3] - self.Ee_pos, p=1, dim=1)

        return torch.exp(
            -(Ee_orient_error + Ee_pos_error) / self.cfg.basic_rewards.tracking_sigma
        )

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.5) * first_contact, dim=1
        )  # reward only on first contact with the ground
        rew_airTime *= (
            torch.norm(self.commands[:, :6], dim=1) > 0.4
        )  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (
            torch.norm(self.commands[:, :2], dim=1) < 0.1
        )

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum(
            (
                torch.norm(self.contact_forces[:, self.feet_indices, :], p=2, dim=-1)
                - self.cfg.basic_rewards.max_contact_force
            ).clip(min=0.0),
            dim=1,
        )

    def _reward_base_yaw_vel_limits(self):
        # Penalize ang z velocities
        x = self.cfg.basic_rewards.max_base_yaw_vel - self.base_ang_vel[:, 2].abs()

        return relexed_barrier_func(x).clip(min=0.0, max=15)

    def _reward_base_lin_vel_limits(self):
        # Penalize ang z velocities
        x = self.cfg.basic_rewards.max_base_lin_vel - torch.norm(
            self.base_lin_vel, p=2, dim=1
        )

        return relexed_barrier_func(x).clip(min=0.0, max=3)

    def _reward_Ee_lin_vel_limits(self):
        # Penalize ang z velocities
        x = self.cfg.basic_rewards.max_Ee_lin_vel - torch.norm(
            self.Ee_lin_vel, p=2, dim=1
        )
        return relexed_barrier_func(x).clip(min=0.0, max=3)

    def _reward_Ee_ang_vel_limits(self):
        # Penalize ang z velocities
        x = self.cfg.basic_rewards.max_Ee_ang_vel - torch.norm(
            self.Ee_ang_vel, p=2, dim=1
        )

        return relexed_barrier_func(x, mu=0.1, delta=0.6).clip(min=0.0, max=3)

    def _reward_wheel_vel_limits(self):
        # Penalize dof velocities
        x = (
            self.dof_vel_limits[self.wheel_j_index]
            * self.cfg.basic_rewards.soft_wheel_vel_limit
            - self.wheel_vel.abs()
        ).clip(min=-5.0)
        result = torch.zeros_like(x[:, 0])
        for i in range(len(self.wheel_j_index)):
            result += relexed_barrier_func(x[:, i], mu=0.1, delta=0.6)

        return result.clip(min=0.0, max=3)
