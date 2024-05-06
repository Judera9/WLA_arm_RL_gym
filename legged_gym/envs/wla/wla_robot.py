from legged_gym import LEGGED_GYM_ROOT_DIR
from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym.utils.math import wrap_to_pi
from legged_gym.envs import LeggedRobot
from .wla_config import WLARoughCfg


class WLA(LeggedRobot):
    cfg: WLARoughCfg

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()

        for _ in range(self.cfg.control.decimation):
             # TODO: only return arm torques
            self.torques = self._compute_torques(self.actions)
            
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.arm_pos))
            self.gym.simulate(self.sim)

            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        # actor_jacobian = self.gym.acquire_jacobian_tensor(self.sim, self.cfg.asset.name)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # self.gym.refresh_jacobian_tensors(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state).view(-1, 13)
        # self.whole_body_jacobian = gymtorch.wrap_tensor(actor_jacobian)
        # self.j_eef = self.whole_body_jacobian[:, self.hand_index, :6, -6:]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        
        # gripper_state
        self.base_handles = self.gym.find_asset_rigid_body_index(self.robot_asset, "base")
        self.gripperMover_handles = self.gym.find_asset_rigid_body_index(self.robot_asset, "L6")
        
        self._gripper_state = self.rigid_body_states[:, self.gripperMover_handles][:, 0:13]
        self._gripper_pos = self.rigid_body_states[:, self.gripperMover_handles][:, 0:3]
        self._gripper_vel = self.rigid_body_states[:, self.gripperMover_handles][:, 7:10]
        self._gripper_rot = self.rigid_body_states[:, self.gripperMover_handles][:, 3:7]

        # dof state
        self.base_align_z_pos = torch.cat([self.root_states[:, :2], self.local_axis_z], dim=1)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # base states (fixed)
        self.base_quat = self.root_states[:, 3:7]
        base_yaw = torch.zeros(self.num_envs, device=self.device)
        self.base_yaw_fixed = wrap_to_pi(base_yaw).view(self.num_envs, 1)
        self.base_yaw_quat[:] = quat_from_euler_xyz(
            torch.zeros(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
            base_yaw,
        )

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) 

        # TODO: check local gripper pos and object pos
        self._local_cube_object_pos = quat_rotate_inverse(self.base_yaw_quat, self._cube_object_pos - self.base_align_z_pos)
        self._local_gripper_pos = quat_rotate_inverse(self.base_yaw_quat, self._gripper_pos - self.base_align_z_pos)
        self._local_gripper_vel[:] = quat_rotate_inverse(
            self.base_quat,
            self._gripper_vel,
        )

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination() # TODO
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        # self.last_dof_pos[:] = self.dof_pos[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        
        if self.viewer:
            sphere_geom = gymutil.WireframeSphereGeometry(
                0.05, 4, 4, None, color=(1, 1, 0)
            )
            sphere_geom2 = gymutil.WireframeSphereGeometry(
                0.05, 4, 4, None, color=(1, 0, 0)
            )
            self.gym.clear_lines(self.viewer)
            for i in range(self.num_envs):
                sphere_pose = gymapi.Transform(
                    gymapi.Vec3(
                        self._cube_object_pos[i, 0],
                        self._cube_object_pos[i, 1],
                        self._cube_object_pos[i, 2],
                    ),
                    r=None,
                )
                sphere_pose1 = gymapi.Transform(
                    gymapi.Vec3(
                        self._gripper_pos[i, 0],
                        self._gripper_pos[i, 1],
                        self._gripper_pos[i, 2],
                    ),
                    r=None,
                )
                gymutil.draw_lines(
                    sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose
                )
                gymutil.draw_lines(
                    sphere_geom2, self.gym, self.viewer, self.envs[i], sphere_pose1
                )
        
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    # def check_termination(self):
    #     """Check if environments need to be reset"""
    #     self.reset_buf = torch.any(
    #         torch.norm(
    #             self.contact_forces[:, self.termination_contact_indices, :], dim=-1
    #         )
    #         > 1.0,
    #         dim=1,
    #     )
    #     self.time_out_buf = (
    #         self.episode_length_buf > self.max_episode_length
    #     )  # no terminal reward for time-outs
    #     self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
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
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    

    def compute_observations(self):
        """Computes observations"""
        self.obs_buf = torch.cat((  self._local_cube_object_pos,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        
        # TODO: add privileged observation
        if self.privileged_obs_buf is not None:
            # self.privileged_obs_buf = self.obs_buf
            self.privileged_obs_buf = torch.cat(
                (
                    self._local_gripper_pos,
                    self._local_gripper_vel,
                    self.obs_buf,
                    self.dof_pos, # TODO: comment this?
                    (self._local_gripper_pos - self._local_cube_object_pos),
                ),
                dim=-1,
            )
            
        # add privilige observations
        if self.cfg.env.num_privileged_obs is not None:
            pass
            self.privileged_obs_buf = torch.cat(
                (
                    self.friction_coeffs.view(self.num_envs, 1),
                ),
                dim=-1,
            )

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        # set target object position
        self.base_align_z_pos[env_ids] = torch.cat(
            [self.root_states[env_ids, :2], self.local_axis_z[env_ids]], dim=1
        ) 
        self._cube_object_pos[env_ids, 0:1] = (
            self.init_local_cube_object_pos[env_ids, 0].view(len(env_ids), 1)
            + torch_rand_float(
                self.command_ranges["obj_pos_x"][0],
                self.command_ranges["obj_pos_x"][1],
                (len(env_ids), 1),
                device=self.device,
            )
            + self.base_align_z_pos[env_ids, 0].view(len(env_ids), 1)
        )
        self._cube_object_pos[env_ids, 1:2] = (
            self.init_local_cube_object_pos[env_ids, 1].view(len(env_ids), 1)
            + torch_rand_float(
                self.command_ranges["obj_pos_y"][0],
                self.command_ranges["obj_pos_y"][1],
                (len(env_ids), 1),
                device=self.device,
            )
            + self.base_align_z_pos[env_ids, 1].view(len(env_ids), 1)
        )
        self._cube_object_pos[env_ids, 2:3] = (
            self.init_local_cube_object_pos[env_ids, 2].view(len(env_ids), 1)
            + torch_rand_float(
                self.command_ranges["obj_pos_z"][0],
                self.command_ranges["obj_pos_z"][1],
                (len(env_ids), 1),
                device=self.device,
            )
            + self.base_align_z_pos[env_ids, 2].view(len(env_ids), 1)
        )

    # def orientation_error(self, desired, current):
    #     pass

    def _compute_torques(self, actions): # TODO: only return the arm_u ?
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # self.arm_u = self.dof_pos[:] + actions[:] # [env_len, 6]
        # self.arm_u[:, :3] = self.arm_u[:, :3] * self.dis_err_rate_inverse # TODO

        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
            # arm_pos = torch.clip(self.arm_u, self.dof_pos_limits[:, 6:, 0], self.dof_pos_limits[:, 6:, 1])
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    # def _reset_dofs(self, env_ids):
    #     """Resets DOF position and velocities of selected environmments
    #     Positions are randomly selected within 0.5:1.5 x default positions.
    #     Velocities are set to zero.

    #     Args:
    #         env_ids (List[int]): Environemnt ids
    #     """
    #     pass

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
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else: # running this one
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids] # set robot to origin

        # base velocities
        self.root_states[env_ids, 7:13] = 0 # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # def _get_noise_scale_vec(self, cfg):
    #     """Sets a vector used to scale the noise added to the observations.
    #         [NOTE]: Must be adapted when changing the observations structure

    #     Args:
    #         cfg (Dict): Environment config file

    #     Returns:
    #         [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
    #     """
    #     pass

    # ----------------------------------------
    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
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
        self.p_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.d_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(
            [0, 0, 0],
            device=self.device,
            requires_grad=False,
        )
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
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
         
        # TODO: delete unused tensors
        # TODO: check the added tensors
        self._local_cube_object_pos = torch.zeros(
            (self.num_envs, 3), dtype=torch.float, device=self.device
        )
        self._cube_object_pos = torch.zeros(
            (self.num_envs, 3), dtype=torch.float, device=self.device
        ) # the global objective position
        self.init_local_cube_object_pos = torch.tensor(
            self.cfg.goal_ee.init_local_cube_object_pos,
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1) # global_pos - robot_base
        self.arm_pos = torch.zeros(
            (self.num_envs, 12), dtype=torch.float, device=self.device
        ) # p, r
        self._local_gripper_pos = torch.zeros(
            (self.num_envs, 3), dtype=torch.float, device=self.device
        )
        self._local_gripper_vel = torch.zeros(
            (self.num_envs, 3), dtype=torch.float, device=self.device
        )
        self._gripper_pos = torch.zeros(
            (self.num_envs, 3), dtype=torch.float, device=self.device
        )
        self._gripper_vel = torch.zeros(
            (self.num_envs, 3), dtype=torch.float, device=self.device
        )
        self.local_axis_z = torch.tensor([0.0], device=self.device).repeat(
            self.num_envs, 1
        ) # used to reset object z position to zero (using root_states for xy)
        self.base_align_z_pos = torch.tensor(
            [0, 0.000, 0.00], dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)
        # self.dis_err_rate_inverse = torch.zeros(
        #     (self.num_envs, 1), dtype=torch.float, device=self.device
        # ) # TODO: what is this
        base_yaw = torch.zeros(self.num_envs, device=self.device)
        self.base_yaw_quat = quat_from_euler_xyz(
            torch.tensor(0), torch.tensor(0), base_yaw
        )
        self.arm_u = torch.zeros(
            (self.num_envs, 12), dtype=torch.float, device=self.device
        )

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(
                        f"PD gain of joint {name} were not defined, setting them to zero"
                    )
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

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
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
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

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.robot_asset = robot_asset
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
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

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
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                            requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = (m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit)
                self.dof_pos_limits[i, 1] = (m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit)
        friction_dof_range = self.cfg.domain_rand.friction_dof_range
        damping_dof_range = self.cfg.domain_rand.damping_dof_range
        for s in range(len(props)):
            if self.cfg.domain_rand.randomize_dof_friction:
                friction = torch_rand_float(friction_dof_range[0], friction_dof_range[1], (1, 1), device='cpu').item()
                props["friction"][s] = friction
            if self.cfg.domain_rand.randomize_dof_damping:
                damping = torch_rand_float(damping_dof_range[0], damping_dof_range[1], (1, 1), device='cpu').item()
                props["damping"][s] = damping
        return props

    # ------------ reward functions----------------
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_object_distance(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Reward for reaching the object."""
        dis_err = torch.sum(torch.square(self._local_gripper_pos - self._local_cube_object_pos), dim=1)
        # print("_object_distance:",dis_err,"value:",torch.exp(-dis_err/self.cfg.rewards.object_sigma).shape)  #[0.7~3.5]
        return torch.exp(-dis_err / self.cfg.rewards.object_sigma)
