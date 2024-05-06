from legged_gym.envs.base.legged_robot_config import (
    BaseConfig,
    LeggedRobotCfgPPO,
    LeggedRobotCfg,
)


class WLAFlatCfg_t(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 200  # 4096
        num_observations = 21  # pos vel actions object_3d
        num_privileged_obs = 6  # TODO: friction? ee_position? diff_ik?
        num_actions = 6

        # No changes
        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds

        # TODO:
        # dof_vel_use_pos_diff = True
        # base_lin_vel_use_pos_diff = True
        history_len = 10
        # use_lp_filter = True
        # lp_alpha = 0.3

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"  # "heightfield" # none, plane, heightfield or trimesh
        measure_heights = False

    class commands(LeggedRobotCfg.commands):
        num_commands = 3  # TODO
        resampling_time = 5.0  # time before command are changed[s]
        curriculum = False  # TODO
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:  # TODO
            obj_pos_x = [-0.1, 0.1]
            obj_pos_y = [-0.3, 0.3]
            obj_pos_z = [-0.2, 0.1]
            
    class goal_ee:
        init_local_cube_object_pos = [0.45, 0, 0.9]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.5]  # joints: J1 J2 J3 J4 J5 J6 [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "J1": 0,  # [rad]
            "J2": 0,  # [rad]
            "J3": 0,  # [rad]
            "J4": 0,  # [rad]
            "J5": 0,  # [rad]
            "J6": 0,  # [rad]
        }  # TODO: initial angles
        
    class control(LeggedRobotCfg.init_state):
        control_type = "P"  # TODO # arm_control_type='position'
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

        # PD Drive parameters:
        stiffness = {
            # "HAA": 90.0,  # [N*m/rad]
            # "HFE": 90.0,
            # "KFE": 90.0,
            # "WHL": 0.0,
            "J1": 20.0,
            "J2": 40.0,
            "J3": 30.0,
            "J4": 10.0,
            "J5": 10.0,
            "J6": 5.0,
        }  # [N*m/rad]
        damping = {
            "J1": 0.5,
            "J2": 1.0,
            "J3": 0.8,
            "J4": 0.25,
            "J5": 0.25,
            "J6": 0.15,
        }  # [N*m*s/rad]
        
    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/wla/urdf/only_arm.urdf"
        name = "wla"
        penalize_contacts_on = ["base", "L"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        fix_base_link = True
            
    class noise(LeggedRobotCfg.noise):
        add_noise = False
        noise_level = 1.0  # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
    
    class domain_rand(LeggedRobotCfg.domain_rand):
        push_robots = False
        randomize_friction = False
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1.0, 1.0]
        push_interval_s = 15
        max_push_vel_xy = 1.0
            
    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False
        object_sigma = 0.4  # TODO
        class scales(LeggedRobotCfg.rewards.scales):
            termination = -1
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            lin_vel_z = -0
            ang_vel_xy = -0
            orientation = -0
            torques = -0.0001  # minimize torques
            dof_vel = -2.5e-5
            dof_acc = -2.5e-7
            base_height = -0.0
            feet_air_time = 0
            collision = -1.0  # penalize collision
            feet_stumble = -0
            action_rate = -0
            stand_still = -0

            # TODO: added rewards
            arm_pos = -0  # TODO
            object_distance = 2
        
class WLAFlatCfgPPO_t(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "flat_wla_t"
        runner_class_name = "OnPolicyRunner_encoder"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        
    class encoder:
        encoder_hidden_dims = [256, 128] # TODO: dimention should not be so big?
        encoder_output_dim = 64
        encoder_type = "teacher"
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        encoder_detach = False  # not used, always false
        orthogonal_init = False
            
        

        
        
