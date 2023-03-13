from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym import gymtorch
from socket import *
import torch
import struct
import time
import math
import multiprocessing
import numpy as np
import os
from Read_trajectory_data import smoothing_windows_filter
import pandas as pd

Ball_Existence = 0
Ball_Time_Stamp = 1
Ball_POS_X = 2
Ball_POS_Y = 3
Ball_POS_Z = 4
Pred_Ball_POS_X = 5
Pred_Ball_POS_Y = 6
Pred_Ball_POS_Z = 7
Pred_Ball_Vel_X = 8
Pred_Ball_Vel_Y = 9
Pred_Ball_Vel_Z = 10
Joint_Time_Stamp = 11
Joint1_POS = 12
Joint2_POS = 13
Joint3_POS = 14
Joint4_POS = None
Joint5_POS = 15
Joint6_POS = 16
Joint7_POS = 17
Table_Contact_Num = 18
Round_Num = 19


class PingPong_Robot_Sim2Real:
    def __init__(self):
        self.gym = gymapi.acquire_gym()
        self._load_recorded_trajectory_path()
        self._create_sim()
        self._create_viewer()
        self._create_camera()
        self._create_plane()
        self._load_asset()
        self._create_env()
        self._load_policy()
        self._socket_init()
        self._reset_target()

    def _create_sim(self):
        if args.use_gpu:
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        sim_params = gymapi.SimParams()
        sim_params.dt = dt = 1.0 / args.frame_rate
        if args.physics_engine == gymapi.SIM_FLEX:
            pass
        elif args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 6
            sim_params.physx.num_velocity_iterations = 6
            sim_params.physx.num_threads = 4
            sim_params.physx.num_subscenes = 4
            sim_params.physx.use_gpu = args.use_gpu
            sim_params.physx.contact_collection = gymapi.CC_ALL_SUBSTEPS
            sim_params.physx.contact_offset = 0
            sim_params.physx.rest_offset = 0
            sim_params.physx.friction_offset_threshold = 0
            sim_params.physx.friction_correlation_distance = 0
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = -9.81
            sim_params.use_gpu_pipeline = args.use_gpu
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine,
                                       sim_params)

    def _create_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_viewer(self):
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "viewer_visualization")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_T, "reset_target")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_B, "reset_ball")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "replay_ball")
        self.viewer_status = 1  # 0: stop render viewer   1: render viewer

    def _load_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.use_mesh_materials = False
        asset_options.override_com = False
        asset_options.override_inertia = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.disable_gravity = False
        asset_options.enable_gyroscopic_forces = True
        print("Loading asset '%s' from '%s'" % (args.robot_asset, args.asset_root))
        self.robot_asset = self.gym.load_asset(self.sim, args.asset_root, args.robot_asset, asset_options)
        asset_options.fix_base_link = False
        self.ball_asset = self.gym.create_sphere(self.sim, 0.02, asset_options)
        asset_options.fix_base_link = True
        self.target_asset = self.gym.create_sphere(self.sim, 0.03, asset_options)  # the target is the green ball with 0.03 radius
        self.num_Robot_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        self.num_Robot_dofs = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_Ball_bodies = self.gym.get_asset_rigid_body_count(self.ball_asset)
        self.num_Ball_dofs = self.gym.get_asset_dof_count(self.ball_asset)

    def _create_camera(self):
        # position the camera
        cam_pos = gymapi.Vec3(0, 2.5, 1.5)
        cam_target = gymapi.Vec3(0, -75, 0.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _create_actor(self):
        # add actor
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0, 0.0)
        self.robot_actor = self.gym.create_actor(self.env, self.robot_asset, pose, "robot", 0, 0)
        joint_props = self.gym.get_actor_dof_properties(self.env, self.robot_actor)
        joint_props["stiffness"] = (5000.0, 5000.0, 500.0, 5000.0, 5000.0, 5000.0, 500.0)
        joint_props["damping"] = (450.0, 450.0, 20.0, 20.0, 200.0, 100.0, 20.0)
        self.gym.set_actor_dof_properties(self.env, self.robot_actor, joint_props)

        # Change PingPong_Table rigid_shape_properties
        Robot_shape = self.gym.get_actor_rigid_shape_properties(self.env, self.robot_actor)
        for index in range(3):
            # Table (link1) consists of 3 shapes, so each shape's property needs to be set or the setting doesn't work.
            Robot_shape[index].restitution = 1.7845  # 1.732 ~ 1.837 (fall from 30cm，bounce to 23~26cm)
            Robot_shape[index].friction = 0.3  # Friction coefficient is 0~0.6
        for index in range(3,
                           len(Robot_shape) - 2):
            Robot_shape[index].restitution = 1.732  # 1.732 ~ 1.837
        Robot_shape[-1].restitution = 1.4785  # 1.415 ~ 1.542
        Robot_shape[-2].restitution = 1.4785
        self.gym.set_actor_rigid_shape_properties(self.env, self.robot_actor, Robot_shape)

        pose.p = gymapi.Vec3(1.0, 0, 1.10)
        self.ball_actor = self.gym.create_actor(self.env, self.ball_asset, pose, "ball", 0, 1)
        self.target_actor = self.gym.create_actor(self.env, self.target_asset, pose, "target", 0, 1)

        self.ball_props = self.gym.get_actor_rigid_body_properties(self.env, self.ball_actor)
        self.ball_props[0].mass = 0.0027  # mass: 2.53g~2.7g
        self.gym.set_actor_rigid_body_properties(self.env, self.ball_actor, self.ball_props, True)
        self.gym.set_rigid_body_color(self.env, self.robot_actor, len(Robot_shape) - 1,
                                      gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))
        self.gym.set_rigid_body_color(self.env, self.robot_actor, len(Robot_shape) - 2,
                                      gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))
        self.gym.set_rigid_body_color(self.env, self.ball_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                      gymapi.Vec3(1, 0, 0))
        self.gym.set_rigid_body_color(self.env, self.target_actor, 0,
                                      gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 1, 0))

        Ball_shape = self.gym.get_actor_rigid_shape_properties(self.env, self.ball_actor)
        Ball_shape[0].restitution = 2
        self.gym.set_actor_rigid_shape_properties(self.env, self.ball_actor, Ball_shape)

    def _create_env(self):

        self.env_lower = torch.tensor([-1.5, -1.8, 0], device=self.device)
        self.env_upper = torch.tensor([3, 1.8, 1.8], device=self.device)
        self.ball_velocity_scaled_upper = torch.tensor([10.0, 5.0, 6.0], device=self.device)
        self.ball_velocity_scaled_lower = torch.tensor([-10.0, -5.0, -6.0], device=self.device)
        self.racket_velocity_scaled_upper = torch.tensor([20,8,7],device=self.device)
        self.racket_velocity_scaled_lower = torch.tensor([-20,-8,-7],device=self.device)
        self.racket_angular_vel_upper_limits = torch.tensor([160,200,100],device=self.device)
        self.racket_angular_vel_lower_limits = torch.tensor([-160,-200,-100],device=self.device)

        self.robot_target_pos_lower_limits = torch.tensor([-0.1, -0.7425, 0.78], device=self.device)
        self.robot_target_pos_upper_limits = torch.tensor([-1.350, 0.7425, 0.82], device=self.device)
        self.ball_pos_lower_limit = torch.tensor([-0.025, -0.7425, 0.78],
                                                 device=self.device)  # the radius of ball is 20 mm
        self.ball_pos_upper_limit = torch.tensor([-1.350, 0.7425, 1.6], device=self.device)  # 1370-20，762.5-20 ，760+840
        self.ball_linear_vel_lower_limit = torch.tensor([3, -4, -1], device=self.device)
        self.ball_linear_vel_upper_limit = torch.tensor([5, 4, 1],
                                                        device=self.device)  # x_max:10 professional player can strike the ball nearly 30 - 40 m/s
        # set up the env grid
        env_lower = gymapi.Vec3(*self.env_lower)
        env_upper = gymapi.Vec3(*self.env_upper)
        self.num_envs = 1
        for _ in range(self.num_envs):
            # create env
            self.env = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self._create_actor()
        ###The following code is important. Without it, many bugs may appear.
        self.gym.prepare_sim(self.sim)  # GPU pipeline needs this code, cpu pipeline doesn't.
        self.ball_handle = self.gym.find_actor_rigid_body_handle(self.env, self.ball_actor, "sphere")
        self.table_handle = self.gym.find_actor_rigid_body_handle(self.env, self.robot_actor, "PingPong_Table")
        self.racket_1_handle = self.gym.find_actor_rigid_body_handle(self.env, self.robot_actor, "Racket_1")
        self.racket_2_handle = self.gym.find_actor_rigid_body_handle(self.env, self.robot_actor, "Racket_2")
        self.robot_target_handle = self.gym.find_actor_rigid_body_handle(self.env, self.target_actor, "sphere")
        self.net_handle = self.gym.find_actor_rigid_body_handle(self.env, self.robot_actor, "Net")

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_tensor = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.ball_root_states = self.root_state_tensor[:, 1]
        self.target_root_states = self.root_state_tensor[:, 2]

        self.ball_pos = self.rigid_body_tensor[:, self.ball_handle, 0:3]
        self.ball_linear_vel = self.rigid_body_tensor[:, self.ball_handle, 7:10]
        self.ball_angular_vel = self.rigid_body_tensor[:, self.ball_handle, 10:13]

        self.racket_pos = self.rigid_body_tensor[:, self.racket_2_handle][:, 0:3]
        self.racket_rot = self.rigid_body_tensor[:, self.racket_2_handle][:, 3:7]
        self.racket_linear_vel = self.rigid_body_tensor[:, self.racket_2_handle][:, 7:10]
        self.racket_angular_vel = self.rigid_body_tensor[:, self.racket_2_handle][:, 10:13]

        self.robot_target_pos = self.rigid_body_tensor[:, self.robot_target_handle][:, 0:3]

        self.Robot_dof_pos = self.dof_state[..., 0]
        self.Robot_dof_vel = self.dof_state[..., 1]
        Robot_dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
        self.Robot_dof_lower_limits = torch.tensor(Robot_dof_props['lower'], device=self.device)
        self.Robot_dof_upper_limits = torch.tensor(Robot_dof_props['upper'], device=self.device)
        self.Robot_dof_vel_limits = torch.tensor(Robot_dof_props['velocity'], device=self.device)

        self.Robot_dof_zeros = (self.Robot_dof_lower_limits + self.Robot_dof_upper_limits) / 2

        self.table_contact_count_buf = torch.zeros(1, device=self.device, dtype=torch.long)
        self.ball_contact_count_buf = torch.zeros(1, device=self.device, dtype=torch.long)

        self.num_bodies = self.rigid_body_tensor.shape[1]
        self.last_contact_progress = torch.zeros(1, device=self.device, dtype=torch.long)
        self.trajectory_state = torch.zeros(1, device=self.device, dtype=torch.long)
        self.trajectory_state_onehot = torch.zeros((1, 4), device=self.device, dtype=torch.long)
        self.round_num = 0
        self.num_actors = self.gym.get_actor_count(self.env)
        self.global_indices = torch.arange(1 * self.num_actors, dtype=torch.int32,
                                           device=self.device).view(1, -1)
        self.ball_last_existence = False  # The last step whether the ball exist or not
        self.absolute_randomize = lambda lower_limits, upper_limits, num_envs, num_properties: \
            lower_limits + torch.rand((num_envs, num_properties), device=self.device) * (upper_limits - lower_limits)
        self.record_data_state_list = []
        self.record_data_paths = []

        for i, path in enumerate(self.selected_trajectory_paths):
            self.record_data_paths.append(path)
            trajectory_data = np.array(pd.read_csv(path)) / 1000.0
            trajectory_data[:, 2] = trajectory_data[:, 2] + 0.76
            trajectory_data[:, 3] = trajectory_data[:, 3] - trajectory_data[0, 3]
            trajectory_data_state = smoothing_windows_filter(trajectory_data.copy(), 3, 7, 7, 0.75)
            self.record_data_state_list.append(trajectory_data_state)
        self.simulation_count = 0
        self.replay_status = False

    def _load_recorded_trajectory_path(self):
        self.selected_trajectory_paths = os.listdir('./Trajectory_Experiments/All_Trajectories_in_Reality')
        for i in range(len(self.selected_trajectory_paths)):
            self.selected_trajectory_paths[i] = os.path.join(
                './Trajectory_Experiments/All_Trajectories_in_Reality/' + self.selected_trajectory_paths[i])

    def replay_data_init(self):
        path_index=torch.randint(0, len(self.record_data_state_list), [1])
        self.record_data_state = self.record_data_state_list[path_index]
        print(self.record_data_paths[path_index])
        self.replay_update_list = []
        for i in range(len(self.record_data_state)):
            self.replay_update_list.append(
                np.around(self.simulation_count + self.record_data_state[i][:, 3] * args.frame_rate))
        self.replay_status = True
        self.trajectory_state[0] = 0
        self.trajectory_state_onehot = torch.zeros_like(self.trajectory_state_onehot, device=self.device)
        self.trajectory_state_onehot[0, self.trajectory_state] = 1
        self.gym.set_rigid_body_color(self.env, self.ball_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                      gymapi.Vec3(0, 0, 1))

    def _socket_init(self):
        self.policy_input_data = multiprocessing.Array('d', [0.0] * 20)
        self.filtered_real_ball_state = multiprocessing.Array('d', [0.0] * 9)  # 3(pos dim)+3(vel dim)+3(force dim)
        self.revc_interval = multiprocessing.Value('d', 10.0)  # 0: lose connection , 1: connection exist

    def scale_observations(self):
        dof_pos_scaled = (2.0 * (self.Robot_dof_pos - self.Robot_dof_lower_limits)
                          / (self.Robot_dof_upper_limits - self.Robot_dof_lower_limits) - 1.0)
        dof_vel_scaled = self.Robot_dof_vel / self.Robot_dof_vel_limits
        ball_pos_scaled = 2.0 * (self.ball_pos - self.env_lower) / (
                self.env_upper - self.env_lower) - 1.0
        racket_pos_scaled = 2.0 * (self.racket_pos - self.env_lower) / (self.env_upper - self.env_lower) - 1.0
        trajectory_state_scaled = (2.0 * self.trajectory_state / 3) - 1.0
        target_pos_scaled = 2.0 * (self.robot_target_pos - self.env_lower) / (self.env_upper - self.env_lower) - 1.0
        ball_linear_vel_scaled = 2.0 * (self.ball_linear_vel - self.ball_velocity_scaled_lower) / (
                self.ball_velocity_scaled_upper - self.ball_velocity_scaled_lower) - 1.0
        racket_linear_vel_scaled = 2.0 * (self.racket_linear_vel - self.racket_velocity_scaled_lower) / (
                self.racket_velocity_scaled_upper - self.racket_velocity_scaled_lower) - 1.0
        racket_angular_vel_scaled = 2.0 * (self.racket_angular_vel - self.racket_angular_vel_lower_limits) / (
                self.racket_angular_vel_upper_limits - self.racket_angular_vel_lower_limits) - 1.0

        self.trajectory_state_onehot = torch.zeros_like(self.trajectory_state_onehot, device=self.device)
        self.trajectory_state_onehot[0, self.trajectory_state] = 1

        return torch.cat([dof_pos_scaled.unsqueeze(0), dof_vel_scaled.unsqueeze(0), ball_pos_scaled, racket_pos_scaled,
                          self.racket_rot, trajectory_state_scaled.unsqueeze(0), target_pos_scaled[:, :2],
                          ball_linear_vel_scaled, racket_linear_vel_scaled, racket_angular_vel_scaled,
                          self.trajectory_state_onehot], dim=-1)

    def _ball_exsitence_insepection(self):
        # self.policy_input_data[0] is used to discriminate whether the ball exist at current time
        # If the ball state at current time and last time is same,it doesn't need to change
        if self.policy_input_data[Ball_Existence] and not args.policy_simulation_test:  # ball exist, disable gravity
            if self.ball_props[0].flags != gymapi.RIGID_BODY_DISABLE_GRAVITY:
                self.gym.set_rigid_body_color(self.env, self.ball_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                              gymapi.Vec3(1, 1, 0))
                self.ball_props[0].flags = gymapi.RIGID_BODY_DISABLE_GRAVITY
            self.root_state_tensor[:, self.ball_actor, 0:3] = self.filtered_real_ball_pos
            self.root_state_tensor[:, self.ball_actor, 7:10] = self.filtered_real_ball_vel
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor))
        else:
            if self.ball_props[0].flags != gymapi.RIGID_BODY_NONE:
                self.ball_props[0].flags = gymapi.RIGID_BODY_NONE  # ball not exist, enable gravity
                self.gym.set_rigid_body_color(self.env, self.ball_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                              gymapi.Vec3(1, 0, 0))


    def _load_policy(self):
        self.policy_init = torch.load(args.loaded_policy[0])
        self.policy_mu = torch.load(args.loaded_policy[1])
        self.policy_log_std = torch.load(args.loaded_policy[2])
        self.obs_mean = torch.load(args.loaded_policy[3])
        self.obs_var = torch.load(args.loaded_policy[4])
        self.epsilon = 1e-5

    def _policy_sample_execution(self, obs):
        obs = torch.clamp(obs, min=-1.0, max=1.0)
        obs = (obs - self.obs_mean.float()) / torch.sqrt(self.obs_var.float() + self.epsilon)
        obs = torch.clamp(obs, min=-5.0, max=5.0)
        actions_mu = self.policy_mu(self.policy_init(obs))
        actions_mu = torch.clamp(actions_mu, -1.0, 1.0)
        if args.deterministic_policy:
            # actions_mu = actions_mu+(actions_mu-self.Robot_dof_pos)*200/args.frame_rate # 200Hz is the original frequency in training environment
            actions_mu = self.Robot_dof_zeros + actions_mu * (
                    self.Robot_dof_upper_limits - self.Robot_dof_lower_limits) / 2
            actions_mu = torch.max(torch.min(actions_mu, self.Robot_dof_upper_limits), self.Robot_dof_lower_limits)
            self.gym.set_dof_position_target_tensor(self.sim,
                                                    gymtorch.unwrap_tensor(actions_mu))
            return

        actions_sigma = torch.exp(self.policy_log_std)
        action_distr = torch.distributions.Normal(actions_mu, actions_sigma)
        targets = self.Robot_dof_zeros + action_distr.sample() * (
                self.Robot_dof_upper_limits - self.Robot_dof_lower_limits) / 2
        targets = torch.max(torch.min(targets, self.Robot_dof_upper_limits), self.Robot_dof_lower_limits)

        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(targets))

    def _pre_simulation(self):
        if self.policy_input_data[Ball_Existence]:
            self.real_ball_pos = torch.tensor(
                self.policy_input_data[Ball_POS_X:Ball_POS_Z + 1], device=self.device) / 1000

            self.filtered_real_ball_pos = torch.tensor(self.filtered_real_ball_state[0:3],
                                                       device=self.device) / 1000  # change millimeter into meter
            self.filtered_real_ball_vel = torch.tensor(self.filtered_real_ball_state[3:6], device=self.device)
            self.filtered_real_ball_force = torch.tensor(self.filtered_real_ball_state[6:9], device=self.device)
        self.real_joint_pos = torch.tensor(self.policy_input_data[Joint1_POS:Joint3_POS + 1] + [0.0] +
                                           self.policy_input_data[Joint5_POS:Table_Contact_Num],
                                           device=self.device) / torch.tensor(
            [1000.0, 1000.0] + [180 / math.pi for _ in range(5)], device=self.device)
        self._ball_exsitence_insepection()

    def _simulation_viewer(self):
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "viewer_visualization" and evt.value > 0:
                self.viewer_status = 1 - self.viewer_status
            if evt.action == "reset_target" and evt.value > 0:
                self._reset_target()
            if evt.action == "reset_ball" and evt.value > 0:
                self._reset_ball()
            if evt.action == "replay_ball" and evt.value > 0:
                self.replay_data_init()

        if self.viewer_status:
            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
        else:
            self.gym.poll_viewer_events(self.viewer)

    def _reset_target(self):
        reset_robot_target_pos = self.absolute_randomize(self.robot_target_pos_lower_limits,
                                                         self.robot_target_pos_upper_limits,
                                                         1, 3)  # reset the target position randomly
        self.target_root_states[:, 0:3] = reset_robot_target_pos
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor))

    def _reset_ball(self):
        reset_ball_pos = self.absolute_randomize(self.ball_pos_lower_limit,
                                                 self.ball_pos_upper_limit,
                                                 1, 3)  # reset the initial state of the ball
        reset_ball_linear_vel = self.absolute_randomize(self.ball_linear_vel_lower_limit,
                                                        self.ball_linear_vel_upper_limit,
                                                        1, 3)  # reset the initial state of the ball
        self.ball_root_states[:, 0:3] = reset_ball_pos
        self.ball_root_states[:, 7:10] = reset_ball_linear_vel
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor))
        self.trajectory_state[0] = 0
        self.round_num = self.policy_input_data[Round_Num]
        self.trajectory_state_onehot = torch.zeros((1, 4), device=self.device,
                                                   dtype=torch.long)
        self.trajectory_state_onehot[0, 0] = 1
        self.table_contact_count_buf[0] = 0
        self.ball_contact_count_buf[0] = 0
        Robot_dof_pos = self.absolute_randomize(self.Robot_dof_lower_limits, self.Robot_dof_upper_limits,
                                                1, self.num_Robot_dofs)
        Robot_dof_vel = torch.zeros_like(self.Robot_dof_vel)
        self.dof_state[..., 0] = Robot_dof_pos
        self.dof_state[..., 1] = Robot_dof_vel
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))

    def _replay_real_ball_trajectory(self):
        if np.where(self.replay_update_list[0] == self.simulation_count)[-1].shape[0] and \
                np.where(self.replay_update_list[1] == self.simulation_count)[-1].shape[0]:
            replay_index = np.where(self.replay_update_list[0] == self.simulation_count)[-1]
            self.ball_root_states[:, 0:3] = torch.tensor(self.record_data_state[0][replay_index][0, :3],
                                                         device=self.device)
            self.ball_root_states[:, 7:10] = torch.tensor(self.record_data_state[1][replay_index][0, :3],
                                                          device=self.device)
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor))
            self.gym.set_rigid_body_color(self.env, self.ball_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          gymapi.Vec3(0, 0, 1))
        else:
            self.gym.set_rigid_body_color(self.env, self.ball_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          gymapi.Vec3(1, 0, 0))

    def simulation(self):
        real_time_all = 0
        while not self.gym.query_viewer_has_closed(self.viewer):
            cur_time = time.time()
            self._pre_simulation()
            self._policy_sample_execution(self.scale_observations())
            if self.replay_status:
                if self.simulation_count > self.replay_update_list[0][-1]:
                    self.gym.set_rigid_body_color(self.env, self.ball_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                                  gymapi.Vec3(1, 0, 0))
                    self.replay_status = False
                if not (np.where(self.replay_update_list[0] == self.simulation_count)[-1].shape[0] and
                        np.where(self.replay_update_list[1] == self.simulation_count)[-1].shape[0]):
                    self.gym.set_rigid_body_color(self.env, self.ball_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                                  gymapi.Vec3(1, 0, 0))
                else:
                    replay_index = np.where(self.replay_update_list[0] == self.simulation_count)[0][-1]

                    self.ball_root_states[:, 0:3] = torch.tensor(self.record_data_state[0][replay_index,:3],
                                                                 device=self.device)
                    if replay_index < self.record_data_state[1].shape[0]:
                        self.ball_root_states[:, 7:10] = torch.tensor(self.record_data_state[1][replay_index,:3],device=self.device)
                    self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor))
                    self.gym.set_rigid_body_color(self.env, self.ball_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                                  gymapi.Vec3(0, 0, 1))
            if 0 < self.ball_root_states[:, 0] < 1.37 and -0.7625 < self.ball_root_states[:, 1] < 0.7625 and \
                    0.76 < self.ball_root_states[:, 2] < 0.9 and self.ball_root_states[:, 9] > 0.00005 and \
                    self.ball_root_states[:, 7] > -0.00005:
                self.trajectory_state[0] = 1
                self.trajectory_state_onehot = torch.zeros((1, 4), device=self.device, dtype=torch.long)
                self.trajectory_state_onehot[0,self.trajectory_state[0]] = 1
            if torch.norm(abs(self.ball_root_states[:, 0:3] - self.racket_pos), p=2, dim=-1) < 0.09 and \
                    self.trajectory_state[0] == 1:
                self.trajectory_state[0] = 2
                self.trajectory_state_onehot = torch.zeros((1, 4), device=self.device, dtype=torch.long)
                self.trajectory_state_onehot[0, self.trajectory_state[0]]= 1
                self.gym.set_rigid_body_color(self.env, self.ball_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                              gymapi.Vec3(1, 0, 0))
                self.replay_status = False
            if self.trajectory_state[0] == 1:
                self.gym.set_rigid_body_color(self.env, self.ball_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                              gymapi.Vec3(0, 1, 1))

            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self._simulation_viewer()  # step the viewer
            self._post_simulation()

            # sync_frame_rate function has delay, so I make the sync by myself
            time.sleep(max(self.simulation_count / args.frame_rate - real_time_all, 0))
            self.simulation_count += 1
            real_time_all += time.time() - cur_time

        print("Done")
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def _post_simulation(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)


if __name__ == '__main__':
    args = gymutil.parse_arguments(
        description="Robot Sim2Real",
        custom_parameters=[
            {"name": "--TCP", "type": bool, "default": False, "help": "Use UDP or TCP to receive or send data"},
            {"name": "--frame_rate", "type": int, "default": 200,
             "help": "The number of simulations during one second"},
            {"name": "--asset_root", "type": str, "default": "./",
             "help": "The root path of robot to to be loaded"},
            {"name": "--robot_asset", "type": str,
             "default": "urdf/PingPong_Robot_Sim2Real/urdf/PingPong_Robot_Sim2Real.urdf",
             "help": "The urdf file of robot to be loaded"},
            {"name": "--ball_asset", "type": str, "default": "urdf/PingPong_Robot/urdf/PingPong_Ball.urdf",
             "help": "The urdf file of ball to be loaded"},
            {"name": "--deterministic_policy", "type": bool, "default": True,
             "help": "Only be available when --use_policy is True. If --determinstic_policy is True, means the policy "
                     "is deterministic policy. If False, means the policy is stochastic."},
            {"name": "--policy_simulation_test", "type": bool, "default": True,
             "help": "This is used to verify whether the policy can run correctly. The ball position won't be "
                     "synchronized with the reality. The policy will just test in simulation"},
            {"name": "--real_ball_pos_buf_length", "type": int, "default": 13,
             "help": "The length of the buffer to receive the balls' positional data "},
            {"name": "--loaded_policy", "type": list,
             "default": ['./Models/stage-based_reward_model/stage-based_reward_init.pth',
                         './Models/stage-based_reward_model/stage-based_reward_mu.pth',
                         './Models/stage-based_reward_model/stage-based_reward_sigma.pth',
                         './Models/stage-based_reward_model/stage-based_reward_obs_mean.pth',
                         './Models/stage-based_reward_model/stage-based_reward_obs_var.pth']
                        , "help": "The policy need to be loaded"},
        ]
    )
    pingpong_robot_sim2real = PingPong_Robot_Sim2Real()
    pingpong_robot_sim2real.simulation()