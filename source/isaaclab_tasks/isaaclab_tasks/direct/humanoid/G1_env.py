from __future__ import annotations

from isaaclab_assets import HUMANOID_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, wrap_to_pi
from .G1_cfg import G1EnvCfg

import torch
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import os
from datetime import datetime




class G1Env(DirectRLEnv):
    cfg: G1EnvCfg

    def __init__(self, cfg: G1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # action offset and scale
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits
        print("niuniuniuniuniuniuniu")
        print(dof_lower_limits)
        self.root_body_index = self.robot.data.body_names.index(self.cfg.root_body)

        
        self.left_foot_index = 24   # left_ankle_roll_link
        self.right_foot_index = 25  # right_ankle_roll_link

        # initialize paramters
        self._init_buffers()

        # Initialize contact forces buffer (for contact reward)
        self.contact_forces = torch.zeros(self.num_envs, len(self.robot.data.body_names), 3, dtype=torch.float, device=self.device)

        self._prepare_reward_function()

    def _init_buffers(self):
        self.commands = torch.zeros(self.num_envs, self.cfg.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        base = torch.tensor([1., 0., 0.], dtype=torch.float32, device=self.device)
        self.forward_vec = base.repeat(self.num_envs, 1)
        
        # Foot indices for ContactSensor
        self.feet_num = 2
        self.feet_indices = torch.tensor([self.left_foot_index, self.right_foot_index], device=self.device)
        
        # Gait phase tracking for contact reward
        # Phase ranges from 0 to 1, < 0.55 means stance phase, >= 0.55 means swing phase
        self.leg_phase = torch.zeros(self.num_envs, self.feet_num, dtype=torch.float, device=self.device)
        self.period = 0.8  # gait period in seconds
        self.offset = 0.5  # phase offset between left and right legs
        
        # Initialize TensorBoard logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"./logs/G1_training_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        self.tensorboard_writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logging to: {log_dir}")
        print(f"Run 'tensorboard --logdir {log_dir}' to view in browser")

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add contact sensor
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add sensor to scene
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model.apply(action)

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        self.post_physics_step()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def post_physics_step(self):
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self._post_physics_step_callback()

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()

        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()
        
        self.last_actions = self.actions[:]

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])

    def _prepare_reward_function(self):
        """Prepares the list of reward functions to be called for computing the total reward.
        to all reward types with non-zero scales specified in the humanoid_cfg.
        """
        self.reward_scales = {
            "alive": self.cfg.alive_reward_scale,
            "base_height": self.cfg.base_height_reward_scale,
            "action_rate": self.cfg.action_rate_reward_scale,
            "tracking_lin_vel": self.cfg.tracking_lin_vel_reward_scale,
            "tracking_ang_vel": self.cfg.tracking_ang_vel_reward_scale,
            "orientation": self.cfg.orientation_reward_scale,
            "default_pose":  self.cfg.default_pose_reward_scale,
            "symmetry": self.cfg.symmetry_reward_scale,
            "contact": self.cfg.contact_reward_scale,
            "lin_vel_z": self.cfg.lin_vel_z_reward_scale,
            "ang_vel_xy": self.cfg.ang_vel_xy_reward_scale,
            "hip_pos": self.cfg.hip_pos_reward_scale,
            "contact_no_vel": self.cfg.contact_no_vel_reward_scale,
            "feet_swing_height": self.cfg.feet_swing_height_reward_scale,
            "dof_pos_limits": self.cfg.dof_pos_limits_reward_scale
        }
        
        # Remove entries with zero scale and multiply non-zero scales by simulation timestep (dt)
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.sim.cfg.dt
        
         # Prepare the list of reward functions and their corresponding names
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))
        
        # Initialize episode reward accumulation buffer for logging
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.sim.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }

    

    def _pre_physics_step(self, actions: torch.Tensor):
        # clip_actions = self.cfg.clip_actions
        # actions = torch.clip(actions, -clip_actions, clip_actions)
        self.actions = actions.clone()

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)

    def _get_observations(self) -> dict:
        # build task observation
        obs = compute_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.projected_gravity_b,
            self.commands[:, :3],
            self.robot.data.root_link_ang_vel_b,
            self.actions,
        )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        total_reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.sim.device)
        
        # Store computed rewards for printing
        computed_rewards = []
        
        env_id = 0  # Use environment 0 for logging
        step = self.common_step_counter
        
        for i, reward_func in enumerate(self.reward_functions):
            name = self.reward_names[i]
            scale = self.reward_scales[name]
            reward = reward_func()
            
            scaled_reward = reward * scale
            total_reward += scaled_reward
            
            # Store for potential printing
            computed_rewards.append((name, reward, scale, scaled_reward))
            
            # Log to TensorBoard immediately
            raw_val = reward[env_id].item()
            scaled_val = scaled_reward[env_id].item()
            
            self.tensorboard_writer.add_scalar(f'Rewards_Raw/{name}', raw_val, step)
            self.tensorboard_writer.add_scalar(f'Rewards_Scaled/{name}', scaled_val, step)

        # Log total reward
        self.tensorboard_writer.add_scalar('Rewards/Total', total_reward[env_id].item(), step)
        
        # Log reward scales for reference (only once at the beginning)
        if self.common_step_counter == 1:
            for name, scale in self.reward_scales.items():
                self.tensorboard_writer.add_scalar(f'Reward_Scales/{name}', scale, 0)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.root_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        
        for key in self.episode_sums.keys():
            self.episode_sums[key][env_ids] = 0.0
            
        # Reset gait phase
        self.leg_phase[env_ids] = 0.0
            
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Use default reset strategy
        root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._resample_commands(env_ids)

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # Update gait phase for contact reward using time-based calculation
        period = 0.8
        offset = 0.5
        # Use step_dt (control timestep) instead of sim.cfg.dt (simulation timestep)
        self.phase = (self.episode_length_buf * self.step_dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        # Get real contact forces from ContactSensor and update contact_forces buffer
        net_contact_forces = self.contact_sensor.data.net_forces_w  # [num_envs, 2, 3]
        # Update contact forces buffer (only feet indices, rest remain zero)
        self.contact_forces.zero_()  # Clear all forces
        self.contact_forces[:, self.feet_indices, :] = net_contact_forces
        
        # Update feet velocities for contact_no_vel reward
        self.feet_vel = self.robot.data.body_lin_vel_w[:, self.feet_indices, :]  # [num_envs, 2, 3]
        
        # Update feet positions for swing height reward
        self.feet_pos = self.robot.data.body_pos_w[:, self.feet_indices, :]  # [num_envs, 2, 3]
        
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.resampling_time / self.sim.cfg.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.heading_command:
            forward = quat_apply(self.robot.data.root_link_quat_w, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
            
            
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.cfg.lin_vel_x[0], self.cfg.lin_vel_x[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.cfg.lin_vel_y[0], self.cfg.lin_vel_y[1], (len(env_ids), 1), device=self.device).squeeze(1)

        if self.cfg.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.cfg.heading[0], self.cfg.heading[1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.cfg.ang_vel_yaw[0], self.cfg.ang_vel_yaw[1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # print(self.commands.shape)

    # -------- Reward Functions --------
    def _reward_alive(self):
        return torch.ones(self.num_envs, dtype=torch.float, device=self.sim.device) 

    def _reward_base_height(self):
        """Penalize base height away from target"""
        base_height = self.robot.data.root_link_state_w[:, 2]
        return torch.square(base_height - self.cfg.base_height_target)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_tracking_lin_vel(self):
        """
        Tracking of linear velocity commands (xy axes).
        Uses exponential reward based on squared error between commanded and actual velocity.
        """
        # Get current base linear velocity in xy plane
        base_lin_vel = self.robot.data.root_link_lin_vel_b[:, :2]  # [num_envs, 2]
        
        # Calculate squared error between command and actual velocity
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - base_lin_vel), dim=1)
        
        # Return exponential reward (higher reward for smaller error)
        return torch.exp(-lin_vel_error / self.cfg.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.tracking_sigma)
    
    def _reward_orientation(self):
        # Penalize deviation from upright (flat) base orientation.
        # Encourages the center of mass to stay balanced between the legs,
        return torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
    
    def _reward_contact(self):
        """
        Reward proper contact phases - stance legs should be in contact, swing legs should not be.
        Uses gait phase to determine expected contact state and rewards when actual contact matches expectation.
        """
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55  # stance phase when phase < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1  # foot in contact
            # Reward when stance phase matches contact state (XOR gives 0 when both are same)
            res += ~(contact ^ is_stance)
        return res.float()

    def _reward_lin_vel_z(self):
        """
        Penalize z axis base linear velocity.
        Discourages jumping or bouncing motions, encouraging stable ground-based locomotion.
        """
        # Get z-axis velocity of the robot base
        base_lin_vel_z = self.robot.data.root_link_lin_vel_b[:, 2]
        return torch.square(base_lin_vel_z)

    def _reward_ang_vel_xy(self):
        """
        Penalize xy axes base angular velocity.
        Discourages excessive roll and pitch angular velocities, encouraging stable upright locomotion.
        """
        # Get x and y axis angular velocity of the robot base (roll and pitch rates)
        base_ang_vel_xy = self.robot.data.root_link_ang_vel_b[:, :2]
        return torch.sum(torch.square(base_ang_vel_xy), dim=1)

    def _reward_hip_pos(self):
        """
        Penalize specific hip and torso joint positions away from zero.
        Encourages stable upright posture by keeping torso straight and limiting hip yaw movements.
        Targets: right_hip_pitch, torso, left_hip_yaw, right_hip_yaw joints.
        """
        # Get joint positions for specific joints: [1,2,7,8]
        # [1]: right_hip_pitch_joint, [2]: torso_joint
        # [7]: left_hip_yaw_joint, [8]: right_hip_yaw_joint
        joint_pos = self.robot.data.joint_pos
        target_joints = joint_pos[:, [1, 2, 7, 8]]
        return torch.sum(torch.square(target_joints), dim=1)

    def _reward_contact_no_vel(self):
        """
        Penalize contact with velocity (foot sliding).
        Encourages stable contact by penalizing feet that are in contact but still moving,
        which indicates sliding or dragging motions.
        """
        # Detect which feet are in contact (force > 1N)
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0  # [num_envs, 2]
        
        # Get feet velocities and mask with contact state
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)  # [num_envs, 2, 3]
        
        # Penalize squared velocity when in contact (sliding penalty)
        penalize = torch.square(contact_feet_vel[:, :, :3])  # [num_envs, 2, 3]
        
        # Sum over feet and velocity dimensions
        return torch.sum(penalize, dim=(1, 2))  # [num_envs]

    def _reward_feet_swing_height(self):
        """
        Encourage proper swing leg height during swing phase based on gait phase.
        Rewards feet being lifted to target height (0.05m) during swing phase (phase >= 0.55).
        Uses gait phase information instead of just contact detection for more accurate timing.
        """
        # Use gait phase to determine swing phase (more accurate than contact detection)
        is_swing = self.leg_phase >= 0.55  # [num_envs, 2]
        
        # Get current foot heights
        foot_heights = self.feet_pos[:, :, 2]  # [num_envs, 2]
        
        # Target height during swing phase (reduced from 8cm to 5cm)
        target_height = 0.08
        
        # Calculate height error only during swing phase
        height_error = torch.square(foot_heights - target_height) * is_swing.float()  # [num_envs, 2]
        
        # Also add a small bonus for achieving target height during swing
        height_bonus = torch.exp(-height_error / 0.01) * is_swing.float()  # [num_envs, 2]
        
        # Return penalty for deviation minus bonus for achieving target
        penalty = torch.sum(height_error, dim=1)  # [num_envs]
        bonus = torch.sum(height_bonus, dim=1)  # [num_envs]
        
        return penalty - 0.1 * bonus  # Small bonus to encourage lifting

    def _reward_dof_pos_limits(self):
        """
        Penalize joint positions too close to limits to prevent unnatural poses.
        """
        dof_pos = self.robot.data.joint_pos
        dof_pos_limits_lower = self.robot.data.soft_joint_pos_limits[:, :, 0]  # Lower limits
        dof_pos_limits_upper = self.robot.data.soft_joint_pos_limits[:, :, 1]  # Upper limits
        
        # Calculate violations of lower and upper limits
        lower_violations = torch.clamp(dof_pos_limits_lower - dof_pos, min=0.0)
        upper_violations = torch.clamp(dof_pos - dof_pos_limits_upper, min=0.0)
        
        # Total violation (will be multiplied by negative scale to penalize)
        total_violations = torch.sum(lower_violations + upper_violations, dim=1)
        
        return total_violations


    def _reward_default_pose(self):
        """
        Rewards joint positions that stay close to the default pose using a Gaussian-shaped reward.
        Maximum reward is given when the joint is exactly at its default position.
        Only applies to upper body joints: torso, shoulder, and elbow joints.
        """
        joint_pos = self.robot.data.joint_pos                                # [num_envs, num_joints]
        default_pos = self.robot.data.default_joint_pos                      # [num_envs, num_joints]

        # Upper body joint indices: torso, shoulder, elbow joints
        # Based on joint order: [2]: torso, [5,6]: shoulder_pitch, [9,10]: shoulder_roll, 
        # [13,14]: shoulder_yaw, [17,18]: elbow_pitch, [21,22]: elbow_roll
        upper_body_indices = [2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22]
        
        # Select only upper body joints
        joint_pos_upper = joint_pos[:, upper_body_indices]                   # [num_envs, num_upper_joints]
        default_pos_upper = default_pos[:, upper_body_indices]               # [num_envs, num_upper_joints]

        # Compute squared error between current and default joint positions
        squared_error = torch.square(joint_pos_upper - default_pos_upper)    # [num_envs, num_upper_joints]

        # Sum over joints
        error_sum = torch.sum(squared_error, dim=1)                          # [num_envs]

        # Gaussian reward: R = exp( - error / alpha )
        alpha = 0.1  # can be tuned depending on the range of joint values
        reward = torch.exp(-error_sum / alpha)

        return reward

    def _reward_symmetry(self):
        """
        Penalize asymmetric joint positions between left and right limbs.
        Encourages symmetric walking gait by comparing corresponding left-right joint pairs.
        """
        joint_pos = self.robot.data.joint_pos  # [num_envs, num_joints]
        
        # Define left-right joint pairs (indices based on actual joint order)
        left_right_pairs = [
            (0, 1),   # left_hip_pitch_joint, right_hip_pitch_joint
            (3, 4),   # left_hip_roll_joint, right_hip_roll_joint  
            (7, 8),   # left_hip_yaw_joint, right_hip_yaw_joint
            (11, 12), # left_knee_joint, right_knee_joint
            (15, 16), # left_ankle_pitch_joint, right_ankle_pitch_joint
            (19, 20), # left_ankle_roll_joint, right_ankle_roll_joint
            (5, 6),   # left_shoulder_pitch_joint, right_shoulder_pitch_joint
            (9, 10),  # left_shoulder_roll_joint, right_shoulder_roll_joint
            (13, 14), # left_shoulder_yaw_joint, right_shoulder_yaw_joint
            (17, 18), # left_elbow_pitch_joint, right_elbow_pitch_joint
            (21, 22), # left_elbow_roll_joint, right_elbow_roll_joint
        ]
        
        symmetry_error = torch.zeros(self.num_envs, device=self.sim.device)
        
        for left_idx, right_idx in left_right_pairs:
            # For hip_yaw and shoulder_yaw, we expect mirrored angles (opposite signs)
            if left_idx in [7, 13]:  # hip_yaw and shoulder_yaw joints
                error = torch.square(joint_pos[:, left_idx] + joint_pos[:, right_idx])
            else:
                # For other joints, we expect similar angles
                error = torch.square(joint_pos[:, left_idx] - joint_pos[:, right_idx])
            
            symmetry_error += error
        
        return symmetry_error





@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)

@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,         # [N, J]
    dof_velocities: torch.Tensor,        # [N, J]
    projected_gravity: torch.Tensor,     # [N, 3]
    commands: torch.Tensor,              # [N, 4]  
    root_angular_velocities: torch.Tensor,# [N, 3]
    actions: torch.Tensor                # [N,J] 
) -> torch.Tensor:
    return torch.cat(
        (
            dof_positions,
            dof_velocities,
            projected_gravity,
            commands,
            root_angular_velocities,
            actions
        ),
        dim=-1,
    )
    return obs

def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(shape, device=device) + lower
