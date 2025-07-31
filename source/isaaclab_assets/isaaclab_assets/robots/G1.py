# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Mujoco Humanoid robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

HUMANOID_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # on server or It can also be loaded from the local path
        usd_path="usd_files/g1_23dof.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
    
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            # Leg joints for natural standing pose
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            "left_knee_joint": 0.3,
            "right_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "right_ankle_pitch_joint": -0.2,
            # All other joints explicitly set to 0
            "left_hip_yaw_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "left_ankle_roll_joint": 0.0,
            "right_ankle_roll_joint": 0.0,
            "torso_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "left_elbow_pitch_joint": 0.0,
            "right_elbow_pitch_joint": 0.0,
            "left_elbow_roll_joint": 0.0,
            "right_elbow_roll_joint": 0.0,
        },
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                "torso_joint": 20.0,
                ".*_shoulder.*": 10.0,
                ".*_elbow.*": 2.0,
                ".*_hip.*": 10.0,
                ".*_knee.*": 5.0,
                ".*_ankle.*": 2.0,
            },
            damping={
                "torso_joint": 5.0,
                ".*_shoulder.*": 5.0,
                ".*_elbow.*": 1.0,
                ".*_hip.*": 5.0,
                ".*_knee.*": 0.1,
                ".*_ankle.*": 1.0,
            },
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)
"""Configuration for the Mujoco Humanoid robot."""
