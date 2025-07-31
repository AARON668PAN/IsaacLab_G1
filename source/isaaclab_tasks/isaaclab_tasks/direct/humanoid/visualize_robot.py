from isaaclab.app import AppLauncher

# Launch Isaac Sim in GUI (non-headless) mode
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# Import your environment and configuration
from isaaclab_tasks.direct.humanoid.G1_env import G1Env
from isaaclab_tasks.direct.humanoid.G1_cfg import G1EnvCfg

import torch

# Create the configuration instance
cfg = G1EnvCfg()

# Temporarily change num_envs to 1
cfg.scene.num_envs = 1

# Create the environment
env = G1Env(cfg)

# Set the joint positions to the default joint pose (static, no control applied)
env.robot.write_joint_position_to_sim(env.robot.data.default_joint_pos.clone())

# Start the rendering loop (no physics simulation step)
while simulation_app.is_running():
    env.sim.render()
