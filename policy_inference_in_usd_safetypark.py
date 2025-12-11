# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates policy inference in a prebuilt USD environment.

In this example, we use a locomotion policy to control the Boston Dynamics Spot robot. The robot was trained
using Isaac-Velocity-Flat-Spot-v0. The robot is commanded to move forward at a constant velocity.

.. code-block:: bash

    # Run the script
    ./isaaclab.sh -p scripts/tutorials/03_envs/policy_inference_in_usd.py --checkpoint /path/to/jit/checkpoint.pt

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on inferencing a policy on a Boston Dynamics Spot robot on a custom terrain mesh.")
parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint exported as jit.", required=True)
parser.add_argument("--keyboard", action="store_true", default=False, help="Use keyboard teleoperation to control the robot.")
parser.add_argument("--gamepad", action="store_true", default=False, help="Use gamepad teleoperation to control the robot.")
parser.add_argument("--terrain_usd", type=str, help="Path to terrain USD file. Default is generic warehouse.", required=False)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import io
import os
import torch

import omni

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.flat_env_cfg import SpotFlatEnvCfg_PLAY

def main():
    """Main function."""
    # load the trained jit policy
    policy_path = os.path.abspath(args_cli.checkpoint)
    file_content = omni.client.read_file(policy_path)[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    policy = torch.jit.load(file, map_location=args_cli.device)

    # setup environment

    if args_cli.terrain_usd:
        try:
            terrain_usd_path = args_cli.terrain_usd
        except:
            raise ValueError("Valid path to terrain USD is required.")
    else:
        terrain_usd_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd"

    env_cfg = SpotFlatEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum = None
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=terrain_usd_path,
    )
    env_cfg.sim.device = args_cli.device
    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False

    if args_cli.keyboard:
        from isaaclab.devices.keyboard.se2_keyboard import Se2KeyboardCfg, Se2Keyboard
        
        keyboard = Se2Keyboard(cfg=Se2KeyboardCfg(v_x_sensitivity=2.0, v_y_sensitivity=1.0, omega_z_sensitivity=1.6))
        
        print(keyboard)

    elif args_cli.gamepad:
        from isaaclab.devices.gamepad.se2_gamepad import Se2GamepadCfg, Se2Gamepad
        gamepad = Se2Gamepad(cfg=Se2GamepadCfg(v_x_sensitivity = 1.0, v_y_sensitivity = 1.0, omega_z_sensitivity = 1.0, dead_zone = 0.01))
        print(gamepad)

    # create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # run inference with the policy
    obs, _ = env.reset()
    with torch.inference_mode():
        while simulation_app.is_running():
            if args_cli.keyboard:
                cmd = keyboard.advance()
                cmd_tensor = torch.tensor(cmd)
                policy_tensor: torch.Tensor = obs["policy"]  # type: ignore
                policy_tensor = policy_tensor.clone()
                policy_tensor[:,9:12] = cmd_tensor.to(policy_tensor.device, policy_tensor.dtype)
                obs["policy"] = policy_tensor
            if args_cli.gamepad:
                cmd = gamepad.advance()
                cmd_tensor = torch.tensor(cmd)
                policy_tensor: torch.Tensor = obs["policy"]  # type: ignore
                policy_tensor = policy_tensor.clone()
                policy_tensor[:,9:12] = cmd_tensor.to(policy_tensor.device, policy_tensor.dtype)
                obs["policy"] = policy_tensor
            action = policy(obs["policy"])
            obs, _, _, _, _ = env.step(action)


if __name__ == "__main__":
    main()
    simulation_app.close()
