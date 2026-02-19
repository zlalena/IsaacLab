# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates policy inference in a prebuilt USD environment with gamepad/keyboard control.

Features:
- Control Boston Dynamics Spot robot with gamepad or keyboard
- Camera automatically follows robot as it moves and rotates
- Scene configuration system for different environments
- Configurable robot spawn positions and camera settings

Usage:
    # Run with gamepad control in airport scene
    ./isaaclab.sh -p gamemode_policy_inference_in_usd_safetypark.py --checkpoint policy.pt --gamepad --scene airport

    # Run with keyboard control in warehouse
    ./isaaclab.sh -p gamemode_policy_inference_in_usd_safetypark.py --checkpoint policy.pt --keyboard --scene warehouse

    # Or use the convenience script
    ./run_scene.sh --scene airport --checkpoint policy.pt --gamepad

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Spot demo with fVDB Scene")
parser.add_argument(
    "--checkpoint",
    type=str,
    help="Path to model checkpoint exported as jit.",
    required=True,
)
parser.add_argument(
    "--keyboard",
    action="store_true",
    default=False,
    help="Use keyboard teleoperation to control the robot.",
)
parser.add_argument(
    "--gamepad",
    action="store_true",
    default=True,
    help="Use gamepad teleoperation to control the robot.",
)
parser.add_argument(
    "--terrain_usd",
    type=str,
    help="Path to terrain USD file. Default is generic warehouse.",
    required=False,
)
parser.add_argument(
    "--scene",
    type=str,
    default="airport",
    help="Scene configuration name from scene_configs.json (e.g. 'airport', 'safetypark')",
)
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="Path to scene_configs.json. Defaults to scene_configs.json in same directory as script.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import io
import json
import math
import os
import torch

import omni
from pxr import Gf, UsdGeom

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.flat_env_cfg import (
    SpotFlatEnvCfg_PLAY,
)


def main():
    """Main function to run robot control with camera following."""

    # ========================================
    # LOAD SCENE CONFIGURATION
    # ========================================
    # Determine path to scene configuration file
    config_path = args_cli.config
    if config_path is None:
        # Default to scene_configs.json in same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "scene_configs.json")

    # Load scene configuration from JSON file
    # This includes: terrain path, robot spawn position, camera settings
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            scene_configs = json.load(f)

        # Check if requested scene exists in config file
        if args_cli.scene not in scene_configs:
            print(
                f"[WARNING] Scene '{args_cli.scene}' not found in config. Available scenes: {list(scene_configs.keys())}"
            )
            print(f"[WARNING] Using default scene configuration")
            # Fallback to default warehouse scene
            scene_config = {
                "terrain_usd": f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
                "robot_position": [0.0, 0.0, 0.6],
                "robot_orientation": [1.0, 0.0, 0.0, 0.0],
                "camera_distance": 10.0,
                "camera_height": 10.0,
            }
        else:
            # Load the requested scene configuration
            scene_config = scene_configs[args_cli.scene]
            print(f"[INFO] Loaded scene configuration: {args_cli.scene}")
            print(f"[INFO] Description: {scene_config.get('description', 'N/A')}")
    else:
        # Config file doesn't exist, use default settings
        print(f"[WARNING] Config file not found at {config_path}")
        print(f"[WARNING] Using default scene configuration")
        scene_config = {
            "terrain_usd": f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
            "robot_position": [0.0, 0.0, 0.6],
            "robot_orientation": [1.0, 0.0, 0.0, 0.0],
            "camera_distance": 10.0,
            "camera_height": 10.0,
        }

    # Command line terrain path overrides config file
    if args_cli.terrain_usd:
        terrain_usd_path = args_cli.terrain_usd
        print(f"[INFO] Using terrain from command line: {terrain_usd_path}")
    else:
        terrain_usd_path = scene_config["terrain_usd"]
        print(f"[INFO] Using terrain from config: {terrain_usd_path}")

    # ========================================
    # LOAD TRAINED POLICY
    # ========================================
    # Load the trained neural network policy (exported as TorchScript JIT)
    policy_path = os.path.abspath(args_cli.checkpoint)
    file_content = omni.client.read_file(policy_path)[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    policy = torch.jit.load(file, map_location=args_cli.device)
    print(f"[INFO] Loaded policy from: {policy_path}")

    # ========================================
    # SETUP ENVIRONMENT
    # ========================================
    # Configure the Spot robot environment for play mode (no training)
    env_cfg = SpotFlatEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1  # Single environment (not parallel training)
    env_cfg.curriculum = None  # No curriculum learning

    # Import custom terrain from USD file
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=terrain_usd_path,
    )

    # Set device (GPU or CPU)
    env_cfg.sim.device = args_cli.device
    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False  # Disable Fabric for CPU mode

    # Disable automatic episode termination to prevent auto-resets
    # This prevents the robot from being reset to origin every ~100 steps
    env_cfg.episode_length_s = 1000000.0  # Very long episode (effectively infinite)

    # Disable termination conditions that would reset the robot
    if hasattr(env_cfg, "terminations"):
        env_cfg.terminations.time_out = None  # Disable timeout termination
        print("[INFO] Disabled timeout termination")

    print(
        f"[INFO] Episode length set to {env_cfg.episode_length_s}s (prevents auto-reset)"
    )

    # Disable command visualization arrows above robot
    if hasattr(env_cfg, "commands"):
        for attr_name in dir(env_cfg.commands):
            if not attr_name.startswith("_"):
                command_term = getattr(env_cfg.commands, attr_name)
                if hasattr(command_term, "debug_vis"):
                    command_term.debug_vis = False

    # ========================================
    # SETUP CONTROL INPUT (KEYBOARD OR GAMEPAD)
    # ========================================
    # Get control sensitivity from scene config
    sensitivity = scene_config.get(
        "sensitivity", [2.0, 1.0, 1.6]
    )  # Default if not specified
    v_x_sens, v_y_sens, omega_z_sens = sensitivity[0], sensitivity[1], sensitivity[2]

    use_keyboard = args_cli.keyboard
    invert_axes = False

    if use_keyboard:
        from isaaclab.devices.keyboard.se2_keyboard import Se2KeyboardCfg, Se2Keyboard

        input_device = Se2Keyboard(
            cfg=Se2KeyboardCfg(
                v_x_sensitivity=v_x_sens,
                v_y_sensitivity=v_y_sens,
                omega_z_sensitivity=omega_z_sens,
            )
        )
        print(
            f"[INFO] Keyboard control enabled with sensitivity: [{v_x_sens}, {v_y_sens}, {omega_z_sens}]"
        )
    else:
        from isaaclab.devices.gamepad.se2_gamepad import Se2GamepadCfg, Se2Gamepad

        input_device = Se2Gamepad(
            cfg=Se2GamepadCfg(
                v_x_sensitivity=v_x_sens,
                v_y_sensitivity=v_y_sens,
                omega_z_sensitivity=omega_z_sens,
                dead_zone=0.01,
            )
        )
        invert_axes = True
        print(
            f"[INFO] Gamepad control enabled with sensitivity: [{v_x_sens}, {v_y_sens}, {omega_z_sens}]"
        )

    # ========================================
    # CREATE ENVIRONMENT
    # ========================================
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print("[INFO] Environment created successfully")

    # ========================================
    # OVERRIDE ENVIRONMENT RESET TO USE CUSTOM SPAWN POSITION
    # ========================================
    # Store the original reset method
    original_reset = env.reset

    # Create custom reset that repositions robot after reset
    def custom_reset(*args, **kwargs):
        """Custom reset that spawns robot at configured position."""
        # Call original reset
        obs, info = original_reset(*args, **kwargs)

        # Reposition robot to configured spawn point
        robot_pose = torch.tensor(
            [
                [
                    float(scene_config["robot_position"][0]),
                    float(scene_config["robot_position"][1]),
                    float(scene_config["robot_position"][2]),
                    float(scene_config["robot_orientation"][0]),
                    float(scene_config["robot_orientation"][1]),
                    float(scene_config["robot_orientation"][2]),
                    float(scene_config["robot_orientation"][3]),
                ]
            ],
            device=env.unwrapped.device,
            dtype=torch.float32,
        )
        root_vel = torch.zeros((1, 6), device=env.unwrapped.device, dtype=torch.float32)

        # Apply position multiple times to ensure it sticks
        for _ in range(20):
            env.unwrapped.scene["robot"].write_root_pose_to_sim(robot_pose)
            env.unwrapped.scene["robot"].write_root_velocity_to_sim(root_vel)
            env.sim.step()

        # Get fresh observation after repositioning
        obs = env.unwrapped.observation_manager.compute()

        return obs, info

    # Replace the environment's reset method with our custom one
    env.reset = custom_reset
    print(
        f"[INFO] Custom reset installed - robot will spawn at {scene_config['robot_position']}"
    )

    # ========================================
    # SETUP FOLLOW CAMERA
    # ========================================
    # Create a camera that will dynamically follow the robot
    stage = omni.usd.get_context().get_stage()
    camera_path = "/World/FollowCamera"

    # Create USD camera with cinematic properties
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.GetHorizontalApertureAttr().Set(36.0)  # 36mm sensor width
    camera.GetVerticalApertureAttr().Set(20.25)  # 20.25mm sensor height (16:9)
    camera.GetFocalLengthAttr().Set(28.51)  # Focal length for ~64.5° FOV
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000000.0))  # Near/far clipping

    # Setup transform operators for dynamic camera movement
    # These allow us to update position and rotation every frame
    xform = UsdGeom.Xformable(camera)
    xform.ClearXformOpOrder()
    translate_op = xform.AddTranslateOp()  # For position updates
    rotate_op = xform.AddRotateXYZOp()  # For rotation updates

    # Set the viewport to use our follow camera
    viewport_api = omni.kit.viewport.utility.get_active_viewport()
    viewport_api.set_active_camera(camera_path)

    # Load camera follow parameters from scene configuration
    camera_distance = scene_config["camera_distance"]  # How far behind robot
    camera_height = scene_config["camera_height"]  # How high above robot
    print(
        f"[INFO] Camera settings - Distance: {camera_distance}m, Height: {camera_height}m"
    )

    # ========================================
    # RUN INFERENCE LOOP
    # ========================================
    # Reset environment to initial state
    obs, _ = env.reset()

    # Position robot at configured spawn point from scene config
    print(f"\n{'='*60}")
    print(f"ROBOT POSITIONING")
    print(f"{'='*60}")
    print(f"Target position: {scene_config['robot_position']}")
    print(f"Target orientation (quat): {scene_config['robot_orientation']}")

    # Check current position before we do anything
    current_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
    print(
        f"Current position before repositioning: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}]"
    )

    # Get robot prim path for debugging
    robot_prim_path = env.unwrapped.scene["robot"].cfg.prim_path
    print(f"Robot prim path: {robot_prim_path}")

    # Create pose tensor: [x, y, z, qw, qx, qy, qz]
    robot_pose = torch.tensor(
        [
            [
                float(scene_config["robot_position"][0]),
                float(scene_config["robot_position"][1]),
                float(scene_config["robot_position"][2]),
                float(scene_config["robot_orientation"][0]),
                float(scene_config["robot_orientation"][1]),
                float(scene_config["robot_orientation"][2]),
                float(scene_config["robot_orientation"][3]),
            ]
        ],
        device=env.unwrapped.device,
        dtype=torch.float32,
    )

    print(f"Pose tensor shape: {robot_pose.shape}")
    print(f"Pose tensor: {robot_pose}")

    # Set robot velocity to zero
    root_vel = torch.zeros((1, 6), device=env.unwrapped.device, dtype=torch.float32)

    # Apply position repeatedly
    print(f"Applying position over 30 simulation steps...")
    for i in range(30):
        env.unwrapped.scene["robot"].write_root_pose_to_sim(robot_pose)
        env.unwrapped.scene["robot"].write_root_velocity_to_sim(root_vel)
        env.sim.step()

        # Check position every 10 steps
        if i % 10 == 9:
            check_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
            print(
                f"  Step {i+1}: position = [{check_pos[0]:.2f}, {check_pos[1]:.2f}, {check_pos[2]:.2f}]"
            )

    # Verify final position
    actual_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
    print(
        f"\nFinal position: [{actual_pos[0]:.2f}, {actual_pos[1]:.2f}, {actual_pos[2]:.2f}]"
    )

    # Check if position was set successfully
    pos_error = [
        abs(actual_pos[0] - scene_config["robot_position"][0]),
        abs(actual_pos[1] - scene_config["robot_position"][1]),
        abs(actual_pos[2] - scene_config["robot_position"][2]),
    ]

    if pos_error[0] > 1.0 or pos_error[1] > 1.0:
        print(f"\n⚠️  WARNING: Robot position NOT set correctly!")
        print(
            f"   Error: X={pos_error[0]:.2f}m, Y={pos_error[1]:.2f}m, Z={pos_error[2]:.2f}m"
        )
    else:
        print(f"\n✓ Robot position set successfully!")

    print(f"{'='*60}\n")

    # Get fresh observation after repositioning
    obs = env.unwrapped.observation_manager.compute()

    # Store target position for respawn detection
    target_position = scene_config["robot_position"]
    last_known_pos = (
        env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy().copy()
    )

    # Main simulation loop with policy inference
    with torch.inference_mode():
        step_count = 0
        last_reposition_step = 0

        while simulation_app.is_running():
            step_count += 1
            # ========================================
            # PROCESS CONTROL INPUT
            # ========================================

            cmd = input_device.advance()  # Returns [v_x, v_y, omega_z]
            if invert_axes:
                cmd[1] = -cmd[1]  # Lateral movement
                cmd[2] = -cmd[2]  # Rotation
            cmd_tensor = torch.tensor(cmd)
            policy_tensor: torch.Tensor = obs["policy"]  # type: ignore
            policy_tensor = policy_tensor.clone()
            policy_tensor[:, 9:12] = cmd_tensor.to(
                policy_tensor.device, policy_tensor.dtype
            )
            obs["policy"] = policy_tensor

            # ========================================
            # RUN POLICY AND STEP SIMULATION
            # ========================================
            # Get action from policy network
            action = policy(obs["policy"])
            # Execute action in environment and get new observation
            obs, rewards, terminated, truncated, info = env.step(action)

            # If episode terminated, manually reset and reposition
            if terminated.any() or truncated.any():
                print(
                    f"\n[INFO] Episode terminated at step {step_count}, resetting to spawn point..."
                )
                obs, _ = env.reset()  # This will use our custom reset hook
                actual_pos = (
                    env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
                )
                print(
                    f"[INFO] Robot reset to: [{actual_pos[0]:.1f}, {actual_pos[1]:.1f}]\n"
                )

            # ========================================
            # RESPAWN DETECTION AND AUTO-REPOSITION
            # ========================================
            # Check if robot has been teleported/respawned back to origin
            current_robot_pos = (
                env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
            )

            # Calculate distance from target position
            distance_from_target = (
                (current_robot_pos[0] - target_position[0]) ** 2
                + (current_robot_pos[1] - target_position[1]) ** 2
            ) ** 0.5

            # If robot is far from target AND near origin, it likely respawned
            distance_from_origin = (
                current_robot_pos[0] ** 2 + current_robot_pos[1] ** 2
            ) ** 0.5

            # Detect respawn: robot is near origin but should be far away
            # More strict detection to avoid false positives
            # Only trigger if robot is VERY close to origin (< 2m) and far from target
            if (
                distance_from_origin < 2.0
                and distance_from_target > 15.0
                and step_count - last_reposition_step > 50
            ):
                print(
                    f"\n[WARNING] Unexpected respawn at origin detected at step {step_count}!"
                )
                print(
                    f"          Current: [{current_robot_pos[0]:.1f}, {current_robot_pos[1]:.1f}], Target: {target_position}"
                )
                print(f"          This should not happen with custom reset enabled.")

                # Reapply target position
                for i in range(20):
                    env.unwrapped.scene["robot"].write_root_pose_to_sim(robot_pose)
                    env.unwrapped.scene["robot"].write_root_velocity_to_sim(root_vel)
                    env.sim.step()

                last_reposition_step = step_count

                # Verify reposition
                new_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
                print(
                    f"          Repositioned to: [{new_pos[0]:.1f}, {new_pos[1]:.1f}]\n"
                )

            # ========================================
            # UPDATE FOLLOW CAMERA
            # ========================================
            # Get robot's current world position and orientation
            robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
            robot_quat = (
                env.unwrapped.scene["robot"].data.root_quat_w[0].cpu().numpy()
            )  # [w, x, y, z]

            # Convert robot quaternion to rotation matrix to get robot's forward direction
            # This allows camera to stay behind robot as it rotates
            w, x, y, z = (
                float(robot_quat[0]),
                float(robot_quat[1]),
                float(robot_quat[2]),
                float(robot_quat[3]),
            )

            # Extract forward direction vector from quaternion (robot's local X-axis)
            # These formulas convert quaternion to the first column of rotation matrix
            forward_x = 1 - 2 * (y * y + z * z)
            forward_y = 2 * (x * y + w * z)
            forward_z = 2 * (x * z - w * y)

            # Calculate camera position: behind and above robot in robot's local frame
            # Subtracting forward vector positions camera behind robot
            # Convert to Python float to avoid numpy type issues with USD API
            camera_pos = [
                float(robot_pos[0] - forward_x * camera_distance),
                float(robot_pos[1] - forward_y * camera_distance),
                float(robot_pos[2] + camera_height),
            ]

            # Apply camera position to USD transform
            translate_op.Set(Gf.Vec3d(camera_pos[0], camera_pos[1], camera_pos[2]))

            # Calculate camera rotation to look at robot (look-at calculation)
            eye = Gf.Vec3d(camera_pos[0], camera_pos[1], camera_pos[2])
            target = Gf.Vec3d(
                float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2] + 0.5)
            )  # Target point 0.5m above robot base (roughly robot's head)
            forward = (target - eye).GetNormalized()

            # Convert look direction to pitch and yaw angles
            pitch = math.asin(-forward[2])  # Vertical angle
            yaw = math.atan2(forward[1], forward[0])  # Horizontal angle

            # Apply camera orientation to USD transform
            # The -90 offset corrects for USD camera's default orientation
            rotate_op.Set(Gf.Vec3d(math.degrees(pitch), 0, math.degrees(yaw) - 90))


if __name__ == "__main__":
    main()
    simulation_app.close()
    simulation_app.close()
