# Scene Configuration Guide

This guide explains how to use the scene configuration system for the Spot robot controller.

## Configuration File

The `scene_configs.json` file contains predefined scene configurations with terrain paths, robot spawn positions, and camera settings.

### Configuration Structure

```json
{
  "scene_name": {
    "terrain_usd": "path/to/terrain.usd",
    "robot_position": [x, y, z],
    "robot_orientation": [w, x, y, z],
    "camera_distance": distance_behind_robot,
    "camera_height": height_above_robot,
    "sensitivity": [v_x, v_y, omega_z],
    "description": "Scene description"
  }
}
```

### Configuration Parameters

- **terrain_usd**: Path to the USD file for the terrain/environment
- **robot_position**: Initial position [x, y, z] where the robot spawns
- **robot_orientation**: Initial orientation as quaternion [w, x, y, z]
- **camera_distance**: How far behind the robot the camera follows (in meters)
- **camera_height**: How high above the robot the camera is positioned (in meters)
- **sensitivity**: Control sensitivity [v_x, v_y, omega_z] for forward/backward, strafe, and rotation speed
- **description**: Human-readable description of the scene

## Usage

### Using a Predefined Scene

Run with a scene name from the config file:

```bash
./isaaclab.sh -p gamemode_policy_inference_in_usd_safetypark.py \
    --checkpoint policy.pt \
    --gamepad \
    --scene airport
```

Available scenes (from default config):
- `warehouse` - Default warehouse scene
- `airport` - Airport mesh scene
- `safetypark` - Safety park scene

### Using a Custom Config File

Specify a different config file:

```bash
./isaaclab.sh -p gamemode_policy_inference_in_usd_safetypark.py \
    --checkpoint policy.pt \
    --gamepad \
    --scene my_scene \
    --config /path/to/my_configs.json
```

### Override Terrain Path

You can override the terrain path from command line:

```bash
./isaaclab.sh -p gamemode_policy_inference_in_usd_safetypark.py \
    --checkpoint policy.pt \
    --gamepad \
    --scene warehouse \
    --terrain_usd /path/to/custom_terrain.usd
```

## Controls

- **Left Stick**: Move robot (forward/backward/left/right)
- **Right Stick**: Rotate robot (left/right)
- The camera automatically follows behind the robot as it moves and rotates

## Control Sensitivity

You can adjust control sensitivity per scene using the `sensitivity` field:

```json
{
  "scene_name": {
    "sensitivity": [v_x, v_y, omega_z]
  }
}
```

- **v_x**: Forward/backward speed (default: 2.0 for keyboard, 1.0 for gamepad)
- **v_y**: Left/right strafe speed (default: 1.0)
- **omega_z**: Rotation speed (default: 1.6 for keyboard, 1.0 for gamepad)

Higher values = more sensitive controls. Adjust based on scene size and navigation needs.


