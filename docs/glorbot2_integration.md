# Glorbot2 Integration

This branch keeps `factr2_next` robot-agnostic. Glorbot2 support comes from a small adapter node in `glorbot2_laptop_deploy` that republishes the robot state and command streams into the plain `sensor_msgs/JointState` topics used by NEXT.

## Adapter Contract

Run one adapter node per arm. Each adapter subscribes to the native Glorbot2 topics:

- `/glorbot2/{robot_name}/nero_{arm}/joint_state`
- `/glorbot2/{robot_name}/nero_{arm}/joint_state_cmd`

and publish:

- `/next_adapter/{robot_name}/nero_{arm}/joint_pos_obs`
- `/next_adapter/{robot_name}/nero_{arm}/joint_vel_obs`
- `/next_adapter/{robot_name}/nero_{arm}/joint_pos_cmd`
- `/next_adapter/{robot_name}/nero_{arm}/joint_effort_obs`

All four adapter outputs should be `sensor_msgs/JointState`, with the numeric vector in `.position`. This matches the existing Piper convention and lets the current recorder, trainer, and inference node run without special Glorbot2 code.

## Expected Mapping

- `joint_pos_obs.position = latest joint_state.position`
- `joint_vel_obs.position = latest joint_state.velocity`
- `joint_effort_obs.position = latest joint_state.effort`
- `joint_pos_cmd.position = latest joint_state_cmd.target.position`

Publish the four adapter messages from one timer using the same ROS timestamp. Wait until both a state message and a command message have arrived before publishing.

Example bimanual startup:

```bash
ros2 run next_adapter glorbot2_next_adapter --ros-args \
  -p robot_name:=glorbot2_1 -p arm:=left -p publish_hz:=60.0

ros2 run next_adapter glorbot2_next_adapter --ros-args \
  -p robot_name:=glorbot2_1 -p arm:=right -p publish_hz:=60.0
```

## Commands

Record bimanual free-motion data:

```bash
ros2 run factr2_next next_record --ros-args \
  -p config_file:=/home/philip/Projects/factr2_next/src/factr2_next/factr2_next/config/record_glorbot2.yaml
```

Train NEXT. This writes one run directory for the left arm and one for the right arm:

```bash
ros2 run factr2_next next_train \
  --config /home/philip/Projects/factr2_next/src/factr2_next/factr2_next/config/train_glorbot2.yaml
```

Run inference once per arm, using the matching checkpoint directory in each config:

```bash
ros2 run factr2_next next_infer --ros-args \
  -p config_file:=/home/philip/Projects/factr2_next/src/factr2_next/factr2_next/config/inference_glorbot2.yaml

ros2 run factr2_next next_infer --ros-args \
  -p config_file:=/home/philip/Projects/factr2_next/src/factr2_next/factr2_next/config/inference_glorbot2_left.yaml
```

Run the visualizer. The default Glorbot2 visualizer config is right-arm, and the left-arm config uses port 8081:

```bash
ros2 run factr2_next next_visualize --ros-args \
  -p config_file:=/home/philip/Projects/factr2_next/src/factr2_next/factr2_next/config/visualize_glorbot2.yaml

ros2 run factr2_next next_visualize --ros-args \
  -p config_file:=/home/philip/Projects/factr2_next/src/factr2_next/factr2_next/config/visualize_glorbot2_left.yaml
```

## Postconditions

- Adapter topics publish 7-DoF vectors in `JointState.position` for both arms.
- `next_record` writes H5 episodes with `left_*` and `right_*` keys for position, velocity, command, and measured torque.
- `next_train` creates separate left and right NEXT checkpoints from the bimanual H5.
- `next_infer` publishes `/next/glorbot2_1/nero_{arm}/external_joint_torque`, `/raw`, `free_joint_torque_pred`, `mse`, `score`, and `contact_state`.
- Gello leader feedback is intentionally separate. It should live in `glorbot2_gello_teleop` and subscribe to the `/next/...` outputs.
