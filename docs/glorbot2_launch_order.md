# Glorbot2 Launch Order

Short version: robot control runs in `glorbot2_laptop_deploy`, Gello teleop runs in `glorbot2_gello_teleop`, and NEXT runs in `factr2_next`.

## 1. Start Robot Bringup

In `glorbot2_laptop_deploy`:

```bash
ros2 launch glorbot_system glorbot2_laptop_bringup.launch.py
```

This starts the Glorbot2 hardware/control stack, including the left and right Nero arm controllers.

## 2. Start Gello Teleop

In `glorbot2_gello_teleop`, either run the local launch:

```bash
ros2 launch teacher_arm nero_teleop.launch.py
```

or use the remote helper:

```bash
./start_all.sh
```

`start_all.sh` starts `nero_teleop` on the NUC and `joystick_publisher` on the Pi.

## 3. Start NEXT Adapter

In `glorbot2_laptop_deploy`, run one adapter per arm:

```bash
ros2 run next_adapter glorbot2_next_adapter --ros-args \
  -p robot_name:=glorbot2_1 -p arm:=left -p publish_hz:=60.0

ros2 run next_adapter glorbot2_next_adapter --ros-args \
  -p robot_name:=glorbot2_1 -p arm:=right -p publish_hz:=60.0
```

The adapter waits until it has both robot state and teleop command messages.

## 4. Record / Train / Infer With NEXT

In `factr2_next`:

```bash
ros2 run factr2_next next_record --ros-args \
  -p config_file:=/home/philip/Projects/factr2_next/src/factr2_next/factr2_next/config/record_glorbot2.yaml
```

After recording, train with `train_glorbot2.yaml`, then run one inference node per arm with the left/right inference configs.

## Topic Flow

Gello teleop publishes leader commands:

```text
/glorbot2/gello_1/nero_left/joint_state_cmd
/glorbot2/gello_1/nero_right/joint_state_cmd
```

`teleop_republisher` forwards them to the robot namespace:

```text
/glorbot2/glorbot2_1/nero_left/joint_state_cmd
/glorbot2/glorbot2_1/nero_right/joint_state_cmd
```

The Nero controllers publish robot state:

```text
/glorbot2/glorbot2_1/nero_left/joint_state
/glorbot2/glorbot2_1/nero_right/joint_state
```

`next_adapter` republishes state and command into NEXT format:

```text
/next_adapter/glorbot2_1/nero_left/*
/next_adapter/glorbot2_1/nero_right/*
```

`factr2_next` publishes external torque estimates:

```text
/next/glorbot2_1/nero_left/external_joint_torque
/next/glorbot2_1/nero_right/external_joint_torque
```
