# FACTR2 Evaluation

The default evaluation configuration records the **right arm** at 60 Hz. Run
each live command in a separate terminal, in this order.

## 1. Robot bringup

From the `glorbot2_laptop_deploy` workspace:

```bash
ros2 launch glorbot_system glorbot2_laptop_bringup.launch.py
```

## 2. Gello teleoperation and feedback

From the `glorbot2_gello_teleop` workspace:

```bash
ros2 launch teacher_arm nero_teleop.launch.py
```

The teleop configuration must have FACTR2 feedback enabled if feedback torque
and gate signals are required in the recording.

## 3. NEXT adapter for the evaluated arm

From the `glorbot2_laptop_deploy` workspace:

```bash
ros2 run next_adapter glorbot2_next_adapter --ros-args \
  -p robot_name:=glorbot2_1 \
  -p arm:=right \
  -p publish_hz:=60.0
```

## 4. NEXT inference

From `/home/laptop/Projects/glorbot2/factr2_next`:

```bash
source install/setup.bash
ros2 run factr2_next next_infer --ros-args \
  -p config_file:=/home/laptop/Projects/glorbot2/factr2_next/src/factr2_next/factr2_next/config/inference_glorbot2.yaml
```

## 5. Evaluation recorder

In another terminal from `/home/laptop/Projects/glorbot2/factr2_next`:

```bash
source install/setup.bash
ros2 run factr2_next next_eval_record --ros-args \
  -p config_file:=/home/laptop/Projects/glorbot2/factr2_next/src/factr2_next/factr2_next/config/eval_glorbot2.yaml
```

Press `r` to start an episode, push the arm, then press `r` again to stop and
flush it. More presses create additional episodes in the same HDF5 file.

## Check and plot

These commands do not require a running ROS graph. Replace `TIMESTAMP` with the
recorded filename:

```bash
ros2 run factr2_next next_check_h5 \
  data/eval/contact_pushes_TIMESTAMP.h5

ros2 run factr2_next next_eval_plot \
  data/eval/contact_pushes_TIMESTAMP.h5 \
  --episode ep_0000 \
  --output figures/contact_push.png \
  --ema-alpha 0.03 \
  --ylim -4 4
```

The figure is one line: the signed sum of all seven external joint torques.
Omit `--ema-alpha` to use the filtering recorded online. Smaller offline EMA
alpha values produce stronger smoothing.

## Left-arm evaluation

Change `arm: right` to `arm: left` in `eval_glorbot2.yaml`, launch the adapter
with `-p arm:=left`, and run inference with `inference_glorbot2_left.yaml`.
