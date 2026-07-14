# factr2_next

## Overview

`factr2_next` is a ROS2 implementation of Neural External Torque Estimation (NEXT), a data-driven method for estimating external joint torque without dedicated force/torque sensors. NEXT learns the torque required for free-space robot motion from contact-free data, then estimates external torque at runtime as:

```text
tau_ext_hat = tau_measured - tau_free_hat
```

This repository contains nodes for recording free-motion data, training NEXT models, running online inference, and visualizing the estimated torque signal. It also includes optional Piper/Gello teleoperation demo packages for testing force-feedback workflows on low-cost robot arms.

## Catalog

- [Repository Layout](#repository-layout)
- [Install](#install)
- [Sanity Check](#sanity-check)
- [Data Recording](#data-recording)
- [Training](#training)
- [Inference](#inference)
- [Optional Piper/Gello Teleop Demo](#optional-pipergello-teleop-demo)
- [Optional FACTR2 Feedback Demo](#optional-factr2-feedback-demo)
- [Config Reference](#config-reference)
- [Hardware Notes](#hardware-notes)
- [License / Citation](#license--citation)

## Repository Layout

- `src/factr2_next`: Core NEXT package for data recording, model training, online inference, and visualization.
- `src/piper_control`: Optional Piper hardware interface used by the demo launch files.
- `src/teacher_arm`: Optional Gello/Piper teacher-arm teleoperation package, including optional FACTR2 torque feedback.
- `src/rdm_bringup`: Optional launch files for Piper/Gello teleoperation demos.

## Install

These instructions assume Ubuntu with ROS2 Jazzy installed at `/opt/ros/jazzy`, Python 3.12, and `uv`. This repository is intended to be used as a ROS2 workspace root.

Create and activate the Python environment:

```bash
uv venv --python 3.12 --prompt ros .venv
source .venv/bin/activate
```

Install the core Python dependencies:

```bash
uv pip install pip wheel setuptools==79.0.1
uv pip install colcon-core colcon-common-extensions
uv pip install "numpy<2.0" pyyaml termcolor h5py torch cffi scipy
```

For the optional Piper/Gello hardware demos, also install:

```bash
uv pip install python-can piper-sdk dynamixel-sdk pyserial pynput
```

Build the workspace:

```bash
source /opt/ros/jazzy/setup.bash
python -m colcon build --symlink-install
source install/setup.bash
```

In each new shell, source the environment before running ROS commands:

```bash
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash
```

## Sanity Check

After building and sourcing the workspace, check that the package and console scripts are visible:

```bash
ros2 pkg list | grep factr2_next
ros2 run factr2_next next_record --help
ros2 run factr2_next next_train --help
ros2 run factr2_next next_infer --help
```

## Data Recording

Record contact-free robot motion with `next_record`. The default `record.yaml` records one robot or arm namespace, such as `/piper/right`, into an H5 file under `data/`.

```bash
ros2 run factr2_next next_record
```

Press `r` to start an episode and `r` again to stop it. Each episode is saved as `ep_0000`, `ep_0001`, etc. Each configured topic is written as:

```text
ep_0000/<key>/data
ep_0000/<key>/timestamps
```

For a single-arm setup, leave `record.yaml` in the default shape and change only the namespace:

```yaml
robot_topic_root: /piper/right
topics:
  joint_pos:
    topic: "{robot_topic_root}/joint_pos_obs"
    field: position
```

For a bimanual setup, either record each arm into a separate file using different `robot_topic_root` values, or define explicit left/right keys in one recording config:

```yaml
topics:
  left_joint_pos:
    topic: /piper/left/joint_pos_obs
    field: position
  right_joint_pos:
    topic: /piper/right/joint_pos_obs
    field: position
```

The important rule is that the H5 keys you record must match the `keys` section in `train.yaml`.

## Training

NEXT is trained on free-motion data. In the paper notation, the model learns the free-space torque term `tau_free_hat`; at runtime, external torque is the residual:

```text
tau_ext_hat = tau_measured - tau_free_hat
```

In code, `NextTorqueDataset._episode_arrays` builds each training input as:

```text
x_t = [joint_pos_t, joint_vel_t, joint_cmd_t - joint_pos_t]
y_t = measured_joint_torque_t
```

The dataset then creates history windows of length `history`, and the model predicts the measured torque at the final timestep of each window. It is not predicting future torque.

For the default single-arm recorder, use:

```yaml
arm_mode: single
keys:
  joint_pos: joint_pos
  joint_vel: joint_vel
  joint_cmd: joint_cmd
  measured_joint_torque: measured_joint_torque
```

For bimanual H5 files with left/right key prefixes, use:

```yaml
arm_mode: both
arms: [left, right]
keys:
  joint_pos: "{arm}_joint_pos"
  joint_vel: "{arm}_joint_vel"
  joint_cmd: "{arm}_joint_cmd"
  measured_joint_torque: "{arm}_measured_joint_torque"
```

Train with:

```bash
ros2 run factr2_next next_train
```

Each run saves `model.pt`, `config.yaml`, `normalization.npz`, and `metrics.json` under `runs/`. When `arm_mode: both`, training writes one run per arm.

## Inference

Inference loads a training run from `checkpoint_dir`, subscribes to the same input signals used during training, and keeps a rolling `HistoryBuffer`. Once the buffer is full, `InferenceNode._callback` predicts free-space torque and publishes:

```text
external_joint_torque_raw = measured_joint_torque - free_joint_torque_pred
external_joint_torque     = smoothed external_joint_torque_raw
```

This is the runtime implementation of the NEXT residual equation above.

Configure one inference node with namespace roots:

```yaml
checkpoint_dir: /path/to/next_run_dir
robot_topic_root: /piper/right
next_topic_root: /next/right
device: cpu
```

Run inference with:

```bash
ros2 run factr2_next next_infer
```

For bimanual inference, run one inference node per arm with a different `checkpoint_dir`, `robot_topic_root`, and `next_topic_root`. The published raw residual is useful for debugging; the smoothed residual is what the score, contact hysteresis, and optional feedback demos consume.

## Optional Piper/Gello Teleop Demo

The Piper/Gello demo packages are provided to reproduce the hardware teleop setup used during development. They are not required for recording, training, or running NEXT on another robot.

- `piper_control` runs the Piper hardware interface.
- `teacher_arm` runs the Gello/Piper leader-follower teleop controller.
- `rdm_bringup` provides launch files that start both pieces together.

Before running the demo, edit the hardware-specific config files:

- `src/piper_control/piper_control/configs/piper_arm.yaml`: set each Piper `can_port`, such as `can0` or `can1`.
- `src/teacher_arm/teacher_arm/configs/piper_gellos.yaml`: set each Dynamixel serial `port` from `/dev/serial/by-id/`.

For a single right-arm demo:

```bash
ros2 launch rdm_bringup right_arm_teleop.launch.py
```

For a bimanual demo:

```bash
ros2 launch rdm_bringup bimanual_teleop.launch.py
```

The launch files pass `name: right` or `name: left` into both nodes. Those names select the matching config blocks and produce topics such as `/piper/right/joint_pos_obs`, `/piper/right/joint_pos_cmd`, `/piper/left/joint_pos_obs`, and `/piper/left/joint_pos_cmd`.

## Optional FACTR2 Feedback Demo

The FACTR2 feedback demo is our Piper/Gello implementation of force feedback from NEXT. It is intentionally optional: the general idea can be adapted to any robot system that can read NEXT torque estimates and command some form of haptic or joint-level feedback.

In this demo, `teacher_arm.factr2_feedback.Factr2TorqueFeedback` subscribes to:

- `/next/{arm}/external_joint_torque`
- `/next/{arm}/contact_state`

It then computes a gated feedback torque for the teacher arm:

```text
gate_target = contact_state and inputs_are_fresh
gate        = ramp_filter(gate, gate_target)
tau_fb      = gate * gains * tau_ext - damping_gains * qdot_teacher
tau_fb      = clip(tau_fb, -max_torque, max_torque)
```

The gates matter. Feedback only turns on when NEXT reports contact, recent torque/contact messages are still fresh, and the ramp filter has moved the gate toward one. This avoids instantly applying stale or discontinuous feedback.

Enable the demo in `src/teacher_arm/teacher_arm/configs/piper_gellos.yaml`:

```yaml
factr2_torque_feedback:
  enable: true
  external_joint_torque_topic: "/next/{arm}/external_joint_torque"
  contact_state_topic: "/next/{arm}/contact_state"
  gains: [-0.06, -0.08, -0.08, -0.03, -0.03, -0.02]
  damping_gains: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  max_torque: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
```

The gains are hardware- and sign-convention-dependent. Start small, verify the sign of each joint one at a time, and keep conservative `max_torque` limits until the feedback direction and magnitude are stable.

`damping_gains` is included as a tuning hook, but defaults to zero because this demo primarily used gated proportional feedback from the estimated external torque.

Typical workflow:

```text
1. Start Piper/Gello teleop.
2. Start NEXT inference for the same arm.
3. Confirm /next/{arm}/external_joint_torque and /next/{arm}/contact_state look sane.
4. Enable FACTR2 feedback with small gains.
5. Increase gains only after checking stability joint by joint.
```

For another robot, keep the same structure: estimate `tau_ext` with NEXT, convert it through a robot-specific feedback map, gate it on contact/freshness, clip it to safe limits, and send it to the hardware interface.

## Config Reference

Most workflows only require editing one or two YAML files:

| File | Purpose | Common fields |
| --- | --- | --- |
| `src/factr2_next/factr2_next/config/record.yaml` | Record synchronized free-motion H5 data. | `output_dir`, `session_name`, `robot_topic_root`, `recording.target_hz`, `topics` |
| `src/factr2_next/factr2_next/config/train.yaml` | Train a NEXT model from H5 data. | `train_h5_paths`, `val_h5_paths`, `arm_mode`, `keys`, `history`, `model`, `train.device` |
| `src/factr2_next/factr2_next/config/inference.yaml` | Run online NEXT inference. | `checkpoint_dir`, `robot_topic_root`, `next_topic_root`, `device`, `smoothing`, `score`, `contact` |
| `src/factr2_next/factr2_next/config/visualize.yaml` | Start the lightweight web visualizer. | `next_topic_root`, `web.port`, `plot.max_points`, `outputs` |
| `src/piper_control/piper_control/configs/piper_arm.yaml` | Configure optional Piper hardware nodes. | `can_port`, `gripper_exist`, `home_pos`, `rest_pos`, loop rates |
| `src/teacher_arm/teacher_arm/configs/piper_gellos.yaml` | Configure optional Gello/Piper teleop and feedback. | Dynamixel `port`, `joint_signs`, teleop limits, `factr2_torque_feedback` |

For single-arm use, prefer namespace roots such as `/piper/right` and `/next/right`. For bimanual use, either run one config per arm or use arm-formatted keys such as `{arm}_joint_pos` when the data is stored in one H5 file.

## Hardware Notes

This code can command real robot hardware. Before running hardware demos:

- Keep `factr2_torque_feedback.enable: false` until teleop and NEXT inference are both working without feedback.
- Start with small feedback `gains` and conservative `max_torque` limits.
- Verify feedback signs one joint at a time before increasing gains.
- Confirm `/next/{arm}/external_joint_torque` and `/next/{arm}/contact_state` are stable before enabling feedback.
- Treat configs in this repository as demo defaults, not universally safe settings for every robot.

## License / Citation

tbd
