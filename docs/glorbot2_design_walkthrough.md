# Glorbot2 FACTR2/NEXT Walkthrough

## Main Idea

I kept the integration split across the repos by ownership:

- `factr2_next` stays robot-agnostic.
- `glorbot2_laptop_deploy` exposes Glorbot2 state/command data through a read-only adapter.
- `glorbot2_gello_teleop` owns the leader torque loop, so feedback is added there.

## Files To Walk Through

### `glorbot2_laptop_deploy`

```text
src/next_adapter/next_adapter/adapter_node.py
```

This is the robot-side bridge. It subscribes to native Nero `joint_state` and `joint_state_cmd` topics and republishes the four `JointState` topics expected by NEXT. It does not command hardware.

### `factr2_next`

```text
src/factr2_next/factr2_next/config/record_glorbot2.yaml
src/factr2_next/factr2_next/config/train_glorbot2.yaml
src/factr2_next/factr2_next/config/inference_glorbot2.yaml
src/factr2_next/factr2_next/config/inference_glorbot2_left.yaml
```

These make the existing NEXT recorder/trainer/inference nodes work with Glorbot2. Recording is bimanual, training produces one model per arm, and inference runs one node per arm.

```text
src/factr2_next/factr2_next/visualization/web_node.py
```

Only small code change here: the visualizer now has configurable joint count, since Nero is 7-DoF and Piper was 6-DoF.

### `glorbot2_gello_teleop`

```text
src/teacher_arm/teacher_arm/factr2_feedback.py
src/teacher_arm/teacher_arm/base_teleop.py
src/teacher_arm/config/gello_1.yaml
```

This mirrors the Piper feedback setup. It subscribes to NEXT external torque and contact state, gates/ramp/clips the signal, then adds feedback at the final Gello leader torque sum. Feedback is disabled by default.

## Design Choices

- Keep `factr2_next` generic instead of adding Glorbot-specific message handling.
- Use a read-only adapter on the robot side, so robot control is not touched.
- Add haptic feedback on the Gello leader side, because that is where torque is commanded.
- Record both arms together, but train and run separate NEXT models per arm.
- Leave feedback off by default and tune gains/signs on hardware.
