# This increment — SONIC encoder motion_* 10frame_step5 look-ahead

Continues ADR-041 (decoder YAML gather). Still no grasp-success number
and no combined T800+Hand weld.

## Done

- **ADR-042.** Official encoder observation names with T800 dims:
  `motion_joint_positions_10frame_step5` 250 + velocities 250 +
  `motion_anchor_orientation_10frame_step5` 60 +
  `motion_root_z_position_10frame_step5` 10 = **570**. G1 650 is refused.
  Look-ahead indices at cursor 0 / step 5 are `[0, 5, …, 45]` (0.9 s).
  Last frame repeats past the clip end. Empty `MotionHold` raises —
  no invented stand clip, no BONES-SEED. `fps` must already be 50;
  30 fps libraries are not resampled. Anchor 6D is Zhou (first two
  columns of `R_robot.T @ R_ref`). Wrist and SMPL names raise.
  `model_encoder.onnx` is refused (`refuse_g1_encoder_onnx`). Hands
  never enter the vector. Decoder 874-D is unchanged.
- `make eval-l1a-encoder` dumps slots. `grasp_success_rate` JSON `null`.

## Why this is not a motion clip

The window is whatever the caller pushed (planner / ZMQ / a validated
50 Hz `motion_lib`). This repo still does not run GMR on BONES-SEED
(blocked on flange SE(3) and wrist CoM).

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. IMU noise, camera delay, and
command latency stay `REQUIRED_INPUT`.

## Not done

- Measured IMU/joint latency (must not be invented)
- Running a T800 encoder ONNX (G1 ONNX is refused)
- PICO / CloudXR / Isaac Teleop SDK
- Isaac Lab PPO launch
- Combined T800+Hand MJCF
- Grasp-success numbers
- Live actor T-pose (q=0 overlay remains)
- GMR on a tiny BONES-SEED clip (blocked on flange/CoM)
- Diagnose a real GR00T / π0.5 dump once provided
- SONIC's trained generative kinematic planner
