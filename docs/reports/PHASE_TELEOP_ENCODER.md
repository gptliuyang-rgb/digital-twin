# This increment — SONIC teleop encoder mode (encoder_mode_4 + lower body)

Continues ADR-042 (570-D motion window). Still no grasp-success number
and no combined T800+Hand weld.

## Done

- **ADR-043.** Official HuggingFace `observation_config.yaml` encoder
  **superset** with T800 dims, SMPL and wrist channels omitted:
  `encoder_mode_4` 4 + full q/dq 250+250 + root z 10+1 + ori 6+60 +
  lower-body q/dq 120+120 + vr_3point 9+12 = **842**. G1 1751 / 650
  are refused. Unused mode slots are **zero-filled**.
- `t800` mode_id 0 (analogue of official `g1`, that name stays refused):
  `encoder_mode_4` + full-body `motion_*_10frame_step5` q/dq +
  `motion_anchor_orientation_10frame_step5`.
- `teleop` mode_id 1 (official list): `encoder_mode_4` +
  `motion_joint_positions_lowerbody_10frame_step5` +
  `motion_joint_velocities_lowerbody_10frame_step5` +
  `vr_3point_local_target` + `vr_3point_local_orn_target` +
  `motion_anchor_orientation`. VR is packed from `command_schema_v1`
  (`push_vr_3point`). Empty VR hold raises — no invented PICO pose.
- `smpl` mode, `motion_*_wrists_*`, and `vr_5point_*` in the encoder
  list are refused. Decoder 874-D is unchanged. Hands bypass WBC.
- `make eval-l1a-encoder` dumps both modes. `grasp_success_rate` JSON `null`.

## Why this is not a PICO stack

The 100 Hz operator loop (ADR-040) already ingest `vr_3point` rows.
This increment only **gathers** those poses into the encoder vector the
official teleop encoder expects. No PICO / CloudXR / Isaac Teleop import.

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. IMU noise, camera delay, and
command latency stay `REQUIRED_INPUT`. Flange SE(3) and wrist CoM still
block GMR-on-BONES-SEED and the combined weld.

## Not done

- Running a T800 encoder ONNX (G1 ONNX is refused)
- Official SMPL teleop encoder (needs smpl_joint.csv + wrists)
- PICO / CloudXR / Isaac Teleop SDK
- Isaac Lab PPO launch
- Combined T800+Hand MJCF
- Grasp-success numbers
- Live actor T-pose (q=0 overlay remains)
- GMR on a tiny BONES-SEED clip (blocked on flange/CoM)
- Diagnose a real GR00T / π0.5 dump once provided
- SONIC's trained generative kinematic planner
