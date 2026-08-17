# This increment — SONIC low-latency encoder (10frame_step1)

Continues ADR-043 (842-D default teleop encoder). Still no grasp-success
number and no combined T800+Hand weld.

## Done

- **ADR-044.** Official `low_latency/observation_config.yaml` with T800
  dims, SMPL and wrist channels omitted, **no** `motion_root_z_*`:
  `encoder_mode_4` 4 + full q/dq 250+250 + ori 60+6 + lower-body
  q/dq 120+120 + vr_3point 9+12 = **831**. G1 **1247** / 640 / 1751
  are refused. Unused mode slots are **zero-filled**.
- Official g1/teleop names stay **10frame_step1** (look-ahead indices
  `[0,1,…,9]`, 0.18 s). The 4-frame / ~80 ms horizon is SMPL-only and
  is refused (`smpl_*_4frame_step1`, `motion_*_wrists_4frame_step1`,
  and a 4-frame body window).
- Default `wbc/obs_gather.yaml` (842-D, `10frame_step5`) is unchanged.
  Mixing step1 and step5 in one YAML raises. Carrying `root_z` into
  the low-latency YAML raises.
- `t800` / `teleop` mode_ids stay 0 / 1. VR packing is still
  `push_vr_3point` from `command_schema_v1`. Empty VR hold raises.
- `make eval-l1a-encoder` dumps both layouts. Decoder 874-D unchanged.
  Hands bypass WBC. `grasp_success_rate` JSON `null`.

## Why this is not a G1 low-latency checkpoint

Official docs require matching encoder ONNX + obs config. G1
`low_latency/model_encoder.onnx` is 1247-D. This repo only **gathers**
the T800 analogue vector. It does not load that ONNX.

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. IMU noise, camera delay, and
command latency stay `REQUIRED_INPUT`. Flange SE(3) and wrist CoM still
block GMR-on-BONES-SEED and the combined weld.

## Not done

- Running a T800 encoder ONNX (G1 default 1751-D and low-latency 1247-D
  ONNX are both refused)
- Official SMPL teleop encoder (needs smpl_joint.csv + wrists)
- PICO / CloudXR / Isaac Teleop SDK
- Isaac Lab PPO launch
- Combined T800+Hand MJCF
- Grasp-success numbers
- Live actor T-pose (q=0 overlay remains)
- GMR on a tiny BONES-SEED clip (blocked on flange/CoM)
- Diagnose a real GR00T / π0.5 dump once provided
- SONIC's trained generative kinematic planner
