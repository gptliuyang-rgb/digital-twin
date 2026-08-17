# This increment — SONIC v1.1 heading-normalized encoder (10frame_step5)

Continues ADR-044 (831-D low-latency step1). Still no grasp-success
number and no combined T800+Hand weld.

## Done

- **ADR-045.** Official `sonic_v1_1/observation_config.yaml` with T800
  dims, SMPL and wrist channels omitted, **no** `motion_root_z_*`,
  heading-normalized anchor ori:
    `encoder_mode_4` 4 + full q/dq 250+250 + heading ori 60+6 +
    lower-body q/dq 120+120 + vr_3point 9+12 = **831**. G1 **1751**
    is refused. Unused mode slots are **zero-filled**.
- Official g1/teleop names stay **10frame_step5** (look-ahead indices
  `[0,5,…,45]`, 0.9 s). Official SMPL/wrist `*_10frame_step1` is
  refused (`smpl_joints_10frame_step1`,
  `smpl_anchor_orientation_heading_10frame_step1`,
  `motion_joint_positions_wrists_10frame_step1`).
- Default `wbc/obs_gather.yaml` (842-D, full ori, with root_z) and
  `wbc/obs_gather_low_latency.yaml` (831-D, step1, full ori) are
  unchanged. Mixing heading names into default, full-ori names into
  v1.1, step1 into v1.1, or root_z into v1.1 raises.
- The T800 v1.1 vector is the **same integer** as low-latency 831 and
  a **different** name list. Tests lock the slot names apart.
  `sonic_v1_1/model_encoder.onnx` is refused (1751 vs 831).
- Heading ori is `R_z(yaw_robot).T @ R_ref`. A 30° robot pitch with
  an identity reference packs identity 6D on v1.1 and a non-identity
  6D on the default YAML.
- Wrist-pose augmentation is **not** implemented. Official docs treat
  it as a training-time transform. `not_wrist_pose_augmentation_sampler`
  stays true.
- `t800` / `teleop` mode_ids stay 0 / 1. VR packing is still
  `push_vr_3point` from `command_schema_v1`. Empty VR hold raises.
- `make eval-l1a-encoder` dumps all three layouts. Decoder 874-D
  unchanged. Hands bypass WBC. `grasp_success_rate` JSON `null`.

## Why this is not a G1 v1.1 checkpoint

Official docs require matching encoder ONNX + obs config. G1
`sonic_v1_1/model_encoder.onnx` is 1751-D. This repo only **gathers**
the T800 analogue vector. It does not load that ONNX. It is also not
the low-latency checkpoint (step1, ~80 ms SMPL lookahead).

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. IMU noise, camera delay, and
command latency stay `REQUIRED_INPUT`. Flange SE(3) and wrist CoM still
block GMR-on-BONES-SEED and the combined weld.

## Not done

- Running a T800 encoder ONNX (G1 default/v1.1 1751-D and low-latency
  1247-D ONNX are all refused)
- Official SMPL teleop encoder (needs smpl_joint.csv + wrists)
- Wrist-pose augmentation sampler (training-time; not invented here)
- PICO / CloudXR / Isaac Teleop SDK
- Isaac Lab PPO launch
- Combined T800+Hand MJCF
- Grasp-success numbers
- Live actor T-pose (q=0 overlay remains)
- GMR on a tiny BONES-SEED clip (blocked on flange/CoM)
- Diagnose a real GR00T / π0.5 dump once provided
- SONIC's trained generative kinematic planner
