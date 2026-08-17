# This increment — SONIC kinematic planner ONNX contract (T800 32-D qpos)

Continues ADR-045 (v1.1 heading encoder). Still no grasp-success
number and no combined T800+Hand weld.

## Done

- **ADR-046.** Official `planner_sonic.onnx` I/O with T800 dims:
  `context_mujoco_qpos` `[1, 4, 32]` = 3 pos + 4 quat wxyz + 25 hinges.
  G1 **36** is refused. V2 keeps 11 inputs / 2 outputs / 27 modes.
  Tokens 6–16 (K=11), 4 frames/token, max 64 padded 30 Hz frames.
- `command_schema` loco_mode `{0,1,2}` maps to official
  `{idle, slowWalk, walk}`. **fast_walk is not run (3).**
  `nav_cmd[2]` (wz) is not `facing_direction` (Eq. 8 already owns heading
  rate). `pelvis_height` does not fill the ONNX height channel; walk/idle
  keep `height = -1`.
- 30 Hz → 50 Hz resample matches the official deployment rule
  (`floor(N*50/30)`, linear pos/q, SLERP quat). Context sampler takes
  4 frames at 30 Hz from a caller-supplied 50 Hz window starting at
  `current_frame + 2`. Empty windows raise; no invented stand clip.
- `planner_sonic.onnx` / last-dim 36 / 29-DoF hinges raise
  `G1CheckpointIncompatible`. `refuse_run_planner_onnx()` never executes
  weights. ADR-018 interpolators stay the runtime L1a.
- `make eval-l1a-planner-onnx` dumps the 11-tensor pack.
  Decoder 874-D and encoder 842/831/831 layouts are unchanged.
  Hands bypass WBC. `grasp_success_rate` JSON `null`.

## Why this is not a G1 planner checkpoint

Official docs require matching encoder + decoder + obs config. The
planner is a fourth ONNX (`planner/target_vel/V2/planner_sonic.onnx`)
trained on G1 36-D MuJoCo qpos. This repo only **packs** the T800
analogue vector. It does not load that ONNX and it is not a clip library.

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. IMU noise, camera delay, and
command latency stay `REQUIRED_INPUT`. Flange SE(3) and wrist CoM still
block GMR-on-BONES-SEED and the combined weld.

## Not done

- Running a T800 planner ONNX (G1 `planner_sonic.onnx` is refused)
- Replacing ADR-018 interpolators with a trained generative planner
- Official clip-library modes beyond command_schema `{0,1,2}`
- PICO / CloudXR / Isaac Teleop SDK
- Isaac Lab PPO launch
- Combined T800+Hand MJCF
- Grasp-success numbers
- Live actor T-pose (q=0 overlay remains)
- GMR on a tiny BONES-SEED clip (blocked on flange/CoM)
- Diagnose a real GR00T / π0.5 dump once provided
