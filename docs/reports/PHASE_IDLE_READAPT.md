# This increment — SONIC idle-mode ADAPTING / RECOVERING readapt

Continues ADR-047 (8-frame blend + replan timer). Still no grasp-success
number and no combined T800+Hand weld.

## Done

- **ADR-048.** Official last-frame idle readapt from
  `g1_deploy_onnx_ref.cpp` `IdleReadaptState`, at T800 25-D hinges:
  - Enters only when `play`, last planner frame, and
    `LocomotionMode::IDLE` (0). Squat / kneel / lying / idleBoxing do
    **not** enter (C++ checks IDLE only).
  - First qualifying tick stores lower-body planner targets
    (J00–J11) and sets IDLE, then applies the same-tick transition.
  - `avg_error` = mean |q_planner − q_motor| over 12 lower-body hinges.
    Upper body is never written. DexHand2 stays off this vector.
  - Thresholds (strict C++ `>` / `<`): `kAdaptTrigger` 0.10 rad,
    `kAdaptStop` 0.05 rad, `kRecoverTrigger` 0.045 rad.
  - ADAPTING: `q ← 0.98 q + 0.02 q_motor`. RECOVERING:
    `q ← 0.98 q + 0.02 q_original`. IDLE: no write.
  - **No `kRecoverStop`.** RECOVERING at 0 rad stays RECOVERING until
    error exceeds 0.10 rad (then ADAPTING).
  - New planner clip clears the stored flag (C++ after blend assign).
- `planner_sonic.onnx` / 29-D hinges still raise
  `G1CheckpointIncompatible`. `refuse_run_idle_readapt_onnx()` never
  executes weights. ADR-047 blend stays a sibling. ADR-018 interpolators
  stay the runtime L1a.
- `make eval-l1a-idle-readapt` dumps the threshold table and skip
  reasons. Decoder 874-D and encoder layouts are unchanged. Hands
  bypass WBC. `grasp_success_rate` JSON `null`.

## Why this is not a G1 planner checkpoint

The C++ stack mutates the **already generated** last-frame joint
targets after TensorRT. This repo only reproduces that arithmetic on
caller-supplied T800 hinges. It does not load
`planner/target_vel/V2/planner_sonic.onnx` and it is not a clip library.

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
