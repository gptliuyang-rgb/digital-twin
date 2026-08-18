# This increment — SONIC 50 Hz planner last-frame playback cursor

Continues ADR-048 (idle ADAPTING/RECOVERING). Still no grasp-success
number and no combined T800+Hand weld.

## Done

- **ADR-049.** Official planner-active `CurrentFrameAdvancement` from
  `g1_deploy_onnx_ref.cpp`, at T800 32-D qpos / 25-D hinges:
  - Owns `current_frame`. Successful ADR-047 assign resets it to 0 and
    clears idle storage, then the **same** 50 Hz tick advances
    `current_frame += 1` (C++ order).
  - `new_frame >= timesteps` clamps to the last frame. Planner path
    never loops to 0 (that is the named-clip reference-motion branch).
  - Idle ADAPTING/RECOVERING runs only after that clamp, only when
    `play` and `LocomotionMode::IDLE` (0). Walk / squat do not enter.
  - `play == false` holds the cursor and does not readapt.
  - One-frame clips clamp on the first tick after assign.
- `planner_sonic.onnx` / 36-D qpos / 29-D hinges still raise
  `G1CheckpointIncompatible`. `refuse_run_playback_onnx()` never
  executes weights. ADR-018 interpolators stay the runtime L1a.
- `make eval-l1a-playback` dumps the clamp table and skip reasons.
  Decoder 874-D and encoder layouts are unchanged. Hands bypass WBC.
  `grasp_success_rate` JSON `null`.

## Why this is not a G1 planner checkpoint

The C++ stack advances a cursor over **already generated** 50 Hz qpos
after TensorRT. This repo only reproduces that cursor + last-frame
hold on caller-supplied T800 clips. It does not load
`planner/target_vel/V2/planner_sonic.onnx` and it is not a clip library.

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. IMU noise, camera delay, and
command latency stay `REQUIRED_INPUT`. Flange SE(3) and wrist CoM still
block GMR-on-BONES-SEED and the combined weld.

## Not done

- Running a T800 planner ONNX (G1 `planner_sonic.onnx` is refused)
- Replacing ADR-018 interpolators with a trained generative planner
- Official clip-library modes beyond command_schema `{0,1,2}`
- Reference-motion named-clip loop-to-zero (refused here on purpose)
- PICO / CloudXR / Isaac Teleop SDK
- Isaac Lab PPO launch
- Combined T800+Hand MJCF
- Grasp-success numbers
- Live actor T-pose (q=0 overlay remains)
- GMR on a tiny BONES-SEED clip (blocked on flange/CoM)
- Diagnose a real GR00T / π0.5 dump once provided
