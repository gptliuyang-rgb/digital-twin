# This increment — SONIC planner 8-frame cross-fade + replan timer

Continues ADR-046 (planner ONNX I/O). Still no grasp-success number
and no combined T800+Hand weld.

## Done

- **ADR-047.** Official control-thread blend from
  `g1_deploy_onnx_ref.cpp` `CurrentFrameAdvancement` and the 10 Hz
  `Planner()` replan gate, at T800 32-D qpos:
  - 8-frame cross-fade: rebase old so `current_frame` is 0; align new at
    `gen_frame - current_frame`; `w_new = clamp((f - blend_start) / 8, 0, 1)`.
    Linear mix on hinges / root xyz; SLERP on root quat. `current_frame`
    resets to 0. Empty old copies new. `new_anim_length <= 0` skips
    (C++ "sit this out").
  - Replan: always on mode / facing / height. Non-static modes also on
    speed / direction or timer with speed ≠ 0. Static set is Idle,
    Squat, Kneel, Lying, Idle Boxing (`{0,4,5,6,7,9}`).
  - Intervals: run 0.1 s, hand-crawl **mode 8 only** 0.2 s, punches/hooks
    1.0 s, else 1.0 s (including elbow crawl 14 and walkBoxing 10).
- `command_schema` loco_mode `{0,1,2}` still maps to idle / slowWalk /
  **walk**. fast_walk is **not** run, so its timer is 1.0 s not 0.1 s.
- `planner_sonic.onnx` / last-dim 36 still raise
  `G1CheckpointIncompatible`. `refuse_run_blend_onnx()` never executes
  weights. ADR-018 interpolators stay the runtime L1a. Idle
  ADAPTING/RECOVERING is refused here.
- `make eval-l1a-planner-blend` dumps the weight ramp `[0, 1/8, …, 1]`
  and the interval table. Decoder 874-D and encoder layouts are
  unchanged. Hands bypass WBC. `grasp_success_rate` JSON `null`.

## Why this is not a G1 planner checkpoint

The C++ stack blends **after** TensorRT. This repo only reproduces that
blend/replan arithmetic on caller-supplied 50 Hz T800 qpos. It does not
load `planner/target_vel/V2/planner_sonic.onnx` and it is not a clip
library.

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. IMU noise, camera delay, and
command latency stay `REQUIRED_INPUT`. Flange SE(3) and wrist CoM still
block GMR-on-BONES-SEED and the combined weld.

## Not done

- Running a T800 planner ONNX (G1 `planner_sonic.onnx` is refused)
- Replacing ADR-018 interpolators with a trained generative planner
- Idle-mode ADAPTING/RECOVERING readapt (`kAdaptTrigger`)
- Official clip-library modes beyond command_schema `{0,1,2}`
- PICO / CloudXR / Isaac Teleop SDK
- Isaac Lab PPO launch
- Combined T800+Hand MJCF
- Grasp-success numbers
- Live actor T-pose (q=0 overlay remains)
- GMR on a tiny BONES-SEED clip (blocked on flange/CoM)
- Diagnose a real GR00T / π0.5 dump once provided
