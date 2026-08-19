# This increment — shared PlannerPlayback.current_frame for encoder + planner

Continues ADR-049 (50 Hz last-frame playback cursor). Still no
grasp-success number and no combined T800+Hand weld.

## Done

- **ADR-050.** Encoder look-ahead (`MotionHold` / `assemble_encoder`)
  and planner context (`sample_context_from_50hz`) read the same
  `PlannerPlayback.current_frame` after each 50 Hz tick:
  - Encoder `MotionHold.cursor` is rebound to that integer.
    10frame_step5 last-frame-repeats (ADR-042).
  - Planner context starts at `current_frame + 2` (4 frames at 30 Hz
    from 50 Hz). Short windows raise; no last-frame-repeat (ADR-046).
  - `play == false` holds both readers.
  - A 16-frame first tick leaves `current_frame == 1`; encoder indices
    `[1, 6, 11, 15, …]`; planner first hinge is clip index 3.
  - At the last frame, encoder repeats the last row; planner context
    raises.
- `MotionCursor` stays a sibling over `motion_lib`. Binding it into
  `PlannerPlayback` still raises. Missing `dq` is zeros, not finite-diff.
- `planner_sonic.onnx` / encoder ONNX / 36-D qpos / 29-D hinges /
  45-D hands still raise. `refuse_run_shared_cursor_onnx()` never
  executes weights. ADR-018 interpolators stay the runtime L1a.
- `make eval-l1a-shared-cursor` dumps both readers. Decoder 874-D on
  this same tick is ADR-051. Hands bypass WBC.
  `grasp_success_rate` JSON `null`.

## Why this is not a G1 checkpoint

Official C++ advances one integer on the control thread. Encoder
look-ahead and planner TensorRT context are sampled from that integer.
This repo only **binds** those two readers onto caller-supplied T800
qpos. It does not load G1 ONNX.

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. IMU noise, camera delay, and
command latency stay `REQUIRED_INPUT`. Flange SE(3) and wrist CoM still
block GMR-on-BONES-SEED and the combined weld.

## Not done

- Running a T800 encoder / planner ONNX (G1 weights are refused)
- Replacing ADR-018 interpolators with a trained generative planner
- Official clip-library modes beyond command_schema `{0,1,2}`
- Reference-motion named-clip loop-to-zero (refused on purpose)
- Mixing `MotionCursor` into `PlannerPlayback` (refused on purpose)
- PICO / CloudXR / Isaac Teleop SDK
- Isaac Lab PPO launch
- Combined T800+Hand MJCF
- Grasp-success numbers
- Live actor T-pose (q=0 overlay remains)
- GMR on a tiny BONES-SEED clip (blocked on flange/CoM)
- Diagnose a real GR00T / π0.5 dump once provided
