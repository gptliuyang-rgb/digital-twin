# This increment — decoder 874-D on the shared 50 Hz playback tick

Continues ADR-050 (encoder look-ahead + planner context share
`PlannerPlayback.current_frame`). Still no grasp-success number and no
combined T800+Hand weld.

## Done

- **ADR-051.** `SharedPlaybackCursor.control_tick` runs the official 50 Hz
  order on caller-supplied T800 state:
  - Advance / clamp `current_frame` (ADR-049).
  - Bind encoder `MotionHold.cursor` and planner context to that integer
    (ADR-050).
  - Assemble decoder **874-D** via `ObsGather.control_tick` /
    `HardwareHold` (ADR-041 / paper S7).
  - Decoder history is **hardware proprioception**, not planner clip
    hinges. After 10 ticks with `q_hw[0] = 0..9`, grouped q-history is
    `[0..9]` while encoder 10frame_step5 look-ahead stays on clip
    indices.
  - `play == false` holds `current_frame` (encoder / planner readers) but
    still logs hardware into the 50 Hz decoder ring.
  - `tick()` without `token` does **not** invent a 64-D token or a
    decoder vector (ADR-050 callers stay valid). `control_tick` requires
    the external token.
- `model_decoder.onnx` / G1 994-D / 45-D hands / clip→decoder copy still
  raise. `refuse_run_shared_cursor_onnx()` never executes weights.
  ADR-018 interpolators stay the runtime L1a.
- `make eval-l1a-decoder-tick` dumps the 874-D vector vs encoder
  look-ahead. Hands bypass WBC. `grasp_success_rate` JSON `null`.

## Why this is not a G1 checkpoint

Official C++ gathers IMU + joints on the same 50 Hz thread that advances
the planner cursor, then runs decoder TensorRT. This repo only **assembles**
the T800 874-D grouped history from `HardwareHold`. It does not load G1
`model_decoder.onnx` (994-D).

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. IMU noise, camera delay, and
command latency stay `REQUIRED_INPUT`. Flange SE(3) and wrist CoM still
block GMR-on-BONES-SEED and the combined weld.

## Not done

- Running a T800 encoder / planner / decoder ONNX (G1 weights are refused)
- Replacing ADR-018 interpolators with a trained generative planner
- Official clip-library modes beyond command_schema `{0,1,2}`
- Feeding decoder last-action from a policy output (still HardwareHold)
- Reference-motion named-clip loop-to-zero (refused on purpose)
- Mixing `MotionCursor` into `PlannerPlayback` (refused on purpose)
- PICO / CloudXR / Isaac Teleop SDK
- Isaac Lab PPO launch
- Combined T800+Hand MJCF
- Grasp-success numbers
- Live actor T-pose (q=0 overlay remains)
- GMR on a tiny BONES-SEED clip (blocked on flange/CoM)
- Diagnose a real GR00T / π0.5 dump once provided
