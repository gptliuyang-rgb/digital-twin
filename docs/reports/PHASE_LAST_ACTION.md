# This increment — caller-supplied 25-D last_action on the 50 Hz tick

Continues ADR-051 (decoder 874-D from HardwareHold on
`SharedPlaybackCursor.control_tick`). Still no grasp-success number and no
combined T800+Hand weld.

## Done

- **ADR-052.** `control_tick(..., last_action=)` pushes a caller-supplied
  T800 **25-D** previous policy action into `HardwareHold` on the same
  50 Hz tick, *before* decoder assemble:
  - After 10 ticks, grouped `his_last_actions` `a[0]` is `[10..19]` while
    `q[0]` hardware history is `[0..9]` and encoder 10frame_step5 stays
    on clip indices.
  - Omit `last_action` to keep HardwareHold (startup zeros are padding,
    not an invented decoder ONNX vector). ADR-051 callers stay valid.
  - `play == false` holds `current_frame` but still logs the caller
    action into the decoder ring.
  - `tick()` without `token` still applies `last_action` to the hold and
    does **not** invent a 64-D token or a decoder vector.
- G1 29-D action / 36-D planner qpos / 32-D T800 qpos / 45-D hands /
  NaN / fake decoder ONNX fill / clip→last_action copy still raise.
  `refuse_run_shared_cursor_onnx()` never executes weights.
- `make eval-l1a-last-action` dumps a-history vs q-history vs encoder
  look-ahead. Hands bypass WBC. `grasp_success_rate` JSON `null`.

## Why this is not a G1 checkpoint

Official C++ feeds the previous decoder output (29-D on G1) back as
`last_action` on the next 50 Hz gather. This repo only **accepts** a
caller-supplied T800 25-D vector. It does not load G1
`model_decoder.onnx` and does not synthesize an action from clip qpos.

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. IMU noise, camera delay, and
command latency stay `REQUIRED_INPUT`. Flange SE(3) and wrist CoM still
block GMR-on-BONES-SEED and the combined weld. A real T800 decoder
output is still missing — this increment only wires the slot.

## Not done

- Running a T800 encoder / planner / decoder ONNX (G1 weights are refused)
- Replacing ADR-018 interpolators with a trained generative planner
- Official clip-library modes beyond command_schema `{0,1,2}`
- Filling last_action from a live policy output (caller must supply it)
- Reference-motion named-clip loop-to-zero (refused on purpose)
- Mixing `MotionCursor` into `PlannerPlayback` (refused on purpose)
- PICO / CloudXR / Isaac Teleop SDK
- Isaac Lab PPO launch
- Combined T800+Hand MJCF
- Grasp-success numbers
- Live actor T-pose (q=0 overlay remains)
- GMR on a tiny BONES-SEED clip (blocked on flange/CoM)
- Diagnose a real GR00T / π0.5 dump once provided
