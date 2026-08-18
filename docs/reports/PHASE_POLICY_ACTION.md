# This increment — delayed 25-D policy_action as next-tick last_action

Continues ADR-052 (same-tick caller-supplied `last_action=`). Still no
grasp-success number and no combined T800+Hand weld.

## Done

- **ADR-053.** `control_tick(..., policy_action=)` stashes this tick's
  T800 **25-D** policy output and feeds it as `last_action` on the
  **next** 50 Hz tick (official `a_{t-1}`):
  - After 10 ticks with `policy_action a0 = 10..19`, grouped
    `his_last_actions` `a[0]` is `[0, 10..18]` while `q[0]` hardware
    history is `[0..9]` and encoder 10frame_step5 stays on clip indices.
  - Tick 0 decoder last_action stays startup zeros (padding, not an
    invented decoder ONNX vector). The stashed `a_t` is 19 after tick 9.
  - Omit `policy_action` to keep the stash; it still feeds `last_action`
    on later ticks. ADR-052 `last_action=` stays a same-tick override.
  - `play == false` holds `current_frame` but still delays the action
    into the decoder ring.
  - `tick()` without `token` still stashes `policy_action` and does
    **not** invent a 64-D token or a decoder vector.
- G1 29-D action / 32-D qpos / 45-D hands / fake decoder ONNX fill /
  same-tick policy_action→obs still raise.
  `refuse_run_shared_cursor_onnx()` never executes weights.
- `make eval-l1a-policy-action` dumps delayed a-history vs q-history vs
  encoder look-ahead. Hands bypass WBC. `grasp_success_rate` JSON `null`.

## Why this is not a G1 checkpoint

Official C++ runs the decoder, then writes that output into
`last_action` for the **next** gather. This repo still does **not** run
ONNX. The caller supplies `a_t`; the cursor only delays it by one 50 Hz
tick. Same-tick `last_action=` (ADR-052) remains for tests that already
pass `a_{t-1}` explicitly.

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. IMU noise, camera delay, and
command latency stay `REQUIRED_INPUT`. Flange SE(3) and wrist CoM still
block GMR-on-BONES-SEED and the combined weld. A real T800 decoder
output is still missing — this increment only delays the caller slot.

## Not done

- Running a T800 encoder / planner / decoder ONNX (G1 weights are refused)
- Replacing ADR-018 interpolators with a trained generative planner
- Filling policy_action from a live policy output (caller must supply it)
- Reference-motion named-clip loop-to-zero (refused on purpose)
- Mixing `MotionCursor` into `PlannerPlayback` (refused on purpose)
- PICO / CloudXR / Isaac Teleop SDK
- Isaac Lab PPO launch
- Combined T800+Hand MJCF
- Grasp-success numbers
- Live actor T-pose (q=0 overlay remains)
- GMR on a tiny BONES-SEED clip (blocked on flange/CoM)
- Diagnose a real GR00T / π0.5 dump once provided
