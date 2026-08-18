# This increment — 25-D policy_action ZOH onto the 500 Hz PD ring

Continues ADR-053 (delayed `policy_action` as next-tick `last_action`).
Still no grasp-success number and no combined T800+Hand weld.

## Done

- **ADR-054.** After `stash_policy_action` on the same 50 Hz tick,
  `a_t` is zero-order held onto `PolicyPdHold` (the 500 Hz PD ring):
  - After 10 ticks with `policy_action a0 = 10..19`, decoder
    `his_last_actions` `a[0]` stays `[0, 10..18]` while PD `q_des[0]`
    is **19**. Encoder 10frame_step5 stays on clip indices.
  - One 50 Hz period is 10 identical 500 Hz samples (ZOH, not Hermite).
  - Two 50 Hz actions → 11 samples: first ten = `a0`, last = `a1`.
  - Omit `policy_action` to keep the PD hold. `last_action=` does
    **not** write PD. `stash_policy_action()` alone does not write PD.
  - `play == false` and `tick()` without `token` still push PD after
    stash.
- G1 29-D / 32-D qpos / 45-D hands / command_schema 75-D / fake
  decoder ONNX / cubic Hermite of joint targets still raise.
  `stream_policy_tokens` on 25-D points at `stream_policy_action_zoh`.
- `make eval-l1a-pd-stream` dumps delayed a-history vs PD q_des vs ZOH
  window. Hands bypass WBC. `grasp_success_rate` JSON `null`.

## Why this is not a G1 checkpoint

Official C++ runs the decoder, sends `a_t` to the 500 Hz PD, then
writes that output into `last_action` for the **next** gather. This
repo still does **not** run ONNX. The caller supplies `a_t`; the cursor
stashes it (ADR-053) and ZOH-holds it on PD (this increment). Hermite
stays on command_schema poses (ADR-039).

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. IMU noise, camera delay, and
command latency stay `REQUIRED_INPUT`. Flange SE(3) and wrist CoM still
block GMR-on-BONES-SEED and the combined weld. A real T800 decoder
output is still missing — this increment only ZOH-holds the caller slot.

## Not done

- Running a T800 encoder / planner / decoder ONNX (G1 weights are refused)
- Replacing ADR-018 interpolators with a trained generative planner
- Filling policy_action from a live policy output (caller must supply it)
- Applying PD `q_des` to a 500 Hz joint-PD plant (this is the hold only)
- Reference-motion named-clip loop-to-zero (refused on purpose)
- Mixing `MotionCursor` into `PlannerPlayback` (refused on purpose)
- PICO / CloudXR / Isaac Teleop SDK
- Isaac Lab PPO launch
- Combined T800+Hand MJCF
- Grasp-success numbers
- Live actor T-pose (q=0 overlay remains)
- GMR on a tiny BONES-SEED clip (blocked on flange/CoM)
- Diagnose a real GR00T / π0.5 dump once provided
