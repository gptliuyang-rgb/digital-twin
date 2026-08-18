# This increment — omit push_hw so next-tick decoder q follows the plant

Continues ADR-056 (10 × 2 ms physics substeps of τ = Kp(a_t − q) − Kd q̇).
Still no grasp-success number and no combined T800+Hand weld.

## Done

- **ADR-057.** After ADR-056 copies post-physics q/dq into `HardwareHold`,
  omitting `push_hw` on later ticks closes the 50 Hz gather onto the plant:
  - Tick 0 still needs a full `HardwareSnapshot` (IMU + initial q).
  - Later ticks that omit `push_hw` assemble decoder 874-D from the
    previous period's measured joints.
  - This tick's decoder q stays pre-*this*-tick physics.
  - IMU (ω, quat) is kept from the last full snapshot. Do not invent
    it from physics body rates. Measured IMU latency stays REQUIRED_INPUT.
  - Passing `push_hw` every tick still overrides the plant (ADR-056 eval).
  - `last_action=` does not write PD or physics. `stash_policy_action()`
    alone does not write PD, τ, or physics.
- G1 29-D / 32-D qpos / 45-D hands / command_schema 75-D / fake decoder
  ONNX / cubic Hermite / finite-diff dq still raise.
- `make eval-l1a-pd-closedloop` dumps delayed a-history vs plant-following
  q vs kept IMU. Hands bypass WBC. `grasp_success_rate` JSON `null`.

## Why this is not a G1 checkpoint

Official C++ applies 500 Hz PD on the robot, then the next 50 Hz gather
reads the resulting joints. This repo still does **not** run ONNX. The
caller supplies `a_t`; the cursor stashes it, ZOH-holds it, evaluates
bring-up PD, steps a 500 Hz plant, and (if `push_hw` is omitted) the
next decoder q is that plant. Hermite stays on command_schema poses
(ADR-039).

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. IMU noise, camera delay, and
command latency stay `REQUIRED_INPUT`. Flange SE(3) and wrist CoM still
block GMR-on-BONES-SEED and the combined weld. A real T800 decoder
output is still missing — this increment only closes gather onto the
caller-supplied `policy_action` plant.

## Not done

- Running a T800 encoder / planner / decoder ONNX (G1 weights are refused)
- Replacing ADR-018 interpolators with a trained generative planner
- Filling policy_action from a live policy output (caller must supply it)
- Writing physics q into this tick's decoder obs (refused on purpose)
- Inventing IMU from body rates (refused on purpose)
- Reference-motion named-clip loop-to-zero (refused on purpose)
- Mixing `MotionCursor` into `PlannerPlayback` (refused on purpose)
- PICO / CloudXR / Isaac Teleop SDK
- Isaac Lab PPO launch
- Combined T800+Hand MJCF
- Grasp-success numbers
- Live actor T-pose (q=0 overlay remains)
- GMR on a tiny BONES-SEED clip (blocked on flange/CoM)
- Diagnose a real GR00T / π0.5 dump once provided
