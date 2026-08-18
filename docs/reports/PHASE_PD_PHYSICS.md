# This increment — 500 Hz joint-PD τ closed onto a physics plant

Continues ADR-055 (τ = Kp(a_t − q) − Kd q̇ from the 50 Hz snapshot).
Still no grasp-success number and no combined T800+Hand weld.

## Done

- **ADR-056.** After the 50 Hz stash + ZOH + plant, `SharedPlaybackCursor`
  optionally applies that τ for **10 physics substeps at 1/500 s**:
  - `wbc/` stays numpy-only. `JointPdPhysics` is a protocol; MuJoCo
    implements it on `T800MujocoEnv.apply_tau_and_step`.
  - q/dq are **measured** each substep. Do not interpolate a 50 Hz
    snapshot. Do not finite-diff dq. Timestep other than 0.002 s raises.
  - This tick's decoder 874-D stays pre-physics. After the period,
    `HardwareHold.push_joints` copies q/dq for the *next* gather and
    **keeps IMU** (not invented).
  - `pd_tau_nm` is still the ADR-055 50 Hz snapshot. `pd_physics_tau_nm`
    is the last 500 Hz sample (q has moved).
  - Omit the backend to keep ADR-055. `last_action=` does not write PD
    or physics. `stash_policy_action()` alone does not write PD or τ
    or physics.
- G1 29-D / 32-D qpos / 45-D hands / command_schema 75-D / fake decoder
  ONNX / cubic Hermite / finite-diff dq still raise.
- `make eval-l1a-pd-physics` dumps delayed a-history vs pre-physics q
  vs 10 substeps. Hands bypass WBC. `grasp_success_rate` JSON `null`.

## Why this is not a G1 checkpoint

Official C++ runs the decoder, sends `a_t` to 500 Hz PD, then joint PD
on the robot/sim. This repo still does **not** run ONNX. The caller
supplies `a_t`; the cursor stashes it, ZOH-holds it, evaluates bring-up
PD, and (if attached) steps a 500 Hz plant. Hermite stays on
command_schema poses (ADR-039).

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. IMU noise, camera delay, and
command latency stay `REQUIRED_INPUT`. Flange SE(3) and wrist CoM still
block GMR-on-BONES-SEED and the combined weld. A real T800 decoder
output is still missing — this increment only applies PD to the caller
slot inside an optional physics backend.

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
