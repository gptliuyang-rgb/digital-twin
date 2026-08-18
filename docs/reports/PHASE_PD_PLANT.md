# This increment — 500 Hz joint-PD plant on held 25-D q_des

Continues ADR-054 (ZOH of `policy_action` onto the 500 Hz PD ring).
Still no grasp-success number and no combined T800+Hand weld.

## Done

- **ADR-055.** After the 50 Hz stash + ZOH, `SharedPlaybackCursor`
  evaluates `τ = Kp (a_t − q) − Kd q̇` with EngineAI `pd_stand`
  bring-up gains (`pd_stand_bringup_not_sonic`, not SONIC tracking):
  - After 10 ticks with `q[0] = i` and `policy_action a0 = 10..19`,
    decoder `his_last_actions` `a[0]` stays `[0, 10..18]`, PD `q_des[0]`
    is **19**, and `τ[0] = 1080 × (19 − 9) = 10800 N·m`.
  - `dq_des` is 0 (ZOH). One 50 Hz period is 10 identical 500 Hz
    torque samples when q/dq are hold-last from the 50 Hz snapshot.
  - Omit `policy_action` to keep the PD hold; τ still updates if q
    changed. `last_action=` does **not** write PD or the plant.
    `stash_policy_action()` alone does not write PD or τ.
  - τ does **not** enter decoder obs.
- G1 29-D / 32-D qpos / 45-D hands / command_schema 75-D / fake
  decoder ONNX / cubic Hermite / finite-diff dq still raise.
  `T800MujocoEnv.apply_pd` calls `joint_pd_torque_nm`.
- `make eval-l1a-pd-plant` dumps delayed a-history vs PD q_des vs τ.
  Hands bypass WBC. `grasp_success_rate` JSON `null`.

## Why this is not a G1 checkpoint

Official C++ runs the decoder, sends `a_t` to 500 Hz PD, then joint PD.
This repo still does **not** run ONNX. The caller supplies `a_t`; the
cursor stashes it (ADR-053), ZOH-holds it (ADR-054), and this plant
applies bring-up PD. Hermite stays on command_schema poses (ADR-039).

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. IMU noise, camera delay, and
command latency stay `REQUIRED_INPUT`. Flange SE(3) and wrist CoM still
block GMR-on-BONES-SEED and the combined weld. A real T800 decoder
output is still missing — this increment only applies PD to the caller slot.

## Not done

- Running a T800 encoder / planner / decoder ONNX (G1 weights are refused)
- Replacing ADR-018 interpolators with a trained generative planner
- Filling policy_action from a live policy output (caller must supply it)
- Closing this τ onto a 500 Hz MuJoCo joint-PD command plant on the
  shared tick (env already uses the same formula; the cursor does not
  step physics)
- Reference-motion named-clip loop-to-zero (refused on purpose)
- Mixing `MotionCursor` into `PlannerPlayback` (refused on purpose)
- PICO / CloudXR / Isaac Teleop SDK
- Isaac Lab PPO launch
- Combined T800+Hand MJCF
- Grasp-success numbers
- Live actor T-pose (q=0 overlay remains)
- GMR on a tiny BONES-SEED clip (blocked on flange/CoM)
- Diagnose a real GR00T / π0.5 dump once provided
