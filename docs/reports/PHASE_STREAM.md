# This increment — SONIC §3.5 500 Hz PD command stream

Continues ADR-038 (Eq. 8 nav spring on L1a). Still no grasp-success number
and no combined T800+Hand weld.

## Done

- **ADR-039.** Paper §3.5 four-rate stack: planner 10 Hz, policy 50 Hz,
  operator 100 Hz (ADR-040), command stream 500 Hz (this module). YAML
  locks integer factors 50 and 10. `stream_planned_ref` interpolates a 10 Hz
  `PlannedRef` with the ADR-018 cubic/SLERP kernel. `stream_policy_tokens`
  interpolates 50 Hz tracker rows. `stream_nav_spring` evaluates Eq. 8 at
  500 Hz timestamps — it does **not** Hermite-resample the 10 Hz spring
  samples. Case A 50-D is refused. Hands still bypass WBC. DexHand2 MIT
  stays 1 kHz. ADR-021 `sonic_command_hz: 50` is renamed in comments to
  *policy/token rate* so it is not confused with this ring.
- 1.6 s L1a window: 17 steps at 10 Hz → 801 steps at 500 Hz. 10 Hz
  timestamps are a subset of the 500 Hz grid.

## Why this is not a Table S4 sweep

Table S4 is closed. The 500 Hz ring is the *deployment* interpolator
between L1a / the 50 Hz tracker and joint PD, the remaining four-rate
piece after ADR-018 (10 Hz) and ADR-038 (Eq. 8).

`grasp_success_rate` JSON `null`. Combined robot stays `PolicyEvalBlocked`.

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. BONES-SEED GMR, live actor T-pose,
and real GR00T/π0.5 dumps stay blocked on flange SE(3) and wrist CoM.
Isaac Lab PPO launch still refused. Pad–cardboard μ remains `REQUIRED_INPUT`.

## Not done

- Operator-input 100 Hz loop — **done in ADR-040** (`wbc/operator.py`)
- Isaac Lab PPO launch
- Combined T800+Hand MJCF
- Grasp-success numbers
- Live actor T-pose (q=0 overlay remains)
- Running `IsaacLabSceneRuntime` inside Isaac Sim python
- GMR on a tiny BONES-SEED clip (blocked on flange/CoM)
- Diagnose a real GR00T / π0.5 dump once provided
- Re-measure official XML MAE for the walk-clip sweep when Native SDK is cloned
- SONIC's trained generative kinematic planner (this is still only interpolators + Eq. 8)
