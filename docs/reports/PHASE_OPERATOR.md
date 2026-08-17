# This increment — SONIC §3.5 100 Hz operator-input loop

Continues ADR-039 (500 Hz PD stream). Still no grasp-success number
and no combined T800+Hand weld.

## Done

- **ADR-040.** Paper §3.5 four-rate stack is now fully represented as
  modules: planner 10 Hz, policy 50 Hz, operator 100 Hz (this module),
  command stream 500 Hz. YAML locks integer factors 5 (→500), 2 (→50),
  10 (→10). `ingest_operator` accepts uniform 100 Hz `command_schema_v1`
  rows from keyboard / gamepad / vr_3point / vr_5point / network.
  VLA / GR00T / π0.5 / policy sources raise `OperatorInputError` (those
  clocks go through L1a). Case A 50-D is refused. `operator_to_stream`
  splines poses and **hold-last** `nav_cmd` (not Hermite). Policy tokens
  and L1a waypoints are aligned strided subsets. Hybrid encoder tokens
  are packed at 50 Hz; hands still bypass WBC. Live `OperatorHold` is
  the 100 Hz push / 500 Hz readout. No PICO / CloudXR / Isaac Teleop
  import. DexHand2 MIT stays 1 kHz.
- 1.6 s window: 161 @ 100 Hz → 81 @ 50 Hz → 17 @ 10 Hz → 801 @ 500 Hz.

## Why this is not a Table S4 sweep

Table S4 is closed. The 100 Hz ring is the *deployment* operator sample
clock between a VR/gamepad/keyboard/network encoder and the 500 Hz PD
stream, the remaining four-rate piece after ADR-018 / ADR-038 / ADR-039.

`grasp_success_rate` JSON `null`. Combined robot stays `PolicyEvalBlocked`.

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. BONES-SEED GMR, live actor T-pose,
and real GR00T/π0.5 dumps stay blocked on flange SE(3) and wrist CoM.
Isaac Lab PPO launch still refused. Pad–cardboard μ remains `REQUIRED_INPUT`.

## Not done

- PICO / CloudXR / Isaac Teleop SDK backend (rate contract only)
- Isaac Lab PPO launch
- Combined T800+Hand MJCF
- Grasp-success numbers
- Live actor T-pose (q=0 overlay remains)
- Running `IsaacLabSceneRuntime` inside Isaac Sim python
- GMR on a tiny BONES-SEED clip (blocked on flange/CoM)
- Diagnose a real GR00T / π0.5 dump once provided
- Re-measure official XML MAE for the walk-clip sweep when Native SDK is cloned
- SONIC's trained generative kinematic planner (this is still only interpolators + Eq. 8 + the 100/500 Hz rings)
