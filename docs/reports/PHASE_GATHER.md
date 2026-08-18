# This increment — SONIC §S7 observation gathering at the four rates

Continues ADR-040 (100 Hz operator loop). Still no grasp-success number
and no combined T800+Hand weld.

## Done

- **ADR-041.** Paper S7 state logger + YAML observation registry. Default
  layout matches official GEAR-SONIC `observation_config.yaml` names with
  T800 dims: token 64 + ω 30 + q 250 + dq 250 + a 250 + g 30 = **874**.
  G1 994 is refused. `sensor_delay_ticks` is locked at 0 (latest-data-wins);
  a non-zero delay raises until SPEC_INTAKE measures one. Hands never enter
  the vector. `HardwareHold` is any-rate push / latest readout (500 Hz PD
  ring compatible). `StateLogger` is the 50 Hz ring with strided lookback
  and startup zero-padding. Grouped YAML order converts to/from the
  interleaved `ProprioHistory` packing (same dim, different order).
- 10 policy ticks fill q-history `[0..9]`. One tick leaves 9 zero frames.

## Why this is not a Table S4 sweep

Table S4 is closed. This is the *deployment* observation assembler the
50 Hz control loop uses to build the decoder vector from IMU + joints.

`grasp_success_rate` JSON `null`. Combined robot stays `PolicyEvalBlocked`.

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. IMU noise, camera delay, and
command latency stay `REQUIRED_INPUT`. BONES-SEED GMR, live actor T-pose,
and real GR00T/π0.5 dumps stay blocked on flange SE(3) and wrist CoM.

## Not done

- Measured IMU/joint latency (must not be invented)
- Running a T800 encoder ONNX (encoder *input* gather is ADR-042)
- PICO / CloudXR / Isaac Teleop SDK
- Isaac Lab PPO launch
- Combined T800+Hand MJCF
- Grasp-success numbers
- Live actor T-pose (q=0 overlay remains)
- GMR on a tiny BONES-SEED clip (blocked on flange/CoM)
- Diagnose a real GR00T / π0.5 dump once provided
- SONIC's trained generative kinematic planner
