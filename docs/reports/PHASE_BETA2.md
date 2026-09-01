# PHASE Beta 2 contract sync — 2026-09-01

Official Wuji docs and `wuji-description` now default to **Hand 2 Beta 2**. This phase
retargets the frozen contracts. No contact parameters were invented. Grasp success
stays forbidden (ADR-004). Combined T800 weld stays blocked on flange SE(3).

## Done

- `dexhand2_spec.yaml` default `sim_model_revision: hand2_beta2`
- Product curb mass **0.800 kg**; sim no-mount **0.6228 kg** (measured URDF/MJCF)
- Official Beta 2 pad bodies: 5, masses thumb 6.3 g / others 3.0 g, **do collide**
- Distal masses re-measured (thumb 10.9 g / others 5.0 g)
- Joint names, limits, actuator order, gen-1 kp/kv: **identical** Beta 1/2
- Mount→wrist SE(3): **identical** `[0.003, 0.00025016, -0.0285]` m
- Tactile contract 40/34 points @ 100 Hz; decode from `FingertipSensorInfo.format`
- Ingest checks both revisions; drift pin watches Beta 2 files
- `hardware_revision` and `hardware_has_tactile` remain REQUIRED_INPUT

## Tests

`make test` (this environment clones `wuji-description`, so ingest is live).

`validate_repo_spec()` still raises `SpecIncompleteError` (expected).

## Not done (still blocked on humans)

Friction, stiffness, hardware torque, hardware kp/kd, command latency, wrist CoM,
T800 flange SE(3), safety rate limits, live pad radius, physical-unit revision.

## Note on changelog vs measurement

Description changelog 2026.08.17 said Beta 2 total mass matches Beta 1 after the
pad-body split. Measured inertial sum is **+2.1 g**. Spec stores 0.6228 kg
(ADR-007).
