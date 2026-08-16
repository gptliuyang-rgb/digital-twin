# PHASE 0 — spec intake and contracts

## Done

- Frozen `interface/command_schema_v1.yaml` (75-D command, 6D Zhou, heading frame)
- Frozen `interface/frames.yaml` (T800 `LINK_WRIST_END_*` as SONIC wrist)
- `assets/dexhand2/meta/dexhand2_spec.yaml` filled from official docs + MJCF/URDF; remaining fields are `REQUIRED_INPUT`
- `joint_name_map.yaml` (doc / MJCF joint / MJCF actuator / SDK label / index)
- `interface/schema.py` with `SpecIncompleteError`, flatten/unflatten, import-time topology checks
- `docs/SPEC_INTAKE.md`

## Tests

See `make test`. `validate_repo_spec()` raises `SpecIncompleteError` listing open fields (expected).

## Open

P0 list is in SPEC_INTAKE.md. Nothing was invented.

## Human input needed (P0)

Friction, stiffness, hardware torque, hardware kp/kd, command latency, wrist CoM, T800 flange SE(3), safety rate limits, pad radius (or accept STL-fitted mesh radius as geometry-only), tactile-or-not on the actual unit.
