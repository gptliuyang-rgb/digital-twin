# PHASE 2 — whole-robot assembly

## Done

- `assets/engineai/meta/t800_joints.yaml` from Native SDK URDF (25 revolute)
- `assets/dexhand2/meta/mount_transform.yaml` with official mount→wrist offset
- `assets/combined/assemble.py` **exits** until T800 flange SE(3) is filled
- `sim/payload.py` mass/CoM API for later SONIC load randomization
- `docs/HW_INTEGRATION.md`

## Blocked

Cannot weld a combined MJCF/USD that pretends the flange transform is known.
T800 has no wrist pitch/roll in this URDF — see ADR-001.
