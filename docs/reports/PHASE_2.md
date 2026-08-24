# PHASE 2 — whole-robot assembly

## Done

- `assets/engineai/meta/t800_joints.yaml` from Native SDK URDF (25 revolute)
- `assets/dexhand2/meta/mount_transform.yaml` with official mount→wrist offset
- **Kinematic-bringup assembly** (`assets/combined/assemble.py`): identity weld of `{l,r}_mount` onto `LINK_WRIST_END_*` (ADR-006). CAD flange remains `REQUIRED_INPUT` / `policy_eval_forbidden`.
- Combined MJCF compiles: nq=65 (pinned base), nu=65 (25 body motors + 40 MIT hand motors), nbody=75
- Combined URDF concatenates official T800 + both with-mount Hand 2 URDFs
- `sim/payload.py` mass/CoM API for later SONIC load randomization
- `docs/HW_INTEGRATION.md`

## How to generate

```bash
make assemble          # MJCF + URDF + compile check
make build-assets      # left + right pad-sphere derived hands
```

Generated files live in `assets/combined/generated/` (gitignored). Official XML is never overwritten.

## Still blocked for policy eval

Do not treat the identity flange as CAD. T800 has no wrist pitch/roll in this URDF — see ADR-001. Contact μ/stiffness are still `REQUIRED_INPUT` (ADR-004).
