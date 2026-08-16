# PHASE 1 — official asset ingest (not a from-scratch generator)

## Done

- `assets/dexhand2/build/ingest_official.py` checks MJCF joint order, limits, mass vs spec
- `assets/dexhand2/build/gen_derived.py` injects pad spheres fitted to official `*_tip.STL`
- Identity `hand/coupling.py`
- `scripts/check_upstream_drift.py`

## Not done (blocked)

- MuJoCo 10 s no-jitter run in CI (optional extra `.[sim]`, not installed by default)
- Convex-piece count / CoACD (official collision stays convex hull until derived simplified variant)
- Hardware gain identification (scan config is in `eval/configs/l2_mujoco.yaml`)

## Official baseline (from cloned wuji-description)

- 20 actuators, 5 fingertip sites, 10 contact excludes, skeleton mass 0.6207 kg
- Tip STLs present, not used as collision in upstream MJCF (confirmed)
- Hand 2 Beta 2 also exists upstream (tactile pad links). This repo targets Beta 1 unless `hardware_has_tactile` says otherwise.
