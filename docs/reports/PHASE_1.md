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

- 20 actuators, 5 fingertip sites, 15 contact excludes, 27 group-2 hulls
- No-cover URDF 0.6878 kg (27 links, includes 0.069 kg mount); with-cover 0.7088 kg (provisional)
- Official 2.2 collides `*_tip_sensor_frame` as a convex hull; derived spheres replace those hulls
- `hand2_1` is a joint-identity / mass drift check only. Beta 1/2 packages are gone upstream.
