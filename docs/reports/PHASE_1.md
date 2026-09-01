# PHASE 1 — official asset ingest (not a from-scratch generator)

## Done

- `assets/dexhand2/build/ingest_official.py` checks MJCF joint order, limits, mass vs spec
- `assets/dexhand2/build/gen_derived.py` injects pad spheres fitted to official `*_tip.STL`
- Identity `hand/coupling.py`
- `scripts/check_upstream_drift.py`

## Not done (blocked)

- MuJoCo 10 s no-jitter run in CI (optional extra `.[sim]`, not installed by default)
- Sphere-fit simplified pads (official Beta 2 collision is convex hull of pad mesh)
- Hardware gain identification (scan config is in `eval/configs/l2_mujoco.yaml`)

## Official baseline (from cloned wuji-description)

- Beta 1: 20 actuators, 5 fingertip sites, 10 contact excludes, sim mass 0.6207 kg, **pads not colliding**
- Beta 2 (sim default): 26 bodies, 5 pad bodies colliding, sim mass 0.6228 kg
- Tip query sites unchanged. Joint limits / actuators / gen-1 gains identical.
- `hardware_has_tactile` still REQUIRED_INPUT for the physical unit.
