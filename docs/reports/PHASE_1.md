# PHASE 1 — official asset ingest (not a from-scratch generator)

## Done

- `assets/dexhand2/build/ingest_official.py` checks MJCF joint order, limits, mass vs spec (both revisions)
- `assets/dexhand2/build/gen_derived.py` **replaces** official Beta 2 pad hulls with 3 mesh-fit spheres; `simplified` also drops distal-bone hulls
- Identity `hand/coupling.py`
- `scripts/check_upstream_drift.py` pin `c2cd7f8` (v2026.9.11); Hand 2 hashes unchanged from 4c1073d0

## Not done (blocked)

- MuJoCo 10 s no-jitter run in CI (optional extra `.[sim]`, not installed by default)
- Hardware gain identification (scan config is in `eval/configs/l2_mujoco.yaml`)
- Live pad radius / μ / k (mesh-fit is not E1/E2)

## Official baseline (from cloned wuji-description)

- Beta 1: 20 actuators, 5 fingertip sites, 10 contact excludes, sim mass 0.6207 kg, **pads not colliding**
- Beta 2 (sim default): 26 bodies, 5 pad bodies colliding as convex hulls, sim mass 0.6228 kg
- Derived: pad hulls disabled, 15 spheres, site-to-surface ≤ 2 mm
- Tip query sites unchanged. Joint limits / actuators / gen-1 gains identical.
- `hardware_has_tactile` still REQUIRED_INPUT for the physical unit.
