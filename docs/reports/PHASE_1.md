# PHASE 1 — official asset ingest (not a from-scratch generator)

## Done

- `assets/dexhand2/build/ingest_official.py` checks MJCF joint order, limits, mass vs spec, 10 contact excludes, 5 tip STLs, 5 sites
- Collision audit: official MJCF has **zero** pad spheres; tip STLs exist on disk and are not collision
- `assets/dexhand2/build/gen_derived.py` injects 3 palmar-pad spheres per finger from the **distal** STL (ADR-006), disables distal hull collision, optional capsule simplified variant, converts `<position>` → `<motor>` for MIT
- Identity `hand/coupling.py`
- `scripts/check_upstream_drift.py` (run via Makefile `check-drift`)
- Baseline markdown: `docs/reports/PHASE_1_baseline.md`

## Measured on this clone (`wuji-description` @ `06e5f14c`)

See PHASE_1_baseline.md. Right-hand URDF skeleton mass **0.6207 kg**, 20 actuators, 5 sites.

Palmar-vertex → pad-sphere surface is asserted **< 2 mm** in `tests/test_derived_assets.py`. Official site → distal hull vertex is already ~0.1 mm (site sits on the bone hull, not the pulp).

## Not done (blocked / optional)

- MuJoCo 10 s no-jitter run in default CI (`.[sim]` extra)
- CoACD convex decomposition (official hulls kept except distal, replaced by spheres)
- Hardware gain identification (9-cell scan in `eval/configs/gain_scan.yaml`)
