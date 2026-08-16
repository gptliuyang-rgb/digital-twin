# PHASE 3 — hand control stack (sim/real same code)

## Done

- `DexHand2Controller` + MIT law + safety (NaN / jump / limits)
- Backends: mock, mujoco, isaac, real (SDK, optional import)
- Primitives: open, power_grasp, pinch, gun_grip, flat_support
- Official retarget wrapper (no invented IK scale)

## Tests

Controller tests use MockBackend. `tests/test_no_sim_imports.py` guards `hand/controller.py`, `runtime/`, `vla/client/`.
