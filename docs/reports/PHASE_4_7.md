# PHASE 4–7 — scene, calibration, VLA glue, eval

## Phase 4 (sim digital twin)

- Parameterised box + Euro pallet constants
- Industrial MuJoCo scene: floor, pallet, two cartons, scan gun (`sim/mujoco_env/scene.py`)
- Scripted dual-arm pipeline: approach / grasp / lift / carry / stack / release / gun grip / QR geometry (`sim/tasks/industrial_pipeline.py`)
- QR PNG generator; `simulate_scan` geometry gate + OpenCV/pyzbar decode
- Sensor calib YAML is REQUIRED_INPUT; delay + JPEG helpers exist
- IBVS / camera decode of the gun view is **not** claimed (no offscreen GL requirement)

## Phase 5

- E1–E3 protocol written
- `fit_params.py` fits μ from E1 and `k` from E2 (`k` column or F=kx on 0.2–1.0 mm)
- Fragment + overlay apply; live spec is not silently patched
- `validate_sim.py` replays Coulomb pull (E1) and pad indent (E2) on an overlay spec
- `make calibrate-synthetic` dry-runs the pipeline; L2.2 uses a 3×3 μ × solref scan
  on the live uncalibrated spec and a single E1/E2 micro episode on an overlay
- Relative slip/drop/contacts only — never `grasp_success_rate` (ADR-004)

## Phase 6

- 6D/quat/RPY conversions with round-trip tests
- Heading-frame helper
- Temporal ensemble (SO(3) slerp on rot6d slices)
- Latency index into action chunks
- `data/build_modality.py` bound to `joint_order`

## Phase 7

- L0 flags a swapped finger channel
- L1 limit/coupling checks
- L2 runs uncalibrated physics when MuJoCo is present; still `blocked_uncalibrated` / no `grasp_success_rate`
- L3 reserved (Isaac / IBVS / GR00T deploy)
- L4 HIL and real-robot trials are **out of scope** until this sim twin is accepted
