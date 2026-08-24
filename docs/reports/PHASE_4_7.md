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
- `fit_params.py` / `validate_sim.py` refuse uncalibrated specs
- L2.2 micro-env 3×3 μ × solref scan reports *relative* slip/drop/contacts only

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
