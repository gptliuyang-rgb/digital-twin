# PHASE 4–7 — scene, calibration, VLA glue, eval

## Phase 4

- Parameterised box + Euro pallet constants
- QR PNG generator; `simulate_scan` geometry gate + OpenCV/pyzbar decode
- Sensor calib YAML is REQUIRED_INPUT; delay + JPEG helpers exist
- Scan envelope heatmap: `eval/qr_envelope.py` (synthetic pinhole + real decode). `make eval-qr`. Not RTX.

## Phase 5

- E1–E3 protocol written
- `fit_params.py` / `validate_sim.py` refuse uncalibrated specs

## Phase 6

- 6D/quat/RPY conversions with round-trip tests
- Heading-frame helper
- Temporal ensemble (SO(3) slerp on rot6d slices)
- Latency index into action chunks
- `data/build_modality.py` bound to `joint_order`

## Phase 7

- L0 flags a swapped finger channel
- L1 limit/coupling checks
- L2 blocked while uncalibrated
- L3 reserved
