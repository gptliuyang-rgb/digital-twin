# L2a — contact honesty (this increment)

## Done

- `eval/gates.py`: recursive forbid of `grasp_success_rate` / `pick_success_rate` / `success_rate` on L2 JSON
- `run_gain_scan`: kp/kv corner (default) or full 3×3 at mid μ/solref; `make eval-l2-gains` for full
- `fit_e3`: max held mass from PROTOCOL CSV; **does not** invent `motor_max_torque_nm`
- Synthetic E3 sheets generated with E1/E2 (`write_synthetic_csvs`)
- `CartesianJumpFilter` (5 cm/step default) in `runtime/safety_filter.py` — sim/real shared, no simulator imports
- `assets/dexhand2/build/pad_metrics.py` + `make phase1-baseline`: official vs derived site→collision distances
  - **2 mm gate not met** (~23 mm pad-sphere vs site). STL frame ≠ distal site frame. Name-stripping bug that dropped 3 fingers’ pads is fixed.

## Not done (blocked on humans)

- Live E1/E2 CSV → `--commit-live` (friction/stiffness still `REQUIRED_INPUT`)
- Human accept of proposed `solref`
- Hardware kp/kd identification (scan is relative only)
- Wiring wrist-jump filter into a SONIC/VLA client (filter exists; L2c runtime not on this branch)

## Tests

`tests/test_l2a_l2b.py` plus existing calibration tests. Live spec remains uncalibrated.
