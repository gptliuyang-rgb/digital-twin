# This increment — IBVS, weld gate, stacking metrics

Cherry-picked pad-sphere / T800-Pro / QR-envelope work from `cursor/dexhand2-fe36`, then added the pieces that were still missing for the industrial task loop.

## Done

- `runtime/ibvs.py` — Chaumette 2006 point-feature IBVS; synthetic pinhole is labelled, not silent
- `sim/qr_visual_servo.py` — IBVS loop until `simulate_scan` decodes
- `runtime/wbc_safety.py` — 5 cm/step wrist jump hold
- `eval/stack_metrics.py` — support ∩ settle (2 cm / 3° / 5 s) + release quality
- `eval/gain_scan.py` — 9-cell gen-1 kp/kv scale grid
- `vla/client/pipeline.py` — infer → ensemble → latency (no sim imports)
- `assets/combined/assemble.py` — `weld_recipe.yaml` + `PolicyEvalBlocked` (no identity weld)
- `sim/hand_mass.py` — Δm = 0.1243 kg is not a CoM; SONIC mass refused
- Pallet (EPAL 1200×800) + scanner placeholder MJCF
- L0 names the swapped joint; L1 optional MuJoCo FK; L2 never emits grasp success
- T800 dummy wrist origins parsed from official URDF (`J_FIXED_WAIST_*`)

## Tests

`make test` (this increment's new files plus prior suite).

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. Mount SE(3), pad–cardboard μ, stiffness, hardware τ/kp/kd, command latency, wrist CoM.

## Not done

- Combined T800+Hand MJCF (correctly refused)
- RTX QR envelope
- SONIC T800 retarget
- Grasp-success numbers
