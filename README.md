# digital-twin

Simulation-side digital twin for **EngineAI T800 + Wuji Hand 2 (DexHand2)** industrial tasks: dual-arm box carry, stacking, and handheld QR scan.

This repository is the **P1 digital-twin layer**: frozen command contracts, official-asset ingest, a sim/real shared hand controller, and L0–L2 eval harnesses. It does **not** train SONIC or a VLA, and it does **not** report grasp success until contact parameters are measured.

Policy client code under `runtime/` and `vla/client/` does not import MuJoCo or Isaac. Only `sim/*_backend` / `hand/backends/mujoco_backend.py` talk to a simulator.

QR scan uses IBVS (`runtime/ibvs.py`) plus real decode. Combined T800+Hand 2 policy eval is refused until the wrist flange SE(3) is CAD-measured — identity is not a substitute.

## Layout

```
interface/     command_schema_v1.yaml, command_schema_v1_5point.yaml, frames.yaml, schema.py
assets/        dexhand2 spec + official ingest; T800 joint table
hand/          controller, coupling, primitives, backends, calibration
runtime/       safety filter, temporal ensemble, latency, IBVS, task FSM
wbc/           T800 SONIC contract, GMR IK export, 3/5-point teleop remap, L1a planner + Eq. 8 nav spring + 100 Hz operator loop + 500 Hz PD stream + S7 YAML obs gather + encoder motion_* look-ahead (default 10frame_step5 + low-latency 10frame_step1 + v1.1 heading 10frame_step5), G1-checkpoint guard
vla/           adapters + policy client (no sim imports)
sim/           payload, QR scanner, URDF FK, hand-only MuJoCo, privileged L2 pallet drop
eval/          L0–L2 harnesses, 9-cell gain scan, L0 ckpt diagnose
docs/          SPEC_INTAKE, DECISIONS, HW_INTEGRATION, RUNBOOK
```

## Quick start

```bash
python3 -m pip install -e ".[dev]"
./scripts/bootstrap_resources.sh
make test
```

`make check-spec` is **supposed to fail** until the P0 `REQUIRED_INPUT` fields in `docs/SPEC_INTAKE.md` are filled. That is intentional.

```bash
./scripts/bootstrap_resources.sh
make ingest-official    # writes docs/reports/PHASE_1_baseline.md
make build-assets       # palmar pad spheres + MIT motors + simplified capsules
make gmr-tpose         # q=0 T800 vs PM01 overlay + rewrite IK JSON
make usd-pads           # USDA pad-sphere overlay (right and left if fitted)
make eval-l2-priv       # privileged pallet drop; grasp_success_rate stays null
make eval-l3-priv       # Isaac Lab privileged cfg dump; still no grasp-success
make eval-gain-scan     # 9-cell MIT kp/kv hold; grasp_success_rate stays null
make eval-l0-diagnose   # classify a ckpt action last-dim (A/B/C); no weights required
make eval-l1-case-a     # 50-D → 75-D FK; requires --apply-fk; uses t800_kinematics.yaml
make extract-kinematics # dump official URDF joints + MJCF range= into t800_kinematics.yaml
make ppo-status         # T800 action_dim 25 vs G1 29; launch blockers
make ppo-train          # supposed to fail until P0 CoM/flange + Isaac Lab
make eval-l2-sim2sim    # kinematic identity MPJPE; grasp_success_rate stays null
make eval-l2-physics-sim2sim  # MuJoCo PD tracking + Table S4 clip joint_jitter ±0.1 rad + pos/ori root jitter + lin_vel/ang_vel walk-clip jitter (negative control, not a height/ori or root_push gate)
make eval-l2-freebase-stand   # floating-base PD stand; fall is reported, not a SONIC gate
make eval-l2-freebase-push    # lean vs fall; Table S4 ±X/±Y/±Z one-shot + F=mv/T + angvel + τ=Iω/T + μ_slide + base CoM ipos + qpos ±0.01 rad; air-drop; μd/restitution recorded-only
make eval-l3-isaac-bind       # Isaac reset/step if Sim python is bound; else unavailable
make eval-l1a             # 10 Hz interpolator smoke
make eval-l1a-spring      # SONIC Eq. 8 reverse-6 diagnostic; grasp_success_rate stays null
make eval-l1a-stream      # SONIC §3.5 500 Hz PD ring; grasp_success_rate stays null
make eval-l1a-operator    # SONIC §3.5 100 Hz operator loop; grasp_success_rate stays null
make eval-l1a-gather      # SONIC §S7 YAML obs gather (T800 874-D); grasp_success_rate stays null
make eval-l1a-encoder     # SONIC encoder 842-D default + 831-D low-latency + 831-D v1.1 heading; no PICO, no G1 ONNX
```

## Facts already taken from official sources

Wuji Hand 2 Beta 1: 20 independent revolute DoF, no coupling, product mass 0.745±0.010 kg (soft body, no cables), skeleton URDF 0.6207 kg, MIT hybrid @ 1 kHz, 12 V, RJ45. Official MJCF uses gen-1 kp/kv and does **not** collide the fingertip pad meshes.

T800 Native SDK URDF: 25 revolute DoF, wrists are dummy `LINK_WRIST_END_*` frames on elbow yaw. T800 Pro is different (wrist pitch/roll + built-in hands).

## Next human inputs

See `docs/SPEC_INTAKE.md`. The blockers for a honest twin are pad–cardboard friction, pad stiffness, hardware torque/gains, command latency, wrist CoM, and the T800 flange SE(3).
