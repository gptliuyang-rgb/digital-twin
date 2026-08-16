# digital-twin

Simulation-side digital twin for **EngineAI T800 + Wuji Hand 2 (DexHand2)** industrial tasks: dual-arm box carry, stacking, and handheld QR scan.

This repository is the **P1 digital-twin layer**: frozen command contracts, official-asset ingest, a sim/real shared hand controller, and L0–L2 eval harnesses. It does **not** train SONIC or a VLA, and it does **not** report grasp success until contact parameters are measured.

Policy client code under `runtime/` and `vla/client/` does not import MuJoCo or Isaac. Only `sim/*_backend` / `hand/backends/mujoco_backend.py` talk to a simulator.

## Layout

```
interface/     command_schema_v1.yaml, frames.yaml, schema.py
assets/        dexhand2 spec + official ingest; T800 joint table
hand/          controller, coupling, primitives, backends, calibration
runtime/       safety filter, temporal ensemble, latency compensation
vla/           adapters + policy client (no sim imports)
sim/           payload, QR scanner, sensor delay/JPEG
eval/          L0–L2 harnesses
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
make eval-qr            # synthetic QR envelope heatmap (needs OpenCV)
```

## Facts already taken from official sources

Wuji Hand 2 Beta 1: 20 independent revolute DoF, no coupling, product mass 0.745±0.010 kg (soft body, no cables), skeleton URDF 0.6207 kg, MIT hybrid @ 1 kHz, 12 V, RJ45. Official MJCF uses gen-1 kp/kv and does **not** collide the fingertip pad meshes.

T800 Native SDK URDF: 25 revolute DoF, wrists are dummy `LINK_WRIST_END_*` frames on elbow yaw. T800 Pro is different (wrist pitch/roll + built-in hands).

## Next human inputs

See `docs/SPEC_INTAKE.md`. The blockers for a honest twin are pad–cardboard friction, pad stiffness, hardware torque/gains, command latency, wrist CoM, and the T800 flange SE(3).
