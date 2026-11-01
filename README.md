# digital-twin

Simulation-side digital twin for **EngineAI T800 + Wuji Hand 2.2 (DexHand2)** industrial tasks: dual-arm box carry, stacking, and handheld QR scan.

This repository is the **P1 digital-twin layer**: frozen command contracts, official-asset ingest, a sim/real shared hand controller, and L0–L2 eval harnesses. It does **not** train SONIC or a VLA, and it does **not** report grasp success until contact parameters are measured.

Policy client code under `runtime/` and `vla/client/` does not import MuJoCo or Isaac. Only `sim/*_backend` / `hand/backends/mujoco_backend.py` talk to a simulator.

## Layout

```
interface/     command_schema_v1.yaml, frames.yaml, schema.py
assets/        dexhand2 spec + official Hand 2.2 ingest; T800 joint table
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
make ingest-official
make check-drift
```

`make check-spec` is **supposed to fail** until the P0 `REQUIRED_INPUT` fields in `docs/SPEC_INTAKE.md` are filled. That is intentional.

## Facts already taken from official sources

Wuji Hand 2.2 (Beta, non-tactile prototype docs): 20 independent revolute DoF, no coupling, product curb **0.800 kg** (bare 0.650, base 0.070 extra), sim no-cover **0.6878 kg** / with-cover **0.7088 kg** (cover provisional), MIT hybrid @ 1 kHz, 11–13 V, RJ45. Model root is `{l,r}_mount`. Official MJCF still uses gen-1 kp/kv and collides `*_tip_sensor_frame` as a convex hull.

Firmware factory MIT after a reset is kp=5.0 / kd=0.1 — recorded, **not** copied into `hardware_kp`. Data port 50001; 50000 is the discovery port and is rejected.

T800 Native SDK URDF: 25 revolute DoF, wrists are dummy `LINK_WRIST_END_*` frames on elbow yaw. T800 Pro is different (wrist pitch/roll + built-in hands).

## Next human inputs

See `docs/SPEC_INTAKE.md`. The blockers for an honest twin are pad–cardboard friction, pad stiffness, hardware torque/gains, command latency, wrist CoM, and the T800 flange SE(3).
