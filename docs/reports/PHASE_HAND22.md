# PHASE — Hand 2.2 retarget (2026-11-01 monthly)

## What changed upstream since the last monthly run (2026-10-01)

wuji-description **v2026.10.12** removed `hand2_beta1` / `hand2_beta2` and the
`_with_mount` variants. Current tree:

| Revision | Path | ROS 2 package |
|---|---|---|
| 2.2 (default) | `hand2/hand2_2/body/` | `wuji_hand2_2_description` |
| 2.1 (compat) | `hand2/hand2_1/body/` | `wuji_hand2_1_description` |

Pinned commit: `5c4dfa1` (tag v2026.10.26).

## Numbers taken from official sources (not invented)

| Item | Value | Source |
|---|---|---|
| Product curb mass | 0.800 kg (cables yes, base no) | overview 2.2 |
| Product bare / base | 0.650 / 0.070 kg | overview 2.2 |
| Sim no-cover / with-cover | 0.6878 / 0.7088 kg | URDF + release notes 2026-10-26 |
| Mount | 0.069 kg, model root `{l,r}_mount` | model-integration + URDF |
| mount→wrist | `[0.00300045189842, 0, -0.0284982808512]` m | 2.2 MJCF |
| Joint / actuator names | 20, identical 2.1↔2.2 | MJCF |
| Collision | 27 group-2 hulls, 15 excludes | 2.2 MJCF / docs |
| Factory IP:port | 192.168.1.110/111:50001 | electrical-network-integration |
| Data port range | 7400–65535, 7447 ok, 50000 forbidden | release notes 2026-10-26 |
| Firmware factory MIT | kp=5.0, kd=0.1 on **new/reset** configs | release notes 2026-10-12 |
| Current firmware (release notes) | v2.13.1 | 2026-10-26 |
| Skin during operation | required | overview + hardware-integration |
| Voltage | 11–13 V (nominal 12 V) on 2.2 | overview / electrical |

Cover mass is **provisional** (official). Soft-body deformation is not simulated.

## Derived assets

`make build-assets` disables the five official `*_tip_sensor_frame` group-2 hulls
and injects 3 mesh-fit spheres per pad (`condim=4`). Fitting uses
`*_tip_sensor_frame.STL` (same frame as the fingertip site). Official `*_tip.STL`
is a different frame (Z flipped/offset) and is not used for collision placement.
The distal sphere is pinned so the official site lies on its surface. Radii are
mesh geometry, not `fingertip_geometry_radius_m`. Grasp success stays blocked
(ADR-004).

## Still REQUIRED_INPUT (do not invent)

Friction, stiffness, hardware torque, hardware kp/kd, command latency, wrist CoM,
T800 flange SE(3), safety rate limits, pad radius, tactile-or-not on the actual
unit, joint-FF enable on this unit.

Official adapter-flange STEP now exists; T800 wrist CAD does not. `assemble.py`
still exits.

## Tests

See `make test`. Ingest and pad-sphere tests run against the local
`third_party/wuji-description` clone (not committed).
