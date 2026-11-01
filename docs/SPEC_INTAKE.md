# DexHand2 / Wuji Hand 2 parameter intake

This file is the human-facing list of every `REQUIRED_INPUT` in the frozen
contracts. Official 2.2 docs and `wuji-description` v2026.10.26 already answered
topology, curb/sim mass, communication, and joint limits.
**Do not invent the remaining numbers.**

Priority:

- **P0** — blocks a correct digital twin (dynamics, contact, command timing, mount).
- **P1** — blocks contact calibration / sim2real claims.
- **P2** — can wait until L3 vision or real backend bring-up.

## P0

| Field | Why it blocks | How to get it | Worst-case substitute if you cannot get it |
|---|---|---|---|
| `friction_vs_cardboard_static/dynamic` | Grasp success is a friction problem. Official soft body / skin is not locked. | E1 incline test (`hand/calibration/PROTOCOL.md`) | Do **not** publish a success rate. Run the 3×3 μ/stiffness scan and report range only. |
| `normal_stiffness_n_per_m` | Sets MuJoCo `solref`/`solimp` and Isaac material. Official pad collision is still a convex hull. | E2 compression test | Same 3×3 scan. Label results "uncalibrated". |
| `motor_max_torque_nm` (hardware) | Official wrist load limits are unpublished. MJCF `forcerange` is sim-only. | E3 grasp-limit test / ask Wuji FAE | Use MJCF `sim_actuator_forcerange_nm` **only** inside sim, never as a payload rating. |
| `hardware_kp` / `hardware_kd` | MJCF gains are gen-1 carry-over. Firmware factory 5.0/0.1 applies only to new/reset configs. | Step / chirp identification, or FAE | Gain scan 3×3. Report range. Do not copy factory 5.0/0.1 into sim actuators. |
| `command_latency_ms` | Finger ring is 1 kHz; unmodelled delay wrecks grasp timing. | Scope / SDK timestamp vs motion | Ring-buffer sweep 5–40 ms; do not pick one value. |
| `com_in_wrist_frame_m` | Product curb 0.800 kg vs URDF 0.6878 kg (includes 0.069 kg mount). Soft-body/cable CoM is unknown. | Hang the physical hand from the wrist flange, measure. | Use URDF CoM and **do not** hang the 0.1122 kg delta at a guessed offset. Do not train SONIC until measured. |
| `t800_wrist_to_hand_mount.{pos,quat}` | SONIC tracks `LINK_WRIST_END_*`. A constant SE(3) bias here is a constant VLA error. Official 2.2 adapter-flange STEP exists; T800 flange CAD does not. | Assemble official adapter-flange STEP against T800 wrist flange CAD | Identity transform **only** for kinematic bring-up; forbidden for policy eval. |
| `controller_safety.max_delta_q_rad` / `velocity_limit_rad_s` | Safety filter is shared sim/real. | From firmware / SDK effort+rate limits, then 50% margin | Soft-clip to `joint_limits_rad` only; log that rate limits are unset. |
| `fingertip_geometry_radius_m` | Sphere-pad collision is the derived-asset point. | Fit `*_tip.STL` or caliper the pad | Fit the official `*_tip.STL` (scripted). That is geometry of the mesh, not the live soft pad. |
| `hardware_has_tactile` | Overview is a 2.2 **non-tactile prototype**; sim 2.2 still has `tip_sensor_frame` links. | Ask FAE / look at the unit | Leave REQUIRED. Do not fill from the overview title. |

P0 count in the table: 10 line-items (friction is one physical experiment producing two coefficients).

## P1

| Field | Why | How |
|---|---|---|
| `fingertip_material` | Documents which pad was on the unit during E1/E2 | Read the batch card |
| `soft_body_batch_id` / `skin_batch_id` / `skin_present` / `firmware_version` | 2.2 pads change; calibration is batch-bound. Operation requires skin. | Photograph + SDK `info.firmware_version` |
| `joint_feedforward_compensation.enabled_on_this_unit` | Default-on only for units shipping from 2026-10 | Ask FAE / read the unit |
| `motor_max_velocity_rad_s` | Rate limit for safety filter | FAE / datasheet. URDF `velocity=` is sim-only. |
| `gear_ratio` | Direct-drive ⇒ likely 1, but do not assume load curves | FAE. Do not fill 1.0 yourself. |
| `nid` vs `sdk_index` | State frames are variable-length and keyed by `nid` | Dump one `joint_states` frame on hardware |
| `tcp_frame` / `gun_tcp_frame` offsets | Grasp and scan poses | CAD of scanner + grasp definition |

## P2

| Field | Why | How |
|---|---|---|
| `image_latency_ms` | L3 VLA closed loop | Instrument camera pipeline |
| Camera `calib_real.yaml` | Visual sim2real | Kalibr / manufacturer |
| `human_to_hand_scale` | Retargeting | Official `wuji-retargeting` default, then measure |
| T800 vs T800 Pro | T800 URDF has **no wrist pitch/roll** (25 revolute DoF). | Team decision. Record in `docs/DECISIONS.md`. |

## Already answered (do not re-ask)

| Item | Value | Source |
|---|---|---|
| Default description revision | Hand 2.2 (`hand2/hand2_2`) | wuji-description v2026.10.12+ |
| Fingers / active DoF / total DoF | 5 / 20 / 20 | Product overview |
| Coupling | none, C = I | Product overview + MJCF |
| Product curb / bare / base | 0.800 / 0.650 / 0.070 kg | Product overview 2.2 |
| Sim no-cover / with-cover | 0.6878 / 0.7088 kg (cover provisional) | URDF + release notes 2026-10-26 |
| Mount mass + root link | 0.069 kg, `{l,r}_mount` | model-integration |
| mount→wrist | `[0.00300045189842, 0, -0.0284982808512]` m | 2.2 MJCF |
| Control | MIT hybrid, 1000 Hz × 20, FOC, 11–13 V, RJ45 100BASE-TX | overview + electrical |
| Backdrivable | true | Product overview |
| Joint limits | MJCF `range=` (matches product deg tables) | `hand2_2/body/mjcf/right.xml` |
| Joint order | THJ/FFJ/MFJ/RFJ/LFJ 0–3 ≡ sdk 0–19 | MJCF actuators + SDK |
| Factory IP | L 192.168.1.110 / R 192.168.1.111 :50001 | electrical-network-integration |
| Data port rules | 7400–65535, 7447 ok, 50000 forbidden | release notes 2026-10-26 |
| Firmware factory MIT | kp=5.0, kd=0.1 on new/reset configs | release notes 2026-10-12 |
| Documented firmware | v2.13.1 (release notes 2026-10-26) | version-compat page still says v2.5.1 — do not mix |
| Skin during operation | required | overview + hardware-integration |
| Sim kp/kv / forcerange | filled, **uncalibrated gen-1 carry-over** | Official 2.2 MJCF |
| SN revision digits 5–7 | A00=2.1, A01=2.2 | version-compatibility |
| Official adapter flange STEP | `hand2/hand2_2/attachment/*adapter-flange*` | hardware-integration |

## Official gaps this repo must close (not missing parameters — missing *models*)

1. Fingertip pad collision is still the convex hull of `*_tip_sensor_frame` — derived spheres replace it.
2. Drive gains in MJCF are gen-1; factory 5.0/0.1 is firmware-only.
3. Soft body / skin unlocked; cover mass provisional; no soft deformation.
4. Wrist load / bending / impact limits are unpublished.
5. External interface form will change after 2.2 (no longer XT30 + RJ45).
6. T800 wrist flange ↔ Hand 2 mount SE(3) is still a mechanical design task.
