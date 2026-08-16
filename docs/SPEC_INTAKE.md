# DexHand2 / Wuji Hand 2 parameter intake

This file is the human-facing list of every `REQUIRED_INPUT` in the frozen
contracts. Official product docs and `wuji-description` already answered
topology, mass (product vs skeleton), communication, and joint limits.
**Do not invent the remaining numbers.**

Priority:

- **P0** — blocks a correct digital twin (dynamics, contact, command timing, mount).
- **P1** — blocks contact calibration / sim2real claims.
- **P2** — can wait until L3 vision or real backend bring-up.

## P0

| Field | Why it blocks | How to get it | Worst-case substitute if you cannot get it |
|---|---|---|---|
| `friction_vs_cardboard_static/dynamic` | Grasp success is a friction problem. Official soft body / skin is not locked. | E1 incline test (`hand/calibration/PROTOCOL.md`) | Do **not** publish a success rate. Run the 3×3 μ/stiffness scan and report range only. |
| `normal_stiffness_n_per_m` | Sets MuJoCo `solref`/`solimp` and Isaac material. Convex-hull distal contact is already wrong; wrong stiffness makes it worse. | E2 compression test | Same 3×3 scan. Label results "uncalibrated". |
| `motor_max_torque_nm` (hardware) | Official usage constraints: measured load is **not** a committed spec. MJCF `forcerange` is sim-only. | E3 grasp-limit test / ask Wuji FAE | Use MJCF `sim_actuator_forcerange_nm` **only** inside sim, never as a payload rating. |
| `hardware_kp` / `hardware_kd` | Official USD/MJCF gains are gen-1 carry-over, not Hand 2 sys-id. | Step / chirp identification, or FAE | Gain scan 3×3 (`eval/configs/gain_scan.yaml`). Report range. |
| `command_latency_ms` | Finger ring is 1 kHz; unmodelled delay wrecks grasp timing. | Scope / SDK timestamp vs motion | Ring-buffer sweep 5–40 ms; do not pick one value. |
| `com_in_wrist_frame_m` | Product mass 0.745 kg vs URDF skeleton 0.6207 kg. Soft-body CoM is unknown. WBC will be wrong if you hang 1.49 kg of hands on the wrists with a skeleton CoM. | Hang the physical hand from the wrist flange, measure. | Use URDF skeleton CoM and **add** a 0.1243 kg point mass at an unknown offset — still mark REQUIRED. Do not train SONIC until measured. |
| `t800_wrist_to_hand_mount.{pos,quat}` | SONIC tracks `LINK_WRIST_END_*` (T800) or `LINK_WRIST_ROLL_*` (T800 Pro). A constant SE(3) bias here is a constant VLA error. | Assemble Hand 2 mount STEP against T800 wrist flange CAD | Identity transform **only** for kinematic bring-up; forbidden for policy eval. |
| `controller_safety.max_delta_q_rad` / `velocity_limit_rad_s` | Safety filter is shared sim/real. Wrong limits either clip every command or pass through dangerous jumps. | From firmware / SDK effort+rate limits, then 50% margin | Soft-clip to `joint_limits_rad` only; log that rate limits are unset. |
| `fingertip_geometry_radius_m` | Sphere-pad collision is the whole point of the derived asset. | Fit `*_tip.STL` or caliper the pad | Fit the official `*_tip.STL` (scripted). That is geometry of the mesh, not the live soft pad. |
| `hardware_has_tactile` | Beta 1 has no tactile; Beta 2 does. Wrong assumption changes the observation spec. | Ask FAE / look at the unit | Assume Beta 1 (`has_tactile: false`) until contradicted. |

P0 count in the table: 10 line-items (friction is one physical experiment producing two coefficients).

## P1

| Field | Why | How |
|---|---|---|
| `fingertip_material` | Documents which pad was on the unit during E1/E2 | Read the batch card |
| `soft_body_batch_id` / `skin_batch_id` / `skin_present` / `firmware_version` | Beta pads change; calibration is batch-bound | Photograph + SDK `info.firmware_version` |
| `motor_max_velocity_rad_s` | Rate limit for safety filter | FAE / datasheet |
| `gear_ratio` | Direct-drive ⇒ likely 1, but usage constraints forbid assuming load curves | FAE. Do not fill 1.0 yourself. |
| `nid` vs `sdk_index` | State frames are variable-length and keyed by `nid` | Dump one `joint_states` frame on hardware |
| `tcp_frame` / `gun_tcp_frame` offsets | Grasp and scan poses | CAD of scanner + grasp definition |

## P2

| Field | Why | How |
|---|---|---|
| `image_latency_ms` | L3 VLA closed loop | Instrument camera pipeline |
| Camera `calib_real.yaml` | Visual sim2real | Kalibr / manufacturer |
| `human_to_hand_scale` | Retargeting | Official `wuji-retargeting` default, then measure |
| T800 vs T800 Pro | T800 URDF has **no wrist pitch/roll** (25 revolute DoF). T800 Pro has wrist + built-in 7DoF hands we will replace. | Team decision. Record in `docs/DECISIONS.md`. |

## Already answered (do not re-ask)

| Item | Value | Source |
|---|---|---|
| Fingers / active DoF / total DoF | 5 / 20 / 20 | Product overview |
| Coupling | none, C = I | Product overview + MJCF (no mimic/tendon) |
| Product mass | 0.745 ± 0.010 kg (soft body, no cables) | Product overview |
| Skeleton mass | 0.6207 kg | URDF inertial sum |
| Mount mass | 0.069 kg | with-mount URDF |
| Control | MIT hybrid, 1000 Hz × 20, FOC, 12 V, RJ45 100BASE-TX | Product overview |
| Backdrivable | true | Product overview |
| Joint limits | MJCF `range=` (matches product thumb/little deg tables) | `hand2_beta1/body/mjcf/right.xml` |
| Joint order | THJ/FFJ/MFJ/RFJ/LFJ 0–3 ≡ sdk 0–19 | MJCF actuators + SDK `JointHandle.index` |
| Tactile on Beta 1 | not provided | Usage constraints |
| Factory IP | L 192.168.1.110 / R 192.168.1.111 :50001 | SDK reference |
| Sim kp/kv / forcerange | filled, **uncalibrated gen-1 carry-over** | Official MJCF |
| Fingertip sites | thumb z=-0.02978 m, others z=-0.02475 m in distal frame | Official MJCF |
| Hand-side mount→wrist | pos `[0.003, 0.00025016, -0.0285]` m | with-mount MJCF |
| T800 wrist links | `LINK_WRIST_END_L/R` | engineai `config/t800/model/default.yaml` |

## Official gaps this repo must close (not missing parameters — missing *models*)

1. Fingertip pad meshes (`*_tip.STL`) ship but are **not** collision geometry.
2. Drive gains are gen-1.
3. Soft body / skin unlocked; sim is skeleton only.
4. Collision is per-link convex hull.
5. External interface form will change after Beta 1.
6. T800 wrist flange ↔ Hand 2 mount SE(3) is a mechanical design task. Hand-side mount STEP **is** now in `hand2/hand2_beta1/attachment/` (RESOURCES.md v1 said it was missing; upstream caught up).
