# DexHand2 / Wuji Hand 2 parameter intake

This file is the human-facing list of every `REQUIRED_INPUT` in the frozen
contracts. Official product docs (current = **Beta 2**) and `wuji-description`
already answered topology, communication, joint limits, **Beta 2 pad collision**,
and product curb mass. **Do not invent the remaining numbers.**

Priority:

- **P0** — blocks a correct digital twin (dynamics, contact, command timing, mount).
- **P1** — blocks contact calibration / sim2real claims.
- **P2** — can wait until L3 vision or real backend bring-up.

## P0

| Field | Why it blocks | How to get it | Worst-case substitute if you cannot get it |
|---|---|---|---|
| `hardware_revision` | Beta 1 has no pad collision and no tactile; Beta 2 has both. Wrong revision changes the observation spec and the contact model. | Look at the unit / ask FAE. Official current docs are Beta 2. | Keep sim on `hand2_beta2` (already the default) and do not claim the physical unit matches. |
| `friction_vs_cardboard_static/dynamic` | Grasp success is a friction problem. Official soft body / skin is still unlocked. | E1 incline test (`hand/calibration/PROTOCOL.md`) | Do **not** publish a success rate. Run the 3×3 μ/stiffness scan and report range only. |
| `normal_stiffness_n_per_m` | Sets MuJoCo `solref`/`solimp` and Isaac material. Official Beta 2 pads collide as **convex hulls of the pad mesh**, not as a measured pad. | E2 compression test | Same 3×3 scan. Label results "uncalibrated". |
| `motor_max_torque_nm` (hardware) | Usage constraints: measured load is **not** a committed spec. MJCF `forcerange` is sim-only. | E3 grasp-limit test / ask Wuji FAE | Use MJCF `sim_actuator_forcerange_nm` **only** inside sim, never as a payload rating. |
| `hardware_kp` / `hardware_kd` | Official USD/MJCF gains are gen-1 carry-over, not Hand 2 sys-id. Unchanged on Beta 2. | Step / chirp identification, or FAE | Gain scan 3×3. Report range. |
| `command_latency_ms` | Finger ring is 1 kHz; unmodelled delay wrecks grasp timing. | Scope / SDK timestamp vs motion | Ring-buffer sweep 5–40 ms; do not pick one value. |
| `com_in_wrist_frame_m` | Product curb mass is now **0.800 kg** (soft + cables, excl. base) vs sim **0.6228 kg**. Soft-body CoM is unknown. | Hang the physical hand from the wrist flange, measure. | Use URDF sim CoM and **add** the 0.1772 kg delta at an unknown offset — still mark REQUIRED. Do not train SONIC until measured. |
| `t800_wrist_to_hand_mount.{pos,quat}` | SONIC tracks `LINK_WRIST_END_*`. Hand-side mount STEP + A3 drawing exist; T800 flange CAD does not. | Assemble Hand 2 mount STEP against T800 wrist flange CAD | Identity transform **only** for kinematic bring-up; forbidden for policy eval. |
| `controller_safety.max_delta_q_rad` / `velocity_limit_rad_s` | Safety filter is shared sim/real. | From firmware / SDK effort+rate limits, then 50% margin | Soft-clip to `joint_limits_rad` only; log that rate limits are unset. |
| `fingertip_geometry_radius_m` | Sphere-pad simplified collision still needs a radius. Official Beta 2 pad mesh is not a sphere. | Fit `*_tip_sensor_frame.STL` or caliper the live pad | Fit the official pad STL (scripted). That is mesh geometry, not the live soft pad. |
| `hardware_has_tactile` | Official Beta 2 **does** ship tactile (40 thumb / 34 other). Your unit may still be Beta 1. | Ask FAE / look at the unit | Do not subscribe to fingertip streams until this is true. |

P0 count in the table: 11 line-items (friction is one physical experiment producing two coefficients).

## P1

| Field | Why | How |
|---|---|---|
| `fingertip_material` | Documents which pad was on the unit during E1/E2 | Read the batch card |
| `soft_body_batch_id` / `skin_batch_id` / `skin_present` / `firmware_version` | Beta pads change; calibration is batch-bound. Skin must be fitted when running. | Photograph + SDK `info.firmware_version` |
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
| Product curb mass (Beta 2) | 0.800 kg (soft + cables, excl. base) | Overview 2026-09-01 |
| Bare hand / base | 0.650 kg / 0.070 kg | Overview 2026-09-01 |
| Sim mass Beta 2 no-mount | 0.6228 kg | URDF/MJCF inertial sum |
| Sim mass Beta 1 no-mount | 0.6207 kg | URDF/MJCF inertial sum |
| Mount mass | 0.069 kg | with-mount − no-mount, both revisions |
| Pad masses (Beta 2) | thumb 6.3 g, others 3.0 g | MJCF + changelog 2026.08.17 |
| Distal masses (Beta 2) | thumb 10.9 g, others 5.0 g | same |
| Pad collision (Beta 2) | **yes** — independent `{side}_{finger}_tip_sensor_frame` bodies, convex hull of pad mesh | Description integration + measured MJCF |
| Pad collision (Beta 1) | **no** — `*_tip.STL` ships but is not collision geometry | Description integration |
| Control | MIT hybrid, 1000 Hz × 20, FOC, 11–13 V (nominal 12 V), RJ45 100BASE-TX | Overview + usage constraints |
| Backdrivable | true | Product overview |
| Joint limits | MJCF `range=` (matches product thumb/little deg tables). **Identical Beta 1/2.** | `hand2_beta2/body/mjcf/right.xml` |
| Joint order | THJ/FFJ/MFJ/RFJ/LFJ 0–3 ≡ sdk 0–19 | MJCF actuators + SDK `JointHandle.index` |
| Tactile (Beta 2 hardware) | 40 pts thumb, 34 others, 100 Hz, 3-axis force + temperature | SDK reference |
| Per-point force scale | fw ≥ v2.4.0 normalized full-scale; older fw newtons. Read `info.format`. | Changelog 2026.08.17 |
| Factory IP | L 192.168.1.110 / R 192.168.1.111 :50001 | SDK reference |
| Sim kp/kv / forcerange | filled, **uncalibrated gen-1 carry-over**, identical Beta 1/2 | Official MJCF |
| Fingertip sites | thumb z=-0.02978 m, others z=-0.02475 m in distal frame | Official MJCF both revisions |
| Hand-side mount→wrist | pos `[0.003, 0.00025016, -0.0285]` m, identical Beta 1/2 | with-mount MJCF |
| Palm mount holes | 1× M3×0.5 + 2× M4×0.5; Beta 1/2 attachment files byte-identical | Changelog 2026.08.17 |
| T800 wrist links | `LINK_WRIST_END_L/R` | engineai `config/t800/model/default.yaml` |
| Skin when running | required; never run without skin | Usage constraints |

## Official gaps this repo must close (not missing parameters — missing *models*)

1. **Beta 1 only:** fingertip pad meshes (`*_tip.STL`) ship but are **not** collision geometry.
2. **Beta 2:** pads **do** collide, but as convex hulls of `*_tip_sensor_frame.STL` — still not a sphere-fit pad and not the live soft body.
3. Drive gains are gen-1 on both revisions.
4. Whole-hand soft body / skin unlocked; sim has skeleton + (Beta 2) pad links only.
5. External interface form will change later in Beta 2 (no longer XT30 + RJ45 two-cable).
6. T800 wrist flange ↔ Hand 2 mount SE(3) is a mechanical design task. Hand-side STEP **is** in `hand2/hand2_beta2/attachment/`.
7. Do not use joint current as contact force (official usage constraints).
