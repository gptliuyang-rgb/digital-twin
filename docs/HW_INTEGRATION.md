# Hardware integration — T800 + Wuji Hand 2

## Mechanical

| Side | Asset | Status |
|---|---|---|
| Hand mount STEP | `wuji-description/hand2/hand2_beta1/attachment/*-mount_beta1_step.STEP` | present (upstream caught up vs older notes) |
| Hand with-mount MJCF offset | `r_wrist` under `r_mount` at `[0.003, 0.00025016, -0.0285]` m | filled |
| T800 wrist flange CAD | Native SDK URDF dummy sphere on `LINK_WRIST_END_*` | **missing as a real flange** |
| T800 ↔ Hand SE(3) | `assets/dexhand2/meta/mount_transform.yaml` | REQUIRED_INPUT (identity weld is sim-only, ADR-006/008) |

Alignment in the robot base (`LINK_BASE`, pinned ≈ world in this sim):

```
T_base_palm = T_base_wrist_end * T_flange * T_mount_wrist
```

- `T_base_wrist_end` — T800 FK of `LINK_WRIST_END_*` (dummy elbow-yaw frame, ADR-001)
- `T_flange` — CAD `t800_wrist_to_hand_mount` when filled; otherwise identity kinematic bring-up
- `T_mount_wrist` — official with-mount MJCF `[0.003, 0.00025, -0.0285]` m

SONIC/VLA wrist poses are in `robot_heading_frame` (pelvis yaw, Z up), not raw world. `make frame-report` prints the chain. Do not guess a palm-forward rotation to make the carton grasp look nicer.
| Impact-separation adapter | Official gen-1 `Impact-Resistant-Adapter.step` is a design reference | design task |

Recommend an impact-separation variant for stacking collisions. Nylon/PETG print is acceptable for a prototype flange (same idea as the Unitree G1 STL adapter).

## Electrical / network

Wuji Hand 2 Beta 1:

- Power: **12 V DC only**, ≥ 200 W per hand (official adapter 12 V 20 A)
- Comms: RJ45 100BASE-TX, static IP, no DHCP
- Factory IP: left `192.168.1.110`, right `192.168.1.111`, port `50001`
- External form **will change** after Beta 1 (no longer XT30 + RJ45 two-cable). Keep the backend swappable.

T800:

- Confirm an internal Ethernet port and a 12 V rail that can feed **two** 20 A hands, or add a switch + DC-DC.
- Two hands ⇒ two IPs on one subnet with the control PC / Orin.

## Software

- Real backend: `hand/backends/real_backend.py` (Wuji SDK `joint_command` / `joint_states` / `mit_params`)
- Control law: MIT `τ = kp(qd−q)+kd(dqd−dq)+τ_ff` in `hand/controller.py`
- Do not use joint current as contact force (official usage constraints)

## Mass that WBC must see

| Item | kg |
|---|---|
| Skeleton URDF (no-mount) | 0.6207 |
| Official mount | 0.069 |
| Product (soft body, no cables) | 0.745 ± 0.010 |
| Two hands, product | ~1.49 |
| Soft-body delta not in sim | 0.1243 each |

Until `com_in_wrist_frame_m` is measured, do not train a load-aware SONIC tracker and claim it matches the robot.
