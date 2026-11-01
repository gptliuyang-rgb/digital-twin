# Hardware integration — T800 + Wuji Hand 2.2

## Mechanical

| Side | Asset | Status |
|---|---|---|
| Wrist adapter flange STEP | `wuji-description/hand2/hand2_2/attachment/wuji-hand2-description-adapter-flange_hand2_2_step.STEP` | present (v2026.10.26) |
| Palm mount STEP / A3 PDF | `hand2/hand2_2/attachment/wuji-hand2-description-{left,right}-mount_hand2_2_*` | present; 2.1 and 2.2 palm interface identical |
| Hand mount→wrist | `r_wrist` under `r_mount` at `[0.00300045189842, 0, -0.0284982808512]` m | filled from 2.2 MJCF |
| Model root | `{l,r}_mount` (69 g). No `_with_mount` variant. | official model-integration |
| T800 wrist flange CAD | Native SDK URDF dummy sphere on `LINK_WRIST_END_*` | **missing as a real flange** |
| T800 ↔ Hand SE(3) | `assets/dexhand2/meta/mount_transform.yaml` | REQUIRED_INPUT |
| Impact-separation adapter | Official gen-1 `Impact-Resistant-Adapter.step` is a design reference | design task |

Palm interface (official, 2.1 = 2.2): M3×0.5 + two M4×0.5 fine-thread holes, 28.00 / 5.15 mm datums. Wrist load, moment, and impact limits are **not published** — do not invent a payload rating.

Recommend an impact-separation variant for stacking collisions.

## Electrical / network

Wuji Hand 2.2:

- Power: **11–13 V DC** (nominal 12 V), ≥ 200 W per hand (official adapter 12 V 20 A). 2.1 is 12 V only.
- Comms: RJ45 100BASE-TX, static IP, no DHCP
- Factory IP: left `192.168.1.110`, right `192.168.1.111`, port `50001`
- Data port: 7400–65535, 7447 compatible, **50000 forbidden** (discovery)
- External form **will change** (no longer XT30 + RJ45 two-cable). Keep the backend swappable.
- On comms loss while enabled: motors stop; must `enable()` again after recover (firmware ≥ v2.1.0)

T800:

- Confirm an internal Ethernet port and a 12 V rail that can feed **two** 20 A hands, or add a switch + DC-DC.
- Two hands ⇒ two IPs on one subnet with the control PC / Orin. Same-handedness multi-device requires `sn`.

## Software

- Real backend: `hand/backends/real_backend.py` (Wuji SDK `joint_command` / `joint_states` / `mit_params`)
- Factory endpoint helper: `interface.schema.factory_tcp_endpoint`
- Control law: MIT `τ = kp(qd−q)+kd(dqd−dq)+τ_ff` in `hand/controller.py`
- Do not use joint current as contact force
- ROS 2: `wuji_hand2_2_description`, `model_revision=2`. Driver does **not** auto-enable joints and has no command-timeout E-stop.

## Mass that WBC must see

| Item | kg |
|---|---|
| Official no-cover URDF (includes mount) | 0.6878 |
| Official with-cover URDF (cover provisional) | 0.7088 |
| Official mount | 0.069 |
| Product curb (soft body + cables, no base) | 0.800 |
| Product bare / base | 0.650 / 0.070 |
| Two hands, curb | 1.60 |
| Curb minus URDF (not a measured CoM) | 0.1122 each |

Until `com_in_wrist_frame_m` is measured, do not train a load-aware SONIC tracker and claim it matches the robot.
