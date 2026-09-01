# Hardware integration — T800 + Wuji Hand 2

## Mechanical

| Side | Asset | Status |
|---|---|---|
| Hand mount STEP (Beta 2) | `wuji-description/hand2/hand2_beta2/attachment/*-mount_beta2_step.STEP` | present; byte-identical to Beta 1 per 2026.08.17 changelog |
| A3 hole drawing | same `attachment/` PDF | 1× M3×0.5 (depth > 5.5 mm) + 2× M4×0.5 (depth > 4 mm) |
| Hand with-mount MJCF offset | `r_wrist` under `r_mount` at `[0.003, 0.00025016, -0.0285]` m | filled, identical Beta 1/2 |
| Official wrist flange | factory-standard on Beta 2, drawings included | Hand-side known |
| T800 wrist flange CAD | Native SDK URDF dummy sphere on `LINK_WRIST_END_*` | **missing as a real flange** |
| T800 ↔ Hand SE(3) | `assets/dexhand2/meta/mount_transform.yaml` | REQUIRED_INPUT |
| Impact-separation adapter | Official gen-1 `Impact-Resistant-Adapter.step` is a design reference | design task |

Recommend an impact-separation variant for stacking collisions. Nylon/PETG print is acceptable for a prototype flange (same idea as the Unitree G1 STL adapter).

## Electrical / network

Wuji Hand 2 **Beta 2**:

- Power: **11–13 V DC** (nominal 12 V), ≥ 200 W per hand (official adapter 12 V 20 A)
- Comms: RJ45 100BASE-TX, static IP, no DHCP
- Factory IP: left `192.168.1.110`, right `192.168.1.111`, port `50001`
- External form **will change later in Beta 2** (no longer XT30 + RJ45 two-cable). Keep the backend swappable.

T800:

- Confirm an internal Ethernet port and a 12 V rail that can feed **two** 20 A hands, or add a switch + DC-DC.
- Two hands ⇒ two IPs on one subnet with the control PC / Orin.

## Software

- Real backend: `hand/backends/real_backend.py` (Wuji SDK `joint_command` / `joint_states` / `mit_params`)
- Tactile contract: `hand/tactile.py` — decode `FingertipSensorInfo.format`, do not hard-code scale
- Control law: MIT `τ = kp(qd−q)+kd(dqd−dq)+τ_ff` in `hand/controller.py`
- Do not use joint current as contact force (official usage constraints)
- Always run with skin fitted

## Mass that WBC must see

| Item | kg |
|---|---|
| Sim no-mount Beta 1 | 0.6207 |
| Sim no-mount Beta 2 (incl. pad links) | 0.6228 |
| Official mount | 0.069 |
| Product curb (Beta 2, soft + cables, excl. base) | 0.800 |
| Bare hand / base (product table) | 0.650 / 0.070 |
| Two hands, product curb | 1.60 |
| Soft+cable delta not in sim | 0.1772 each |

Until `com_in_wrist_frame_m` is measured, do not train a load-aware SONIC tracker and claim it matches the robot.
