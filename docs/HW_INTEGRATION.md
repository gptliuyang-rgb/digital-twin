# Hardware integration — T800 + Wuji Hand 2

## Mechanical

| Side | Asset | Status |
|---|---|---|
| Hand mount STEP + PDF | `wuji-description/hand2/hand2_beta1/attachment/*-mount_beta1_step.STEP` and `*-mount_beta1.pdf` | present (clone @ 06e5f14) |
| Hand with-mount MJCF offset | `r_wrist` under `r_mount` at `[0.003, 0.00025016, -0.0285]` m | filled |
| T800 wrist flange CAD | Native SDK URDF dummy sphere on `LINK_WRIST_END_*` | **missing as a real flange** |
| T800 ↔ Hand SE(3) | `assets/dexhand2/meta/mount_transform.yaml` | REQUIRED_INPUT |
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
- Do not use joint current as external contact force (usage constraints: forward path exists, calibration not converged).
- Wrist flange drawings ship with the product (`腕部官方标配法兰，并随产品提供腕部安装图纸`); description-repo current-revision adapter STEP was previously missing and is now under `hand2/hand2_beta1/attachment/`. T800 mating CAD is still REQUIRED_INPUT.

## Mass that WBC must see

| Item | kg |
|---|---|
| Skeleton URDF (no-mount) | 0.6207 |
| Official mount | 0.069 |
| Product (soft body, no cables) | 0.745 ± 0.010 |
| Two hands, product | ~1.49 |
| Soft-body delta not in sim | 0.1243 each |

Until `com_in_wrist_frame_m` is measured, do not train a load-aware SONIC tracker and claim it matches the robot. `sim/hand_mass.py:require_sonic_mass()` enforces that.

Combined-robot policy eval is refused (`PolicyEvalBlocked`) until `t800_wrist_to_hand_mount` is CAD-measured. `make weld-recipe` writes the recipe anyway.
