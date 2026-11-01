# Architecture decision records

## ADR-001 — Wrist frame for SONIC / VLA

- **Status:** accepted for T800 bring-up, revisit if the robot is T800 Pro
- **Context:** 3-point teleop (head + two wrists) is the SONIC/GR00T contract. The tracked body must be identical in training and deployment.
- **Options:** (A) T800 `LINK_WRIST_END_L/R` from Native SDK `end_effector_frames`; (B) T800 Pro `LINK_WRIST_ROLL_L/R`; (C) Wuji `{l,r}_wrist` after the flange.
- **Decision:** (A) for the T800 URDF in this repo. That URDF has **no wrist pitch/roll**; `LINK_WRIST_END_*` is a 1 g dummy sphere welded to elbow yaw.
- **Consequences:** VLA wrist poses are the elbow-yaw frame, not a 7-DoF wrist. Box handling workspace is reduced. Prefer T800 Pro if the real robot has wrist joints. Record the physical robot SKU before SONIC retarget.

## ADR-002 — Do not regenerate official Hand 2 URDF/MJCF/USD

- **Status:** accepted (patch v2)
- **Context:** Wuji publishes three consistent formats.
- **Decision:** ingest + validate + derive (pad spheres). Never hand-maintain a second skeleton.
- **Consequences:** `scripts/check_upstream_drift.py` must run when `wuji-description` updates.

## ADR-003 — Coupling is identity

- **Status:** accepted for Hand 2.1 / 2.2
- **Context:** Official kinematics are serial direct-drive, no nonlinear coupling.
- **Decision:** `hand/coupling.py` is `C = I`, interface kept.
- **Consequences:** Gen-1 tendon/mimic hands need a different Coupling subclass.

## ADR-004 — Grasp success is forbidden until E1/E2

- **Status:** accepted
- **Context:** N3 in the agent prompt. Official contact is convex-hull distal, pad not colliding, soft body unlocked.
- **Decision:** `eval/l2_mujoco_closedloop.py` reports `blocked_uncalibrated` while friction/stiffness are `REQUIRED_INPUT`.
- **Consequences:** No dashboard number that looks like a pick-success rate.

## ADR-005 — SDK index order

- **Status:** accepted with hardware confirmation pending
- **Context:** SDK `JointHandle.index` is 0..19, labels `{finger}_S{1..4}`, fingertip API 0=thumb…4=pinky.
- **Decision:** Map S1=J0 … S4=J3 in TH/FF/MF/RF/LF actuator order. `nid` assumed equal to index until a live dump.
- **Consequences:** One `joint_states` frame on hardware must be checked against `joint_name_map.yaml`.

## ADR-010 — Simulation default is Hand 2.2

- **Status:** accepted (2026-11-01)
- **Context:** wuji-description v2026.10.12 deleted `hand2_beta1` / `hand2_beta2`. Official docs default to Wuji Hand 2.2 (Beta) 非触觉版样机. SN digits 5–7 `A01` = 2.2, `A00` = 2.1.
- **Options:** (A) stay on deleted Beta paths and vendor a snapshot; (B) retarget contracts to `hand2/hand2_2` and keep `hand2/hand2_1` as a joint-identity drift check; (C) support both as first-class embodiments.
- **Decision:** (B). `model_revision: hand2_2`. 2.1 is ingested only to assert identical 20-DoF names/limits and the same 0.6878 kg no-cover mass.
- **Consequences:** `hardware_has_tactile` stays REQUIRED_INPUT — the overview title is not a unit inspection. Cover mass 0.7088 kg is provisional. Do not hang the 0.1122 kg curb-minus-URDF delta until CoM is measured.

## ADR-011 — Firmware factory MIT is not hardware_kp

- **Status:** accepted
- **Context:** 2026-10-12 release notes set default MIT `kp=5.0`, `kd=0.1` for **new or reset** joint configs. Upgrades keep saved parameters. Official MJCF still ships gen-1 kp/kv (~0.2–0.7 / ~0.01–0.03).
- **Decision:** Record factory defaults under `firmware_factory_mit`. `hardware_kp` / `hardware_kd` remain REQUIRED_INPUT until sys-id on the bench unit. Do not write factory 5.0/0.1 into the MJCF `<position>` actuators.
- **Consequences:** Sim and real compliance will differ until the identified gains are filled. Gain-scan remains the uncalibrated fallback.

