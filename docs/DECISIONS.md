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

- **Status:** accepted for Hand 2 Beta 1/2
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

## ADR-006 — Palmar pad spheres from distal STL, not raw `*_tip.STL`

- **Status:** accepted
- **Context:** Official docs say `*_tip.STL` is the fingertip pad mesh but is not used as collision. Vertex-for-vertex, those files are the **distal bone** mesh in a CAD frame (Z flipped about a plane). Injecting raw tip coordinates into the MJCF distal body puts primitives on the wrong side of the finger.
- **Decision:** Fit 3 spheres per finger from the palmar half of `{l,r}_*_distal.STL` in the MJCF distal frame. Disable distal hull collision on the derived MJCF. Keep `fingertip_geometry_radius_m` as `REQUIRED_INPUT` (live soft pad).
- **Consequences:** Derived XML is a skeleton-pulp approximation. E1/E2 still required before any grasp-success number.

## ADR-007 — T800 vs T800 Pro body DoF

- **Status:** accepted
- **Context:** Product copy says 29 DoF excluding hands. Native SDK `serial_t800.urdf` has **25** revolute and dummy `LINK_WRIST_END_*`. `serial_t800pro.urdf` has **43** revolute, of which 14 are a built-in 7-DoF hand we replace, leaving **29** body DoF including wrist pitch/roll.
- **Decision:** Keep ADR-001 (T800 dummy wrist) unless the physical SKU is T800 Pro. SONIC body link on Pro is `LINK_WRIST_ROLL_*`.
- **Consequences:** Box-handling workspace on non-Pro T800 is elbow-yaw only. Record the SKU before retargeting BONES-SEED.
