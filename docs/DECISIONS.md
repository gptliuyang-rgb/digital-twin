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

## ADR-006 — Kinematic-bringup flange vs CAD flange

- **Status:** accepted for simulation-only digital twin
- **Context:** T800 wrist flange CAD is still `REQUIRED_INPUT`. A combined T800 + DexHand2 model is required to exercise dual-arm box / QR-scan strategies in MuJoCo.
- **Decision:** `kinematic_bringup_identity` welds `{l,r}_mount` onto `LINK_WRIST_END_*` with identity SE(3). Combined MJCF/URDF are generated under `assets/combined/generated/` (gitignored). Official Hand 2 and T800 files are never overwritten. Hand `<position>` actuators are converted to MIT `<motor>` plants in the derived combined model only.
- **Forbidden:** publishing SONIC/VLA/sim2real numbers, or a `grasp_success_rate`, from this weld or from the uncalibrated μ/solref scan. Replace the weld with CAD SE(3) before any policy-eval claim.
- **Consequences:** Box-handling palm orientation is the T800 elbow-yaw dummy frame (ADR-001). The scripted industrial FSM is a digital-twin playback, not a trained policy.

## ADR-007 — Industrial demo is kinematic playback until E1/E2

- **Status:** accepted for simulation-only digital twin
- **Context:** Uncalibrated hand–object contact (ADR-004) cannot support a believable physics grasp; floating prop slabs also read as unfinished set dressing.
- **Decision:** The scripted industrial FSM (`sim/tasks/industrial_pipeline.py`) plays back with `mj_forward` + IK + explicit carton/gun assists. Scene benches are grounded (legs to floor). The scan gun is a **placeholder pistol mesh** (grip/housing/barrel), not vendor CAD.
- **Forbidden:** treating lift/carry/scan visuals as contact-validated sim2real evidence or publishing `grasp_success_rate`.
- **Consequences:** Demo answers “does the twin look like the cell?” not “does the hand pick?”. Replace assists with E1/E2 friction/stiffness and a real scanner mesh when available.

