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

## ADR-006 — Default sim model is Hand 2 Beta 2

- **Status:** accepted (2026-09-01 monthly contract sync)
- **Context:** Official docs and `wuji-description` now default to Beta 2. Beta 2 adds five tactile-pad bodies that participate in collision, with measured pad/distal masses. Product curb mass is 0.800 kg (soft + cables, excl. base), not the Beta 1 0.745 kg.
- **Options:** (A) stay on Beta 1 until the physical unit is confirmed; (B) default *simulation* to Beta 2 and keep `hardware_revision` / `hardware_has_tactile` as REQUIRED_INPUT.
- **Decision:** (B). `sim_model_revision: hand2_beta2`. Joint names, limits, actuators, and gen-1 kp/kv are identical, so command_schema_v1 does not change (still 75-D).
- **Consequences:** Ingest must check both revisions. Do not subscribe to fingertip streams until `hardware_has_tactile` is filled. Pad collision being present does **not** authorize a grasp-success number (ADR-004): official pads are convex hulls, μ/k still REQUIRED_INPUT.

## ADR-007 — Record measured Beta 2 sim mass 0.6228 kg

- **Status:** accepted
- **Context:** Changelog 2026.08.17 said total mass matches Beta 1 to floating-point noise after splitting pad bodies. Measured URDF/MJCF inertial sum is 0.6228 kg vs Beta 1 0.6207 kg (+2.1 g).
- **Decision:** Store the measured 0.6228 kg. Do not overwrite it with the changelog prose.
- **Consequences:** Soft+cable delta vs product curb 0.800 kg is 0.1772 kg, still not a measured CoM.
