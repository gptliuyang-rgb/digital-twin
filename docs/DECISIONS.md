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

## ADR-008 — QR scan is IBVS, never open-loop wrist pose

- **Status:** accepted
- **Context:** SONIC 3-point teleop mean wrist error is ~6 cm. QR stickers are 2–8 cm. Open-loop VLA wrist commands cannot register a box.
- **Decision:** Coarse VLA approach to ±10 cm, then `runtime/ibvs.py` (Chaumette 2006 point feature) at 20–30 Hz until `simulate_scan` actually decodes. Camera intrinsics come from `calib_real.yaml` or an explicitly labelled `CameraIntrinsics.synthetic_pinhole()` — never a silent webcam default.
- **Consequences:** L3/real scan success is a decode, not a distance. Synthetic pinhole envelope is geometry+decode, not RTX.

## ADR-009 — No identity T800↔Hand weld for policy eval

- **Status:** accepted
- **Context:** Hand-side mount→wrist offset is in official with-mount MJCF. T800 flange SE(3) is not. An identity weld would inject a constant VLA wrist bias into every ckpt.
- **Decision:** `assets/combined/assemble.py` writes `weld_recipe.yaml` and raises `PolicyEvalBlocked` for policy eval until CAD fills `t800_wrist_to_hand_mount`. Dummy elbow→`LINK_WRIST_END_*` from the T800 URDF is recorded and is **not** the hand flange.
- **Consequences:** Hand-only MuJoCo tracking may run. Combined-robot L2/L3 may not.

## ADR-010 — Soft-body Δm is not a CoM

- **Status:** accepted
- **Context:** Product 0.745 kg vs skeleton 0.6207 kg. Δ = 0.1243 kg is arithmetic, not a hang-test.
- **Decision:** `sim/hand_mass.py` exposes the budget and refuses SONIC load-aware training until `com_in_wrist_frame_m` is measured.
- **Consequences:** Do not attach the delta at the skeleton CoM and call it the physical hand.

## ADR-011 — T800 SONIC is 25-DoF; G1 checkpoints are unloadable

- **Status:** accepted
- **Context:** Official GEAR-SONIC ONNX is Unitree G1 29-DoF. Decoder input is 994 = 64 token + 10×(ω3 + g3 + 3×29). T800 Native SDK URDF is 25 revolute (decoder input 874).
- **Decision:** `wbc/t800_sonic.yaml` is the T800 contract. `refuse_g1_checkpoint()` raises `G1CheckpointIncompatible` on G1 robot name, 29 DoF, or decoder dim 994. Fine-tuning G1 `last.pt` is forbidden.
- **Consequences:** BONES-SEED must be GMR-retargeted onto T800 and PPO trained from scratch. `make sonic-status` prints the blockers (flange SE(3), wrist CoM).

## ADR-012 — vr_3point field order is SONIC's, not command_schema's

- **Status:** accepted
- **Context:** `command_schema_v1.yaml` stores head, left wrist, right wrist (rot6d). SONIC `vr_3point_local_target` is `[left_wrist xyz, right_wrist xyz, head xyz]` and orientations are 3× quaternion wxyz.
- **Decision:** `wbc/teleop.py` is the only remapper. Hands never enter the WBC token.
- **Consequences:** A head-first flatten fed to SONIC would swap the head into the left-wrist slot. Tests lock the order.

## ADR-013 — T800 GMR IK config is exported, not copied from G1, and not a PM01 clone

- **Status:** accepted
- **Context:** SONIC training needs BONES-SEED retargeted onto T800 via GMR. GMR already ships `engineai_pm01` with `LINK_BASE` / `LINK_ELBOW_END_*` / `LINK_TORSO_YAW`. T800 official MJCF uses `LINK_WAIST_YAW`, dummy `LINK_WRIST_END_*`, and `LINK_FOOT_*` (SONIC body set B). G1 29-DoF libraries are unloadable (ADR-011).
- **Options:** (A) copy PM01 JSON and rename; (B) generate from `t800_sonic.yaml` tracked bodies + a labelled PM01 offset table; (C) wait for a measured T-pose.
- **Decision:** (B). `wbc/gmr/body_map.yaml` is the source. Wrists/feet/head must match `t800_sonic.yaml`. `human_scale` is 1.0 (not PM01's 0.85) until a T-pose pass. Quat/pos offsets copied from GMR's PM01 configs are tagged `uncalibrated_copied_from_gmr_pm01`. Head is new vs PM01; its quat is identity until T-pose. DexHand2 is not in the GMR skeleton (`hand_bypass: true`).
- **Consequences:** `make gmr-export` writes `smplx_to_t800.json` and `bvh_lafan1_to_t800.json`. Running GMR on BONES-SEED is still blocked on flange SE(3) and wrist CoM. Do not check a G1 `motion_lib.pkl` into this repo.

## ADR-014 — Privileged L2 may drop a box on a pallet, never a grasp-success number

- **Status:** accepted
- **Context:** L2 needs a physics path that does not wait for pad–cardboard calibration, without violating ADR-004.
- **Decision:** `PrivilegedL2Env` is pallet+box only. Floor friction is an explicit fixture constant and is **not** written into `dexhand2_spec.yaml`. `grasp_success_rate` is always JSON `null`. Combined T800+Hand remains `PolicyEvalBlocked`.
- **Consequences:** Stack-settle diagnostics can be physics-backed. Pick-success dashboards cannot.

