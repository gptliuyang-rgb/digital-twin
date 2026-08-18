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

## ADR-015 — T800 GMR T-pose overlay is q=0 XML composition, not a live actor T-pose

- **Status:** accepted
- **Context:** ADR-013 copied GMR PM01 quat offsets and left `human_scale=1.0`. A real T-pose needs a human SMPL-X rest pose plus the robot at a defined pose. T800 and PM01 MJCF body tags are translation-only, so at q=0 every tracked body shares the pelvis orientation.
- **Decision:** `wbc/gmr/tpose.py` measures T800 vs PM01 body frames at q=0 (no MuJoCo import). Offsets are composed `R_off_t800 = R_off_pm01 * inv(R_pm01) * R_t800`. Head (absent from GMR PM01 IK) takes the pelvis offset after that composition, matching GMR tienkung/hi. Scale is PM01's official table times measured T800/PM01 link-length ratios. Results live in `tpose_offsets.yaml`; `body_map.yaml` stays the PM01-copy source.
- **Consequences:** `make gmr-tpose` rewrites IK JSON. Running GMR on BONES-SEED is still blocked on flange SE(3) and wrist CoM. Do not call this a calibrated actor T-pose.

## ADR-016 — Privileged Isaac Lab env is the same pallet+box recipe

- **Status:** accepted
- **Context:** L3 needs a privileged physics path that does not wait for Isaac Sim in CI and does not violate ADR-004.
- **Decision:** `PrivilegedIsaacCfg` is pallet+box, privileged state only, no RTX cameras. `grasp_success_rate` is always JSON `null`. Constructing `PrivilegedIsaacEnv` without Isaac Lab raises `IsaacLabUnavailable`. Combined T800+Hand remains `PolicyEvalBlocked`.
- **Consequences:** Unit tests never import Isaac Lab. RTX visual closed loop stays `eval/l3_isaac_closedloop.py`.


## ADR-017 — 5-point teleop tracks elbow *pitch* bodies, position-only

- **Status:** accepted
- **Context:** 3-point (head + two wrists) leaves the elbow underconstrained. Industrial boxes collide with the elbow bulge. SONIC §2.4 / agent prompt §4.3 option A is "extend hybrid encoder to 5 points, add dual elbow positions" — a retrain, not a deploy-time concat.
- **Options:** (A) add `left_elbow_pos` / `right_elbow_pos` (6 D) on `LINK_ELBOW_PITCH_*`; (B) full elbow SE(3); (C) track `LINK_ELBOW_YAW_*` (forearm / dummy-wrist parent).
- **Decision:** (A). `command_schema_v1` stays 75-D. `command_schema_v1_5point.yaml` is opt-in (+6 D → 81). Packing order is SONIC's: left_wrist, right_wrist, head, left_elbow, right_elbow. Orn stays 12 (wrists+head quats). Hybrid encoder cmd dim 21 → 27. `refuse_teleop_mode_mismatch()` raises if a 3-point checkpoint sees 5-point commands. Elbow body is `LINK_ELBOW_PITCH_*` (GMR role), not `LINK_ELBOW_YAW_*` (forearm, parent of dummy `LINK_WRIST_END_*`).
- **Consequences:** A T800 SONIC trained on 3-point cannot consume 5-point tokens. Flange SE(3) and wrist CoM still block PPO. Hands still bypass WBC.

## ADR-018 — L1a planner is 10 Hz cubic/SLERP, shared sim/real

- **Status:** accepted
- **Context:** SONIC deploys a 10 Hz kinematic planner producing a 0.8–2.4 s reference. VLA chunks are 5–10 Hz. Zhou 6D is not a vector space.
- **Decision:** `wbc/planner.py` upsamples waypoints at 10 Hz. Positions: cubic Hermite. Rotations: rot6d → SO(3) → SLERP → rot6d. Hands: linear (bypass WBC but share the clock). Enums (loco_mode, hand_mode, trigger): nearest waypoint. Horizon is clamped to [0.8, 2.4] s. No MuJoCo/Isaac import.
- **Consequences:** Cubic Hermite can overshoot a step; tests lock the linear option and SLERP geodesic. This is not SONIC's trained planner — it is the interface the trained planner must match.

## ADR-014 — Privileged L2 may drop a box on a pallet, never a grasp-success number

- **Status:** accepted
- **Context:** L2 needs a physics path that does not wait for pad–cardboard calibration, without violating ADR-004.
- **Decision:** `PrivilegedL2Env` is pallet+box only. Floor friction is an explicit fixture constant and is **not** written into `dexhand2_spec.yaml`. `grasp_success_rate` is always JSON `null`. Combined T800+Hand remains `PolicyEvalBlocked`.
- **Consequences:** Stack-settle diagnostics can be physics-backed. Pick-success dashboards cannot.

## ADR-020 — Case A FK is an explicit call, never a PolicyClient hook

- **Status:** accepted
- **Context:** ADR-019 classifies a 50-D last-dim as Case A (T800 2×5 arm q + 2×20 fingers). `JointToWristAdapter` already FKs one arm. Loading 50-D into `PolicyClient` looks like a joint-order bug. Silently padding head/nav or calling FK inside `require_command_schema_vector` would hide a missing L3 command.
- **Decision:** `CaseAToCommandSchema.convert(..., apply_fk=True, head_nav=HeadNavCommand)` is the only A→B path. `apply_fk` is keyword-only; False raises. `HeadNavCommand` must name its `source`. Neck FK from two head joints with torso at zero is forbidden (those joints are not in the 50-D vector). `PolicyClient`, `DeployPipeline`, π0.5 glue, L0 replay, and 50 Hz upsample still refuse 50-D.
- **Consequences:** `make eval-l1-case-a --apply-fk` is the kinematic gate. L3 must supply pelvis height, nav, loco mode, tool trigger, and head pose. Combined T800+Hand weld remains `PolicyEvalBlocked`.

## ADR-021 — VLA→50 Hz upsample is integer-factor SLERP, after Case A conversion

- **Status:** accepted
- **Context:** GR00T typical is 10 Hz × 16; π0.5 typical is 5 Hz × 50; SONIC instruction stream is 50 Hz. Zhou 6D is not a vector space. Non-integer rate ratios were already refused by `upsample_factor`.
- **Decision:** `vla/adapters/upsample.py` interpolates `command_schema_v1` rows with the same kernel as L1a (`interpolate_command_matrix`): cubic Hermite (or linear) on positions/fingers, SLERP on SO(3), nearest-neighbour on enums. Output length is `(H-1)*factor+1`. Case A chunks raise.
- **Consequences:** Confirm `infer_hz` against the real ckpt after L0 diagnose. GPU latency is still unmeasured (`chunk_clock.yaml` is labelled typicals). L1a planner stays 10 Hz (ADR-018); this module is the 50 Hz command stream after that, or a direct VLA→50 Hz path when the planner is not in the loop.


- **Status:** accepted
- **Context:** Existing VLA ckpts may be dual-arm joints (case A), wrist SE(3) (case B), or velocity/delta (case C). `command_schema_v1` is 75-D case B. Loading the wrong last-dim into `PolicyClient` looks like a joint-order bug in L0.
- **Decision:** `vla/adapters/action_space.py` classifies last-dim against a frozen layout table. Unknown dims raise `ActionSpaceMismatch` (no pad/slice). G1 29-DoF raises `G1CheckpointIncompatible`. `PolicyClient` and L0 replay call this before flatten. π0.5 is pass-through only when D=75 (`pi05_glue.py`); it is not native to SONIC.
- **Consequences:** `make eval-l0-diagnose` can run without weights. Case A still needs `CaseAToCommandSchema.convert(apply_fk=True, head_nav=...)` (ADR-020). Case C is a retrain. SONIC PPO remains blocked on flange SE(3) and wrist CoM. Decoder history packing (`ProprioHistory`) is 874-D for T800 and refuses 29-DoF construction.

## ADR-020 — Case A FK is an explicit call, never a PolicyClient hook

- **Status:** accepted
- **Context:** ADR-019 classifies a 50-D last-dim as Case A (T800 2×5 arm q + 2×20 fingers). `JointToWristAdapter` already FKs one arm. Loading 50-D into `PolicyClient` looks like a joint-order bug. Silently padding head/nav or calling FK inside `require_command_schema_vector` would hide a missing L3 command.
- **Decision:** `CaseAToCommandSchema.convert(..., apply_fk=True, head_nav=HeadNavCommand)` is the only A→B path. `apply_fk` is keyword-only; False raises. `HeadNavCommand` must name its `source`. Neck FK from two head joints with torso at zero is forbidden (those joints are not in the 50-D vector). `PolicyClient`, `DeployPipeline`, π0.5 glue, L0 replay, and 50 Hz upsample still refuse 50-D.
- **Consequences:** `make eval-l1-case-a` with `--apply-fk` is the kinematic gate. L3 must supply pelvis height, nav, loco mode, tool trigger, and head pose. Combined T800+Hand weld remains `PolicyEvalBlocked`.

## ADR-021 — VLA→50 Hz upsample is integer-factor SLERP, after Case A conversion

- **Status:** accepted
- **Context:** GR00T typical is 10 Hz × 16; π0.5 typical is 5 Hz × 50; SONIC instruction stream is 50 Hz. Zhou 6D is not a vector space. Non-integer rate ratios were already refused by `upsample_factor`.
- **Decision:** `vla/adapters/upsample.py` interpolates `command_schema_v1` rows with the same kernel as L1a (`interpolate_command_matrix`): cubic Hermite (or linear) on positions/fingers, SLERP on SO(3), nearest-neighbour on enums. Output length is `(H-1)*factor+1`. Case A chunks raise.
- **Consequences:** Confirm `infer_hz` against the real ckpt after L0 diagnose. GPU latency is still unmeasured (`chunk_clock.yaml` is labelled typicals). L1a planner stays 10 Hz (ADR-018); this module feeds the 50 Hz command stream.

## ADR-022 — T800 SONIC PPO recipe is paper Table S1–S4 with action_dim=25; launch is refused

- **Status:** accepted
- **Context:** SONIC trains in Isaac Lab with appendix Tables S1–S4 (arXiv:2511.07820v3). Paper action dim is 29 (G1). T800 Native SDK is 25 revolute. G1 `last.pt` fine-tune is forbidden (ADR-011). Flange SE(3) and wrist CoM still block a real train job. Official T800 MJCF puts joint `range=` (rad) and `actuatorfrcrange=` (N·m) on the same tag; a greedy `[^>]*range=` latch reads the force range and disables the clip-limit gate.
- **Options:** (A) copy the G1 Isaac Lab env; (B) freeze T800 YAML from S1–S4 with `action_dim=25`, implement reward kernels in numpy, refuse PPO launch while blockers remain, run kinematic identity Sim2Sim; (C) wait for a GPU cluster.
- **Decision:** (B). `wbc/ppo/` is the contract. `refuse_ppo_launch()` wraps `assert_retarget_ready()`. `parse_mjcf_joint_limits` matches `\\brange=`. Sim2Sim MPJPE is URDF FK (identity tracker = 0). The G1 hardware ~6 cm wrist error is **not** a T800 gate. Table S4 floor friction is WBC domain rand, not DexHand2 pad–cardboard (must not enter `dexhand2_spec.yaml`). 5-point PPO is a different `teleop_mode` and a retrain.
- **Consequences:** `make ppo-train` exits non-zero until SPEC_INTAKE P0 CoM/flange are filled **and** Isaac Lab is wired. Kinematic identity Sim2Sim and the clip filter can run now on `t800_kinematics.yaml`.

## ADR-023 — Physics Sim2Sim is pinned-base PD on official T800 MJCF; 5-point is an overlay

- **Status:** accepted
- **Context:** Kinematic Sim2Sim (ADR-022) cannot catch actuator/PD/contact issues. Official `serial_t800.xml` is now cloned locally. 5-point teleop is a retrain (ADR-017) and must not silently change `action_dim`. Privileged Isaac `reset`/`step` were stubs.
- **Options:** (A) wait for a trained SONIC policy and run free-base tracking; (B) pin the floating base, PD-track a synthetic clip with official motors + `pd_stand` gains, keep a kinematics fixture for CI without the SDK; (C) weld DexHand2 and publish grasp-success.
- **Decision:** (B). `T800MujocoEnv` loads official XML via `MjSpec` (delete freejoint to pin; add a floor only for free-base). Fixture inertias are 1 kg / 0.01 kg·m² placeholders and use mild PD — never reported as sim2real. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`. `wbc/ppo/network_5point.yaml` overlays `default_teleop_mode: vr_5point` while `action_dim` remains 25; `refuse_ppo_launch(teleop_mode=vr_5point)` still raises `PpoLaunchBlocked`. `PrivilegedIsaacEnv.reset`/`step` bind `IsaacLabSceneRuntime` (pallet+box USD) when `omni.usd` is running; without Isaac they still raise `IsaacLabUnavailable`.
- **Consequences:** `make eval-l2-physics-sim2sim` is the physics gate. Free-base fall_rate without a trained tracker is not a pass/fail for SONIC. G1 ~6 cm wrist error remains not a T800 gate. Floor friction on the official XML is WBC collision default, not pad–cardboard. Official MJCF `LINK_FOOT_*` sits at the ankle-roll origin; the URDF `J_FIXED_FOOT_*` offsets it by z=−64.53 mm. Physics Sim2Sim reports `foot_urdf_mjcf_delta_z_m` instead of mixing the two frames. Wrists match. Foot frame for GMR/SONIC is locked in ADR-024.

## ADR-024 — GMR/SONIC feet are MJCF `LINK_FOOT_*` (ankle-roll origin)

- **Status:** accepted
- **Context:** Official T800 MJCF `LINK_FOOT_*` is a child of `LINK_ANKLE_ROLL_*` with `pos=0`. Official URDF `J_FIXED_FOOT_*` welds a same-named `LINK_FOOT_*` at `xyz="0 0 -0.06453"`. MJCF collision boxes on that body are centered at z=−0.054 m (`collision_*_fore/hind_foot` in `serial_t800.xml`). ADR-023 left the choice open. Mixing the URDF sole into GMR would bias every foot track by 64.53 mm.
- **Options:** (A) track MJCF `LINK_FOOT_*` (physics/SONIC body); (B) track URDF `LINK_FOOT_*` (sole, 64.53 mm below); (C) invent a mid-point.
- **Decision:** (A). `wbc/t800_sonic.yaml` `foot_frame.decision: mjcf_link_foot_at_ankle_roll`. GMR `body_map.yaml` already names `LINK_FOOT_*`. URDF sole world pose is `xpos_foot + R_foot @ [0,0,-0.06453]` for diagnostics only. Collision-box z is recorded and is **not** the SONIC body.
- **Consequences:** `make eval-l2-freebase-stand` reports both MJCF foot z and URDF sole z. Do not retarget BONES-SEED onto URDF `LINK_FOOT_*`. Do not average the two frames. Isaac Lab PPO still refused until flange SE(3) and wrist CoM.

## ADR-025 — Free-base PD stand is a diagnostic, not a SONIC gate

- **Status:** accepted
- **Context:** Pinned-base physics Sim2Sim cannot catch balance failure. EngineAI `pd_stand/default.yaml` is joint PD to a published stand pose, not a learned tracker. Official `serial_t800.xml` has a `freejoint` and **no** floor plane.
- **Options:** (A) wait for a trained SONIC policy; (B) run free-base PD stand on a floor, report fall/hold honestly, never treat it as tracker success; (C) claim bring-up PD as a balance controller.
- **Decision:** (B). `T800MujocoEnv(pinned_base=False, add_floor=True)` adds the official collision-default floor (WBC floor, not pad–cardboard). `q_des` is EngineAI `desired_joint_position` on official XML and zeros on the kinematics fixture. `local_tracking_success` is always false. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`.
- **Consequences:** Fixture 1 kg bodies in free air (no floor) fall under gravity, which is the proof the freejoint is active. On a floor they may hold on the placeholder foot boxes; that is not a static-stand rating. Official XML may fall or briefly hold; neither outcome is a SONIC pass. `make eval-l2-freebase-stand`. Isaac Sim python is still required for `PrivilegedIsaacEnv.reset/step`; CI reports `isaac_runtime_status()["runtime"]=="unavailable"`.

## ADR-026 — Lean is not a stand; lateral root-push and air-drop are diagnostics

- **Status:** accepted
- **Context:** Official 3 s free-base PD hold on `serial_t800.xml` ended at pelvis z=0.864 m and tilt 0.72 rad (~41°) with `fallen=false` at the 0.40 m / 0.80 rad bands (ADR-025). That lean is easy to misread as a static stand. SONIC Table S4 randomizes a root linear-velocity push of ±0.5 m/s. Official XML has a freejoint and no plane; a floor-on hold cannot prove the freejoint.
- **Options:** (A) raise the fall tilt band so 0.72 rad counts as fallen; (B) add an explicit `upright | leaned | fallen` label, keep the fall bands, add a one-shot +Y 0.5 m/s root `qvel` (Table S4 max |y|, not a sustained force) and a no-floor air-drop; (C) treat bring-up PD recovery after the push as a balance controller.
- **Decision:** (B). `eval/lean_classify.py` labels posture. Lean tilt band is 0.20 rad (diagnostic, not CAD). `T800MujocoEnv.apply_root_linvel` sets freejoint linear velocity. `make eval-l2-freebase-push` runs hold + push + air-drop. `local_tracking_success` stays false. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`. Floor friction remains the official WBC collision default, not pad–cardboard. `make eval-l3-isaac-bind` calls `IsaacLabSceneRuntime.reset/step` only when `isaaclab` and `omni.usd` import; otherwise it reports `isaac_bind_unavailable`.
- **Consequences:** A 41° lean is `leaned`, not a SONIC pass. A push that knocks the robot over is also not a SONIC fail. Air-drop `freejoint_moved` is the freejoint proof. Isaac bind does not load T800+Hand. Flange SE(3) and wrist CoM still block PPO / GMR-on-BONES-SEED / combined weld.

## ADR-027 — Table S4 planar push sweep is ±X/±Y extrema, still not a SONIC gate

- **Status:** accepted
- **Context:** ADR-026 injected only +Y 0.5 m/s. SONIC Table S4 randomizes root linear velocity independently on x and y in ±0.5 m/s (z is ±0.2 m/s). The official 3 s PD hold leans ~41°, likely sagittal. A single +Y hit can hide an axis that stays leaned or one that falls sooner.
- **Options:** (A) keep the single +Y case; (B) sweep the four planar extrema from `wbc/ppo/domain_rand.yaml` as one-shot `qvel` after the same settle, keep +Y as the `lateral_push` compatibility key; (C) also sweep z and treat any fall as a SONIC fail.
- **Decision:** (B). `wbc/ppo/table_s4.py` is the single source of extrema. Default sweep is `+x -x +y -y`. Z is recorded and excluded (vertical impulse is a different diagnostic). `local_tracking_success` stays false. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`. Floor friction remains the official WBC collision default, not pad–cardboard.
- **Consequences:** `make eval-l2-freebase-push` wall time is ~4× the single-axis case. Fall on any axis is **not** a SONIC fail. PPO launch, GMR-on-BONES-SEED, and the combined weld remain blocked on flange SE(3) and wrist CoM. Linear-z is a separate diagnostic (ADR-030), not mixed into this planar list.

## ADR-028 — Table S4 duration is a sustained force, same impulse as one-shot qvel

- **Status:** accepted
- **Context:** Table S4 (arXiv:2511.07820v3) publishes root linear velocity *and* push duration Δt ∼ [1, 3] s. ADR-027 applied the planar velocity extrema as one-shot freejoint `qvel`. That is an instantaneous Δv, not the paper's duration. The agent prompt's domain-rand table is "根部线速度扰动 ±0.5 m/s，持续 1–3 s". Table S4 does not publish Newtons.
- **Options:** (A) re-clamp root `qvel` every physics step for T (a velocity hold, harsher than one-shot); (B) apply a constant world force F = m v / T on `LINK_BASE` for T, using the loaded MJCF subtree mass, so the linear impulse equals the one-shot Δp = m v; (C) invent a Newton range or treat duration as the interval *between* one-shot pushes.
- **Decision:** (B). `wbc/ppo/table_s4.py` `sustained_force_cases()` is planar extrema × duration extrema (8 cases). Mass is `body_subtreemass[LINK_BASE]` from the compiled model (fixture placeholder or official XML) — not a T800 datasheet guess. Floor friction stays the official WBC collision default, not pad–cardboard. `+y_T1.0s` is the `sustained_force` compatibility key. One-shot sweep is unchanged. `local_tracking_success` stays false. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`. Z remains unswept.
- **Consequences:** A 1 s force is 3× stronger than the 3 s force for the same Table S4 |v|. Bring-up PD falling under either duration is **not** a SONIC fail. Gravity, contact, and joint PD still act, so realized Δv will not equal the free-space identity. PPO / GMR-on-BONES-SEED / combined weld remain blocked on flange SE(3) and wrist CoM. Linear-z duration uses the same F = m v / T formula under ADR-030; the planar 8-case list is unchanged.

## ADR-029 — Table S4 angular-velocity push is one-shot qvel and τ = I ω / T

- **Status:** accepted
- **Context:** Table S4 also randomizes root angular velocity: roll/pitch ±0.52 rad/s, yaw ±0.78 rad/s, with the same duration Δt ∼ [1, 3] s. ADR-027/028 only swept linear velocity. The official 3 s PD hold already leans ~41° in pitch; a roll or yaw impulse can be a different failure mode than another planar shove.
- **Options:** (A) skip angvel until a trained tracker exists; (B) one-shot freejoint `qvel[3:6]` at the six signed extrema, plus a sustained body-frame torque τ = I ω / T for duration extrema, with I taken from the compiled `mj_fullM` angular block at the inject pose; (C) invent Newton-metre ranges or apply world-frame Euler rates.
- **Decision:** (B). Roll/pitch/yaw map to LINK_BASE body X/Y/Z, matching MuJoCo freejoint angular `qvel`. `+yaw` is the `angvel_push` compatibility key; `+yaw_T1.0s` is the `sustained_torque` key. Inertia is pose-dependent and must be read from the loaded MJCF at inject time — not a T800 datasheet guess. Floor friction stays the official WBC collision default, not pad–cardboard. Linear one-shot and F = m v / T sweeps are unchanged. `local_tracking_success` stays false. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`. Linear z remains unswept.
- **Consequences:** `make eval-l2-freebase-push` adds 6 one-shot angvel cases and 12 sustained-torque cases. Falling under bring-up PD is **not** a SONIC fail. Realized Δω will not equal the free-space identity. PPO / GMR-on-BONES-SEED / combined weld remain blocked on flange SE(3) and wrist CoM. Linear-z is ADR-030, not this record.

## ADR-030 — Table S4 linear-z is a separate vertical diagnostic

- **Status:** accepted
- **Context:** Table S4 root_push lin_vel z is ±0.2 m/s, not the planar ±0.5 m/s. ADR-027/028 recorded z and excluded it from the planar list because a vertical impulse is a different diagnostic (into the floor vs a shove). ADR-029 left linear z unswept. The agent prompt's domain-rand table also lists a root linear-velocity disturbance; z is part of the paper table and must be exercised, but not by folding it into `push_sweep`.
- **Options:** (A) keep z recorded-only; (B) sweep ±Z 0.2 m/s as one-shot world `qvel[0:3]` plus F = m v / T for duration extrema 1 s and 3 s, with `+z` / `+z_T1.0s` as compatibility keys, leaving the planar 4+8 and angvel 6+12 lists unchanged; (C) mix z into `SWEEP_AXES` so `push_sweep` becomes 6 cases.
- **Decision:** (B). `VERTICAL_AXES = ("z",)`. `vertical_linvel_extrema_mps()` / `vertical_force_cases()` are the sources. Mass is still compiled MJCF `body_subtreemass[LINK_BASE]`. Floor friction stays the official WBC collision default, not pad–cardboard. Planar `push_sweep` stays 4 cases; `force_sweep` stays 8. `local_tracking_success` stays false. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`.
- **Consequences:** `make eval-l2-freebase-push` adds 2 one-shot z cases and 4 sustained vertical-force cases. +Z is a lift; −Z drives into the floor. Either outcome under bring-up PD is **not** a SONIC fail. PPO / GMR-on-BONES-SEED / combined weld remain blocked on flange SE(3) and wrist CoM.

## ADR-031 — Table S4 static friction is WBC floor+foot sliding, not pad–cardboard

- **Status:** accepted
- **Context:** Table S4 `physical.static_friction` is [0.3, 1.6] and `dynamic_friction` is [0.3, 1.2] (arXiv:2511.07820v3). ADR-027–030 swept root-push velocity/force/torque while leaving the eval floor at the official collision default `(1.0, 0.005, 0.0001)`. The 41° PD lean may be friction-sensitive. MuJoCo has a single sliding coefficient (`geom_friction[0]`) and contact friction is the **element-wise max** of the two geoms — retuning only the plane would be a no-op against 1.0 feet. PhysX static vs dynamic is not a MuJoCo feature. These numbers must not enter `dexhand2_spec.yaml` (ADR-004).
- **Options:** (A) keep floor friction recorded-only; (B) sweep static-friction extrema as MuJoCo sliding on the eval floor *and* `LINK_FOOT_*` collision geoms during the 3 s PD hold, leave spin/roll official, leave dynamic friction and restitution recorded-only, keep `mu_s_0.3` as the `friction_hold` compatibility key, do not mix into `push_sweep`; (C) copy 0.3/1.6 into DexHand2 pad–cardboard or invent a solref mapping for restitution.
- **Decision:** (B). `static_friction_extrema()` is the source. `T800MujocoEnv.set_wbc_slide_friction` refuses a missing plane and `mu_slide <= 0`. Default `hold` stays the official collision default. `local_tracking_success` stays false. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`.
- **Consequences:** `make eval-l2-freebase-push` adds 2 friction-hold cases. A change in lean vs fall across μ is a diagnostic, **not** a SONIC fail and **not** an E1 pad–cardboard result. Dynamic friction, restitution, and default joint offset stay unswept. Base CoM offset is ADR-032. PPO / GMR-on-BONES-SEED / combined weld remain blocked on flange SE(3) and wrist CoM.

## ADR-032 — Table S4 base CoM offset is LINK_BASE ipos, not wrist CoM

- **Status:** accepted
- **Context:** Table S4 `physical.base_com_offset_m` is x ±0.075 m and y/z ±0.1 m (arXiv:2511.07820v3). ADR-031 left this recorded-only. Isaac Lab randomizes the *root* rigid-body CoM. MuJoCo's matching channel is `body_ipos[LINK_BASE]` (that body's inertial frame in the body frame). Redistributing every link's mass, or writing these numbers into DexHand2 `com_in_wrist_frame_m`, would invent a mapping. Official compiled ipos is not zero; the paper range is an *offset*. Restitution still has no non-invented `solref` map. `default_joint_pos_offset_rad` is a reset-qpos term, not this diagnostic.
- **Options:** (A) keep CoM offset recorded-only; (B) sweep the six axis-aligned signed extrema as additive `body_ipos[LINK_BASE]` during the 3 s PD hold, keep `com_+x` as the `com_offset` compatibility key, do not mix into `push_sweep`; (C) apply the offset to every body, or fill wrist CoM with Table S4 numbers.
- **Decision:** (B). `base_com_offset_extrema()` is the source. `T800MujocoEnv.set_base_com_offset` adds to the compiled ipos and reports own-body mass vs subtree mass so a light pelvis is visible. `restore_base_ipos` returns to the compiled origin. Default `hold` stays the compiled ipos. `local_tracking_success` stays false. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`.
- **Consequences:** `make eval-l2-freebase-push` adds 6 CoM-hold cases. A change in lean vs fall across CoM offset is a diagnostic, **not** a SONIC fail and **not** a wrist hang-test. Restitution stays unswept. Joint-pos offset is ADR-033. PPO / GMR-on-BONES-SEED / combined weld remain blocked on flange SE(3) and wrist CoM.

## ADR-033 — Table S4 default joint-pos offset is reset qpos, not restitution

- **Status:** accepted
- **Context:** Table S4 `physical.default_joint_pos_offset_rad` is [−0.01, 0.01] rad (arXiv:2511.07820v3). ADR-032 left this recorded-only. Isaac Lab samples the offset independently per actuated joint at reset and uses the same default as the zero-action bias. A 2^25 corner grid is not an extrema sweep. Mapping restitution onto `solref` would still invent a PhysX→MuJoCo conversion. These numbers must not enter `dexhand2_spec.yaml` (ADR-004) and are not Hand 2 command-latency.
- **Options:** (A) keep joint-pos offset recorded-only; (B) sweep the two published scalar extrema as a *uniform* additive offset on all 25 actuated hinges' reset qpos **and** PD target during the 3 s PD hold, keep `qpos_+0.01` as the `joint_offset` compatibility key, leave the freejoint unchanged, do not mix into `push_sweep`; (C) independently extremize each hinge (50 cases) or clip through joint limits.
- **Decision:** (B). `default_joint_pos_offset_extrema()` is the source. `T800MujocoEnv.offset_default_joint_pos` adds to the bring-up `q_des` and **raises** if any hinge would leave its MJCF `range`. Out-of-limit clipping would invent a map. Default `hold` stays the un-offset bring-up pose. `local_tracking_success` stays false. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`. Restitution stays recorded-only.
- **Consequences:** `make eval-l2-freebase-push` adds 2 joint-offset hold cases. A change in lean vs fall across ±0.01 rad is a diagnostic, **not** a SONIC fail and **not** a Hand 2 calibration. Restitution still has no non-invented `solref` map. PPO / GMR-on-BONES-SEED / combined weld remain blocked on flange SE(3) and wrist CoM.

## ADR-034 — Table S4 target-motion joint jitter is clip dof_pos, not reset qpos

- **Status:** accepted
- **Context:** Table S4 `target_motion.joint_jitter_rad` is [−0.1, 0.1] rad (arXiv:2511.07820v3). ADR-033 swept `physical.default_joint_pos_offset_rad` (±0.01 rad) as a free-base *reset* qpos / PD-target offset. Target-motion jitter is a 10× larger perturbation of the *reference clip* the tracker is asked to follow. Pos/ori jitter of that clip's root cannot be tracked with a pinned pelvis. Mapping restitution onto `solref` would still invent a PhysX→MuJoCo conversion. These numbers must not enter `dexhand2_spec.yaml` (ADR-004) and are not Hand 2 command-latency.
- **Options:** (A) keep target_motion recorded-only; (B) sweep the two published scalar joint-jitter extrema as a *uniform* additive offset on all 25 clip hinges during pinned-base physics Sim2Sim, keep `q_jit_+0.1` as the `joint_jitter` compatibility key, leave pos/ori/lin_vel/ang_vel target jitter recorded-only, do not mix into `push_sweep` or ADR-033; (C) independently extremize each hinge, or apply pos/ori jitter as a pinned-base pass/fail.
- **Decision:** (B). `target_motion_joint_jitter_extrema()` is the source. `apply_clip_joint_jitter` adds to clip `dof_pos`. `T800MujocoEnv.assert_hinges_in_mjcf_range` **raises** if any frame would leave MJCF `range` — clipping would invent a map. Unjittered pinned PD tracking stays the CI gate. Jittered `local_tracking_success` is a diagnostic. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`. Restitution stays recorded-only. Pos/ori helpers exist in `table_s4.py` so the ranges are not re-guessed.
- **Consequences:** `make eval-l2-physics-sim2sim` adds 2 joint-jitter cases on pinned-base. A change in joint MAE across ±0.1 rad is **not** a SONIC fail and **not** a Hand 2 calibration. PPO / GMR-on-BONES-SEED / combined weld remain blocked on flange SE(3) and wrist CoM.

## ADR-035 — Table S4 target-motion pos/ori jitter is clip root, not a height/ori gate

- **Status:** accepted
- **Context:** Table S4 `target_motion.pos_jitter_m` is ±xy 0.05 m / z 0.01 m and `ori_jitter_rad` is ±roll/pitch 0.1 rad / yaw ±0.2 rad (arXiv:2511.07820v3). ADR-034 swept `joint_jitter_rad` as clip `dof_pos` and left pos/ori recorded-only because a pinned pelvis cannot follow a jittered root. The 0.25 m / 1.0 rad Sim2Sim height/ori bands swallow those paper ranges, so treating pos/ori jitter as a pass/fail against those gates would be a false green. Mapping restitution onto `solref` would still invent a PhysX→MuJoCo conversion. These numbers must not enter `dexhand2_spec.yaml` (ADR-004) and are not Hand 2 command-latency, not a root_push, and not ADR-034 joint jitter.
- **Options:** (A) keep pos/ori recorded-only; (B) sweep the six pos + six ori axis-aligned signed extrema as an additive offset on clip `root_pos` / a world-frame RPY on clip `root_rot` during pinned-base physics Sim2Sim, keep `pos_+x` / `ori_+yaw` as compatibility keys, do not mix into `joint_jitter_sweep` or `push_sweep`, do not use height/ori gates as pass/fail; (C) treat `local_tracking_success` against 0.25 m / 1.0 rad as a SONIC gate.
- **Decision:** (B). This is a *negative control*: joint MAE must stay; MPJPE (and pelvis ori error) vs the jittered root must move by about the jitter. Unjittered pinned PD tracking stays the CI gate. Jittered `local_tracking_success` is ignored as a gate (`not_a_height_ori_gate`). `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`. Restitution stays recorded-only. `lin_vel`/`ang_vel` target jitter stay recorded (need a velocity clip, not a stand clip).
- **Consequences:** `make eval-l2-physics-sim2sim` adds 6 pos + 6 ori cases on pinned-base. A change in MPJPE across ±0.05 m is **not** a SONIC fail and **not** a Hand 2 calibration. PPO / GMR-on-BONES-SEED / combined weld remain blocked on flange SE(3) and wrist CoM. lin_vel/ang_vel target jitter is ADR-036.

## ADR-036 — Table S4 target-motion lin_vel/ang_vel jitter is clip velocity, not a root_push

- **Status:** accepted
- **Context:** Table S4 `target_motion.lin_vel_jitter` is ±xy 0.5 m/s / z 0.2 m/s and `ang_vel_jitter` is ±roll/pitch 0.52 rad/s / yaw ±0.78 rad/s (arXiv:2511.07820v3). Those numeric bounds match `root_push`, but they perturb the *reference clip* `s^g`, not robot `qvel`. ADR-035 left them recorded-only because the synthetic stand clip has identically zero velocity — adding jitter there would *be* the velocity (degenerate). Mapping restitution onto `solref` would still invent a PhysX→MuJoCo conversion. These numbers must not enter `dexhand2_spec.yaml` (ADR-004) and are not Hand 2 command-latency, not a root_push, and not ADR-035 pos/ori jitter. The 0.25 m / 1.0 rad height/ori bands are irrelevant to velocity.
- **Options:** (A) keep lin_vel/ang_vel recorded-only; (B) add a synthetic walk clip whose pose matches stand (so pinned PD joint-MAE stays comparable) but whose `root_linvel` / `root_angvel` are a labelled diagnostic carrier, sweep the six + six axis-aligned signed extrema as an additive offset on those fields, keep `lin_vel_+x` / `ang_vel_+yaw` as compatibility keys, refuse a stand clip, do not mix into `push_sweep` or pos/ori/joint jitter, do not pull BONES-SEED; (C) treat a translating root as a SONIC gate, or copy Table S4 velocity numbers into DexHand2 latency.
- **Decision:** (B). Carrier is `SYNTHETIC_WALK_LINVEL_MPS = (0.35, 0, 0)` m/s and `SYNTHETIC_WALK_ANGVEL_RAD_S = (0, 0, 0.25)` rad/s — not a T800 gait, chosen so −X / −yaw cannot cancel through zero. Pose is **not** integrated from the carrier (that would confound the joint-MAE negative control). Unjittered pinned PD tracking on the stand clip stays the CI gate. Jittered `local_tracking_success` is ignored as a gate (`not_a_height_ori_gate`). `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`. Restitution stays recorded-only.
- **Consequences:** `make eval-l2-physics-sim2sim` adds 6 lin-vel + 6 ang-vel cases on pinned-base using the walk clip. A change in target linvel across ±0.5 m/s is **not** a SONIC fail, **not** a root_push, and **not** a Hand 2 calibration. PPO / GMR-on-BONES-SEED / combined weld remain blocked on flange SE(3) and wrist CoM. `dynamic_friction` / `restitution` stay recorded-only (ADR-037).

## ADR-037 — Table S4 dynamic friction and restitution stay recorded-only

- **Status:** accepted
- **Context:** Table S4 `physical.dynamic_friction` is [0.3, 1.2] and `physical.restitution` is [0.0, 0.5] (arXiv:2511.07820v3). ADR-031 swept static friction as MuJoCo `geom_friction[0]` (sliding) on the eval floor and foot geoms. Isaac Lab / PhysX has a separate dynamic coefficient and a restitution scalar. MuJoCo has one sliding coefficient; spin/roll live in `geom_friction[1:]` and are **not** μd. Bounce is a `solref`/`solimp` pair, not a restitution number. Mapping those paper ranges onto those arrays would invent a PhysX→MuJoCo conversion. These numbers must not enter `dexhand2_spec.yaml` (ADR-004) and are not Hand 2 pad–cardboard E1/E2.
- **Options:** (A) leave the YAML ranges unguarded (a later sweep could silently write them); (B) load the ranges, refuse any write onto `geom_friction[1:]` / `geom_solref` / `geom_solimp`, AST-scan sim/eval/wbc for those assignments, keep `set_wbc_slide_friction` slide-only with a runtime solref/solimp/spin-roll equality check; (C) invent a solref map or a second μ channel and sweep it.
- **Decision:** (B). `recorded_only_physical()` is the source. `refuse_recorded_only_physical_map` always raises. `T800MujocoEnv.set_wbc_dynamic_friction` / `set_wbc_restitution` are the forbidden APIs. `set_wbc_slide_friction` still writes only `geom_friction[gid, 0]`. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`.
- **Consequences:** CI fails if someone maps μd or restitution onto a geom. There is still no restitution sweep and no pad–cardboard number. PPO / GMR-on-BONES-SEED / combined weld remain blocked on flange SE(3) and wrist CoM.

## ADR-038 — L1a nav_cmd uses SONIC Eq. 8, not a raw 1 s integrate

- **Status:** accepted
- **Context:** SONIC §3.3 (arXiv:2511.07820v3) filters navigation commands with a critically damped spring on pelvis *x*, pelvis *y*, and projected heading. Damping is 5 ln 2 (position) and 20 ln 2 (heading). Velocity commands become a target after 1.0 s; the planner keyframe is *x*(1.0). Command range is 0–6.0 m/s any heading. Equation 8 as typeset equals (x_T−x_0+(v_0+(c/2)(x_T−x_0))t) e^{−ct/2}, which at t=0 is x_T−x_0, not x_0. Table S4 is finished as sweeps + recorded-only μd/restitution (ADR-037). DexHand2 pad–cardboard remains REQUIRED_INPUT (ADR-004).
- **Options:** (A) pass `nav_cmd` straight into L1a cubic Hermite (abrupt 6↔−6 m/s is exactly the case the paper guards); (B) implement the IC-correct critically damped solution with the paper's c and 1.0 s / 6.0 m/s numbers, clamp planar speed only, wrap heading on the shortest arc, keep the typeset formula as a documented non-runtime helper; (C) invent a different ζ or a wz clamp.
- **Decision:** (B). `wbc/spring.py` is shared sim/real (no MuJoCo/Isaac). `KinematicPlanner.plan()` is unchanged for upper-body waypoints. `plan_nav_root()` is the nav path. Hands still bypass WBC. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`. Table S4 is not touched.
## ADR-039 — 500 Hz command stream is the PD ring, not the 50 Hz token rate

- **Status:** accepted
- **Context:** SONIC §3.5 (arXiv:2511.07820v3) deploys four concurrent loops: kinematic planner 10 Hz, policy inference 50 Hz, operator input 100 Hz, command stream 500 Hz. ADR-021 named `sonic_command_hz: 50` the "instruction stream" because that is the tracker/token rate VLA chunks upsample onto. `t800_sonic.yaml` already listed `command_stream_hz: 500` with no interpolator. Hermite-resampling a 10 Hz Eq. 8 sampling is not the spring. DexHand2 MIT is 1 kHz and must not be collapsed onto 500 Hz. Table S4 and pad–cardboard are unrelated.
- **Options:** (A) treat 50 Hz as the PD clock and drop the paper 500 Hz loop; (B) interpolate 10 Hz L1a refs and 50 Hz policy tokens onto a 500 Hz ring with the same cubic/SLERP kernel as ADR-018, evaluate Eq. 8 at 500 Hz timestamps, keep ADR-021 `sonic_command_hz: 50` as the policy/token rate, record operator 100 Hz without implementing that loop; (C) invent a 333 Hz or 1 kHz body stream.
- **Decision:** (B). `wbc/stream.py` is shared sim/real (no MuJoCo/Isaac). Integer factors only (10→500 = 50, 50→500 = 10). Case A 50-D is refused. Hands still bypass WBC and ride the clock. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`.
- **Consequences:** `make eval-l1a-stream` dumps the factors. The operator 100 Hz loop is ADR-040. This is **not** SONIC's trained generative planner, **not** the Hand 2 1 kHz MIT ring, **not** a pad–cardboard number, and **not** a flange/CoM substitute. PPO / GMR-on-BONES-SEED / combined weld remain blocked on flange SE(3) and wrist CoM.

## ADR-040 — 100 Hz operator-input loop is the VR/gamepad/keyboard/network ring

- **Status:** accepted
- **Context:** SONIC §3.5 (arXiv:2511.07820v3) deploys four concurrent loops. ADR-039 implemented 500 Hz and recorded 100 Hz without a module. Paper: "switching between input interfaces (keyboard, gamepad, VR, network streams) by changing the active encoder, with no retraining required." VLA inference is 5–10 Hz and already has L1a (ADR-018) plus the 50 Hz token upsample (ADR-021). Hermite-interpolating `nav_cmd` would fight Eq. 8 (ADR-038). A PICO / CloudXR / Isaac Teleop import would put a simulator/SDK into `wbc/`. DexHand2 MIT is 1 kHz. Table S4 and pad–cardboard are unrelated.
- **Options:** (A) leave 100 Hz recorded-only; (B) ingest uniform 100 Hz `command_schema_v1` rows, upsample poses to 500 Hz with the ADR-018 cubic/SLERP kernel, hold-last `nav_cmd`, stride-downsample to 50 Hz (every 2nd) and 10 Hz (every 10th), refuse VLA/Case A/non-integer rates/3-point↔5-point mix, keep sources as a YAML allow-list without a PICO SDK; (C) invent a 90 Hz OpenVR default or average rotations when downsampling.
- **Decision:** (B). `wbc/operator.py` is shared sim/real (no MuJoCo/Isaac/PICO). Integer factors only (100→500 = 5, 100→50 = 2, 100→10 = 10). Live `OperatorHold` is hold-last between 100 Hz pushes onto 500 Hz reads. Hands still bypass WBC and ride the clock. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`.
- **Consequences:** `make eval-l1a-operator` dumps the 1.6 s counts (161 / 81 / 17 / 801). This is **not** SONIC's trained generative planner, **not** a PICO SDK, **not** the Hand 2 1 kHz MIT ring, **not** a pad–cardboard number, and **not** a flange/CoM substitute. PPO / GMR-on-BONES-SEED / combined weld remain blocked on flange SE(3) and wrist CoM.

## ADR-041 — Observation gathering is YAML-driven grouped history, not invented latency

- **Status:** accepted
- **Context:** SONIC paper S7 (arXiv:2511.07820v3) gathers IMU orientation / angular velocity and joint q/dq at the 50 Hz control tick, remaps to policy joint order, and pushes a snapshot into a dual-purpose logger (ring + CSV). Observation order is a YAML registry compiled to (function, offset, dim) triples. Official GEAR-SONIC `observation_config.yaml` concatenates grouped blocks `token | ω_hist | q_hist | dq_hist | a_hist | g_hist`. Existing `ProprioHistory` packs interleaved 10×(ω,q,dq,a,g). Both are 874-D on T800 / 994-D on G1. Inventing a 40–150 ms sensor delay would violate SPEC_INTAKE. DexHand2 joints must not enter the WBC vector. Hardware may publish at 500 Hz; the paper uses latest-data-wins.
- **Options:** (A) keep only interleaved `ProprioHistory` and skip the YAML registry; (B) compile `wbc/obs_gather.yaml` to the official grouped layout with T800 25-DoF dims, 50 Hz ring + 500 Hz `HardwareHold`, delay ticks locked at 0, G1 994 / 29-DoF / hand names refused, provide grouped↔interleaved converters; (C) copy G1 29-DoF names and fill typical IMU noise.
- **Decision:** (B). `wbc/gather.py` is shared sim/real (no MuJoCo/Isaac/PICO). Default total is 64+30+250+250+250+30 = 874. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`.
- **Consequences:** `make eval-l1a-gather` dumps the compiled slots. Loading a T800 ONNX trained on interleaved packing needs `grouped_history_to_interleaved` (or a YAML that lists per-frame slots). Do not invent IMU bias or camera delay. Encoder `motion_*` observations are ADR-042. A measured latency model is still future work. PPO / GMR-on-BONES-SEED / combined weld remain blocked on flange SE(3) and wrist CoM.

## ADR-042 — Encoder motion_* is a 50 Hz look-ahead, not an invented clip

- **Status:** accepted
- **Context:** Official GEAR-SONIC `encoder_observations` are `motion_joint_positions_10frame_step5` (G1 290-D), `motion_joint_velocities_10frame_step5` (290), `motion_anchor_orientation_10frame_step5` (60), `motion_root_z_position_10frame_step5` (10) = **650-D**. Docs: 10 future frames, every 5 ticks at 50 Hz (0.1 s), 0.9 s horizon; last frame repeats past the clip end; 6D is the first two columns of the heading-corrected relative rotation. ADR-041 assembled the *decoder* 874-D robot-state history. Feeding G1 `model_encoder.onnx` or a 29-DoF window into T800 would smash dimensions. Inventing a BONES-SEED clip or silently resampling 30 fps → 50 Hz would invent motion. T800 has **0** wrist joints (ADR-001); G1 `motion_*_wrists_*` is 6-DoF. SMPL joints are not in the T800 motion_lib schema. DexHand2 still bypasses WBC. Table S4 and pad–cardboard are unrelated.
- **Options:** (A) leave encoder input recorded-only until a trained T800 ONNX exists; (B) assemble the official names with T800 25-DoF dims (**570-D**), require a caller-supplied 50 Hz reference (`MotionHold` / `MotionCursor`), last-frame-repeat, Zhou 6D relative orientation, refuse G1 650 / 29-DoF / wrist / SMPL / mode name `g1` / encoder ONNX, do not resample; (C) copy G1 29-DoF names and generate a stand clip when the hold is empty.
- **Decision:** (B). `wbc/motion_ref.py` + `ObsGather.assemble_encoder()` are shared sim/real (no MuJoCo/Isaac/PICO). Default encoder list matches the docs example. Token_state stays an external 64-D — this increment does **not** run ONNX. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`.
- **Consequences:** `make eval-l1a-encoder` dumps the 570-D slots and look-ahead indices `[0,5,…,45]`. A 30 fps `motion_lib` raises. Empty hold raises instead of inventing a stand pose. Lower-body slice is J00–J11 (12), same count as G1 legs, not a G1 copy. PPO / GMR-on-BONES-SEED / combined weld remain blocked on flange SE(3) and wrist CoM.

## ADR-043 — Teleop encoder mode is lower-body + VR 3-point, not a PICO SDK

- **Status:** accepted
- **Context:** Official `nvidia/GEAR-SONIC` `observation_config.yaml` concatenates a **superset** of encoder observations and zero-fills slots that are not in the active `encoder_modes[].required_observations`. `encoder_mode_4` is mode_id + 3 zeros. Official modes are `g1` (id 0, full-body 10frame_step5), `teleop` (id 1, lower-body 10frame_step5 + `vr_3point_local_target` / `vr_3point_local_orn_target` + current-frame `motion_anchor_orientation`), and `smpl` (id 2, SMPL joints + G1 wrists). ADR-042 assembled only the docs-example 570-D motion window and did not implement mode zero-fill. A PICO / CloudXR / Isaac Teleop import would put an SDK into `wbc/`. T800 has 0 wrist DoF; SMPL is not in the T800 motion_lib schema. DexHand2 still bypasses WBC. Table S4 and pad–cardboard are unrelated.
- **Options:** (A) keep the 570-D packed t800-only list until a trained T800 encoder ONNX exists; (B) adopt the official HuggingFace superset with T800 dims (**842-D** = official list minus SMPL/wrists), `t800` as the G1 analogue (mode_id 0), `teleop` as official mode_id 1, pack VR 3-point from `command_schema_v1` via `push_vr_3point`, refuse `g1` / `smpl` / wrists / 5-point / G1 1751-D ONNX, do not invent a headset pose when the VR hold is empty; (C) copy G1 29-DoF names and import a PICO SDK.
- **Decision:** (B). `ObsGather.assemble_encoder(mode)` is shared sim/real (no MuJoCo/Isaac/PICO). Token_state stays an external 64-D — this increment does **not** run ONNX. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`.
- **Consequences:** `make eval-l1a-encoder` dumps both modes. T800 motion window remains 570-D (ADR-042 formula). Encoder ONNX input is 842-D with unused slots zero-filled. Official G1 ONNX (1751) is refused. SMPL teleop remains blocked until a T800 SMPL clip exists. Low-latency `10frame_step1` is ADR-044. PPO / GMR-on-BONES-SEED / combined weld remain blocked on flange SE(3) and wrist CoM.

## ADR-044 — Low-latency encoder is 10frame_step1, not G1 4-frame ONNX

- **Status:** accepted
- **Context:** Official `nvidia/GEAR-SONIC` `low_latency/observation_config.yaml` keeps the 3-mode tokenizer (`g1` / `teleop` / `smpl`) but changes the look-ahead. G1 and teleop future-reference names are `*_10frame_step1` (indices `[0,1,…,9]` at 50 Hz, 0.18 s to last index). SMPL/wrist names are `*_4frame_step1` (~80 ms). The YAML **omits** `motion_root_z_*`. Encoder ONNX input is **1247-D** on G1 vs default **1751-D**. Official docs: use the matching encoder, decoder, and obs config together; the low-latency name is reference lookahead, not end-to-end latency. ADR-043 assembled only the default `10frame_step5` 842-D T800 superset. Loading `low_latency/model_encoder.onnx` on T800 would smash 1247 vs 831. Packing SMPL 4-frame would invent `smpl_joint.csv` and G1 wrists (ADR-001). Mixing step5 and step1 in one concat would not match either official YAML.
- **Options:** (A) leave step1 recorded-only until a trained T800 low-latency ONNX exists; (B) add `wbc/obs_gather_low_latency.yaml` as the official low-latency list with T800 dims (**831-D** = official list minus SMPL/wrists), keep default 842-D YAML, refuse 4-frame body/SMPL/wrist names, refuse G1 1247-D ONNX, do not invent a PICO pose; (C) copy G1 4-frame SMPL names and load the HuggingFace low-latency ONNX.
- **Decision:** (B). `ObsGather(cfg=load_gather_cfg(GATHER_LOW_LATENCY_YAML))` is shared sim/real (no MuJoCo/Isaac/PICO). Token_state stays an external 64-D — this increment does **not** run ONNX. Decoder 874-D is unchanged. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`.
- **Consequences:** `make eval-l1a-encoder` dumps both YAMLs. T800/teleop modes still use **10** frames (official g1/teleop names). The 4-frame horizon is SMPL-only and stays refused. Do not pair the default 842-D vector with a future low-latency T800 ONNX, or the reverse. PPO / GMR-on-BONES-SEED / combined weld remain blocked on flange SE(3) and wrist CoM.

## ADR-045 — SONIC v1.1 encoder is heading-normalized step5, not G1 1751-D ONNX

- **Status:** accepted
- **Context:** Official `nvidia/GEAR-SONIC` `sonic_v1_1/observation_config.yaml` (2026-07-23) is a robot-heading-normalized controller trained with wrist-pose augmentation for 3-point teleop and SONIC-backed VLA. G1/teleop future-reference names stay `*_10frame_step5`. Anchor ori is `motion_anchor_orientation_heading_*` (`R_z(yaw_robot).T @ R_ref`), not the default full `R_robot.T @ R_ref`. The YAML **omits** `motion_root_z_*`. SMPL/wrist names are `*_10frame_step1` (~200 ms), not the low-latency 4-frame horizon. Official header: encoder **1751-D** / decoder 994-D on G1. Docs: this is **not** the low-latency checkpoint. ADR-043 assembled default full-ori 842-D. ADR-044 assembled step1 831-D. Loading `sonic_v1_1/model_encoder.onnx` on T800 would smash 1751 vs 831. Pairing the T800 v1.1 831-D vector with a low-latency step1 ONNX would silently mis-index the same integer. Wrist-pose augmentation is a training-time transform; inventing a sampler here would invent data. T800 has 0 wrist DoF; SMPL is not in the T800 motion_lib schema. DexHand2 still bypasses WBC.
- **Options:** (A) leave heading ori recorded-only until a trained T800 v1.1 ONNX exists; (B) add `wbc/obs_gather_sonic_v1_1.yaml` as the official v1.1 list with T800 dims (**831-D** = official list minus SMPL/wrists/root_z), keep default 842-D and low-latency 831-D YAMLs, refuse full-ori names / step1 / root_z / G1 1751-D ONNX, do not invent a wrist-pose augmenter; (C) copy G1 SMPL/wrist 10frame_step1 names and load the HuggingFace v1.1 ONNX.
- **Decision:** (B). `ObsGather(cfg=load_gather_cfg(GATHER_SONIC_V1_1_YAML))` is shared sim/real (no MuJoCo/Isaac/PICO). Token_state stays an external 64-D — this increment does **not** run ONNX. Decoder 874-D is unchanged. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`.
- **Consequences:** `make eval-l1a-encoder` dumps three YAMLs. T800/teleop modes still use **10** frames at **step5**. Heading ori strips robot pitch/roll; default full ori does not. The two 831-D layouts are not interchangeable. Official v1.1 SMPL/wrist 10frame_step1 stays refused. Do not pair this vector with `low_latency/model_encoder.onnx` or `sonic_v1_1/model_encoder.onnx`. PPO / GMR-on-BONES-SEED / combined weld remain blocked on flange SE(3) and wrist CoM.

## ADR-046 — Planner ONNX is T800 32-D qpos, not G1 planner_sonic.onnx

- **Status:** accepted
- **Context:** Official `planner_sonic.onnx` (nvlabs planner_onnx.html) is the trained generative L1a. G1 `context_mujoco_qpos` is `[1, 4, 36]` = 3 pos + 4 quat wxyz + 29 hinges. Output is 30 Hz, resampled to 50 Hz. V2 has 11 inputs / 27 modes / tokens 6–16 (K=11). ADR-018 is cubic/SLERP interpolators. ADR-038 is Eq. 8. Loading G1 36-D on T800 25-DoF would smash the hinge block. Inventing a clip library or running the G1 ONNX would violate ADR-011. `command_schema` loco_mode is `{stand, slow_walk, fast_walk}`; official mode 3 is `run`. `nav_cmd[2]` is heading rate, not `facing_direction`. `pelvis_height` is a VLA field; official height is disabled (`-1`) except squat/kneel/lying.
- **Options:** (A) leave the ONNX I/O recorded-only until a T800 planner exists; (B) freeze the official V2 tensor list with T800 dims (**32** = 7+25), map loco_mode 0/1/2 → idle/slowWalk/walk (not run), pack movement/facing from heading-frame `nav_cmd` + caller yaw, keep height disabled from command_schema, implement the documented 30→50 resample and 4-frame 30 Hz context sampler, refuse G1 36-D / `planner_sonic.onnx` / empty context, do not run ONNX; (C) load HuggingFace `planner/target_vel/V2/planner_sonic.onnx` and zero-pad 4 hinge slots.
- **Decision:** (B). `wbc/planner_onnx.py` is shared sim/real (no MuJoCo/Isaac/PICO/onnxruntime). ADR-018 interpolators stay the runtime L1a. Token_state / encoder layouts are unchanged. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`.
- **Consequences:** `make eval-l1a-planner-onnx` dumps the 11-tensor pack. Official G1 planner remains unloadable. Do not treat the interpolator as the trained planner. PPO / GMR-on-BONES-SEED / combined weld remain blocked on flange SE(3) and wrist CoM.

## ADR-047 — Planner 8-frame cross-fade and replan timer, not a G1 ONNX run

- **Status:** accepted
- **Context:** Official planner_onnx.html and `g1_deploy_onnx_ref.cpp` blend successive 50 Hz planner clips over 8 frames and gate TensorRT calls with a 10 Hz replan table. ADR-046 packed T800 32-D I/O and the 30→50 resample but left that control-thread logic recorded-only. Loading `planner_sonic.onnx` is still G1 36-D. Inventing a 4-frame fade, a 0.2 s elbow-crawl interval, or mapping `command_schema` loco_mode 2 onto official run (0.1 s) would not match C++. Idle ADAPTING/RECOVERING is a different path (`kAdaptTrigger`). DexHand2 still bypasses WBC. Table S4 and pad–cardboard are unrelated.
- **Options:** (A) leave blend/replan recorded-only until a T800 planner exists; (B) implement the C++ formulas on caller-supplied 50 Hz T800 qpos: `w_new = clamp((f - blend_start) / 8, 0, 1)`, `blend_start = max(0, gen_frame - current_frame)`, linear joints/root xyz, SLERP quat, `current_frame` reset to 0; replan category 1 = mode/facing/height, category 2 = non-static speed/direction/timer with speed ≠ 0; intervals run 0.1 s / crawl **mode 8 only** 0.2 s / punches-hooks 1.0 s / else 1.0 s; refuse G1 36-D / ONNX run / interpolator substitute / idle-readapt mix; (C) run HuggingFace `planner_sonic.onnx` and treat fast_walk as run.
- **Decision:** (B). `wbc/planner_blend.py` is shared sim/real (no MuJoCo/Isaac/PICO/onnxruntime). ADR-018 interpolators stay the runtime L1a. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`.
- **Consequences:** `make eval-l1a-planner-blend` dumps the weight ramp and interval table. `command_schema` loco_mode 2 keeps the **walk** 1.0 s timer, not run. Elbow crawling (14) stays on the default 1.0 s timer because that is what C++ does. Do not treat this blend as a trained planner. Idle ADAPTING/RECOVERING is ADR-048.

## ADR-048 — Idle ADAPTING/RECOVERING last-frame hold, not a G1 ONNX run

- **Status:** accepted
- **Context:** Official `g1_deploy_onnx_ref.cpp` mutates the last planner frame while `LocomotionMode::IDLE` with a double-threshold machine: IDLE / ADAPTING (toward motor q) / RECOVERING (toward the stored original last-frame targets). Thresholds are `kAdaptTrigger` 0.10 rad, `kAdaptStop` 0.05 rad, `kRecoverTrigger` 0.045 rad. Blend is `0.98 q + 0.02 target`. ADR-047 refused this path from the 8-frame blend. Loading `planner_sonic.onnx` is still G1 36-D. Inventing a recover-stop, applying the machine to squat/walk, averaging all 25 hinges, or writing DexHand2 q would not match C++. C++ only runs this after playback clamps to the last frame. DexHand2 still bypasses WBC. Table S4 and pad–cardboard are unrelated.
- **Options:** (A) leave idle readapt recorded-only until a T800 planner exists; (B) implement the C++ formulas on caller-supplied T800 25-D hinges: lower-body J00–J11 only, strict `>`/`<`, no recover-stop, skip unless play + last frame + idle, clear stored on new clip, refuse G1 29-D / 32-D qpos / 45-D hands / ONNX run / blend mix-in; (C) run HuggingFace `planner_sonic.onnx` and add a recover-stop so RECOVERING falls through to IDLE.
- **Decision:** (B). `wbc/idle_readapt.py` is shared sim/real (no MuJoCo/Isaac/PICO/onnxruntime). ADR-047 blend stays a sibling. ADR-018 interpolators stay the runtime L1a. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`.
- **Consequences:** `make eval-l1a-idle-readapt` dumps the threshold table. RECOVERING at 0 rad stays RECOVERING. Squat (4) does not enter. `command_schema` loco_mode 0 is the only schema value that maps to this path. Do not treat this hold as a trained planner. The 50 Hz cursor that owns `current_frame` and calls this after blend assign is ADR-049.

## ADR-049 — 50 Hz last-frame playback cursor, not a G1 ONNX run

- **Status:** accepted
- **Context:** Official `g1_deploy_onnx_ref.cpp` `CurrentFrameAdvancement` runs at the end of each 50 Hz control tick. After an ADR-047 clip assign (`current_frame = 0`, `idle_readapt_stored_ = false`), it always does `new_frame = current_frame + 1` while `play` and `timesteps > 0`, then clamps to `timesteps - 1`. Idle ADAPTING/RECOVERING (ADR-048) runs only inside that clamp, and only for `LocomotionMode::IDLE`. Named-clip playback loops to 0 in a *different* branch; the planner-active path never does. Loading `planner_sonic.onnx` is still G1 36-D. Inventing a loop, advancing when paused, calling idle readapt mid-clip, or mixing `MotionCursor` (encoder look-ahead) into this hold would not match C++. DexHand2 still bypasses WBC. Table S4 and pad–cardboard are unrelated.
- **Options:** (A) leave the cursor recorded-only until a T800 planner exists; (B) implement the C++ order on caller-supplied 50 Hz T800 qpos: assign then advance+clamp, idle hold only after clamp + play + idle, refuse named-clip loop / G1 36-D / 29-D / 45-D hands / ONNX run / interpolator substitute; (C) run HuggingFace `planner_sonic.onnx` and loop planner clips to frame 0.
- **Decision:** (B). `wbc/playback.py` is shared sim/real (no MuJoCo/Isaac/PICO/onnxruntime). ADR-047 blend and ADR-048 idle hold stay the callees. ADR-018 interpolators stay the runtime L1a. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`.
- **Consequences:** `make eval-l1a-playback` dumps the clamp table. A successful first assign of a 16-frame clip with `play` leaves `current_frame == 1` on that same tick. One-frame clips clamp (and may readapt) on the first tick. Do not treat this cursor as a trained planner. Encoder look-ahead and planner context sampling that *read* this integer are ADR-050.

## ADR-050 — Encoder look-ahead and planner context share playback.current_frame

- **Status:** accepted
- **Context:** ADR-049 owns `current_frame` on `PlannerPlayback`. Encoder `MotionHold` / `assemble_encoder` (ADR-042) and planner `sample_context_from_50hz` (ADR-046) each took a caller-supplied integer. Two independent cursors can drift: encoder look-ahead from frame 0 while planner context starts at `current_frame + 2` of a different clock. Mixing `MotionCursor` (motion_lib sibling) into `PlannerPlayback` is the ADR-049 refusal. Loading encoder/planner ONNX is still G1. Inventing last-frame-repeat on the planner context, looping planner clips to 0, or finite-diffing missing `dq` would not match official behavior. DexHand2 still bypasses WBC. Table S4 and pad–cardboard are unrelated.
- **Options:** (A) leave the two readers independent until a T800 ONNX exists; (B) bind both readers to `PlannerPlayback.current_frame` after each 50 Hz tick: encoder `MotionHold.cursor` last-frame-repeats, planner context starts at `current_frame + 2` and raises on short windows, refuse MotionCursor mix / G1 36-D / 45-D hands / ONNX run / invented clip; (C) store a `MotionCursor` inside `PlannerPlayback` and last-frame-repeat planner context.
- **Decision:** (B). `wbc/shared_cursor.py` is shared sim/real (no MuJoCo/Isaac/PICO/onnxruntime). `MotionCursor` stays a sibling over a validated 50 Hz `motion_lib`. ADR-018 interpolators stay the runtime L1a. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`.
- **Consequences:** `make eval-l1a-shared-cursor` dumps both readers at the same integer. After a 16-frame first tick, `current_frame == 1`, encoder indices are `[1,6,11,15,…]`, planner context starts at index 3. At the last frame, encoder repeats; planner context raises. Do not treat this wiring as a trained planner or encoder ONNX run. Decoder 874-D assembled on the same 50 Hz tick is ADR-051.

## ADR-051 — Decoder 874-D control_tick shares the SharedPlaybackCursor clock

- **Status:** accepted
- **Context:** Official `g1_deploy_onnx_ref.cpp` gathers IMU + joints and assembles the decoder observation on the same 50 Hz thread as `CurrentFrameAdvancement`. ADR-041 already builds T800 874-D grouped history from `HardwareHold`. ADR-050 bound encoder look-ahead and planner context to `playback.current_frame` but left decoder assemble as a separate caller. Two clocks can drift: decoder ring advancing while `current_frame` is paused, or clip hinges copied into proprioception. Loading `model_decoder.onnx` is still G1 994-D. Inventing a 64-D token, copying planner qpos into `q_hw`, or skipping the decoder ring when `play == false` would not match C++ (the robot still has IMU/joints while the clip is held). DexHand2 still bypasses WBC. Table S4 and pad–cardboard are unrelated.
- **Options:** (A) leave decoder gather on a separate caller until a T800 decoder ONNX exists; (B) add `SharedPlaybackCursor.control_tick`: after ADR-049/050, call `ObsGather.control_tick` from HardwareHold, require an external 64-D token, refuse clip→decoder copy / G1 994-D / 45-D hands / ONNX run, keep `tick()` without token as the ADR-050 path; (C) run HuggingFace `model_decoder.onnx` and fill token/last_action from clip qpos.
- **Decision:** (B). `wbc/shared_cursor.py` stays shared sim/real (no MuJoCo/Isaac/PICO/onnxruntime). Decoder history is hardware, not clip. ADR-018 interpolators stay the runtime L1a. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`.
- **Consequences:** `make eval-l1a-decoder-tick` dumps 874-D vs encoder look-ahead. After 10 ticks, decoder `q[0]` history is the hardware sequence; encoder 10frame_step5 stays on clip indices. `play == false` holds `current_frame` but still logs hardware. Do not treat this wiring as a trained decoder ONNX run. Caller-supplied 25-D `last_action` on this same tick is ADR-052.

## ADR-052 — Caller-supplied 25-D last_action on the SharedPlaybackCursor tick

- **Status:** accepted
- **Context:** Official `g1_deploy_onnx_ref.cpp` feeds the previous decoder output back as `last_action` in the next 50 Hz gather (`his_last_actions_*`). ADR-051 assembled T800 874-D from `HardwareHold` but left `last_action` as whatever the snapshot carried (usually zeros). Filling it from G1 `model_decoder.onnx` (29-D action), from planner clip qpos (32-D), or from a synthetic ONNX vector would invent a policy output. DexHand2 still bypasses WBC (45-D refused). PPO `action_dim` is 25. Table S4 and pad–cardboard are unrelated.
- **Options:** (A) leave last_action as snapshot zeros until a T800 decoder ONNX exists; (B) accept a caller-supplied 25-D vector on `control_tick` / `tick`, push it into `HardwareHold` before decoder assemble, omit-to-keep-hold so ADR-051 callers stay valid, refuse G1 29-D / 32-D qpos / 45-D hands / fake ONNX fill / clip copy; (C) run HuggingFace `model_decoder.onnx` and write its 29-D action into the T800 ring.
- **Decision:** (B). `wbc/shared_cursor.py` + `HardwareHold.push_last_action` stay shared sim/real (no MuJoCo/Isaac/PICO/onnxruntime). Startup zeros are padding, not an invented decoder output. ADR-018 interpolators stay the runtime L1a. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`.
- **Consequences:** `make eval-l1a-last-action` dumps a-history vs q-history vs encoder look-ahead. After 10 ticks, decoder `a[0]` is the caller sequence while `q[0]` is hardware and encoder 10frame_step5 stays on clip indices. Do not treat this slot as a trained decoder ONNX run. A live T800 policy output is still missing — the caller must supply it. Stashing this tick's 25-D output as next-tick `last_action` is ADR-053.

## ADR-053 — Delayed 25-D policy_action becomes next-tick last_action

- **Status:** accepted
- **Context:** Official `g1_deploy_onnx_ref.cpp` writes the decoder output into `last_action` for the **next** 50 Hz gather (`his_last_actions_*` is `a_{t-1}`). ADR-052 accepted a caller-supplied 25-D vector on the **same** tick so ADR-051 tests could fill the slot without an ONNX run. That is not the C++ delay. Filling `a_t` from G1 `model_decoder.onnx` (29-D), from planner clip qpos (32-D), or from a synthetic ONNX vector would invent a policy output. Writing `policy_action` into this tick's decoder obs would also miss the official delay. DexHand2 still bypasses WBC (45-D refused). PPO `action_dim` is 25. Table S4 and pad–cardboard are unrelated.
- **Options:** (A) leave the one-tick delay to the caller until a T800 decoder ONNX exists; (B) accept `policy_action` on `control_tick` / `tick`, stash it, apply as `last_action` on the next tick, keep `last_action=` as a same-tick override so ADR-052 callers stay valid, refuse G1 29-D / 32-D qpos / 45-D hands / fake ONNX fill / same-tick obs write; (C) run HuggingFace `model_decoder.onnx` and write its 29-D action into the T800 ring this tick.
- **Decision:** (B). `wbc/shared_cursor.py` `stash_policy_action` stays shared sim/real (no MuJoCo/Isaac/PICO/onnxruntime). Startup zeros on tick 0 are padding, not an invented decoder output. ADR-018 interpolators stay the runtime L1a. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`.
- **Consequences:** `make eval-l1a-policy-action` dumps delayed a-history vs q-history vs encoder look-ahead. After 10 ticks, decoder `a[0]` is `[0, 10..18]` while the stashed `a_t` is 19, `q[0]` is `[0..9]`, and encoder 10frame_step5 stays on clip indices. Do not treat this delay as a trained decoder ONNX run. A live T800 policy output is still missing — the caller must supply `policy_action`. Wiring that `a_t` onto the 500 Hz PD ring is ADR-054.

## ADR-054 — Delayed 25-D policy_action is ZOH-held on the 500 Hz PD ring after stash

- **Status:** accepted
- **Context:** Official `g1_deploy_onnx_ref.cpp` sends decoder `a_t` to the 500 Hz PD instruction ring on the same 50 Hz tick that produced it, while `his_last_actions_*` stays `a_{t-1}`. ADR-053 stashed caller-supplied 25-D `policy_action` for the next gather but left the PD ring on command_schema cubic/SLERP (ADR-039). Hermite of joint targets would change the learned 50 Hz tracking. Filling PD from G1 `model_decoder.onnx` (29-D), planner clip qpos (32-D), DexHand2 concat (45-D), or a synthetic ONNX vector would invent a policy output. Writing `policy_action` into this tick's decoder obs would miss the official delay. `last_action=` is `a_{t-1}` and must not overwrite PD. DexHand2 still bypasses WBC. PPO `action_dim` is 25. Table S4 and pad–cardboard are unrelated.
- **Options:** (A) leave PD q_des to the caller until a T800 decoder ONNX exists; (B) after ADR-053 stash on `control_tick` / `tick`, ZOH-hold the same 25-D `a_t` on `PolicyPdHold`, keep `stash_policy_action()` from writing PD so ADR-053 callers stay explicit, refuse G1 29-D / 32-D qpos / 45-D hands / 75-D command_schema / fake ONNX fill / Hermite; (C) run HuggingFace `model_decoder.onnx` and spline its 29-D action onto the T800 PD ring.
- **Decision:** (B). `wbc/stream.py` `PolicyPdHold` / `stream_policy_action_zoh` stay shared sim/real (no MuJoCo/Isaac/PICO/onnxruntime). Startup zeros are padding, not an invented decoder output. ADR-018 interpolators stay the runtime L1a. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`.
- **Consequences:** `make eval-l1a-pd-stream` dumps delayed a-history vs PD q_des vs a 10-sample ZOH period. After 10 ticks, decoder `a[0]` is `[0, 10..18]` while PD `q_des[0]` is 19. One 50 Hz command occupies 10 identical 500 Hz samples. Do not treat this hold as a trained decoder ONNX run. A live T800 policy output is still missing — the caller must supply `policy_action`. Applying `q_des` to a 500 Hz joint-PD plant is ADR-055.

## ADR-055 — Held 25-D PD q_des is applied by a 500 Hz joint-PD plant on the same tick

- **Status:** accepted
- **Context:** Official `g1_deploy_onnx_ref.cpp` sends decoder `a_t` to the 500 Hz PD ring, then joint PD computes `τ = Kp (a_t − q) − Kd q̇` (no velocity target under ZOH). ADR-054 ZOH-held caller-supplied 25-D `policy_action` on `PolicyPdHold` but left τ to the MuJoCo env. Hermite of joint targets would invent `dq_des`. Filling τ from G1 `model_decoder.onnx` (29-D), planner clip qpos (32-D), DexHand2 concat (45-D), or a synthetic ONNX vector would invent a policy output. Finite-differencing a 50 Hz `q` snapshot onto 500 Hz would invent proprioception. SONIC PPO learns its own actuator PD; EngineAI `pd_stand` is bring-up only (ADR-022). DexHand2 still bypasses WBC. Table S4 and pad–cardboard are unrelated.
- **Options:** (A) leave τ to `T800MujocoEnv.apply_pd` until a T800 decoder ONNX exists; (B) evaluate `τ = Kp(a_t − q) − Kd q̇` on the same 50 Hz tick after ADR-054 ZOH, freeze Kp/Kd from `pd_stand` labelled `pd_stand_bringup_not_sonic`, hold-last 50 Hz `q/dq` unless the caller supplies a 500 Hz stream, refuse G1 29-D / 32-D qpos / 45-D hands / 75-D command_schema / fake ONNX fill / Hermite / finite-diff dq, do not write τ into decoder obs; (C) run HuggingFace `model_decoder.onnx` and spline 29-D τ onto the T800 plant.
- **Decision:** (B). `wbc/pd_plant.py` `PolicyPdPlant` / `joint_pd_torque_nm` stay shared sim/real (no MuJoCo/Isaac/PICO/onnxruntime). `T800MujocoEnv.apply_pd` calls the same function. Startup-zero q_des is padding, not an invented decoder output. ADR-018 interpolators stay the runtime L1a. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`.
- **Consequences:** `make eval-l1a-pd-plant` dumps delayed a-history vs PD q_des vs τ. After 10 ticks with `q[0]=i` and `a_t[0]=10+i`, decoder `a[0]` is `[0, 10..18]`, PD `q_des[0]` is 19, and `τ[0] = 1080 × (19 − 9) = 10800 N·m`. One 50 Hz command occupies 10 identical 500 Hz torque samples. Do not treat this plant as a trained decoder ONNX run or as SONIC tracking PD. A live T800 policy output is still missing — the caller must supply `policy_action`. Closing the 500 Hz plant onto a joint-PD command plant *inside MuJoCo on this same tick* (without treating τ as an ONNX run) is the next increment.


