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
- **Consequences:** `make eval-l2-freebase-push` wall time is ~4× the single-axis case. Fall on any axis is **not** a SONIC fail. PPO launch, GMR-on-BONES-SEED, and the combined weld remain blocked on flange SE(3) and wrist CoM.

## ADR-028 — Table S4 duration is a sustained force, same impulse as one-shot qvel

- **Status:** accepted
- **Context:** Table S4 (arXiv:2511.07820v3) publishes root linear velocity *and* push duration Δt ∼ [1, 3] s. ADR-027 applied the planar velocity extrema as one-shot freejoint `qvel`. That is an instantaneous Δv, not the paper's duration. The agent prompt's domain-rand table is "根部线速度扰动 ±0.5 m/s，持续 1–3 s". Table S4 does not publish Newtons.
- **Options:** (A) re-clamp root `qvel` every physics step for T (a velocity hold, harsher than one-shot); (B) apply a constant world force F = m v / T on `LINK_BASE` for T, using the loaded MJCF subtree mass, so the linear impulse equals the one-shot Δp = m v; (C) invent a Newton range or treat duration as the interval *between* one-shot pushes.
- **Decision:** (B). `wbc/ppo/table_s4.py` `sustained_force_cases()` is planar extrema × duration extrema (8 cases). Mass is `body_subtreemass[LINK_BASE]` from the compiled model (fixture placeholder or official XML) — not a T800 datasheet guess. Floor friction stays the official WBC collision default, not pad–cardboard. `+y_T1.0s` is the `sustained_force` compatibility key. One-shot sweep is unchanged. `local_tracking_success` stays false. `grasp_success_rate` stays JSON `null`. Combined T800+Hand stays `PolicyEvalBlocked`. Z remains unswept.
- **Consequences:** A 1 s force is 3× stronger than the 3 s force for the same Table S4 |v|. Bring-up PD falling under either duration is **not** a SONIC fail. Gravity, contact, and joint PD still act, so realized Δv will not equal the free-space identity. PPO / GMR-on-BONES-SEED / combined weld remain blocked on flange SE(3) and wrist CoM.


