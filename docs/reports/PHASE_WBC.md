# This increment — T800 SONIC contract, URDF FK, task FSM, hand-only MuJoCo

Builds on the IBVS / weld-gate / pad-sphere work. Still no grasp-success number and no combined T800+Hand weld.

## Done

- `wbc/t800_sonic.yaml` — 25-DoF joint order from Native SDK URDF; stand PD copied from EngineAI `pd_stand/default.yaml` (bring-up only)
- Decoder dim formula: T800 874 vs G1 994. `refuse_g1_checkpoint()` raises on G1 ONNX / 29 DoF
- `wbc/teleop.py` — remaps command_schema (head, left, right, rot6d) → SONIC vr_3point (left, right, head, quat wxyz)
- `wbc/observation.py` — heading-frame proprio `(q, dq, ω, g, a_prev)`
- `sim/urdf_fk.py` + `JointToWristAdapter` — numpy URDF FK; dummy `LINK_WRIST_END_*` matches the official elbow→wrist fixed joint
- `runtime/task_fsm.py` — pick_box / stack_to / scan_qr; scan DONE requires `decode_ok`, not distance
- `sim/mujoco_env/hand_env.py` — hand-only MIT env; combined robot still `PolicyEvalBlocked`
- `vla/server/policy_server.py` — in-process replay; GR00T/π0.5 weights not downloaded
- L1 report includes T800 wrist FK when the URDF is cloned

## Tests

`make test`. Official-asset tests run when `third_party/wuji-description` and the T800 URDF are present. MuJoCo hold-test skips if `mujoco` is not installed.

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. SONIC train/retarget additionally needs flange SE(3) and `com_in_wrist_frame_m`.

## Not done

- Combined T800+Hand MJCF (correctly refused)
- BONES-SEED GMR run / PPO
- RTX QR envelope
- Grasp-success numbers
