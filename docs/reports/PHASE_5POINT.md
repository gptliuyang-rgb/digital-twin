# This increment — 5-point elbow teleop, L1a planner, MIT gain-scan hold

Builds on GMR T-pose / left pads / privileged Isaac cfg. Still no grasp-success number and no combined T800+Hand weld.

## Done

- `interface/command_schema_v1_5point.yaml` — opt-in +6 D elbow positions. v1 stays 75-D / 3-point.
- `wbc/teleop.py` — `command_to_vr_5point` packs SONIC order left_wrist, right_wrist, head, left_elbow, right_elbow. Orn stays 12. `refuse_teleop_mode_mismatch()` (ADR-017).
- Elbow body is `LINK_ELBOW_PITCH_*` (GMR), not `LINK_ELBOW_YAW_*` (forearm / dummy-wrist parent). `JointToWristAdapter.elbow_pose()` FKs that body.
- Hybrid encoder cmd dim: 21 (3-point) vs 27 (5-point). Decoder history is still 874 (T800 25-DoF) either way.
- `wbc/planner.py` — L1a 10 Hz, horizon clamped to [0.8, 2.4] s. Cubic Hermite on positions, SLERP on SO(3), linear on finger joints (ADR-018).
- `eval/gain_scan.py` `run_hold_scan()` — 9-cell kp/kv hold on derived MIT motors when MuJoCo + official MJCF exist. `grasp_success_rate` always null.
- `runtime/wbc_safety.py` rejects elbow jumps > 5 cm/step when 5-point extras are present.

## Tests

`make test`. Elbow FK skips unless `third_party/engineai-native-sdk` is cloned. Gain-scan hold skips unless MuJoCo and `wuji-description` are present. Planner / 5-point packing / mode mismatch do not need clones.

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. 5-point is a *retrain* of the T800 hybrid encoder; flange SE(3) and `com_in_wrist_frame_m` still block SONIC PPO. A 3-point checkpoint must not be concatenated with elbows at deploy time.

## Not done

- Running GMR on BONES-SEED / writing `t800_motion_lib.pkl`
- SONIC PPO (3-point or 5-point)
- Combined T800+Hand MJCF
- Isaac Lab env `reset`/`step`
- Grasp-success numbers
