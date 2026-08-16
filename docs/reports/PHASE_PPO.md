# This increment — T800 SONIC PPO recipe, clip filter, kinematic Sim2Sim

Continues Case A FK / 50 Hz upsample (ADR-020/021). Still no grasp-success number and no combined T800+Hand weld.

## Done

- `assets/engineai/meta/t800_kinematics.yaml` — kinematics-only extract of the official Native SDK URDF (29 joints = 25 revolute + 4 fixed feet/dummy wrists) plus MJCF `range=` in radians.
- Fixed `parse_mjcf_joint_limits`: `\brange=` so it does not latch `actuatorfrcrange` (N·m). Hip pitch is [-3.316, 2.269] rad, not ±415.
- `wbc/ppo/` — paper Table S1–S4 frozen for T800 (`action_dim: 25`). `refuse_ppo_launch()` raises `PpoLaunchBlocked` while flange SE(3) / wrist CoM are open. G1 29-D action is refused. Table S4 friction is marked `not_dexhand2_contact`.
- Numpy Table S3 kernels. Identity tracking terms = 1.0. 0.3 m body error at scale 0.3 → exp(-1).
- `wbc/gmr/filter_lib.py` — batch filter using committed MJCF ranges (not ±π).
- `eval/l2_sim2sim.py` — kinematic MPJPE / wrist error. Identity tracker ≈ 0. Grasp-success stays JSON null. G1 6 cm wrist figure is recorded as *not a gate*.
- `wbc/export_onnx.py` — T800 decoder 874 vs G1 994; export refused without a trained ckpt.
- Case A FK falls back to the committed kinematics YAML, so `make eval-l1-case-a --apply-fk` no longer needs the SDK clone.

## Tests

`tests/test_ppo_sim2sim.py`. `make ppo-train` is supposed to fail.

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. BONES-SEED GMR run and Isaac Lab PPO stay blocked on flange SE(3) and wrist CoM. No live actor T-pose. No real GR00T/π0.5 dump in this repo.

## Not done

- Wiring `PrivilegedIsaacEnv.reset`/`step` inside Isaac Sim python
- Combined T800+Hand MJCF
- Grasp-success numbers
