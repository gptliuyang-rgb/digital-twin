# This increment — GMR T800 export, USD pad overlay, privileged L2

Builds on the SONIC contract / IBVS / pad-sphere work. Still no grasp-success number and no combined T800+Hand weld.

## Done

- `wbc/gmr/body_map.yaml` — T800 MJCF bodies (`LINK_WAIST_YAW`, `LINK_WRIST_END_*`, `LINK_FOOT_*`, `LINK_HEAD_YAW`) mapped to SMPL-X and LAFAN1. Human scale is 1.0 (uncalibrated). Quat offsets copied from GMR's EngineAI PM01 configs and labelled as such (ADR-013).
- `wbc/gmr/export.py` — writes `smplx_to_t800.json` / `bvh_lafan1_to_t800.json` / `params_overlay.yaml`. Refuses PM01 names `LINK_TORSO_YAW` and `LINK_ELBOW_END_*`.
- `wbc/gmr/motion_lib.py` — GMR pkl schema check; 29-DoF libraries raise `G1CheckpointIncompatible`.
- `wbc/filter.py` — joint-limit / foot-penetration / NaN / dq-blowup clip filter. Limits come from official T800 MJCF `range=` when cloned.
- `assets/dexhand2/build/gen_usd_pads.py` — USDA overlay of 15 palmar spheres for the right hand. Does not mirror left (no left fit in `fitted_pad_spheres.yaml`). Official USD is not overwritten.
- `sim/mujoco_env/privileged_l2.py` — pallet+box drop with labelled fixture floor friction. `grasp_success_rate` is always `null` (ADR-014).

## Tests

`make test`. GMR body-existence and MJCF limit parse run when `third_party/engineai-native-sdk` is cloned. Privileged drop skips if `mujoco` is not installed.

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. GMR *execution* (not export) still needs a T-pose pass to replace quat offsets and scale, plus flange SE(3) and `com_in_wrist_frame_m` before SONIC PPO.

## Not done

- Running GMR on BONES-SEED / writing `t800_motion_lib.pkl`
- SONIC PPO
- Left-hand USD overlay (no left palmar fit committed)
- Combined T800+Hand MJCF
- Grasp-success numbers
