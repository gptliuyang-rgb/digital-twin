# This increment — MuJoCo physics Sim2Sim, 5-point PPO overlay, Isaac reset/step

Continues kinematic Sim2Sim / PPO recipe (ADR-022). Still no grasp-success number and no combined T800+Hand weld.

## Done

- `sim/mujoco_env/t800_env.py` — body-only T800 MuJoCo env. Official `serial_t800.xml` via `MjSpec` (pin = delete freejoint). Kinematics fixture from `t800_kinematics.yaml` when the SDK is missing. `include_hands=True` raises `PolicyEvalBlocked`.
- Official MJCF vs URDF: `LINK_FOOT_*` is 64.53 mm higher in MJCF (`t800_frame_offsets.yaml`). Wrists match. Reported as `foot_urdf_mjcf_delta_z_m`, not averaged away.
- `wbc/ppo/network_5point.yaml` — overlay: `vr_5point`, hybrid cmd 27, `action_dim` stays 25. `refuse_ppo_launch(teleop_mode=vr_5point)` still `PpoLaunchBlocked`.
- `sim/isaaclab_env/scene_spec.py` + `PrivilegedIsaacEnv.reset`/`step` — pallet+box USD spawn when `omni.usd` is running. Constructor without Isaac Lab still raises `IsaacLabUnavailable`.
- `wbc/pd_stand.py` — flatten EngineAI bring-up PD to 25-D. Not SONIC tracking gains.

## Tests

`tests/test_physics_sim2sim.py`, extra cases in `tests/test_isaaclab_privileged.py`.

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. Free-base physics without a trained SONIC policy will fall. BONES-SEED GMR, live actor T-pose, and real GR00T/π0.5 dumps stay blocked on flange SE(3) and wrist CoM.

## Not done

- Isaac Lab PPO launch (still refused)
- Combined T800+Hand MJCF
- Grasp-success numbers
- Official XML free-base standing without a learned tracker
