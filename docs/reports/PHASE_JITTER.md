# This increment — Table S4 target-motion joint-jitter extrema

Continues ADR-033 (reset-qpos ±0.01 rad on the free-base hold). Still no
grasp-success number and no combined T800+Hand weld.

## Done

- **ADR-034.** Table S4 `target_motion.joint_jitter_rad` is [−0.1, 0.1] rad.
  The pinned-base physics Sim2Sim applies those extrema as a **uniform**
  additive offset on all 25 clip hinges (`dof_pos`), every frame. The clip
  root is not jittered: a pinned pelvis cannot follow pos/ori target jitter.
  This is not ADR-033 reset-qpos, not a per-joint 2^25 corner grid, not
  mixed into `push_sweep`, and not Hand 2 command latency. Restitution stays
  recorded-only (no non-invented `solref` map). `q_jit_+0.1` is the
  `joint_jitter` compatibility key.
- `wbc/ppo/table_s4.py` — `target_motion_joint_jitter_range_rad`,
  `target_motion_joint_jitter_extrema`, plus recorded pos/ori extrema
  helpers. No MuJoCo import.
- `T800MujocoEnv.assert_hinges_in_mjcf_range` — raises rather than clips.
- `make eval-l2-physics-sim2sim` still one target. Unjittered pinned PD
  tracking stays the CI gate. `local_tracking_success` on jittered cases is
  a diagnostic. `grasp_success_rate` JSON `null`.

## Official XML measurement (this machine)

`python3 -m eval.l2_physics_sim2sim --source official` path with Native SDK
`serial_t800.xml`. All 25 hinges are `jnt_limited`; ±0.1 rad about the
synthetic stand clip (amplitude 0.05 rad, 40 frames @ 50 Hz, root z = 1.03 m)
stays inside MJCF `range`. Gains are `pd_stand` bring-up, not SONIC.

Pinned-base PD tracking:

| clip | joint MAE | wrist err | FK consist (no feet) | local_tracking_success |
|---|---|---|---|---|
| unjittered (CI gate) | 0.0095 rad | 0.0191 m | 0.00034 m | true |
| `q_jit_-0.1` (−0.1 rad) | 0.0116 rad | 0.0280 m | 0.00030 m | true |
| `q_jit_+0.1` (+0.1 rad, `joint_jitter`) | 0.0126 rad | 0.0315 m | 0.00033 m | true |

Bring-up PD still tracks the ±0.1 rad clip under the pinned-base MAE gate
(0.15 rad). Wrist error rises from 1.9 cm to ~3 cm. That is a clip-reference
diagnostic, **not** a SONIC fail, and **not** a DexHand2 command-latency result.

`grasp_success_rate` JSON `null`. Combined robot stays `PolicyEvalBlocked`.
Isaac bind is unchanged (`runtime=unavailable` on this machine unless Sim
python is bound).

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. BONES-SEED GMR, live actor T-pose,
and real GR00T/π0.5 dumps stay blocked on flange SE(3) and wrist CoM.
Isaac Lab PPO launch still refused. Pad–cardboard μ remains `REQUIRED_INPUT`.

## Not done

- Isaac Lab PPO launch
- Combined T800+Hand MJCF
- Grasp-success numbers
- Live actor T-pose (q=0 overlay remains)
- Running `IsaacLabSceneRuntime` inside Isaac Sim python
- GMR on a tiny BONES-SEED clip (blocked on flange/CoM)
- Diagnose a real GR00T / π0.5 dump once provided
- Table S4 `dynamic_friction`, `restitution`
  (recorded; restitution has no non-invented MuJoCo map)
- Table S4 `target_motion` pos/ori/lin_vel/ang_vel jitter
  (recorded; pinned-base cannot follow a jittered root)
