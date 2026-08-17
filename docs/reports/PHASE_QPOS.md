# This increment — Table S4 default joint-pos offset extrema

Continues ADR-032 (LINK_BASE ipos CoM offset). Still no grasp-success
number and no combined T800+Hand weld.

## Done

- **ADR-033.** Table S4 `physical.default_joint_pos_offset_rad` is
  [−0.01, 0.01] rad. The free-base diagnostic applies those extrema as a
  **uniform** additive offset on all 25 actuated hinges: reset qpos *and*
  PD target. The freejoint is not offset. This is not a per-joint 2^25
  corner grid, not Hand 2 command latency, and not mixed into
  `push_sweep`. Restitution stays recorded-only (no non-invented `solref`
  map). `qpos_+0.01` is the `joint_offset` compatibility key.
- `wbc/ppo/table_s4.py` — `physical_default_joint_pos_offset_range_rad`,
  `default_joint_pos_offset_extrema`. No MuJoCo import.
- `T800MujocoEnv.offset_default_joint_pos` — adds to bring-up `q_des`,
  checks MJCF joint `range`, raises rather than clips.
- `make eval-l2-freebase-push` still one target. Default `hold` stays the
  un-offset bring-up pose. `local_tracking_success` always false.
  `grasp_success_rate` JSON `null`.

## Official XML measurement (this machine)

`python3 -m eval.l2_freebase_push --source official` path with Native SDK
`serial_t800.xml`. All 25 hinges are `jnt_limited`; ±0.01 rad about the
bring-up `q_des` stays inside MJCF `range`.

3 s bring-up PD hold:

| offset | posture | fallen | pelvis z | tilt | time_to_fall |
|---|---|---|---|---|---|
| un-offset (official default `hold`) | **leaned** | false | 1.03→0.864 m | 0.722 rad (~41°) | — |
| `qpos_-0.01` (−0.01 rad) | **fallen** | true | →0.109 m | 1.584 rad (max 1.623) | 1.782 s |
| `qpos_+0.01` (+0.01 rad, `joint_offset`) | **upright** | false | 1.013 m | 0.005 rad (max 0.035) | — |

The sticky 41° lean sits on a pose knife-edge: a uniform +0.01 rad on every
hinge lets the same bring-up PD stay upright; −0.01 rad makes it fall at
1.782 s. That is a reset-qpos diagnostic, **not** a SONIC fail, and **not**
a DexHand2 command-latency result.

`grasp_success_rate` JSON `null`. `local_tracking_success=false`.
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
- Table S4 `target_motion` jitter (recorded; not a root-push / physical hold)
