# This increment — Table S4 static-friction extrema

Continues ADR-030 (linear-z one-shot + F = m v / T). Still no grasp-success
number and no combined T800+Hand weld.

## Done

- **ADR-031.** Table S4 `physical.static_friction` is [0.3, 1.6]. The free-base
  diagnostic applies those extrema as MuJoCo `geom_friction[0]` (sliding) on
  the eval floor **and** `LINK_FOOT_*` collision geoms. MuJoCo contact friction
  is the element-wise max of the two geoms, so a floor-only write would be a
  no-op against 1.0 feet. Spin/roll stay the official collision default
  `(0.005, 0.0001)`. Dynamic friction and restitution stay recorded-only —
  MuJoCo has no separate μd, and mapping restitution onto `solref` would be
  invented. `mu_s_0.3` / `mu_s_1.6` are the case names; `friction_hold` is
  `mu_s_0.3`. Not mixed into `push_sweep`. Not pad–cardboard.
- `wbc/ppo/table_s4.py` — `physical_static_friction_range`,
  `static_friction_extrema`. No MuJoCo import.
- `T800MujocoEnv.set_wbc_slide_friction` — refuses a missing plane and
  `mu_slide <= 0`. Restores geom friction after the sweep.
- `make eval-l2-freebase-push` still one target. Default `hold` stays official
  μ_slide = 1.0. `local_tracking_success` always false. `grasp_success_rate`
  JSON `null`.

## Official XML measurement (this machine)

`python3 -m eval.l2_freebase_push --source official` with Native SDK
`serial_t800.xml`. Subtree mass of `LINK_BASE` is **84.917 kg**. WBC slide
geoms: `floor` (world) + three collision boxes on `LINK_FOOT_L` + three on
`LINK_FOOT_R` (7 geoms). Official unnamed foot geoms compile as `geom_13..15`
and `geom_26..28`. Spin/roll stay `(0.005, 0.0001)`.

3 s bring-up PD hold:

| μ_slide | posture | fallen | pelvis z | tilt | time_to_fall |
|---|---|---|---|---|---|
| 1.0 (official default `hold`) | **leaned** | false | 1.03→0.864 m | 0.722 rad (~41°) | — |
| 0.3 (`friction_hold`) | **upright** | false | 1.013 m | 0.023 rad (max 0.179) | — |
| 1.6 | **fallen** | true | →0.108 m | 1.586 rad (max 1.614) | 2.568 s |

Low sliding friction lets the feet slip; the 41° sticky lean disappears and
the pose stays upright. High sliding friction locks the feet harder and the
same bring-up PD falls at 2.57 s. That is a diagnostic of the PD-hold contact
regime, **not** a SONIC fail, and **not** an E1 pad–cardboard result.

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
- Table S4 `dynamic_friction`, `restitution`, `default_joint_pos_offset_rad`,
  `base_com_offset_m` (recorded, unswept)
