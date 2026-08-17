# This increment — Table S4 base CoM offset extrema

Continues ADR-031 (WBC floor+foot sliding friction). Still no grasp-success
number and no combined T800+Hand weld.

## Done

- **ADR-032.** Table S4 `physical.base_com_offset_m` is x ±0.075 m and
  y/z ±0.1 m. The free-base diagnostic applies those extrema as an
  **additive** offset on compiled MuJoCo `body_ipos[LINK_BASE]`. That is the
  pelvis body's own inertial frame, not a redistribution of every link, and
  not DexHand2 `com_in_wrist_frame_m`. Restitution stays recorded-only (no
  non-invented `solref` map). `default_joint_pos_offset_rad` stays unswept.
  `com_+x` is the `com_offset` compatibility key. Not mixed into `push_sweep`.
- `wbc/ppo/table_s4.py` — `physical_base_com_offset_ranges_m`,
  `base_com_offset_extrema`. No MuJoCo import.
- `T800MujocoEnv.set_base_com_offset` — adds to compiled ipos, reports
  own-body mass vs subtree mass. `restore_base_ipos` returns to compile time.
- `make eval-l2-freebase-push` still one target. Default `hold` stays the
  compiled ipos. `local_tracking_success` always false. `grasp_success_rate`
  JSON `null`.

## Official XML measurement (this machine)

`python3 -m eval.l2_freebase_push --source official` with Native SDK
`serial_t800.xml`. Compiled `LINK_BASE` inertial origin is
**[0.01487, −0.00012, 0.16732] m**. Own-body mass **10.083 kg**; subtree mass
**84.917 kg** (unchanged by ipos). An ipos offset only moves that 10.083 kg;
the rest of the tree stays where the joints put it.

3 s bring-up PD hold:

| offset | posture | fallen | pelvis z | tilt | time_to_fall |
|---|---|---|---|---|---|
| compiled (official default `hold`) | **leaned** | false | 1.03→0.864 m | 0.722 rad (~41°) | — |
| `com_-x` (−0.075 m) | **upright** | false | 1.013 m | 0.005 rad (max 0.129) | — |
| `com_+x` (+0.075 m, `com_offset`) | **fallen** | true | →0.108 m | 1.583 rad (max 1.612) | 2.064 s |
| `com_-y` (−0.1 m) | **leaned** | false | 0.837 m | 0.774 rad | — |
| `com_+y` (+0.1 m) | **leaned** | false | 0.839 m | 0.771 rad | — |
| `com_-z` (−0.1 m) | **upright** | false | 1.012 m | 0.049 rad (max 0.184) | — |
| `com_+z` (+0.1 m) | **fallen** | true | →0.426 m | 1.293 rad | 2.776 s |

The compiled pelvis CoM already sits +0.167 m along body Z. Pushing it further
forward (+X) or up (+Z) makes the 41° sticky lean fall. Pulling it back (−X)
or down (−Z) lets the same bring-up PD stay upright. Lateral ±Y stays leaned.
That is a diagnostic of the PD-hold mass distribution, **not** a SONIC fail,
and **not** a DexHand2 wrist hang-test.

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
- Table S4 `dynamic_friction`, `restitution`, `default_joint_pos_offset_rad`
  (recorded; restitution has no non-invented MuJoCo map)
