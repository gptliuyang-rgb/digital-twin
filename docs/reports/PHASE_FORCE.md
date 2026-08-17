# This increment — Table S4 sustained-force duration extrema

Continues ADR-027 (planar one-shot `qvel` sweep). Still no grasp-success
number and no combined T800+Hand weld.

## Done

- **ADR-028.** Table S4 push duration is Δt ∼ [1, 3] s. Spread the same
  linear impulse as the one-shot Δv as a constant world force
  `F = m v / T` on `LINK_BASE` for duration extrema 1.0 s and 3.0 s.
  Mass is the compiled MJCF subtree mass, not a guessed datasheet number.
- `wbc/ppo/table_s4.py` — `force_n_from_impulse`, `sustained_force_cases`.
  No MuJoCo import.
- `T800MujocoEnv.apply_root_force_n` / `clear_root_force` / `root_subtree_mass_kg`.
  Pinned-base raises.
- `eval/l2_freebase_push.py` — `force_sweep` (8 cases) plus `sustained_force`
  kept as the `+y_T1.0s` compatibility key. One-shot `push_sweep` unchanged.
- `make eval-l2-freebase-push` still one target. `local_tracking_success`
  always false. `grasp_success_rate` JSON `null`.

## Official XML measurement (this machine)

`python3 -m eval.l2_freebase_push --source official` with Native SDK
`serial_t800.xml`. Subtree mass of `LINK_BASE` is **84.917 kg** (from the
compiled MJCF, not a product-page guess). For |v| = 0.5 m/s:

| T | |F| |
|---|---|
| 1.0 s | 42.459 N |
| 3.0 s | 14.153 N |

| case | posture | fallen | notes |
|---|---|---|---|
| 3 s PD hold | **leaned** | false | pelvis 1.03→0.864 m, tilt 0.722 rad (~41°) |
| one-shot ±X/±Y 0.5 m/s (ADR-027) | **fallen** | true | time_to_fall 3.056 s (56 ms after inject) |
| −X T=1 s / T=3 s | **fallen** | true | pre_push=leaned; time_to_fall 3.050 s; pelvis →0.108 m, tilt 1.58 rad |
| +X T=1 s / T=3 s | **fallen** | true | same 50 ms after inject |
| −Y T=1 s / T=3 s | **fallen** | true | same 50 ms after inject |
| +Y T=1 s (`sustained_force`) | **fallen** | true | F = [0, 42.459, 0] N |
| +Y T=3 s | **fallen** | true | F = [0, 14.153, 0] N |
| air-drop 0.4 s, no floor | n/a | n/a | `freejoint_moved=true`, drop 0.781 m |

All eight duration×axis extrema fall 50 ms after inject — the same leaned
pose that one-shot knocks over in 56 ms. The 3 s (weaker) force does not
save bring-up PD: 0.722 rad already has only 0.078 rad of margin under the
0.80 rad fall band. That is **not** a SONIC fail. It does show that Table S4
duration is not a hidden “easy” axis relative to one-shot `qvel` on this
controller.

`grasp_success_rate` JSON `null`. `local_tracking_success=false`.
Isaac bind is unchanged (`runtime=unavailable` on this machine).

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. BONES-SEED GMR, live actor T-pose,
and real GR00T/π0.5 dumps stay blocked on flange SE(3) and wrist CoM.
Isaac Lab PPO launch still refused.

## Not done

- Isaac Lab PPO launch
- Combined T800+Hand MJCF
- Grasp-success numbers
- Live actor T-pose (q=0 overlay remains)
- Running `IsaacLabSceneRuntime` inside Isaac Sim python
- Angular-velocity Table S4 push (roll/pitch ±0.52 rad/s, yaw ±0.78 rad/s)
