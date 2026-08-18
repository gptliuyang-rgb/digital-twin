# This increment — Table S4 linear-z extrema

Continues ADR-029 (angvel one-shot + τ = I ω / T). Still no grasp-success
number and no combined T800+Hand weld.

## Done

- **ADR-030.** Table S4 root_push lin_vel z is ±0.2 m/s — not the planar
  ±0.5 m/s. One-shot world `qvel[0:3]` applies Δv. Sustained force spreads
  the same linear impulse as `F = m v / T` for duration extrema 1 s and
  3 s. Mass is compiled MJCF `body_subtreemass[LINK_BASE]`, not a datasheet
  guess. Planar `push_sweep` stays 4 cases; `force_sweep` stays 8.
  `+z` / `+z_T1.0s` are the compatibility keys.
- `wbc/ppo/table_s4.py` — `VERTICAL_AXES`, `vertical_linvel_extrema_mps`,
  `vertical_force_cases`. Default `SWEEP_AXES` remains `("x", "y")`.
  No MuJoCo import.
- `eval/l2_freebase_push.py` — `vertical_sweep` (2 cases) plus
  `vertical_push` kept as `+z`; `vertical_force_sweep` (4 cases) plus
  `sustained_vertical` kept as `+z_T1.0s`. Planar and angvel lists unchanged.
- `make eval-l2-freebase-push` still one target. `local_tracking_success`
  always false. `grasp_success_rate` JSON `null`.

## Official XML measurement (this machine)

`python3 -m eval.l2_freebase_push --source official` with Native SDK
`serial_t800.xml`. Subtree mass of `LINK_BASE` is **84.917 kg** (same as
ADR-028). For |v_z| = 0.2 m/s:

| T | |F_z| |
|---|---|
| 1.0 s | 16.983 N |
| 3.0 s | 5.661 N |

| case | posture | fallen | notes |
|---|---|---|---|
| 3 s PD hold | **leaned** | false | pelvis 1.03→0.864 m, tilt 0.722 rad (~41°) |
| one-shot −z 0.2 m/s | **fallen** | true | 56 ms after inject (into the floor) |
| one-shot +z 0.2 m/s (`vertical_push`) | **fallen** | true | 56 ms after inject (lift; still falls) |
| all 4 F = m v / T z extrema | **fallen** | true | 50 ms after inject; pelvis →0.108 m, tilt 1.58 rad |
| air-drop 0.4 s, no floor | n/a | n/a | `freejoint_moved=true`, drop 0.781 m |

One-shot z knocks the leaned pose in 56 ms — the same clock as planar
±0.5 m/s (ADR-027). Sustained |F_z| is 2.5× weaker than planar |F_y|
(0.2 vs 0.5 m/s) and still falls in 50 ms. +Z does not catch the 41°
lean. That is **not** a SONIC fail. It does show that Table S4 z is not
a hidden “easy” axis on this bring-up PD.

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
- GMR on a tiny BONES-SEED clip (blocked on flange/CoM)
- Diagnose a real GR00T / π0.5 dump once provided
