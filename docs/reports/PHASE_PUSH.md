# This increment — lean vs fall, lateral root-push, air-drop, Isaac bind entry

Continues ADR-025 free-base PD stand. Still no grasp-success number and no combined T800+Hand weld.

## Done

- **ADR-026.** Posture is `upright | leaned | fallen`. The official 3 s hold at ~0.72 rad (~41°) is **leaned**, not a stand. Fall bands stay 0.40 m / 0.80 rad. Lean band is 0.20 rad (diagnostic, not CAD).
- `eval/lean_classify.py` — pure function, no MuJoCo.
- `T800MujocoEnv.apply_root_linvel` — one-shot freejoint world linear velocity. Pinned-base raises.
- `eval/l2_freebase_push.py` — hold + Table S4 +Y 0.5 m/s one-shot `qvel` injected **after** the 3 s leaned hold + no-floor air-drop. `local_tracking_success` always false.
- `eval/l3_isaac_bind.py` — calls `IsaacLabSceneRuntime.reset/step` only when Isaac Sim python is bound; otherwise `isaac_bind_unavailable`.
- `make eval-l2-freebase-push`, `make eval-l3-isaac-bind`.

## Official XML measurement (this machine)

`make eval-l2-freebase-push` with Native SDK `serial_t800.xml`:

| case | posture | fallen | notes |
|---|---|---|---|
| 3 s PD hold | **leaned** | false | pelvis 1.03→0.864 m, tilt 0.722 rad (~41°), ncon=2 |
| +Y 0.5 m/s one-shot after that hold | **fallen** | true | `pre_push_posture=leaned`; time_to_fall 3.056 s; pelvis →0.108 m, tilt 1.58 rad. Bring-up PD is not a recovery controller |
| air-drop 0.4 s, no floor | n/a | n/a | `freejoint_moved=true`, drop 0.781 m, n_plane=0 |

`grasp_success_rate` JSON `null`. `local_tracking_success=false`. Isaac bind: `runtime=unavailable`, `reset_step_ran=false`.

A 41° lean is not a stand. A push that then exceeds the 0.80 rad / 0.40 m fall bands is **not** a SONIC fail.

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. BONES-SEED GMR, live actor T-pose, and real GR00T/π0.5 dumps stay blocked on flange SE(3) and wrist CoM. Isaac Lab PPO launch still refused.

## Not done

- Isaac Lab PPO launch
- Combined T800+Hand MJCF
- Grasp-success numbers
- Live actor T-pose (q=0 overlay remains)
- Running `IsaacLabSceneRuntime` inside Isaac Sim python on this machine (bind reports unavailable)
