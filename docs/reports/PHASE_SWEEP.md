# This increment — Table S4 planar (±X / ±Y) root-push sweep

Continues ADR-026 (lean vs fall, single +Y push, air-drop). Still no
grasp-success number and no combined T800+Hand weld.

## Done

- **ADR-027.** Sweep Table S4 planar extrema (`±0.5` m/s on X and Y) as
  one-shot freejoint `qvel` after the same 3 s leaned hold. Z (`±0.2` m/s)
  is recorded in `wbc/ppo/domain_rand.yaml` and excluded from this planar
  list: a vertical impulse is a different diagnostic (now ADR-030).
- `wbc/ppo/table_s4.py` — single source of extrema. No MuJoCo import.
- `eval/l2_freebase_push.py` — `push_sweep` (4 cases) plus `lateral_push`
  kept as the `+y` compatibility key.
- `make eval-l2-freebase-push` still one target. `local_tracking_success`
  always false. `grasp_success_rate` JSON `null`.

## Official XML measurement (this machine)

`make eval-l2-freebase-push --source official` with Native SDK `serial_t800.xml`:

| case | posture | fallen | notes |
|---|---|---|---|
| 3 s PD hold | **leaned** | false | pelvis 1.03→0.864 m, tilt 0.722 rad (~41°), ncon=2 |
| −X 0.5 m/s after that hold | **fallen** | true | `pre_push=leaned`; time_to_fall 3.056 s; pelvis →0.108 m, tilt 1.58 rad |
| +X 0.5 m/s | **fallen** | true | same 56 ms after inject; max tilt 1.61 rad |
| −Y 0.5 m/s | **fallen** | true | same 56 ms after inject |
| +Y 0.5 m/s (`lateral_push`) | **fallen** | true | matches ADR-026 +Y case |
| air-drop 0.4 s, no floor | n/a | n/a | `freejoint_moved=true`, drop 0.781 m, n_plane=0 |

All four planar extrema fall 56 ms after inject. The 0.722 rad lean has only
0.078 rad of margin under the 0.80 rad fall band; bring-up PD does not recover
from any Table S4 planar extremum. That is **not** a SONIC fail, and it
falsifies the idea that +Y was a special surviving/failing axis.

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
- Sustained-force (1–3 s) Table S4 push — see ADR-028 / `docs/reports/PHASE_FORCE.md`
