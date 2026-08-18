# This increment — Table S4 dynamic_friction / restitution stay recorded-only

Continues ADR-036 (walk-clip lin_vel/ang_vel jitter). Still no grasp-success
number and no combined T800+Hand weld.

## Done

- **ADR-037.** Table S4 `physical.dynamic_friction` is [0.3, 1.2] and
  `physical.restitution` is [0.0, 0.5]. Both still load from
  `wbc/ppo/domain_rand.yaml`. Neither is written onto a MuJoCo geom.
  MuJoCo has one sliding coefficient (`geom_friction[0]`); PhysX μd has no
  non-invented map. PhysX restitution has no non-invented `solref`/`solimp`
  map. `recorded_only_physical` in YAML is `[dynamic_friction, restitution]`.
  `refuse_recorded_only_physical_map` always raises. Not mixed into
  `push_sweep` or the static-friction hold. Not pad–cardboard.
- `T800MujocoEnv.set_wbc_slide_friction` still writes only `geom_friction[gid, 0]`
  and now asserts `geom_solref`, `geom_solimp`, and `geom_friction[:, 1:]`
  are unchanged. `set_wbc_dynamic_friction` / `set_wbc_restitution` raise.
- Source scan of `sim/`, `eval/`, `wbc/` forbids assignments to `geom_solref`
  / `geom_solimp` and to `geom_friction` except slide index 0 or a full-array
  restore. `make eval-l2-freebase-push` is unchanged as a target.
  `grasp_success_rate` JSON `null`.

## Why this is not a sweep

A later agent mapping `restitution: [0.0, 0.5]` onto `solref` timeconst, or
writing μd onto `geom_friction[:, 1]` (torsional), would look like a Table S4
completion and silently change contact. This increment is the lock that
prevents that map. There is still no restitution diagnostic and no E1
pad–cardboard result.

`grasp_success_rate` JSON `null`. Combined robot stays `PolicyEvalBlocked`.

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
- Re-measure official XML MAE for the walk-clip sweep when Native SDK is cloned
