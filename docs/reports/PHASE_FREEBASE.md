# This increment — free-base PD stand and MJCF foot-frame lock

Continues physics Sim2Sim (ADR-023). Still no grasp-success number and no combined T800+Hand weld.

## Done

- **ADR-024.** GMR/SONIC tracked feet are official MJCF `LINK_FOOT_*` (coincident with `LINK_ANKLE_ROLL_*`). URDF `J_FIXED_FOOT_*` z=−0.06453 m is a sole offset, not a second tracked body. MJCF collision boxes at z=−0.054 m are recorded and are not the SONIC body.
- `wbc/foot_frame.py` — `urdf_sole_world_m`, `assert_foot_frame`, refuse URDF sole as a SONIC body.
- EngineAI `pd_stand/default.yaml` `desired_joint_position` copied into `wbc/t800_sonic.yaml` (rad, 25-D).
- `eval/l2_freebase_stand.py` — floating base + floor plane. Official XML has a freejoint and no plane; the eval adds the official collision-default floor. `local_tracking_success` is always false. Fall is reported; standing is not a SONIC pass (ADR-025).
- `sim/isaaclab_env/probe.py` — `isaac_runtime_status()` without importing Isaac at package load. CI is `unavailable`.

## Official XML measurement (this machine, 3 s hold)

`make eval-l2-freebase-stand --source official` with Native SDK `serial_t800.xml`:

- `nq=32`, `n_plane=1`, `gains_source=pd_stand_bringup_not_sonic`
- start pelvis z = 1.03 m; end = 0.864 m; max tilt = 0.72 rad (~41°)
- `fallen=false` at the 0.40 m / 0.80 rad thresholds; `local_tracking_success=false`
- start MJCF `LINK_FOOT` z = 80.0 mm; URDF sole z = 15.5 mm (Δz = 64.5 mm, ADR-024)
- end `ncon=2` (both feet contacting the added floor)
- `grasp_success_rate` JSON `null`

Bring-up PD settled onto the feet and leaned. That is **not** a SONIC tracking pass.

`tests/test_foot_frame.py`, `tests/test_freebase_stand.py`. Fixture free-base in free air (no floor) drops vs pinned (proves the freejoint). On a floor the placeholder boxes may hold — that is not a stand rating. Official XML, when cloned, reports MJCF-foot vs URDF-sole Δz ≈ 64.53 mm and does **not** gate on fallen vs stood.

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. BONES-SEED GMR, live actor T-pose, and real GR00T/π0.5 dumps stay blocked on flange SE(3) and wrist CoM. Isaac Lab PPO launch still refused.

## Not done

- Isaac Lab PPO launch
- Combined T800+Hand MJCF
- Grasp-success numbers
- Live actor T-pose (q=0 overlay remains)
- Running `IsaacLabSceneRuntime` inside Isaac Sim python (probe reports unavailable here)

## Follow-on

`make eval-l2-freebase-push` labels the 41° hold as **leaned** (ADR-026), sweeps Table S4 planar (±X/±Y) one-shot `qvel` (ADR-027), and spreads the same impulse as F = m v / T for 1 s and 3 s (ADR-028). None of these is a SONIC gate.
