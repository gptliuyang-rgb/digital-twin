# This increment — SONIC Eq. 8 root spring on L1a nav_cmd

Continues ADR-037 (Table S4 μd / restitution recorded-only). Still no
grasp-success number and no combined T800+Hand weld.

## Done

- **ADR-038.** SONIC §3.3 critically damped spring on pelvis *x*, pelvis *y*,
  and projected heading. YAML locks `5 ln 2` / `20 ln 2`, 1.0 s target
  horizon, and 6.0 m/s planar-speed clamp. Equation 8 as typeset is kept as
  `paper_eq8_as_typeset` (x(0)=x_T−x_0). Runtime `critically_damped` restores
  x(0)=x_0, ẋ(0)=v_0, equilibrium x_T. Heading uses the shortest-arc wrap.
  `KinematicPlanner.plan_nav_root` emits a 10 Hz root ref. Hands still bypass
  WBC. Not Table S4. Not pad–cardboard.
- Reverse 6 m/s → −6 m/s keyframe at 1.0 s is strictly between the ballistic
  target (−6 m) and the current-velocity ballistic (+6 m). Rest-at-target is
  identity. Rest-to-step has no overshoot.

## Why this is not a Table S4 sweep

Table S4 is closed (swept channels + recorded-only μd/restitution). The
spring is the *deployment* filter between `nav_cmd` and L1a, the missing
piece of the four-rate stack after the 10 Hz interpolator (ADR-018).

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
- SONIC's trained generative kinematic planner (this is only Eq. 8)
