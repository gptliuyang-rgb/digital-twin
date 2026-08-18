# This increment — Table S4 target-motion lin_vel/ang_vel jitter extrema

Continues ADR-035 (clip `root_pos` / `root_rot`). Still no grasp-success
number and no combined T800+Hand weld.

## Done

- **ADR-036.** Table S4 `target_motion.lin_vel_jitter_mps` is ±xy 0.5 m/s /
  z 0.2 m/s and `ang_vel_jitter_rad_s` is ±roll/pitch 0.52 rad/s / yaw
  ±0.78 rad/s. The pinned-base physics Sim2Sim applies those extrema as an
  additive offset on clip `root_linvel` / `root_angvel` of a **synthetic
  walk clip**. A stand clip is refused (velocity identically zero would make
  jitter degenerate). Clip pose and `dof_pos` match the stand clip so joint
  MAE stays a negative control. Not mixed into `push_sweep`, pos/ori jitter,
  or joint jitter. Not a root_push (robot `qvel`). Not a 0.25 m / 1.0 rad
  height/ori gate. Restitution stays recorded-only. `lin_vel_+x` /
  `ang_vel_+yaw` are the compatibility keys.
- `synthetic_walk_clip` — stand pose + labelled carrier
  `(0.35, 0, 0)` m/s and `(0, 0, 0.25)` rad/s. Not a T800 gait. Pose is not
  integrated from the carrier.
- `apply_clip_lin_vel_jitter` / `apply_clip_ang_vel_jitter`.
- `make eval-l2-physics-sim2sim` still one target. Unjittered pinned PD
  tracking on the stand clip stays the CI gate. `grasp_success_rate` JSON
  `null`.

## Fixture XML measurement (this machine)

Native SDK `serial_t800.xml` is not cloned here. `python3 -m eval.l2_physics_sim2sim --source fixture` uses the kinematics fixture (placeholder 1 kg / 0.01 kg·m², mild PD). Gains are **not** SONIC. Walk-clip carrier is 0.35 m/s +x and 0.25 rad/s yaw. 40 frames @ 50 Hz.

Pinned-base PD tracking (negative control on the walk clip):

| clip | joint MAE | target linvel (x,y,z) | target angvel (r,p,y) | root linvel err |
|---|---|---|---|---|
| unjittered stand (CI gate) | 0.0192 rad | ~0 (no field) | ~0 | ~0 |
| `lin_vel_+x` (+0.5 m/s) | **0.0192 rad** | **(0.85, 0, 0)** m/s | (0, 0, 0.25) rad/s | 0.85 m/s |
| `lin_vel_-x` | 0.0192 rad | (−0.15, 0, 0) m/s | (0, 0, 0.25) | 0.15 m/s |
| `lin_vel_+y` / `lin_vel_-y` | 0.0192 rad | (0.35, ±0.5, 0) | (0, 0, 0.25) | 0.610 m/s |
| `lin_vel_+z` / `lin_vel_-z` | 0.0192 rad | (0.35, 0, ±0.2) | (0, 0, 0.25) | 0.403 m/s |
| `ang_vel_+yaw` (+0.78 rad/s) | **0.0192 rad** | (0.35, 0, 0) | **(0, 0, 1.03)** rad/s | 0.35 m/s |
| `ang_vel_-yaw` | 0.0192 rad | (0.35, 0, 0) | (0, 0, −0.53) | 0.35 m/s |
| `ang_vel_+roll` / `±pitch` | 0.0192 rad | (0.35, 0, 0) | (±0.52 on that axis, 0.25 yaw) | 0.35 m/s |

Joint MAE is **identical** to unjittered stand (0.019166… rad) on every lin_vel and ang_vel case. −X / −yaw do **not** cancel through zero. `local_tracking_success` stays true — this sweep is **not** a pass/fail against 0.25 m / 1.0 rad.

That is a clip-velocity diagnostic, **not** a SONIC fail, **not** a root_push, and **not** a DexHand2 command-latency result.

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
- Table S4 `dynamic_friction`, `restitution`
  (recorded; restitution has no non-invented MuJoCo map)
