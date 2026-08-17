# This increment — Table S4 target-motion pos/ori jitter extrema

Continues ADR-034 (clip `dof_pos` ±0.1 rad). Still no grasp-success number
and no combined T800+Hand weld.

## Done

- **ADR-035.** Table S4 `target_motion.pos_jitter_m` is ±xy 0.05 m / z 0.01 m
  and `ori_jitter_rad` is ±roll/pitch 0.1 rad / yaw ±0.2 rad. The pinned-base
  physics Sim2Sim applies those extrema as an additive offset on clip
  `root_pos` or a world-frame RPY on clip `root_rot`. Clip `dof_pos` is not
  changed. The robot stays pinned: this is a **negative control** (joint MAE
  stays; MPJPE / pelvis ori error vs the jittered root move by about the
  jitter). The 0.25 m / 1.0 rad height/ori bands swallow these paper ranges,
  so `local_tracking_success` is **not** a gate (`not_a_height_ori_gate`).
  Not mixed into `joint_jitter_sweep` or `push_sweep`. Not ADR-034 joint
  jitter, not a root_push, not Hand 2 command latency. Restitution stays
  recorded-only (no non-invented `solref` map). `pos_+x` / `ori_+yaw` are
  the compatibility keys.
- `apply_clip_pos_jitter` / `apply_clip_ori_jitter` — clip root only.
- `make eval-l2-physics-sim2sim` still one target. Unjittered pinned PD
  tracking stays the CI gate. `grasp_success_rate` JSON `null`.

## Official XML measurement (this machine)

`python3 -m eval.l2_physics_sim2sim --source official` path with Native SDK
`serial_t800.xml`. Gains are `pd_stand` bring-up, not SONIC. Synthetic stand
clip: amplitude 0.05 rad, 40 frames @ 50 Hz, root z = 1.03 m.

Pinned-base PD tracking (negative control):

| clip | joint MAE | MPJPE | root pos err | pelvis ori err | local_tracking_success |
|---|---|---|---|---|---|
| unjittered (CI gate) | 0.0095 rad | 0.0307 m | 0.000 m | 0.000 rad | true |
| `pos_+x` (+0.05 m, `pos_jitter`) | 0.0095 rad | 0.0609 m | **0.050 m** | 0.000 rad | true |
| `pos_-x` | 0.0095 rad | 0.0644 m | 0.050 m | 0.000 rad | true |
| `pos_+y` / `pos_-y` | 0.0095 rad | 0.0646 / 0.0632 m | 0.050 m | 0.000 rad | true |
| `pos_+z` / `pos_-z` | 0.0095 rad | 0.0321 / 0.0382 m | **0.010 m** | 0.000 rad | true |
| `ori_+yaw` (+0.2 rad, `ori_jitter`) | 0.0095 rad | 0.0451 m | 0.000 m | **0.200 rad** | true |
| `ori_-yaw` | 0.0095 rad | 0.0437 m | 0.000 m | 0.200 rad | true |
| `ori_+roll` / `ori_-roll` | 0.0095 rad | 0.0620 / 0.0610 m | 0.000 m | **0.100 rad** | true |
| `ori_+pitch` / `ori_-pitch` | 0.0095 rad | 0.0585 / 0.0547 m | 0.000 m | 0.100 rad | true |

Joint MAE is identical to unjittered at 0.0095 rad on every pos/ori case.
Root pos error matches the published ±0.05 / ±0.01 m. Pelvis ori error
matches the published ±0.1 / ±0.2 rad. `local_tracking_success` stays true
with zero height/ori fail frames — the 0.25 m / 1.0 rad bands swallow the
paper range, which is why this sweep is **not** a pass/fail against those
gates.

That is a clip-root diagnostic, **not** a SONIC fail, and **not** a
DexHand2 command-latency result.

`grasp_success_rate` JSON `null`. Combined robot stays `PolicyEvalBlocked`.
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
- Table S4 `dynamic_friction`, `restitution`
  (recorded; restitution has no non-invented MuJoCo map)
