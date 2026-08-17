# This increment — Table S4 angular-velocity extrema

Continues ADR-028 (sustained linear force). Still no grasp-success
number and no combined T800+Hand weld.

## Done

- **ADR-029.** Table S4 root_push ang_vel is roll/pitch ±0.52 rad/s and
  yaw ±0.78 rad/s, with the same duration Δt ∼ [1, 3] s. One-shot
  `qvel[3:6]` applies Δω in the LINK_BASE body frame. Sustained torque
  spreads the same angular impulse as `τ = I ω / T`, where `I` is the
  3×3 freejoint angular block of `mj_fullM` at the inject pose — not a
  datasheet guess. World `xfrc_applied` torque is `R @ τ_body`.
- `wbc/ppo/table_s4.py` — `axis_aligned_angvel_extrema_rad_s`,
  `torque_nm_from_impulse`, `sustained_torque_cases`. No MuJoCo import.
- `T800MujocoEnv.apply_root_angvel` / `apply_root_torque_nm` /
  `root_ang_inertia_kgm2`. Pinned-base raises. `mj_fullM(model, data, dst)`
  is the MuJoCo 3.11 signature (`data.qM` is gone).
- `eval/l2_freebase_push.py` — `angvel_sweep` (6 cases) plus
  `angvel_push` kept as `+yaw`; `torque_sweep` (12 cases) plus
  `sustained_torque` kept as `+yaw_T1.0s`. Linear sweeps unchanged.
- `make eval-l2-freebase-push` still one target. `local_tracking_success`
  always false. `grasp_success_rate` JSON `null`.

## Official XML measurement (this machine)

`python3 -m eval.l2_freebase_push --source official` with Native SDK
`serial_t800.xml`. After the 3 s leaned hold, `mj_fullM` angular diag is
**[17.326, 15.521, 2.098] kg·m²** (roll, pitch, yaw). That is compiled
MJCF inertia at the inject pose, not a product-page guess. For |ω_yaw| =
0.78 rad/s:

| T | |τ_yaw| (body Z) |
|---|---|
| 1.0 s | 1.636 N·m |
| 3.0 s | 0.545 N·m |

Roll/pitch |ω| = 0.52 rad/s at T = 1 s gives |τ_roll| = 9.009 N·m and
|τ_pitch| = 8.071 N·m (off-diagonal coupling is small).

| case | posture | fallen | notes |
|---|---|---|---|
| 3 s PD hold | **leaned** | false | pelvis 1.03→0.864 m, tilt 0.722 rad (~41°) |
| one-shot ±roll 0.52 rad/s | **fallen** | true | time_to_fall 3.082 s (82 ms after inject) |
| −pitch 0.52 rad/s | **fallen** | true | 92 ms after inject |
| +pitch 0.52 rad/s | **fallen** | true | 72 ms after inject (already leaned in pitch) |
| ±yaw 0.78 rad/s (`angvel_push` = +yaw) | **fallen** | true | 82 ms after inject |
| all 12 τ = I ω / T extrema | **fallen** | true | 48–50 ms after inject; pelvis →0.108 m, tilt 1.58 rad |
| air-drop 0.4 s, no floor | n/a | n/a | `freejoint_moved=true`, drop 0.781 m |

One-shot angvel knocks the leaned pose in 72–92 ms (linear one-shot was
56 ms). Sustained torque still falls in 50 ms — the same 0.078 rad of
margin under the 0.80 rad fall band. Weaker 3 s torque does not save
bring-up PD. That is **not** a SONIC fail. It does show that Table S4
angvel is not a hidden “easy” axis relative to planar linvel on this
controller, and that +pitch (the lean axis) is the fastest one-shot.

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
- Table S4 linear-z (±0.2 m/s) root push
