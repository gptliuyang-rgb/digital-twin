# This increment — DexHand2 1 kHz MuJoCo HandMitPhysics backend

Continues ADR-058 (numpy MIT ring, bypassing WBC last_action).
Still no grasp-success number and no combined T800+Hand weld.

## Done

- **ADR-059.** `sim/mujoco_env/hand_mit_physics.py` implements
  `HandMitPhysics` on a hand-only MuJoCo plant:
  - 20-D motor actuators, `dt = 1/1000` s.
  - Fixture XML is a CI stand-in (placeholder mass, gravity off, no
    contact). Official `*_mit.xml` is optional after `make build-assets`.
  - Official `<position>` XML is refused — MIT τ must own the gains.
  - 25-D / 29-D / 32-D / 45-D / 75-D τ raises. 500 Hz `dt` raises.
  - `include_t800=True` raises `PolicyEvalBlocked`.
  - Lives in `sim/`. `hand/mit_ring.py` / `runtime/` / `wbc/` stay
    numpy-only.
- `make eval-l1c-mit-physics` dumps 10 ticks × 20 substeps on the
  fixture. After 10 ticks with `q_des[0]=0.5`, measured `q[0]` moves
  toward the hold (not the numpy increment double). Without MuJoCo the
  eval reports `runtime=unavailable`.

## Why this is not a G1 checkpoint and not a contact eval

Official Hand 2 is 1 kHz MIT hybrid and is **not** inside SONIC. The
fixture does **not** use CAD inertias, pad spheres, or cardboard
friction. `hardware_kp` stays REQUIRED_INPUT; this plant still uses
MJCF gen-1 `sim_kp`/`sim_kv` labelled
`mjcf_gen1_carryover_not_hand2_sysid`. Body 500 Hz PD (ADR-054–057)
is unchanged.

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. Hardware kp/kd, command latency,
per-joint torque, pad–cardboard friction, wrist CoM, and flange SE(3)
stay REQUIRED_INPUT. A real T800 decoder output is still missing — the
caller must supply `policy_action` on the body path.

## Not done

- Running a T800 encoder / planner / decoder ONNX (G1 weights are refused)
- Enabling the delay deque (`command_latency_ms` unfilled)
- Force mode / current-to-tau map
- Combined T800+Hand MJCF
- Grasp-success numbers
- Inventing IMU from body rates
- Writing hand q into last_action
- Resampling the 1 kHz ring onto 0.002 s
