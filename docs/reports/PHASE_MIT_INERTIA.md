# This increment — official-inertia 1 kHz MIT plant (mesh-free)

Continues ADR-059 (fixture MuJoCo HandMitPhysics). Still no grasp-success
number and no combined T800+Hand weld.

## Done

- **ADR-060.** Official Hand 2 MJCF is rewritten into a mesh-free MIT
  plant (`to_meshfree_mit`):
  - 20 `<position>` → `<motor gear=1>` so the MIT law owns the gains.
  - Official `timestep=0.002` is rewritten to `0.001`. Loading the
    unconverted XML is refused.
  - Meshes and `<contact>` excludes are stripped (no STL, not a
    pad–cardboard eval). CAD `<inertial>` tags are kept.
  - Gravity off. Skeleton mass 0.6207 kg.
  - `source=auto` selects `official_inertia` when cloned, **never**
    `official_position`.
  - Fixture inertias stay placeholders; treating them as CAD raises.
  - Soft-body CoM is still not in the plant.
- `make eval-l1c-mit-physics` dumps 10 ticks on official inertias when
  `wuji-description` is cloned (else the fixture). Without MuJoCo the
  eval reports `runtime=unavailable`.

## Why this is not a G1 checkpoint and not a contact eval

Official Hand 2 is 1 kHz MIT hybrid and is **not** inside SONIC. The
mesh-free plant does **not** collide pad spheres or cardboard.
`hardware_kp` stays REQUIRED_INPUT; this plant still uses MJCF gen-1
`sim_kp`/`sim_kv` labelled `mjcf_gen1_carryover_not_hand2_sysid`.

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
- Treating fixture inertias as CAD
- Putting the 0.1243 kg soft-body delta on the wrist (CoM unmeasured)
