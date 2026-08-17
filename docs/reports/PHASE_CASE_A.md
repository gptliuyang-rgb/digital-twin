# This increment — explicit Case A FK + 50 Hz chunk upsample

Continues L0 diagnose (ADR-019). Still no grasp-success number and no combined T800+Hand weld.

## Done

- `vla/adapters/case_a_layout.yaml` — frozen 50-D packing (left 5 + right 5 arm q, left 20 + right 20 fingers). Dims verified against `t800_joints.yaml` / `dexhand2_spec.yaml`.
- `vla/adapters/case_a.py` — `CaseAToCommandSchema.convert(apply_fk=True, head_nav=...)`. Keyword `apply_fk` is required; False raises. `HeadNavCommand.source` is required. Head/nav/pelvis/trigger are **not** invented from the 50-D vector.
- `PolicyClient` / `DeployPipeline` / π0.5 glue / L0 replay still refuse 50-D (no silent hook).
- `eval/l1_case_a.py` — `make eval-l1-case-a` (must pass `--apply-fk`). Uses a labelled synthetic head/nav fixture. Skips if T800 URDF is not cloned.
- `vla/adapters/upsample.py` — integer-factor upsample of 75-D chunks to 50 Hz. SLERP on Zhou 6D. Case A refused until convert.
- `wbc/planner.py` `interpolate_command_matrix` shared by L1a and the 50 Hz upsample.

## Tests

`tests/test_case_a.py`, `tests/test_upsample.py`. Planner tests still cover the 10 Hz path.

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. A real GR00T/π0.5 dump is still not in this repo. GMR on BONES-SEED and SONIC PPO stay blocked on flange SE(3) and wrist CoM. Isaac Lab `reset`/`step` not wired.

## Not done

- Running diagnose on a real weight file
- Live actor T-pose (q=0 overlay remains)
- Combined T800+Hand MJCF
- Grasp-success numbers
