# This increment — L0 ckpt action-space diagnostic + T800 decoder history

Continues the 5-point / L1a / gain-scan stack. Still no grasp-success number and no combined T800+Hand weld.

## Done

- `vla/adapters/action_space.py` — classify a ckpt last-dim as case A (arm q + fingers, 50-D), B (`command_schema_v1` 75-D), or C (velocity/delta via modality). Frozen layout table also names 21/27 SONIC tokens, 81-D 5-point, 25-D T800 `a_t`, and 29-D G1 (refused).
- `eval/l0_ckpt_diagnose.py` — `make eval-l0-diagnose`. No weights required. Unknown dims exit 2. Catalog mode if nothing is passed.
- `PolicyClient` and L0 replay refuse a non-75-D vector with the diagnosis instead of a reshape error.
- `vla/adapters/pi05_glue.py` — pass-through only when D=75. π0.5 is not native to SONIC (ADR-019).
- `vla/chunk_clock.yaml` — labelled typicals (GR00T 10 Hz × 16, π0.5 5 Hz × 50 → SONIC 50 Hz). Non-integer upsample is refused.
- `wbc/observation.py` `ProprioHistory` — token + 10×(ω, q, dq, a, g) = 874 on T800. G1 29-DoF construction raises.
- `wbc/gmr/synthetic_clip.py` — standing 25-DoF `motion_lib` for schema+filter tests. Not BONES-SEED.
- `wbc/load_rand.py` — 0–20 kg wrist payload sampling raises `PolicyEvalBlocked` until `com_in_wrist_frame_m` and flange SE(3) are filled.

## Tests

`make test` includes `tests/test_l0_diagnose.py`. Official clones are optional except where existing ingest tests skip/run.

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. Diagnosing a ckpt does not make SONIC PPO legal. A case-A ckpt still needs FK + heading-frame convert before L1. A case-C ckpt needs a retrain.

## Not done

- Running GMR on a real BONES-SEED clip
- Loading a real GR00T / π0.5 weight file (none in this repo)
- Isaac Lab `reset`/`step`
- Combined T800+Hand MJCF
- Grasp-success numbers
