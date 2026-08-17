# This increment — GMR T-pose overlay, left palmar pads, privileged Isaac cfg

Builds on GMR export / USD pads / privileged MuJoCo L2. Still no grasp-success number and no combined T800+Hand weld.

## Done

- `wbc/gmr/tpose.py` — q=0 MJCF FK (no MuJoCo import). Composes PM01 offsets with T800 vs PM01 body frames. Head uses the pelvis offset (GMR tienkung/hi) after measuring `R_head == R_pelvis` at q=0. Human scale is GMR PM01's table × measured hip–foot / shoulder–wrist length ratios.
- `wbc/gmr/tpose_offsets.yaml` — overlay consumed by `export.py`. `body_map.yaml` stays the PM01-copy source (ADR-015).
- Left-hand palmar spheres from `l_*_distal.STL` (not a Y-mirror of the right fit). USDA overlay for left is generated once the fit exists.
- `sim/isaaclab_env/privileged.py` — pallet+box cfg, `grasp_success_rate` always `null`, `IsaacLabUnavailable` if constructed without Isaac Lab (ADR-016).

## Tests

`make test`. T-pose composition tests do not need clones. Frame-delta / scale tests skip unless `third_party/engineai-native-sdk` and `third_party/GMR` are cloned. Left pad tests skip unless `wuji-description` is cloned.

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. A live BONES-SEED T-pose (actor + robot) is **not** this overlay. Flange SE(3) and `com_in_wrist_frame_m` still block SONIC PPO.

## Not done

- Running GMR on BONES-SEED / writing `t800_motion_lib.pkl`
- SONIC PPO
- Combined T800+Hand MJCF
- Grasp-success numbers
- Isaac Lab env `reset`/`step` (needs Isaac Sim python)
