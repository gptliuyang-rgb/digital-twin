# External resources

Read official pages before editing `dexhand2_spec.yaml`. Local clones: `scripts/bootstrap_resources.sh`.

## Wuji Hand 2 (current docs = Beta 2)

- Overview: https://docs.wuji.tech/docs/en/wuji-hand/latest/overview/
- Usage constraints: https://docs.wuji.tech/docs/en/wuji-hand/latest/usage-constraints/
- SDK: https://docs.wuji.tech/docs/en/wuji-hand/latest/sdk-reference/
- Hand changelog (firmware/SDK/description): https://docs.wuji.tech/docs/en/wuji-hand/latest/release-notes/
- Description integration: https://docs.wuji.tech/docs/zh/wuji-description/latest/integration/
- Models: https://github.com/wuji-technology/wuji-description
  - Beta 2 body: `hand2/hand2_beta2/body/`
  - Beta 1 body: `hand2/hand2_beta1/body/` (kept for drift comparison)
- Retargeting: https://github.com/wuji-technology/wuji-retargeting
- Description release notes: https://docs.wuji.tech/docs/zh/release-notes/

Facts taken from those pages on **2026-09-01** (do not re-ask):

- 20 independent revolute DoF, serial direct-drive, no coupling
- Product curb mass 800 g (bare 650 g, base 70 g separately), soft + cables, excl. base
- MIT hybrid @ 1 kHz, 11–13 V DC (nominal 12 V), RJ45 100BASE-TX
- Factory-standard wrist flange + palm-mount STEP/PDF in `hand2/hand2_beta2/attachment/`
- Beta 2 sim includes fingertip pad links (measured mass, collide, F/T-sensor frames). Whole-hand soft body is not in sim.
- Tactile: 40 pts thumb / 34 others, 100 Hz; fw ≥ v2.4.0 per-point force is normalized — decode `FingertipSensorInfo.format`
- Gains still gen-1 carry-over. Load curves are not a committed spec.

## EngineAI T800

- Native SDK (URDF + MuJoCo): https://github.com/engineai-robotics/engineai_robotics_native_sdk
- Gym: https://github.com/engineai-robotics/engineai_gym

## SONIC / VLA (not trained in this phase)

- https://github.com/NVlabs/GR00T-WholeBodyControl
- https://github.com/NVIDIA/Isaac-GR00T
- https://github.com/Physical-Intelligence/openpi

## Papers

- SONIC: https://arxiv.org/abs/2511.07820
- GMR: https://arxiv.org/abs/2510.02252
- 6D rotation: Zhou et al., CVPR 2019
