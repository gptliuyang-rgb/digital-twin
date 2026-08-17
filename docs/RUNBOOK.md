# Runbook

## Setup

```bash
python3 -m pip install -e ".[dev]"
./scripts/bootstrap_resources.sh   # official Hand 2 + T800 URDF
make test
make check-spec                    # expected to FAIL until REQUIRED_INPUT is filled
```

Optional: `pip install -e ".[sim]"` for MuJoCo, `".[vision]"` for OpenCV QR decode.

## After filling a spec field

1. Edit `assets/dexhand2/meta/dexhand2_spec.yaml` (or mount / calib YAML).
2. `make test && make check-spec`
3. If contact params changed: regenerate derived MJCF `make build-assets` and re-run Phase 1 ingest.

## After Wuji upstream model change

```bash
python3 scripts/check_upstream_drift.py
# if it fails: inspect CHANGELOG, re-run ingest, update upstream_pin.json, re-fit E1/E2
```

## After pad / skin swap

Re-run E1 and E2 (`hand/calibration/PROTOCOL.md`). Store under
`hand/calibration/results/<batch>_<fw>/`. Do not mix batches in one spec file.

## Eval

```bash
make eval-l0            # works without a ckpt (synthetic demo flags a swapped channel)
make eval-l0-diagnose   # classify action last-dim (A/B/C); no weights required
make eval-l1            # limits + coupling; IK skipped without Pinocchio
make eval-l1-case-a     # Case A FK (must pass --apply-fk); labelled head/nav fixture
make eval-l2            # refuses success rates while uncalibrated; records 9-cell gain scan + weld gate
make eval-l2-sim2sim    # kinematic MPJPE on T800 tracked bodies; grasp_success_rate stays null
make eval-l2-physics-sim2sim  # MuJoCo PD tracking; official XML or kinematics fixture
make eval-l2-freebase-stand   # free-base PD stand on a floor; fall is not a SONIC gate
make eval-l2-freebase-push    # lean vs fall + Table S4 ±X/±Y one-shot + F=mv/T + angvel + τ=Iω/T + air-drop
make eval-l3-isaac-bind       # IsaacLabSceneRuntime.reset/step if bound; else unavailable
make ppo-status         # frozen Table S1–S4 recipe; action_dim 25
make ppo-train          # exits non-zero until SPEC_INTAKE P0 + Isaac Lab
make extract-kinematics # official URDF → assets/engineai/meta/t800_kinematics.yaml
make eval-qr    # synthetic pinhole QR envelope; decode is real, renderer is not RTX
make eval-report
make weld-recipe
make sonic-status          # T800 decoder dim vs G1; retarget blockers
make gmr-export            # write smplx_to_t800.json / bvh_lafan1_to_t800.json from body_map.yaml
make usd-pads              # USDA overlay of palmar pad spheres (right hand)
make eval-l2-priv          # privileged pallet drop; grasp_success_rate stays null
make ingest-official && make build-assets && make check-drift
```

IBVS for scan lives in `runtime/ibvs.py` (shared). Combined T800+hand policy eval is refused until mount SE(3) is CAD-measured.

## What this repo will not do yet

- Train SONIC on T800 (`wbc/ppo/` freezes Table S1–S4; `make ppo-train` refuses; G1 checkpoints are refused; quat offsets still need a live T-pose)
- Download GR00T / π0.5 weights
- Claim a box-pick success rate
- Weld Hand 2 onto T800 (`assets/combined/assemble.py` exits until mount SE(3) is filled)
