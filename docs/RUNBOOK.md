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
make eval-l1a-spring    # SONIC Eq. 8 nav spring; grasp_success_rate stays null
make eval-l1a-stream    # SONIC §3.5 500 Hz PD stream; grasp_success_rate stays null
make eval-l1a-operator  # SONIC §3.5 100 Hz operator loop; grasp_success_rate stays null
make eval-l1a-gather    # SONIC §S7 YAML decoder gather (874-D)
make eval-l1a-encoder   # SONIC encoder 842-D default + 831-D low-latency step1 + 831-D v1.1 heading step5; no PICO / G1 ONNX
make eval-l1a-planner-onnx  # official planner V2 I/O, T800 32-D qpos; G1 36-D / planner_sonic.onnx refused
make eval-l1a-planner-blend  # 8-frame cross-fade + replan timer; does not run ONNX
make eval-l1a-idle-readapt   # idle ADAPTING/RECOVERING last-frame hold; does not run ONNX
make eval-l1a-playback       # 50 Hz current_frame clamp + idle hold after blend; does not run ONNX
make eval-l1a-shared-cursor  # encoder + planner context share playback.current_frame; does not run ONNX
make eval-l1a-decoder-tick   # decoder 874-D on the same 50 Hz tick (HardwareHold); does not run ONNX
make eval-l1a-last-action    # caller-supplied 25-D last_action on that tick; does not invent ONNX
make eval-l1a-policy-action  # delay this tick's 25-D policy_action into next-tick last_action; does not invent ONNX
make eval-l1a-pd-stream      # ZOH a_t onto the 500 Hz PD ring after stash; not Hermite / not ONNX
make eval-l1a-pd-plant       # 500 Hz τ = Kp(a_t − q) − Kd q̇; pd_stand bring-up, not SONIC tracking
make eval-l1a-pd-physics     # 10×2 ms physics substeps of that τ; decoder q stays pre-physics
make eval-l1a-pd-closedloop  # omit push_hw: next decoder q follows the plant; IMU kept, not invented
make eval-l1-case-a     # Case A FK (must pass --apply-fk); labelled head/nav fixture
make eval-l2            # refuses success rates while uncalibrated; records 9-cell gain scan + weld gate
make eval-l2-sim2sim    # kinematic MPJPE on T800 tracked bodies; grasp_success_rate stays null
make eval-l2-physics-sim2sim  # MuJoCo PD tracking; official XML or kinematics fixture; Table S4 joint/pos/ori + lin_vel/ang_vel walk-clip jitter
make eval-l2-freebase-stand   # free-base PD stand on a floor; fall is not a SONIC gate
make eval-l2-freebase-push    # lean vs fall + Table S4 ±X/±Y + ±Z one-shot + F=mv/T + angvel + τ=Iω/T + μ_slide + base CoM ipos + qpos ±0.01 rad + air-drop
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
