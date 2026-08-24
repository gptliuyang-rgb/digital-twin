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
make eval-l0    # works without a ckpt (synthetic demo flags a swapped channel)
make eval-l1    # limits + coupling; IK skipped without Pinocchio
make assemble   # T800 + both DexHand2, identity flange, MIT motors, compile check
make eval-l2    # uncalibrated μ/stiffness scan + scripted industrial pipeline (needs MuJoCo)
make view-industrial   # MuJoCo GUI: pallet, boxes, gun, full FSM playback (needs display)
```

### Industrial viewer tips

While contact is uncalibrated, the pipeline uses **kinematic demo playback**
(`mj_forward` only — no contact physics): the carton tracks the wrists during
carry/stack, and the scan gun mocap sticks to the right hand during scan.
Benches have legs to the floor; the scanner is a placeholder pistol mesh, not
vendor CAD. This is not a validated grasp (ADR-004/006/007).

```bash
make assemble
python3 scripts/view_industrial_twin.py --steps-per-phase 100 --real-time --sync-every 8
# Headless GIF (no window):
make render-industrial-gif
# → artifacts/industrial_demo.gif
```

- On **Wayland**, GLFW may warn about window position; run the viewer once per session.
  A second back-to-back run can segfault on EGL teardown — quit fully before re-launching.
- Headless VMs: `sudo apt install xvfb` then
  `xvfb-run -a python3 scripts/view_industrial_twin.py --steps-per-phase 80` (no window).
- If pytest picks up ROS plugins: `unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_DISTRO`
  and/or `python -m pytest tests -p no:launch_testing`.

`make eval-l2` still sets `status: blocked_uncalibrated` and never writes a computed `grasp_success_rate`.

## What this repo will not do yet

- Train SONIC on T800
- Download GR00T / π0.5 weights
- Claim a box-pick success rate
- Treat the identity T800↔Hand weld as CAD (`kinematic_bringup_identity` is sim-only; ADR-006)
- Real-robot / HIL experiments
