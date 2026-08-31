# Runbook

## Setup

```bash
python3 -m pip install -e ".[dev]"
./scripts/bootstrap_resources.sh   # official Hand 2 + T800 URDF
make test
make check-spec                    # expected to FAIL until REQUIRED_INPUT is filled
```

Optional: `pip install -e ".[sim]"` for MuJoCo, `".[vision]"` for OpenCV QR decode.

## L2a / L2b (relative physics)

```bash
make phase1-baseline          # official vs pad-sphere site distances
make eval-l2                  # μ×solref + gain corners + bimanual + kinematic industrial
make eval-l2-physics          # mj_step industrial (gravity-comp PD + constraint welds)
make eval-l2-gains            # full 3×3 kp/kv scan
```

E3 synthetic template (not a payload rating):

```bash
python3 -c "from hand.calibration.synthetic import write_synthetic_csvs, SKIN_ON_DIR; print(write_synthetic_csvs(SKIN_ON_DIR)['e3'])"
python3 -m hand.calibration.fit_params --e3 hand/calibration/results/synthetic_batch_v1.0/skin_on/e3.csv --out /tmp/e3_fragment.yaml
```

Do not copy E3 mass into `motor_max_torque_nm`.

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

### E1/E2 pipeline (synthetic dry-run first)

Live `dexhand2_spec.yaml` stays `REQUIRED_INPUT` until a **human accepts** a hardware
fragment. The dry-run proves the full chain end-to-end:

```
CSV → fit μ/k → fingertip radius (from STL) → proposed solref
  → overlay spec  →  pad friction/solref in derived MJCF
  → MuJoCo Coulomb pull (E1) + pad indent (E2) replay
  → L2 micro episode with calibrated contact (relative, no grasp_success_rate)
```

```bash
# Dry-run: synthetic sheets, both skin states, real STL tip radius
make calibrate-synthetic          # CSV → fragment → overlay → validate_sim
make calibrate-synthetic-l2       # same + MuJoCo micro episode via overlay

# Output (gitignored, do not commit):
#  hand/calibration/results/synthetic_batch_v1.0/generated/
#    fragment.yaml          — fit numbers + STL radius
#    overlay.yaml           — merged spec (live spec unchanged)
#    validate_sim.json      — e1_ok/e2_ok/ok/human_must_accept_solref
#    e1_e2_summary.json     — skin_on vs skin_off comparison
#    l2_overlay.json        — micro metrics (with --with-l2 / calibrate-synthetic-l2)
```

Interpreting the skin comparison (synthetic reference):

| field | skin_on | skin_off | Δ |
|---|---|---|---|
| μ_s | ~0.75 | ~0.58 | −22% (skin adds friction) |
| k (N/m) | ~2500 | ~3800 | +50% (skin softens pad) |
| solref_s | ~0.022 | ~0.018 | −18% |

These are synthetic ground-truth deltas. Hardware will differ. **Re-run if pad/skin batch changes.**

Hardware sheets (after collecting real CSVs):

```bash
# One-shot with --both-skins and STL radius:
python3 scripts/run_e1_e2_pipeline.py \
  --e1  hand/calibration/results/<batch>/skin_on/e1.csv \
  --e2  hand/calibration/results/<batch>/skin_on/e2.csv \
  --e1-off hand/calibration/results/<batch>/skin_off/e1.csv \
  --e2-off hand/calibration/results/<batch>/skin_off/e2.csv \
  --both-skins --fit-tip-radius --with-l2 \
  --out-dir hand/calibration/results/<batch>/generated

# Review generated/e1_e2_summary.json and generated/validate_sim.json.
# After human accepts solref:
python3 scripts/apply_calibration_fragment.py \
  --fragment hand/calibration/results/<batch>/generated/fragment.yaml \
  --out assets/dexhand2/meta/dexhand2_spec.yaml \
  --commit-live
make build-assets
```

`make eval-l2` still uses the live spec (`blocked_uncalibrated`) until contact
fields are filled. Overlay L2 — relative metrics only, still no `grasp_success_rate`:

```bash
make eval-l2-overlay
# → eval/report/generated/l2_overlay.json
#   status: ready, physics: ran_calibrated_contact, l2_2_calibrated_micro
```

## Eval

```bash
make sim        # simulation-only stack (no real robot): assemble, pads, record, L0–L2, QR envelope, replay
make sim-quick  # same without the L2 μ×solref scan; shorter episode
make eval-l0    # synthetic demo flags a swapped channel; pass --dataset for a recorded npz
make eval-l1    # limits + coupling; MuJoCo DLS IK + ncon when assets compile
make assemble   # T800 + both DexHand2, identity flange, MIT motors, compile check
make frame-report  # DexHand poses in LINK_BASE (YAML chain; live FK if trees cloned)
make eval-l2    # uncalibrated μ/stiffness scan + scripted industrial pipeline (needs MuJoCo)
make calibrate-synthetic  # E1/E2 CSV→fit→overlay→MuJoCo replay (does not patch live spec)
make eval-l2-overlay      # L2 against the synthetic overlay spec (still no grasp_success_rate)
make view-industrial   # MuJoCo GUI: physics industrial (PD + welds; needs display)
make view-industrial-kinematic  # mj_forward geometry FSM
```

### Industrial viewer tips

The viewer/GIF default is **physics playback**: `mj_step`, gravity-compensated arm PD,
and equality welds that snapshot the grasp/gun pose **after DexHand pad contact**
(constraint grasp, not E1/E2). The first phase **raises** the wrists so hanging
pads clear the bench (T800 has no wrist pitch). `--kinematic` is the L2.3 geometry
FSM (`mj_forward` + prop assists). `--no-welds` drops the carton under uncalibrated
contact (honesty path). Benches have legs to the floor; the scanner is a lofted STL
barcode-gun with a freejoint hull (`assets/objects/scan_gun/`, not vendor CAD).
This is not a validated Coulomb grasp (ADR-004/006/007).

```bash
make assemble
python3 scripts/view_industrial_twin.py --steps-per-phase 80 --substeps 24 --real-time
# Kinematic geometry playback:
python3 scripts/view_industrial_twin.py --kinematic --steps-per-phase 100 --real-time
# Headless GIF (physics, no window):
make render-industrial-gif
# → artifacts/industrial_demo.gif
# Scanner still (bench rest pose after a short settle):
python3 scripts/render_industrial_gif.py --skip-gif --gun-closeup artifacts/scan_gun_closeup.png
# Overlay L2 (contact numbers from synthetic E1/E2; still no grasp_success_rate):
make eval-l2-overlay
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
