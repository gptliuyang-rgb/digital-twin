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

Ingest now checks **both** `hand2_beta1` and `hand2_beta2`. Default sim revision is Beta 2.

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
make eval-l2    # refuses success rates while uncalibrated
```

## What this repo will not do yet

- Train SONIC on T800
- Download GR00T / π0.5 weights
- Claim a box-pick success rate
- Weld Hand 2 onto T800 (`assets/combined/assemble.py` exits until mount SE(3) is filled)
