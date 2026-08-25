# L2b — physics task loop (this increment)

## Done

- Feature path: `make eval-l2-physics` → `l2_3_physics_industrial` with **kinematic_assist=false**, `mj_step` + MIT + arm PD
- Kinematic demo (L2.3) unchanged (ADR-007); it now attempts `gun_cam` RGB decode when GL works (`scan_decode_ok` may be null)
- Real QR texture on the carton plate (`assets/objects/generated/qr_BOX-0.png`, gitignored, generated at scene attach)
- `assets/objects/flex.py`: 2×2 slide-joint cardboard tiles when max edge > 0.4 m (default industrial box is 0.18 m, still rigid)
- `sim/mujoco_env/bimanual_box.py`: dual-hand squeeze micro, relative slip/drop/contacts
- Stack process fields on the physics run: release speed, gap_z, XY alignment (no success rate)

## Honest limits

- Physics industrial will usually drop the box. That is the point of turning assists off.
- Scan decode in physics is whatever the unheld mocap gun actually sees.
- Still **forbidden** to publish `grasp_success_rate` (ADR-004).
- Identity flange still `policy_eval_forbidden` (ADR-006).

## How to run

```bash
make test
make eval-l2                 # μ×solref + gain corners + bimanual + kinematic industrial
make eval-l2-physics         # also mj_step industrial
make phase1-baseline         # pad distance markdown + JSON
```
