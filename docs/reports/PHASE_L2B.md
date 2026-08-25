# L2b — physics task loop (this increment)

## Done

- Feature path: `make eval-l2-physics` → `l2_3_physics_industrial` with **kinematic_assist=false**, `mj_step` + MIT + gravity-compensated arm PD (`qfrc_bias`) + optional payload `Jᵀmg` + constraint welds
- Viewer/GIF default to that physics path; `--kinematic` keeps L2.3 geometry playback
- Scan gun is a freejoint body with collision hulls (rests on the pick bench); not mocap
- Kinematic demo (L2.3) still used for `scan_geometry_ok`; it attempts `gun_cam` RGB decode when GL works (`scan_decode_ok` may be null)
- Real QR texture on the carton plate (`assets/objects/generated/qr_BOX-0.png`, gitignored, generated at scene attach)
- `assets/objects/flex.py`: 2×2 slide-joint cardboard tiles when max edge > 0.4 m (default industrial box is 0.18 m, still rigid)
- `sim/mujoco_env/bimanual_box.py`: dual-hand squeeze micro, relative slip/drop/contacts
- Stack process fields on the physics run: release speed, gap_z, XY alignment, `constraint_weld`, `max_abs_tau_nm` (no success rate)

## Honest limits

- Welds are **constraint grasps**, not E1/E2 Coulomb. `--no-welds` / `use_welds=false` is the honesty path (box expected to drop).
- Scan decode in physics is whatever the held/welded gun actually sees.
- Still **forbidden** to publish `grasp_success_rate` (ADR-004).
- Identity flange still `policy_eval_forbidden` (ADR-006).

## How to run

```bash
make test
make eval-l2                 # μ×solref + gain corners + bimanual + kinematic industrial
make eval-l2-physics         # also mj_step industrial (welds on)
make view-industrial         # physics viewer
make phase1-baseline         # pad distance markdown + JSON
```
