# Contact calibration protocol (E1–E3)

Beta 1 soft body and skin are **not locked**. Record `soft_body_batch_id`,
`skin_batch_id`, `skin_present`, and `firmware_version` on every sheet.
Re-run E1/E2 after any pad or skin change (`docs/RUNBOOK.md`).

Do not use these results as a published payload rating (Wuji usage constraints).

## Common setup

| Item | Spec |
|---|---|
| Room | 20–25 °C, dry |
| Cardboard | Same lot as the warehouse boxes (record GSM / flute) |
| Hand fixture | Wrist flange bolted to a rigid stand, palm down or as specified |
| Logging | CSV columns listed per experiment; 200 Hz or faster |
| Photos | Pad, skin on/off, box face |

---

## E1 — Incline slip (μ_s, μ_d)

**Goal:** fingertip–cardboard static and dynamic friction.

**Gear:** inclinable plate, digital inclinometer (±0.1°), test box panel, 1 kg
calibration mass, video at 60 fps.

**Steps:**

1. Mount one distal pad on the plate with a known normal force (start 5 N, then 10 N).
2. Increase incline at ≤ 1°/s until first slip. Record `theta_s_deg`.
3. After slip, record steady sliding speed and `theta_d_deg` if the pad re-sticks / kinetic regime is visible.
4. Repeat n=10 per {skin on, skin off} × {pad index, middle, thumb}.
5. `mu_s = tan(theta_s)`. `mu_d` from kinetic angle or force-sensor pull if available.

**CSV:** `trial,finger,skin,normal_n,theta_s_deg,theta_d_deg,mu_s,mu_d,batch,fw,notes`

---

## E2 — Pad compression (normal stiffness)

**Goal:** `normal_stiffness_n_per_m` → MuJoCo `solref`/`solimp`.

**Gear:** linear stage + load cell (0–50 N), displacement ±0.01 mm.

**Steps:**

1. Indent the pad along the site normal to 2 mm, 1 mm/s, hold 2 s, retract.
2. Fit F = k x on the 0.2–1.0 mm window (avoid the first-contact toe).
3. Repeat n=5 per finger, skin on and off.

**CSV:** `trial,finger,skin,disp_m,force_n,k_n_per_m,batch,fw`

Either fill `k_n_per_m` per trial, or log the `disp_m`/`force_n` series (multiple
rows per trial). `fit_params.py` fits `F = k x` on the 0.2–1.0 mm window when
`k_n_per_m` is empty.

**MuJoCo mapping (proposal, must be accepted by a human):**
`solref[0] ≈ 2π / sqrt(k / m_eff)` with default `m_eff = 0.03 kg` (pad + fixture,
not a measured CoM). The fitter writes `k` and a **proposed** `solref_timeconst_s`
into a spec *fragment*. `validate_sim.py` replays a Coulomb pull (E1) and a pad
indent (E2). Do **not** copy the fragment into live `dexhand2_spec.yaml` until a
human accepts the conversion. Synthetic dry-run:

```bash
make calibrate-synthetic
# → overlay spec under hand/calibration/results/synthetic_batch_v1.0/generated/
# live assets/dexhand2/meta/dexhand2_spec.yaml is unchanged
```

---

## E3 — Grasp limit (effective torque / max mass)

**Goal:** largest cardboard mass held 5 s in `power_grasp` without slip.

**Gear:** instrumented box 2–20 kg in 1 kg steps, overhead hoist, e-stop.

**Steps:**

1. Command `power_grasp` closure 0.7, 0.85, 1.0.
2. Raise the box 5 cm, hold 5 s, lower.
3. Stop at first slip or thermal warning. Do **not** run long full-load tests
   (Wuji thermal constraint).
4. Record peak joint current from `joint_diagnostics`.

**CSV:** `trial,closure,mass_kg,held_s,slipped,max_current_a,finger,batch,fw`

---

## Skin on vs off

Run E1 and E2 in both states. Store under
`hand/calibration/results/<batch>_<fw>/skin_on` and `skin_off`.
