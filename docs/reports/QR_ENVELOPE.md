# QR scan envelope (synthetic pinhole)

Not RTX / Isaac. A 4 cm QR is projected with `fx=600 px` and decoded with OpenCV.
Geometry gate: distance 5–30 cm, incidence ≤ 45°. Cells outside the gate are `·` even if a decoder would succeed.

`make eval-qr` regenerates `eval/report/generated/qr_envelope.md`.

| d\θ deg | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 |
|---|---|---|---|---|---|---|---|---|
| 0.03 m | · | · | · | · | · | · | · | · |
| 0.06 m | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | · |
| 0.10 m | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | · |
| 0.13 m | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | · |
| 0.16 m | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | · |
| 0.20 m | ✓ | ✓ | ✓ | ✓ | · | · | · | · |
| 0.23 m | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | · |
| 0.27 m | ✓ | · | ✓ | ✓ | · | · | · | · |
| 0.30 m | ✓ | ✓ | ✓ | · | · | · | · | · |
| 0.33 m | · | · | · | · | · | · | · | · |
| 0.37 m | · | · | · | · | · | · | · | · |
| 0.40 m | · | · | · | · | · | · | · | · |

success_rate=0.365  (35/96)

Implication for IBVS: the capture region is roughly **6–30 cm** and **≤ 40°**. VLA coarse approach to ±10 cm is inside this envelope; 50°+ incidence is a hard fail.
