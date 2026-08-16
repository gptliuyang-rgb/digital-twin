# PHASE 1 baseline — official Wuji Hand 2 Beta 1 (right)

Source: `third_party/wuji-description/hand2/hand2_beta1/body/mjcf/right.xml`
SHA256: `ee860215e6ae0b2f6ded7fa69d3b7ced36ab6d8a72aace99d20259aa78dc1742`

## Counts

| Item | Value |
|---|---|
| Actuators | 20 |
| Joints | 20 |
| Fingertip sites | 5 |
| Skeleton mass (URDF sum) | 0.6207 kg |
| Contact excludes | 10 |
| Tip STL files on disk | 5 |
| Pad spheres in official MJCF | 0 |

## Fingertip sites

- `r_thumb_tip` pos_m=[0.0, 0.0, -0.02978]
- `r_index_finger_tip` pos_m=[0.0, 0.0, -0.02475]
- `r_middle_finger_tip` pos_m=[0.0, 0.0, -0.02475]
- `r_ring_finger_tip` pos_m=[0.0, 0.0, -0.02475]
- `r_pinky_tip` pos_m=[0.0, 0.0, -0.02475]

## Official gaps (unchanged by ingest)

- Tip STLs are **not** collision geometry (`tip_stl_used_as_collision=False`).
- Collision geoms are per-link convex hulls.
- Drive kp/kv are gen-1 carry-over.

Ingest ok: **True**. Mismatches: none.
