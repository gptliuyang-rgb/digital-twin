# PHASE pad-spheres — 2026-10-01 monthly

Official docs still default to Hand 2.2 / `hand2_beta2`. Description v2026.9.11
did **not** change Hand 2 MJCF/URDF (hashes match the 2026-09-01 pin). This
phase does the increment that pin forbade inventing last month: replace the
official pad convex hull with a mesh-fit sphere collision for batch sim.

## Done

- Retargeted `main` contracts to Beta 2 (same facts as draft PR #49: mass
  0.800 / 0.6228 kg, 5 pad bodies, tactile 40/34, 11–13 V).
- `gen_derived.py` **disables** group-2 pad hulls and injects 3 PCA spheres
  (`condim=4`) in the pad body frame. Distal-biased last station.
- `simplified` variant also disables distal-bone hulls (fingertip contact =
  spheres only). Wrist / phalanx hulls stay.
- Right-hand site-to-sphere-surface **1.79 mm** thumb / **0.70 mm** other fingers
  (gate 2 mm). Left thumb 1.65 mm.
- Collision count: pad_spheres 36 (21 hull + 15 spheres); simplified 31
  (16 hull + 15 spheres). Budget 60.
- Firmware v2.10.0: default port **50001** (was 7447), configurable
  `hand.port()`, joint FF compensation only on units shipping from 2026-10.
  `enabled_on_this_unit` is REQUIRED_INPUT. `factory_tcp_endpoint()` is the
  single `ip:port` source.
- Product overview fetched 2026-10-01 is a **non-tactile 2.2 prototype**.
  `hardware_has_tactile` is still REQUIRED_INPUT.

## Tests

`make test` with cloned `wuji-description` HEAD `c2cd7f8`. Ingest of both
revisions green. `validate_repo_spec()` still raises `SpecIncompleteError`.

## Not done (still blocked on humans)

Friction, stiffness, hardware torque, hardware kp/kd, command latency, wrist
CoM, T800 flange SE(3), safety rate limits, live pad radius, physical-unit
revision, tactile-or-not, FF-compensation-or-not.

No grasp-success number (ADR-004).
