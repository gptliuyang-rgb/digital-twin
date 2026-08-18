# This increment — DexHand2 1 kHz MIT ring, bypassing WBC

Continues ADR-057 (omit-`push_hw` closed-loop T800 decoder gather).
Still no grasp-success number and no combined T800+Hand weld.

## Done

- **ADR-058.** `command_schema_v1` `left_hand_q` / `right_hand_q` are
  ZOH-held onto a 1 kHz MIT ring for 20 substeps on each 50 Hz tick:
  - τ = kp (qd − q) + kd (0 − dq) + τ_ff (official MIT form).
  - Gains are official MJCF `sim_kp` / `sim_kv`, labelled
    `mjcf_gen1_carryover_not_hand2_sysid`. `hardware_kp` stays REQUIRED_INPUT.
  - τ is clipped to MJCF `sim_actuator_forcerange_nm` (sim-only, not a
    payload rating).
  - Lives in `hand/mit_ring.py` + `runtime/hand_bypass.py`. `wbc/` keeps
    `not_hand_mit_ring: true`. last_action stays T800 25-D.
  - 25-D / 29-D / 32-D / 45-D / 75-D vectors raise. 500 Hz dt on the
    hand ring raises. Hermite / finite-diff dq / invented latency /
    force-mode-without-E3 raise.
- `make eval-l1c-mit-ring` dumps 10 ticks × 20 substeps on an increment
  double. After 10 ticks, q[0] is 0.20 and last τ matches kp0 × (0.5 − 0.199).
  Hands bypass WBC. `grasp_success_rate` JSON `null`.

## Why this is not a G1 checkpoint and not a contact eval

Official Hand 2 is 1 kHz MIT hybrid and is **not** inside SONIC. This
repo still does **not** run ONNX and does **not** weld the hand onto
T800. The caller supplies `command_schema` hand q; the ring ZOH-holds
it. Body 500 Hz PD (ADR-054–057) is unchanged.

## Still blocked on humans

Same P0 list as `docs/SPEC_INTAKE.md`. Hardware kp/kd, command latency,
per-joint torque, pad–cardboard friction, and flange SE(3) stay
REQUIRED_INPUT. A real T800 decoder output is still missing — the
caller must supply `policy_action` on the body path.

## Not done

- Running a T800 encoder / planner / decoder ONNX (G1 weights are refused)
- Enabling the delay deque (command_latency_ms unfilled)
- Force mode / current-to-tau map
- Combined T800+Hand MJCF
- Grasp-success numbers
- Inventing IMU from body rates
- Writing hand q into last_action
