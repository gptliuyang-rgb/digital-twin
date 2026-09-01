"""Convert official Hand 2 <position> actuators into MIT-style <motor> plants.

Official MJCF uses position actuators (kp/kv on the XML). Hardware is MIT
hybrid: τ = kp(qd−q) + kd(dqd−dq) + τ_ff. After conversion, kp/kv live in the
DexHand2Controller, not in the XML. Official files are never overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from interface.schema import REPO_ROOT

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None  # type: ignore[assignment]


@dataclass(frozen=True)
class MitGains:
    names: tuple[str, ...]
    kp: np.ndarray
    kd: np.ndarray
    tau_lim: np.ndarray


def extract_position_gains(spec: Any) -> MitGains:
    names = []
    kp = []
    kd = []
    tau = []
    for act in spec.actuators:
        names.append(act.name)
        kp.append(float(act.gainprm[0]))
        kd.append(float(-act.biasprm[2]))
        tau.append(float(abs(act.forcerange[1])))
    return MitGains(
        names=tuple(names),
        kp=np.asarray(kp, dtype=np.float64),
        kd=np.asarray(kd, dtype=np.float64),
        tau_lim=np.asarray(tau, dtype=np.float64),
    )


def convert_spec_to_motors(spec: Any) -> MitGains:
    """In-place: position actuators → unit-gain motors with force ctrlrange.

    T800 <motor> actuators (bias none) are left unchanged.
    """
    if mujoco is None:
        raise ImportError("mujoco is required to convert Hand 2 plants")
    names = []
    kp = []
    kd = []
    tau = []
    for act in spec.actuators:
        if int(act.biastype) != int(mujoco.mjtBias.mjBIAS_AFFINE):
            continue
        names.append(act.name)
        kp.append(float(act.gainprm[0]))
        kd.append(float(-act.biasprm[2]))
        tau.append(float(abs(act.forcerange[1])))
        lo, hi = float(act.forcerange[0]), float(act.forcerange[1])
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm[0] = 1.0
        act.biastype = mujoco.mjtBias.mjBIAS_NONE
        act.biasprm[:] = 0.0
        act.dyntype = mujoco.mjtDyn.mjDYN_NONE
        act.ctrlrange = (lo, hi)
        act.ctrllimited = True
        act.forcelimited = True
        act.forcerange = (lo, hi)
    return MitGains(
        names=tuple(names),
        kp=np.asarray(kp, dtype=np.float64),
        kd=np.asarray(kd, dtype=np.float64),
        tau_lim=np.asarray(tau, dtype=np.float64),
    )


def official_hand_xml(side: str) -> Any:
    if mujoco is None:
        raise ImportError("mujoco is required")
    from assets.dexhand2.build.ingest_official import official_mjcf

    path = official_mjcf(side, with_mount=True)
    if not path.is_file():
        raise FileNotFoundError(path)
    return mujoco.MjSpec.from_file(str(path))


def dump_gains_sidecar(gains: MitGains, side: str) -> None:
    out = REPO_ROOT / "assets" / "dexhand2" / "derived" / f"{side}_mit_gains.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, names=np.array(gains.names), kp=gains.kp, kd=gains.kd, tau_lim=gains.tau_lim)
