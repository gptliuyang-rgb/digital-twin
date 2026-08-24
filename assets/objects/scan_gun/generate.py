"""Build a lofted industrial barcode-scanner STL set (not vendor CAD).

Gun frame: optical +X, grip −Y. Units are metres.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from interface.schema import REPO_ROOT

OUT_DIR = REPO_ROOT / "assets" / "objects" / "scan_gun"
TIP_X = 0.128


def _rounded_rect(hy: float, hz: float, radius: float, n_arc: int = 5) -> np.ndarray:
    """Closed polyline in the YZ plane, CCW as seen along +X."""
    r = min(radius, hy * 0.95, hz * 0.95)
    pts: list[list[float]] = []
    corners = (
        (hy - r, hz - r, 0.0, 0.5 * np.pi),
        (-(hy - r), hz - r, 0.5 * np.pi, np.pi),
        (-(hy - r), -(hz - r), np.pi, 1.5 * np.pi),
        (hy - r, -(hz - r), 1.5 * np.pi, 2.0 * np.pi),
    )
    for cy, cz, a0, a1 in corners:
        for k in range(n_arc):
            ang = a0 + (a1 - a0) * k / n_arc
            pts.append([cy + r * np.cos(ang), cz + r * np.sin(ang)])
    return np.asarray(pts, dtype=np.float64)


def _slice_at(x: float, hy: float, hz: float, radius: float, y_off: float = 0.006) -> np.ndarray:
    yz = _rounded_rect(hy, hz, radius)
    out = np.zeros((len(yz), 3), dtype=np.float64)
    out[:, 0] = x
    out[:, 1] = yz[:, 0] + y_off
    out[:, 2] = yz[:, 1]
    return out


def _loft(slices: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rings = np.stack(slices, axis=0)
    s_count, n_count, _ = rings.shape
    verts = [pt.tolist() for ring in rings for pt in ring]
    faces: list[tuple[int, int, int]] = []
    for i in range(s_count - 1):
        for j in range(n_count):
            a = i * n_count + j
            b = i * n_count + (j + 1) % n_count
            c = (i + 1) * n_count + j
            d = (i + 1) * n_count + (j + 1) % n_count
            faces.append((a, c, b))
            faces.append((b, c, d))

    def cap(ring: np.ndarray, offset: int, flip: bool) -> None:
        ci = len(verts)
        verts.append(ring.mean(axis=0).tolist())
        for j in range(n_count):
            a = offset + j
            b = offset + (j + 1) % n_count
            faces.append((ci, b, a) if flip else (ci, a, b))

    cap(slices[0], 0, flip=True)
    cap(slices[-1], (s_count - 1) * n_count, flip=False)
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _ellipse_slice(origin: np.ndarray, tangent: np.ndarray, ry: float, rz: float, n: int = 20) -> np.ndarray:
    t = tangent / (np.linalg.norm(tangent) + 1e-12)
    helper = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    b1 = np.cross(t, helper)
    b1 /= np.linalg.norm(b1) + 1e-12
    b2 = np.cross(t, b1)
    ang = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return origin + np.outer(np.cos(ang), b1) * ry + np.outer(np.sin(ang), b2) * rz


def _write_stl(path: Path, verts: np.ndarray, faces: np.ndarray, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = name.encode("ascii", "replace")[:80].ljust(80, b"\0")
    n = int(len(faces))
    chunks = [header, struct.pack("<I", n)]
    for a, b, c in faces:
        p, q, r = verts[a], verts[b], verts[c]
        nrm = np.cross(q - p, r - p)
        ln = float(np.linalg.norm(nrm))
        if ln < 1e-18:
            nrm = np.array([0.0, 0.0, 1.0])
        else:
            nrm = nrm / ln
        chunks.append(struct.pack("<3f", *nrm))
        chunks.append(struct.pack("<3f", *p))
        chunks.append(struct.pack("<3f", *q))
        chunks.append(struct.pack("<3f", *r))
        chunks.append(struct.pack("<H", 0))
    path.write_bytes(b"".join(chunks))


def _head_slices() -> list[np.ndarray]:
    # (x, hy, hz, r) — lofted rounded-rect engine housing.
    spec = [
        (0.000, 0.024, 0.028, 0.008),
        (0.018, 0.027, 0.032, 0.010),
        (0.040, 0.029, 0.034, 0.011),
        (0.070, 0.028, 0.035, 0.011),
        (0.098, 0.027, 0.034, 0.010),
        (0.112, 0.026, 0.033, 0.008),
    ]
    return [_slice_at(x, hy, hz, r) for x, hy, hz, r in spec]


def _bumper_slices() -> list[np.ndarray]:
    spec = [
        (0.112, 0.027, 0.035, 0.007),
        (0.120, 0.028, 0.036, 0.006),
        (0.124, 0.026, 0.033, 0.005),
    ]
    return [_slice_at(x, hy, hz, r) for x, hy, hz, r in spec]


def _window_slices() -> list[np.ndarray]:
    spec = [
        (0.124, 0.020, 0.024, 0.004),
        (0.128, 0.018, 0.022, 0.003),
        (TIP_X, 0.016, 0.020, 0.002),
    ]
    return [_slice_at(x, hy, hz, r) for x, hy, hz, r in spec]


def _grip_slices() -> list[np.ndarray]:
    start = np.array([0.030, -0.012, 0.0])
    end = np.array([-0.012, -0.122, 0.0])
    slices = []
    n = 10
    for i in range(n):
        t = i / (n - 1)
        # Ease back so the grip is pistol-angled, not a sausage on +X.
        pos = (1 - t) * start + t * end
        pos[0] += -0.008 * t * t
        ry = 0.021 * (1.0 - 0.22 * t)
        rz = 0.017 * (1.0 - 0.18 * t)
        tan = end - start + np.array([-0.016 * t, 0.0, 0.0])
        slices.append(_ellipse_slice(pos, tan, ry, rz, n=18))
    return slices


def _guard_slices() -> list[np.ndarray]:
    # U-shaped trigger guard in the XY plane.
    path = np.array(
        [
            [0.022, -0.022, 0.0],
            [0.022, -0.048, 0.0],
            [0.034, -0.058, 0.0],
            [0.048, -0.058, 0.0],
            [0.058, -0.046, 0.0],
            [0.058, -0.024, 0.0],
        ]
    )
    slices = []
    for i, p in enumerate(path):
        if i == 0:
            tan = path[1] - path[0]
        elif i == len(path) - 1:
            tan = path[i] - path[i - 1]
        else:
            tan = path[i + 1] - path[i - 1]
        slices.append(_ellipse_slice(p, tan, 0.0055, 0.0045, n=12))
    return slices


def _cable_slices() -> list[np.ndarray]:
    start = np.array([-0.012, -0.122, 0.0])
    end = np.array([-0.048, -0.168, 0.0])
    slices = []
    for i in range(6):
        t = i / 5
        pos = (1 - t) * start + t * end
        r = 0.006 * (1.0 - 0.35 * t)
        slices.append(_ellipse_slice(pos, end - start, r, r, n=12))
    return slices


def _stripe_slices() -> list[np.ndarray]:
    spec = [
        (0.018, 0.008, 0.016, 0.003),
        (0.055, 0.007, 0.018, 0.003),
        (0.095, 0.006, 0.016, 0.003),
    ]
    return [_slice_at(x, hy, hz, r, y_off=0.038) for x, hy, hz, r in spec]


def generate(out_dir: Path | None = None) -> dict[str, Path]:
    dest = out_dir or OUT_DIR
    dest.mkdir(parents=True, exist_ok=True)
    body_parts = [
        _loft(_head_slices()),
        _loft(_grip_slices()),
        _loft(_guard_slices()),
        _loft(_cable_slices()),
    ]
    accent_parts = [
        _loft(_bumper_slices()),
        _loft(_stripe_slices()),
    ]
    window_parts = [_loft(_window_slices())]

    def _merge(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        verts = []
        faces = []
        offset = 0
        for v, f in parts:
            verts.append(v)
            faces.append(f + offset)
            offset += len(v)
        return np.vstack(verts), np.vstack(faces)

    paths = {
        "body": dest / "body.stl",
        "accent": dest / "accent.stl",
        "window": dest / "window.stl",
    }
    bv, bf = _merge(body_parts)
    av, af = _merge(accent_parts)
    wv, wf = _merge(window_parts)
    _write_stl(paths["body"], bv, bf, "scan_gun_body")
    _write_stl(paths["accent"], av, af, "scan_gun_accent")
    _write_stl(paths["window"], wv, wf, "scan_gun_window")
    (dest / "README.txt").write_text(
        "Generated lofted scanner STLs. Not vendor CAD. Optical +X, grip -Y.\n"
        f"Regenerate: python3 -m assets.objects.scan_gun.generate\n"
        f"tip_x={TIP_X}\n",
        encoding="utf-8",
    )
    return paths


def main() -> None:
    paths = generate()
    for key, path in paths.items():
        size = path.stat().st_size
        print(f"{key}: {path} ({size} bytes)")


if __name__ == "__main__":
    main()
