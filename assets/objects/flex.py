"""2×2 split-flex cardboard approximation (large boxes only)."""

from __future__ import annotations

from assets.objects.boxes import BoxSpec


def split_flex_mjcf(box: BoxSpec, name: str = "box") -> str:
    """Four rigid sub-bodies + weak joints. Not a calibrated paper model.

    Used only when `box.split_flex` is true (max edge > 0.4 m). Contact
    parameters stay REQUIRED_INPUT; this XML is a geometry scaffold.
    """
    if not box.split_flex:
        sx, sy, sz = box.size_m
        return (
            f'<body name="{name}">\n'
            f'  <geom type="box" size="{sx/2:.4f} {sy/2:.4f} {sz/2:.4f}" mass="{box.mass_kg:.4f}"/>\n'
            f"</body>\n"
        )
    sx, sy, sz = box.size_m
    hx, hy, hz = sx / 4.0, sy / 4.0, sz / 2.0
    m = box.mass_kg / 4.0
    parts = []
    idx = 0
    for ix in (-1.0, 1.0):
        for iy in (-1.0, 1.0):
            parts.append(
                f'  <body name="{name}_q{idx}" pos="{ix * sx/4:.4f} {iy * sy/4:.4f} 0">\n'
                f'    <joint name="{name}_q{idx}_flex" type="slide" axis="0 0 1" '
                f'stiffness="80" damping="8" range="-0.01 0.01"/>\n'
                f'    <geom type="box" size="{hx:.4f} {hy:.4f} {hz:.4f}" mass="{m:.4f}"/>\n'
                f"  </body>"
            )
            idx += 1
    return f'<body name="{name}">\n' + "\n".join(parts) + "\n</body>\n"
