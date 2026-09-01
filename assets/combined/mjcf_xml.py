"""XML helpers for inlining MuJoCo includes and rewriting mesh paths.

MjSpec.attach is unstable on MuJoCo 3.2.x (segfaults). Combined models are
built by expanding <include> and nesting official Hand 2 worldbody trees.
Official files are never overwritten.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path


def load_mjcf(path: Path) -> ET.Element:
    path = path.resolve()
    root = ET.parse(path).getroot()
    _absolutize_assets(root, path.parent)
    _expand_includes(root, path.parent)
    return root


def _expand_includes(elem: ET.Element, basedir: Path) -> None:
    i = 0
    while i < len(list(elem)):
        child = list(elem)[i]
        if child.tag == "include":
            inc_path = (basedir / child.get("file", "")).resolve()
            if not inc_path.is_file():
                raise FileNotFoundError(inc_path)
            inc_root = ET.parse(inc_path).getroot()
            _absolutize_assets(inc_root, inc_path.parent)
            _expand_includes(inc_root, inc_path.parent)
            subs = list(inc_root)
            elem.remove(child)
            for j, sub in enumerate(subs):
                elem.insert(i + j, sub)
            i += len(subs)
        else:
            _expand_includes(child, basedir)
            i += 1


def _absolutize_assets(root: ET.Element, basedir: Path) -> None:
    compiler = root.find("compiler")
    meshdir = basedir
    texturedir = basedir
    if compiler is not None:
        if compiler.get("meshdir"):
            meshdir = (basedir / compiler.get("meshdir")).resolve()
        if compiler.get("texturedir"):
            texturedir = (basedir / compiler.get("texturedir")).resolve()
    for mesh in root.iter("mesh"):
        _abs_file(mesh, meshdir)
    for tex in root.iter("texture"):
        _abs_file(tex, texturedir)
    for hfield in root.iter("hfield"):
        _abs_file(hfield, meshdir)


def _abs_file(elem: ET.Element, root: Path) -> None:
    raw = elem.get("file")
    if not raw:
        return
    path = Path(raw)
    if not path.is_absolute():
        path = (root / path).resolve()
    elem.set("file", str(path))


def find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise KeyError(f"body {name!r} not found")


def find_or_create(parent: ET.Element, tag: str) -> ET.Element:
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def deepcopy_children(src: ET.Element) -> list[ET.Element]:
    return [copy.deepcopy(c) for c in list(src)]


def dump_mjcf(root: ET.Element) -> str:
    if hasattr(ET, "indent"):
        ET.indent(root, space="  ")
    xml = ET.tostring(root, encoding="unicode")
    if not xml.startswith("<?xml"):
        xml = '<?xml version="1.0" encoding="utf-8"?>\n' + xml
    return xml
