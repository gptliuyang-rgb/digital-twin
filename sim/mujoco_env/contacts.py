"""Contact queries for DexHand pads vs props (relative, not a pick rate)."""

from __future__ import annotations


def iter_contact_names(model, data):
    for k in range(int(data.ncon)):
        con = data.contact[k]
        n1 = model.geom(int(con.geom1)).name
        n2 = model.geom(int(con.geom2)).name
        yield con, n1, n2


def count_contacts(model, data, left_pred, right_substr: str) -> int:
    """Count contacts where one geom matches ``left_pred`` and the other contains ``right_substr``."""
    n = 0
    for _con, n1, n2 in iter_contact_names(model, data):
        if (left_pred(n1) and right_substr in n2) or (left_pred(n2) and right_substr in n1):
            n += 1
    return n


def hand_object_contacts(model, data, side: str, object_substr: str) -> int:
    """Contacts between one DexHand tree (``l_`` / ``r_`` geoms, including pads) and an object."""
    prefix = "l_" if side == "left" else "r_"
    return count_contacts(model, data, lambda n: n.startswith(prefix), object_substr)


def pad_object_contacts(model, data, side: str, object_substr: str) -> int:
    prefix = "l_" if side == "left" else "r_"

    def pred(name: str) -> bool:
        return name.startswith(prefix) and "_pad_" in name

    return count_contacts(model, data, pred, object_substr)
