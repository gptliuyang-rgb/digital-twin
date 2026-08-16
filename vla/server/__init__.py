"""Optional GR00T / π0.5 process wrappers. No weights are downloaded here."""

from __future__ import annotations


class PolicyServer:
    def __init__(self, kind: str) -> None:
        if kind not in {"groot", "pi05", "replay"}:
            raise ValueError(kind)
        self.kind = kind

    def serve(self) -> None:
        raise NotImplementedError(f"{self.kind} server is wired in a later phase (P6+)")
