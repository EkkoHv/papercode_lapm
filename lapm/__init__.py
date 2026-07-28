"""Local Association Prediction Method for spatial soil-property prediction."""

from __future__ import annotations

from typing import Any

__all__ = ["LAPM", "LAPMConfig"]


def __getattr__(name: str) -> Any:
    """Import the PyTorch implementation only when it is requested."""

    if name in __all__:
        from .model import LAPM, LAPMConfig

        return {"LAPM": LAPM, "LAPMConfig": LAPMConfig}[name]
    raise AttributeError(name)
