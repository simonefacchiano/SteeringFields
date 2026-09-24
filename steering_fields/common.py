"""Small model-independent types and helpers."""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SteeringParameters:
    """Parameters shared by the Flux and SD minimal-transport samplers."""

    mode: str = "add"
    alpha_schedule: str = "linear"
    alpha: float = 0.5
    alpha_start: float = 0.9
    alpha_end: float = 0.1
    mu_schedule: str = "constant"
    mu: float = 0.5
    mu_start: float = 0.5
    mu_end: float = 0.5


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def unique_ints(values: list[int] | None, fallback: int) -> list[int]:
    return list(dict.fromkeys(values if values else [fallback]))


def json_safe(value: Any) -> Any:
    """Convert paths, dataclasses and common scalar containers to JSON values."""
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value
