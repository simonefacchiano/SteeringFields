"""Runtime selection helpers. Heavy dependencies are imported lazily."""

from __future__ import annotations

import random
from contextlib import nullcontext
from typing import Any


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for generation but is not installed in this environment") from exc
    return torch


def resolve_device(device: str | None = None, device_number: int | None = None):
    torch = _torch()
    requested = (device or "auto").lower()
    number = 0 if device_number is None else int(device_number)
    if requested == "auto":
        requested = f"cuda:{number}" if torch.cuda.is_available() else "cpu"
    elif requested == "cuda":
        requested = f"cuda:{number}"
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({resolved}) but CUDA is unavailable")
    return resolved


def resolve_dtype(dtype_name: str | None, device, *, model: str | None = None):
    torch = _torch()
    name = (dtype_name or "auto").lower()
    if name == "auto":
        if device.type != "cuda":
            return torch.float32
        return torch.bfloat16 if model == "flux" else torch.float16
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in aliases:
        raise ValueError(f"Unsupported dtype {dtype_name!r}; expected auto, float32, float16, or bfloat16")
    if device.type == "cpu" and aliases[name] != torch.float32:
        raise ValueError(f"dtype {dtype_name!r} is not supported for CPU inference; use float32 or auto")
    return aliases[name]


def set_global_seed(seed: int) -> None:
    torch = _torch()
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_generator(device, seed: int):
    return _torch().Generator(device=device).manual_seed(int(seed))


def autocast_context(device) -> Any:
    torch = _torch()
    return torch.autocast("cuda") if device.type == "cuda" else nullcontext()
