"""Average-embedding payload I/O and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to load embedding payloads") from exc
    return torch


def resolve_embedding_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_dir():
        resolved = resolved / "avg_embedding.pt"
    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Average embedding payload not found: {resolved}")
    return resolved


def validate_embedding_payload(payload: Any, required_key_groups: Iterable[Iterable[str]] = ()) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Average embedding payload must be a dictionary")
    groups = [tuple(group) for group in required_key_groups]
    if groups and not any(all(key in payload for key in group) for group in groups):
        expected = " or ".join("{" + ", ".join(group) + "}" for group in groups)
        raise ValueError(f"Average embedding payload is missing required keys; expected {expected}")
    return payload


def load_embedding_payload(
    path: str | Path,
    *,
    required_key_groups: Iterable[Iterable[str]] = (),
) -> dict[str, Any]:
    torch = _torch()
    resolved = resolve_embedding_path(path)
    payload = torch.load(resolved, map_location="cpu")
    return validate_embedding_payload(payload, required_key_groups)


def save_embedding_payload(payload: dict[str, Any], path: str | Path, *, overwrite: bool = False) -> Path:
    torch = _torch()
    output = Path(path).expanduser()
    if output.suffix.lower() != ".pt":
        output = output / "avg_embedding.pt"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; enable overwrite explicitly")
    torch.save(payload, output)
    return output


def load_initial_noise(path: str | Path):
    torch = _torch()
    resolved = Path(path).expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu")
    noise = payload.get("noise") if isinstance(payload, dict) else payload
    if not isinstance(noise, torch.Tensor):
        raise ValueError(f"Noise payload must be a tensor or a dict containing 'noise': {resolved}")
    if noise.ndim not in (3, 4) or noise.shape[0] != 1:
        raise ValueError(f"Noise tensor must have batch size 1 and 3 or 4 dimensions; got {tuple(noise.shape)}")
    return noise.contiguous()
