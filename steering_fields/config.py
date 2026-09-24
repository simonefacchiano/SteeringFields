"""Configuration loading, precedence and portable path resolution."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
PATH_KEYS = {
    "flux_checkpoint",
    "sd3_checkpoint",
    "sd35_checkpoint",
    "output_root",
    "dataset_config",
}
ENV_OVERRIDES = {
    "FLUX_CHECKPOINT": "paths.flux_checkpoint",
    "SD3_CHECKPOINT": "paths.sd3_checkpoint",
    "SD35_CHECKPOINT": "paths.sd35_checkpoint",
    "STEERING_FIELDS_OUTPUT_ROOT": "paths.output_root",
}


BUILTIN_DEFAULTS: dict[str, Any] = {
    "paths": {
        "flux_checkpoint": None,
        "sd3_checkpoint": None,
        "sd35_checkpoint": None,
        "output_root": str(PROJECT_ROOT / "outputs"),
        "dataset_config": str(PROJECT_ROOT / "data" / "dataset_config.json"),
    },
    "runtime": {"device": "auto", "dtype": "auto", "device_number": 0},
    "defaults": {"height": 1024, "width": 1024, "num_steps": 28, "seed": 42},
    "flux": {"cfg_src": 1.5, "cfg_tar": 5.5},
    "sd3": {"cfg": 7.0},
    "sd35": {"cfg": 3.5},
    "steering": {
        "mode": "add",
        "alpha_schedule": "linear",
        "alpha": 0.5,
        "alpha_start": 0.9,
        "alpha_end": 0.1,
        "mu_schedule": "constant",
        "mu": 0.5,
        "mu_start": 0.5,
        "mu_end": 0.5,
    },
    "image_to_image": {"strength": 0.8, "timestep_start": 1, "box_feather": 0.0},
}


@dataclass(frozen=True)
class AppConfig:
    data: dict[str, Any]
    config_files: tuple[Path, ...]

    def get(self, dotted_key: str, default: Any = None) -> Any:
        value: Any = self.data
        for part in dotted_key.split("."):
            if not isinstance(value, Mapping) or part not in value:
                return default
            value = value[part]
        return value

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)


def resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _deep_merge(target: dict[str, Any], update: Mapping[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return payload


def _resolve_paths_in_layer(layer: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    result = copy.deepcopy(layer)
    paths = result.get("paths")
    if isinstance(paths, dict):
        for key in PATH_KEYS:
            if paths.get(key) is not None:
                paths[key] = str(resolve_path(paths[key], base_dir=base_dir))
    return result


def _set_dotted(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = target
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def _normalize_overrides(overrides: Mapping[str, Any], base_dir: Path) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, value in overrides.items():
        if value is None:
            continue
        key = raw_key if "." in raw_key else (
            f"paths.{raw_key}" if raw_key in PATH_KEYS else raw_key
        )
        if key.startswith("paths."):
            value = str(resolve_path(value, base_dir=base_dir))
        _set_dotted(normalized, key, value)
    return normalized


def load_config(
    config_path: Path | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    """Load defaults, YAML, environment and CLI overrides in that order."""
    data = copy.deepcopy(BUILTIN_DEFAULTS)
    loaded: list[Path] = []

    if DEFAULT_CONFIG.exists():
        default_path = DEFAULT_CONFIG.resolve()
        _deep_merge(data, _resolve_paths_in_layer(_read_yaml(default_path), default_path.parent))
        loaded.append(default_path)

    selected = config_path
    if selected is None and os.environ.get("STEERING_FIELDS_CONFIG"):
        selected = Path(os.environ["STEERING_FIELDS_CONFIG"])
    if selected is not None:
        selected = resolve_path(selected, base_dir=Path.cwd())
        if selected != DEFAULT_CONFIG.resolve():
            _deep_merge(data, _resolve_paths_in_layer(_read_yaml(selected), selected.parent))
            loaded.append(selected)

    env_layer: dict[str, Any] = {}
    for env_name, dotted_key in ENV_OVERRIDES.items():
        value = os.environ.get(env_name)
        if value:
            _set_dotted(env_layer, dotted_key, str(resolve_path(value, base_dir=Path.cwd())))
    _deep_merge(data, env_layer)

    if cli_overrides:
        _deep_merge(data, _normalize_overrides(cli_overrides, Path.cwd()))

    return AppConfig(data=data, config_files=tuple(loaded))


def get_model_checkpoint(config: AppConfig, model: str) -> Path:
    normalized = model.lower().replace(".", "")
    key = {"flux": "flux_checkpoint", "sd3": "sd3_checkpoint", "sd35": "sd35_checkpoint"}.get(normalized)
    if key is None:
        raise ValueError(f"Unsupported model {model!r}; expected flux, sd3, or sd35")
    value = config.get(f"paths.{key}")
    if not value:
        env_name = key.upper()
        raise ValueError(
            f"No checkpoint configured for {normalized}. Set paths.{key} in --config, "
            f"set {env_name}, or pass --checkpoint."
        )
    checkpoint = Path(value)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint path not found for {normalized}: {checkpoint}")
    return checkpoint
