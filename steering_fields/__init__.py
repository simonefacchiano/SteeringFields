"""Portable inference entry points for SteeringFields."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

from .common import SteeringParameters
from .config import AppConfig, get_model_checkpoint, load_config


def _load_algorithm_module():
    """Load the repository-root algorithm file as a package submodule."""
    module_name = f"{__name__}._algorithm"
    module_path = Path(__file__).resolve().parent.parent / "steering_fields.py"
    spec = spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Steering Fields algorithm from {module_path}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_algorithm = _load_algorithm_module()
blend_velocities = _algorithm.blend_velocities
sample_min_transport_flux = _algorithm.sample_min_transport_flux

__all__ = [
    "AppConfig",
    "SteeringParameters",
    "blend_velocities",
    "get_model_checkpoint",
    "load_config",
    "sample_min_transport_flux",
]
