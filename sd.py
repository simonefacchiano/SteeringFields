"""Compatibility wrapper for the historical SD3/SD3.5 command."""

import os
from pathlib import Path

from steering_fields.cli.sd_t2i import main as _main, parse_args as _parse_args
from steering_fields.common import str2bool
from steering_fields.embeddings import load_embedding_payload as load_avg_target_embeddings
from steering_fields.embeddings import save_embedding_payload as _save_embedding_payload
from steering_fields.schedules import alpha_at_step, mu_at_step
from steering_fields.sd_utils import (
    blend_velocities,
    build_avg_target_embeddings_from_csv,
    calc_v_sd3_single,
    condition_from_avg,
    decode_sd3_latent_to_pil,
    default_cfg_for_version,
    get_cfg_condition,
    load_or_build_avg_target_embeddings,
    sample_min_transport_sd,
)

__all__ = [
    "alpha_at_step", "blend_velocities", "build_avg_target_embeddings_from_csv",
    "calc_v_sd3_single", "condition_from_avg", "decode_sd3_latent_to_pil",
    "default_cfg_for_version", "get_cfg_condition", "load_avg_target_embeddings",
    "load_or_build_avg_target_embeddings", "main", "mu_at_step", "parse_args",
    "resolve_checkpoint", "sample_min_transport_sd", "save_avg_embeddings_payload", "str2bool",
]


def parse_args():
    return _parse_args(legacy_cli=True)


def resolve_checkpoint(version: str, checkpoint_override: str | None) -> Path:
    if checkpoint_override:
        return Path(checkpoint_override)
    value = os.environ.get("SD3_CHECKPOINT" if version == "sd3" else "SD35_CHECKPOINT")
    if not value:
        raise ValueError(f"No checkpoint configured for {version}; pass --checkpoint or set {version.upper()}_CHECKPOINT")
    return Path(value)


def save_avg_embeddings_payload(payload, out_arg: str, *, overwrite: bool):
    return _save_embedding_payload(payload, out_arg, overwrite=overwrite)


def main() -> None:
    _main(legacy_cli=True)


if __name__ == "__main__":
    main()
