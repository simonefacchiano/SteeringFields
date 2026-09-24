"""Compatibility wrapper for the historical Flux baseline command."""

from pathlib import Path

from steering_fields.cli.legacy_flux import main, parse_args
from steering_fields.datasets import load_dataset_config, read_prompt_csv, resolve_dataset_entry
from steering_fields.flux_utils import load_flux_img2img_pipeline, load_flux_pipeline
from steering_fields.outputs import find_existing_image, prompt_first8words_slug, sanitize_filename_token

__all__ = [
    "find_existing_image", "load_csv_entries", "load_dataset_config",
    "load_flux_img2img_pipeline", "load_flux_pipeline", "main", "parse_args",
    "prompt_first8words_slug", "resolve_dataset_entry", "sanitize_filename_token",
]


def load_csv_entries(csv_path, source_column, seed_cfg, default_seed, max_rows):
    records = read_prompt_csv(
        Path(csv_path),
        source_column=source_column,
        seed_strategy=seed_cfg,
        base_seed=default_seed,
        max_rows=max_rows,
    )
    return [
        {
            "row_index": record.row_index,
            "filename_id": record.row_id,
            "prompt": record.prompt,
            "seed": record.seed,
        }
        for record in records
    ]


if __name__ == "__main__":
    main()
