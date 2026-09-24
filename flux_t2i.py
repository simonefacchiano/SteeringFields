"""Flux text-to-image command."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from steering_fields.common import unique_ints
from steering_fields.datasets import load_prompt_records
from steering_fields.embeddings import load_embedding_payload, load_initial_noise
from steering_fields.flux_utils import decode_flux_latent_to_pil, load_flux_pipeline, sample_flux
from steering_fields.outputs import (
    create_run_directory,
    make_paired_image,
    prompt_first8words_slug,
    sanitize_filename_token,
    save_images_and_metadata,
    slugify,
    steering_target_label,
    timestamped_run_name,
    write_metadata,
)
from steering_fields.runtime import make_generator, resolve_device, resolve_dtype, set_global_seed
from steering_fields.cli._shared import (
    add_config_arguments,
    add_generation_arguments,
    add_mode_arguments,
    effective_dimensions,
    resolved_config,
    steering_parameters,
    validate_prompt_selection,
    value,
)


def build_parser(*, legacy_cli: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flux text-to-image baseline and minimal-transport steering.")
    add_config_arguments(parser)
    add_generation_arguments(parser, image_to_image=False, legacy=legacy_cli)
    add_mode_arguments(parser, legacy_steering=legacy_cli)
    parser.add_argument("--dataset", type=str, default=None, help="CSV path or configured dataset name.")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--source-column", type=str, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--initial-noise-pt", type=str, default=None)
    return parser


def parse_args(argv: Sequence[str] | None = None, *, legacy_cli: bool = False) -> argparse.Namespace:
    return build_parser(legacy_cli=legacy_cli).parse_args(argv)


def _generate_image(
    *, pipe, prompt: str, args, seed: int, steering, height: int, width: int,
    steps: int, cfg_src: float, cfg_tar: float, avg_payload, initial_noise, device,
):
    latent = sample_flux(
        pipe=pipe,
        source_prompt=prompt,
        target_prompt=args.target_prompt,
        height=height,
        width=width,
        num_steps=steps,
        cfg_src=cfg_src,
        cfg_tar=cfg_tar,
        steering=steering,
        generator=make_generator(device, seed),
        avg_target_embeddings=avg_payload,
        safe_prompt=args.safe_prompt,
        unsafe_prompt=args.unsafe_prompt,
        initial_noise=initial_noise,
    )
    vae_dtype = next(pipe.vae.parameters()).dtype
    return decode_flux_latent_to_pil(pipe, latent, device, vae_dtype)


def main(argv: Sequence[str] | None = None, *, legacy_cli: bool = False) -> None:
    args = parse_args(argv, legacy_cli=legacy_cli)
    if legacy_cli and args.prompt is None and args.dataset is None:
        args.prompt = "a dog"
    validate_prompt_selection(args, dataset_allowed=True)
    config, checkpoint, output_root = resolved_config(args, "flux")
    height, width, steps, default_seed = effective_dimensions(args, config)
    seeds = unique_ints(args.seeds, default_seed)
    if args.dataset and args.seeds:
        raise ValueError("--seeds is not supported with --dataset; use --seed")

    device = resolve_device(value(args, "device", config, "runtime.device", "auto"), value(args, "device_number", config, "runtime.device_number", 0))
    dtype = resolve_dtype(value(args, "dtype", config, "runtime.dtype", "auto"), device, model="flux")
    cfg_src = float(value(args, "cfg_src", config, "flux.cfg_src", 1.5))
    configured_tar = float(value(args, "cfg_tar", config, "flux.cfg_tar", 5.5))
    cfg_tar = cfg_src if args.cfg_tar_eq_src else configured_tar
    steering = steering_parameters(args, config, steps) if args.generation_mode == "steer" else None
    set_global_seed(seeds[0])
    pipe = load_flux_pipeline(checkpoint, device, dtype)
    avg_payload = load_embedding_payload(args.avg_target_embeddings) if args.avg_target_embeddings else None
    initial_noise = load_initial_noise(args.initial_noise_pt) if args.initial_noise_pt else None

    common_meta = {
        "model": "flux",
        "workflow": "t2i",
        "generation_mode": args.generation_mode,
        "checkpoint": str(checkpoint),
        "target_prompt": args.target_prompt,
        "safe_prompt": args.safe_prompt,
        "unsafe_prompt": args.unsafe_prompt,
        "height": height,
        "width": width,
        "num_steps": steps,
        "cfg_src": cfg_src,
        "cfg_tar": cfg_tar,
        "steering": steering,
        "steering_mode": steering.mode if steering else None,
        "mode": steering.mode if steering else "baseline",
        "alpha_schedule": steering.alpha_schedule if steering else None,
        "alpha": steering.alpha if steering else None,
        "alpha_start": steering.alpha_start if steering else None,
        "alpha_end": steering.alpha_end if steering else None,
        "mu_schedule": steering.mu_schedule if steering else None,
        "mu": steering.mu if steering else None,
        "mu_start": steering.mu_start if steering else None,
        "mu_end": steering.mu_end if steering else None,
        "device": str(device),
        "dtype": str(dtype),
        "avg_target_embeddings": str(Path(args.avg_target_embeddings).expanduser().resolve()) if args.avg_target_embeddings else None,
        "initial_noise_pt": str(Path(args.initial_noise_pt).expanduser().resolve()) if args.initial_noise_pt else None,
        "include_baseline": bool(args.include_baseline),
        "resolved_config": config.as_dict(),
        "config_files": list(config.config_files),
    }

    if args.dataset:
        dataset_config = Path(args.dataset_config).expanduser().resolve() if args.dataset_config else Path(config.get("paths.dataset_config"))
        dataset_path, dataset_entry, records = load_prompt_records(
            args.dataset,
            dataset_config=dataset_config,
            source_column=args.source_column,
            base_seed=default_seed,
            cli_seed_override=args.seed is not None,
            max_rows=args.max_rows,
        )
        run_name = args.run_name or timestamped_run_name(
            f"{dataset_path.stem}_seed{default_seed}_cfg{cfg_src}"
        )
        if legacy_cli:
            run_dir = output_root.expanduser().resolve() / slugify(run_name, max_len=80)
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = create_run_directory(output_root, model="flux", workflow="t2i", mode=args.generation_mode, run_name=run_name)
        rows_meta: list[dict] = []
        skipped = 0
        for record in records:
            base = f"{sanitize_filename_token(record.row_id)}_{prompt_first8words_slug(record.prompt)}"
            primary_path = run_dir / f"{base}.png"
            if primary_path.exists():
                skipped += 1
                rows_meta.append({"row_index": record.row_index, "row_id": record.row_id, "prompt": record.prompt, "seed": record.seed, "image": primary_path.name, "skipped_existing": True})
                continue
            image = _generate_image(
                pipe=pipe, prompt=record.prompt, args=args, seed=record.seed, steering=steering,
                height=height, width=width, steps=steps, cfg_src=cfg_src, cfg_tar=cfg_tar,
                avg_payload=avg_payload, initial_noise=initial_noise, device=device,
            )
            image.save(primary_path)
            row_meta = {"row_index": record.row_index, "row_id": record.row_id, "prompt": record.prompt, "seed": record.seed, "image": primary_path.name}
            if steering is not None and args.include_baseline:
                baseline = _generate_image(
                    pipe=pipe, prompt=record.prompt, args=args, seed=record.seed, steering=None,
                    height=height, width=width, steps=steps, cfg_src=cfg_src, cfg_tar=cfg_src,
                    avg_payload=None, initial_noise=initial_noise, device=device,
                )
                baseline_path = run_dir / f"{base}_baseline.png"
                paired_path = run_dir / f"{base}_paired.png"
                baseline.save(baseline_path)
                make_paired_image(baseline, image).save(paired_path)
                row_meta.update({"baseline": baseline_path.name, "paired": paired_path.name})
            rows_meta.append(row_meta)
            write_metadata(run_dir / "meta.json", {**common_meta, "dataset": str(dataset_path), "dataset_entry": dataset_entry, "rows": rows_meta, "skipped_existing": skipped})
        write_metadata(run_dir / "meta.json", {**common_meta, "dataset": str(dataset_path), "dataset_entry": dataset_entry, "rows": rows_meta, "skipped_existing": skipped})
        if skipped:
            print(f"[resume] skipped existing images: {skipped}")
        print(f"[done] run dir: {run_dir}")
        return

    images = {}
    per_seed = []
    for seed in seeds:
        image = _generate_image(
            pipe=pipe, prompt=args.prompt, args=args, seed=seed, steering=steering,
            height=height, width=width, steps=steps, cfg_src=cfg_src, cfg_tar=cfg_tar,
            avg_payload=avg_payload, initial_noise=initial_noise, device=device,
        )
        key = ("steered" if steering else "baseline") + (f"_{seed}" if len(seeds) > 1 else "")
        images[key] = image
        if steering is not None and args.include_baseline:
            baseline = _generate_image(
                pipe=pipe, prompt=args.prompt, args=args, seed=seed, steering=None,
                height=height, width=width, steps=steps, cfg_src=cfg_src, cfg_tar=cfg_src,
                avg_payload=None, initial_noise=initial_noise, device=device,
            )
            suffix = f"_{seed}" if len(seeds) > 1 else ""
            images[f"baseline{suffix}"] = baseline
            images[f"paired{suffix}"] = make_paired_image(baseline, image)
        per_seed.append({"seed": seed})
    name_target = steering_target_label(
        args.target_prompt, args.safe_prompt, args.avg_target_embeddings
    )
    default_name = f"{args.prompt}_to_{name_target}" if steering else args.prompt
    run_name = args.run_name or timestamped_run_name(default_name)
    if legacy_cli:
        run_dir = output_root.expanduser().resolve() / slugify(run_name, max_len=80)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_directory(output_root, model="flux", workflow="t2i", mode=args.generation_mode, run_name=run_name)
    save_images_and_metadata(run_dir, images, {**common_meta, "prompt": args.prompt, "source_prompt": args.prompt, "seed": default_seed, "seeds": seeds, "per_seed": per_seed})


if __name__ == "__main__":
    main()
