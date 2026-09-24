"""Flux image-to-image command."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ..common import unique_ints
from ..embeddings import load_embedding_payload
from ..flux_utils import (
    decode_flux_latent_to_pil,
    load_flux_img2img_pipeline,
    parse_box,
    sample_flux,
)
from ..outputs import (
    create_run_directory,
    make_paired_image,
    save_images_and_metadata,
    slugify,
    steering_target_label,
    timestamped_run_name,
)
from ..runtime import make_generator, resolve_device, resolve_dtype, set_global_seed
from ._shared import (
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
    parser = argparse.ArgumentParser(description="Flux image-to-image baseline and minimal-transport steering.")
    add_config_arguments(parser)
    add_generation_arguments(parser, image_to_image=True, legacy=legacy_cli)
    add_mode_arguments(parser, legacy_steering=legacy_cli)
    parser.add_argument("--box", type=str, default=None, help="ROI as x0,y0,x1,y1 in output pixels.")
    parser.add_argument("--box-feather", type=float, default=None)
    parser.add_argument("--timestep-start", type=int, default=None)
    return parser


def parse_args(argv: Sequence[str] | None = None, *, legacy_cli: bool = False) -> argparse.Namespace:
    args = build_parser(legacy_cli=legacy_cli).parse_args(argv)
    if legacy_cli:
        legacy_defaults = {
            "cfg_tar": 6.0,
            "mu_start": 0.8,
            "mu_end": 0.4,
            "alpha_start": 0.8,
            "alpha_end": 0.6,
        }
        for name, default in legacy_defaults.items():
            if getattr(args, name) is None:
                setattr(args, name, default)
    return args


def _sample_image(*, pipe, image, prompt, args, seed, steering, height, width, steps, cfg_src, cfg_tar, avg_payload, device, strength, timestep_start, box, feather):
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
        conditioning_image=image,
        image_strength=strength,
        timestep_start=timestep_start,
        box_xyxy=box,
        box_feather=feather,
    )
    return decode_flux_latent_to_pil(pipe, latent, device, next(pipe.vae.parameters()).dtype)


def main(argv: Sequence[str] | None = None, *, legacy_cli: bool = False) -> None:
    args = parse_args(argv, legacy_cli=legacy_cli)
    if legacy_cli and args.prompt is None:
        args.prompt = "a dog"
    validate_prompt_selection(args, dataset_allowed=False)
    config, checkpoint, output_root = resolved_config(args, "flux")
    height, width, steps, default_seed = effective_dimensions(args, config)
    seeds = unique_ints(args.seeds, default_seed)
    strength = float(value(args, "image_strength", config, "image_to_image.strength", 0.8))
    timestep_start = int(value(args, "timestep_start", config, "image_to_image.timestep_start", 1))
    feather = float(value(args, "box_feather", config, "image_to_image.box_feather", 0.0))
    if not 0.0 < strength <= 1.0:
        raise ValueError("--image-strength must be in (0, 1]")
    if timestep_start < 1:
        raise ValueError("--timestep-start must be >= 1")
    if feather < 0.0:
        raise ValueError("--box-feather must be non-negative")
    box = parse_box(args.box) if args.box else None

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Conditioning image not found: {image_path}")
    from PIL import Image
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    input_image = Image.open(image_path).convert("RGB").resize((width, height), resampling)

    device = resolve_device(value(args, "device", config, "runtime.device", "auto"), value(args, "device_number", config, "runtime.device_number", 0))
    dtype = resolve_dtype(value(args, "dtype", config, "runtime.dtype", "auto"), device, model="flux")
    cfg_src = float(value(args, "cfg_src", config, "flux.cfg_src", 1.5))
    cfg_tar = cfg_src if args.cfg_tar_eq_src else float(value(args, "cfg_tar", config, "flux.cfg_tar", 5.5))
    steering = steering_parameters(args, config, steps) if args.generation_mode == "steer" else None
    set_global_seed(seeds[0])
    pipe = load_flux_img2img_pipeline(checkpoint, device, dtype)
    avg_payload = load_embedding_payload(args.avg_target_embeddings) if args.avg_target_embeddings else None

    images = {"input": input_image.copy()}
    for seed in seeds:
        generated = _sample_image(
            pipe=pipe, image=input_image, prompt=args.prompt, args=args, seed=seed,
            steering=steering, height=height, width=width, steps=steps,
            cfg_src=cfg_src, cfg_tar=cfg_tar, avg_payload=avg_payload, device=device,
            strength=strength, timestep_start=timestep_start, box=box, feather=feather,
        )
        suffix = f"_{seed}" if len(seeds) > 1 else ""
        images[f"{'steered' if steering else 'baseline'}{suffix}"] = generated
        # The input/result pair remains useful in both modes and matches the former i2i CLI.
        images[f"paired{suffix}"] = make_paired_image(input_image, generated)
        if steering is not None and args.include_baseline:
            baseline = _sample_image(
                pipe=pipe, image=input_image, prompt=args.prompt, args=args, seed=seed,
                steering=None, height=height, width=width, steps=steps,
                cfg_src=cfg_src, cfg_tar=cfg_src, avg_payload=None, device=device,
                strength=strength, timestep_start=1, box=box, feather=feather,
            )
            baseline_key = f"vanilla_img2img{suffix}" if legacy_cli else f"baseline{suffix}"
            images[baseline_key] = baseline
            if not legacy_cli:
                images[f"paired{suffix}"] = make_paired_image(baseline, generated)

    target = steering_target_label(
        args.target_prompt, args.safe_prompt, args.avg_target_embeddings
    )
    default_name = f"{args.prompt}_to_{target}" if steering else args.prompt
    run_name = args.run_name or timestamped_run_name(default_name)
    if legacy_cli:
        run_dir = output_root.expanduser().resolve() / slugify(run_name, max_len=80)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_directory(output_root, model="flux", workflow="i2i", mode=args.generation_mode, run_name=run_name)
    metadata = {
        "model": "flux",
        "workflow": "i2i",
        "generation_mode": args.generation_mode,
        "checkpoint": str(checkpoint),
        "prompt": args.prompt,
        "source_prompt": args.prompt,
        "target_prompt": args.target_prompt,
        "safe_prompt": args.safe_prompt,
        "unsafe_prompt": args.unsafe_prompt,
        "image": str(image_path),
        "image_strength": strength,
        "box": args.box,
        "box_feather": feather,
        "timestep_start": timestep_start,
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
        "seed": default_seed,
        "seeds": seeds,
        "device": str(device),
        "dtype": str(dtype),
        "avg_target_embeddings": str(Path(args.avg_target_embeddings).expanduser().resolve()) if args.avg_target_embeddings else None,
        "include_baseline": bool(args.include_baseline),
        "resolved_config": config.as_dict(),
        "config_files": list(config.config_files),
    }
    save_images_and_metadata(run_dir, images, metadata)


if __name__ == "__main__":
    main()
