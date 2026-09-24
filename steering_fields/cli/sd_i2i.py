"""SD3/SD3.5 image-to-image command."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ..common import unique_ints
from ..outputs import (
    create_run_directory,
    make_paired_image,
    save_images_and_metadata,
    steering_target_label,
    timestamped_run_name,
)
from ..runtime import make_generator, resolve_device, resolve_dtype, set_global_seed
from ..sd_utils import (
    decode_sd3_latent_to_pil,
    default_cfg_for_model,
    load_or_build_avg_target_embeddings,
    load_sd_pipeline,
    sample_sd,
)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SD3/SD3.5 image-to-image baseline and minimal-transport steering.")
    add_config_arguments(parser)
    parser.add_argument("--model", "--version", dest="model", choices=("sd3", "sd35"), required=True)
    add_generation_arguments(parser, image_to_image=True)
    add_mode_arguments(parser)
    parser.add_argument("--cfg", type=float, default=None, help="Baseline CFG override.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _generate(*, pipe, image, model, prompt, args, seed, steering, height, width, steps, cfg_src, cfg_tar, avg_payload, device, strength):
    latent = sample_sd(
        pipe=pipe,
        model=model,
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
    )
    return decode_sd3_latent_to_pil(pipe, latent, device, next(pipe.vae.parameters()).dtype)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_prompt_selection(args, dataset_allowed=False)
    config, checkpoint, output_root = resolved_config(args, args.model)
    height, width, steps, default_seed = effective_dimensions(args, config)
    seeds = unique_ints(args.seeds, default_seed)
    strength = float(value(args, "image_strength", config, "image_to_image.strength", 0.8))
    if not 0.0 < strength <= 1.0:
        raise ValueError("--image-strength must be in (0, 1]")

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Conditioning image not found: {image_path}")
    from PIL import Image
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    input_image = Image.open(image_path).convert("RGB").resize((width, height), resampling)

    default_cfg = float(config.get(f"{args.model}.cfg", default_cfg_for_model(args.model)))
    cfg_src = float(args.cfg_src) if args.cfg_src is not None else (float(args.cfg) if args.cfg is not None else default_cfg)
    cfg_tar = cfg_src if args.cfg_tar_eq_src else (float(args.cfg_tar) if args.cfg_tar is not None else cfg_src)
    steering = steering_parameters(args, config, steps) if args.generation_mode == "steer" else None
    device = resolve_device(value(args, "device", config, "runtime.device", "auto"), value(args, "device_number", config, "runtime.device_number", 0))
    dtype = resolve_dtype(value(args, "dtype", config, "runtime.dtype", "auto"), device, model=args.model)
    set_global_seed(seeds[0])
    pipe = load_sd_pipeline(args.model, checkpoint, device, dtype, image_to_image=True)
    avg_payload = load_or_build_avg_target_embeddings(args.avg_target_embeddings, pipe=pipe, device=device, dtype=dtype) if args.avg_target_embeddings else None

    images = {"input": input_image.copy()}
    for seed in seeds:
        generated = _generate(
            pipe=pipe, image=input_image, model=args.model, prompt=args.prompt, args=args,
            seed=seed, steering=steering, height=height, width=width, steps=steps,
            cfg_src=cfg_src, cfg_tar=cfg_tar, avg_payload=avg_payload, device=device, strength=strength,
        )
        suffix = f"_{seed}" if len(seeds) > 1 else ""
        images[f"{'steered' if steering else 'baseline'}{suffix}"] = generated
        images[f"paired{suffix}"] = make_paired_image(input_image, generated)
        if steering is not None and args.include_baseline:
            baseline = _generate(
                pipe=pipe, image=input_image, model=args.model, prompt=args.prompt, args=args,
                seed=seed, steering=None, height=height, width=width, steps=steps,
                cfg_src=cfg_src, cfg_tar=cfg_src, avg_payload=None, device=device, strength=strength,
            )
            images[f"baseline{suffix}"] = baseline
            images[f"paired{suffix}"] = make_paired_image(baseline, generated)

    target = steering_target_label(
        args.target_prompt, args.safe_prompt, args.avg_target_embeddings
    )
    default_name = f"{args.prompt}_to_{target}" if steering else f"{args.model}_{args.prompt}"
    run_name = args.run_name or timestamped_run_name(default_name)
    run_dir = create_run_directory(output_root, model=args.model, workflow="i2i", mode=args.generation_mode, run_name=run_name)
    metadata = {
        "model": args.model,
        "model_version": args.model,
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
