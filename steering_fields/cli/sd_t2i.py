"""SD3/SD3.5 text-to-image command."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ..common import str2bool, unique_ints
from ..datasets import load_prompt_records
from ..embeddings import save_embedding_payload
from ..outputs import (
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


def build_parser(*, legacy_cli: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SD3/SD3.5 text-to-image baseline and minimal-transport steering.")
    add_config_arguments(parser)
    parser.add_argument("--model", "--version", dest="model", choices=("sd3", "sd35"), required=True)
    add_generation_arguments(parser, image_to_image=False, legacy=legacy_cli)
    add_mode_arguments(parser, legacy_steering=legacy_cli)
    if legacy_cli:
        parser.add_argument("--steer", type=str2bool, default=False)
    parser.add_argument("--cfg", type=float, default=None, help="Baseline CFG override.")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--source-column", type=str, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--meta-name", type=str, default="meta.json")
    parser.add_argument("--export-avg-embeddings", type=str, default=None)
    parser.add_argument("--overwrite-export-avg-embeddings", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None, *, legacy_cli: bool = False) -> argparse.Namespace:
    args = build_parser(legacy_cli=legacy_cli).parse_args(argv)
    if legacy_cli:
        args.generation_mode = "steer" if args.steer else "baseline"
    return args


def _generate(*, pipe, model, prompt, args, seed, steering, height, width, steps, cfg_src, cfg_tar, avg_payload, device):
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
    )
    return decode_sd3_latent_to_pil(pipe, latent, device, next(pipe.vae.parameters()).dtype)


def main(argv: Sequence[str] | None = None, *, legacy_cli: bool = False) -> None:
    args = parse_args(argv, legacy_cli=legacy_cli)
    # Historical launchers sometimes pass --source-prompt together with
    # --dataset. Dataset rows remain authoritative in that compatibility mode.
    legacy_prompt = args.prompt
    if legacy_cli and args.dataset is not None:
        args.prompt = None
    validate_prompt_selection(args, dataset_allowed=True)
    args.prompt = legacy_prompt
    config, checkpoint, output_root = resolved_config(args, args.model)
    height, width, steps, default_seed = effective_dimensions(args, config)
    seeds = unique_ints(args.seeds, default_seed)
    if args.dataset and args.seeds:
        raise ValueError("--seeds is not supported with --dataset; use --seed")
    default_cfg = float(config.get(f"{args.model}.cfg", default_cfg_for_model(args.model)))
    baseline_cfg = float(args.cfg) if args.cfg is not None else default_cfg
    cfg_src = float(args.cfg_src) if args.cfg_src is not None else default_cfg
    cfg_tar = cfg_src if args.cfg_tar_eq_src else (float(args.cfg_tar) if args.cfg_tar is not None else cfg_src)
    if args.generation_mode == "baseline":
        cfg_src = baseline_cfg
        cfg_tar = baseline_cfg
    steering = steering_parameters(args, config, steps) if args.generation_mode == "steer" else None

    device = resolve_device(value(args, "device", config, "runtime.device", "auto"), value(args, "device_number", config, "runtime.device_number", 0))
    dtype = resolve_dtype(value(args, "dtype", config, "runtime.dtype", "auto"), device, model=args.model)
    set_global_seed(seeds[0])
    pipe = load_sd_pipeline(args.model, checkpoint, device, dtype)
    avg_payload = load_or_build_avg_target_embeddings(args.avg_target_embeddings, pipe=pipe, device=device, dtype=dtype) if args.avg_target_embeddings else None
    if args.export_avg_embeddings:
        if avg_payload is None:
            raise ValueError("--export-avg-embeddings requires --avg-target-embeddings")
        saved = save_embedding_payload(avg_payload, args.export_avg_embeddings, overwrite=args.overwrite_export_avg_embeddings)
        print(f"[done] wrote {saved}")

    common_meta = {
        "model": args.model,
        "model_version": args.model,
        "workflow": "t2i",
        "generation_mode": args.generation_mode,
        "steer": steering is not None,
        "checkpoint": str(checkpoint),
        "target_prompt": args.target_prompt,
        "safe_prompt": args.safe_prompt,
        "unsafe_prompt": args.unsafe_prompt,
        "height": height,
        "width": width,
        "num_steps": steps,
        "cfg": baseline_cfg,
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
            cli_seed_override=False,
            max_rows=args.max_rows,
            default_row_seed_strategy=True,
        )
        run_name = args.run_name or timestamped_run_name(
            f"{dataset_path.stem}_seed{default_seed}_cfgsrc{cfg_src}_cfgtar{cfg_tar}"
        )
        if legacy_cli:
            run_dir = output_root.expanduser().resolve() / args.model / ("steered" if steering else "baseline") / slugify(run_name, max_len=80)
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = create_run_directory(output_root, model=args.model, workflow="t2i", mode=args.generation_mode, run_name=run_name)
        rows_meta: list[dict] = []
        skipped = 0
        for record in records:
            base = f"{sanitize_filename_token(record.row_id)}_{prompt_first8words_slug(record.prompt)}"
            out_path = run_dir / f"{base}.png"
            if out_path.exists():
                skipped += 1
                rows_meta.append({"row_index": record.row_index, "row_id": record.row_id, "seed": record.seed, "prompt": record.prompt, "image": out_path.name, "skipped_existing": True})
                continue
            image = _generate(
                pipe=pipe, model=args.model, prompt=record.prompt, args=args, seed=record.seed,
                steering=steering, height=height, width=width, steps=steps,
                cfg_src=cfg_src, cfg_tar=cfg_tar, avg_payload=avg_payload, device=device,
            )
            image.save(out_path)
            row_meta = {"row_index": record.row_index, "row_id": record.row_id, "seed": record.seed, "prompt": record.prompt, "image": out_path.name}
            if steering is not None and args.include_baseline:
                baseline = _generate(
                    pipe=pipe, model=args.model, prompt=record.prompt, args=args, seed=record.seed,
                    steering=None, height=height, width=width, steps=steps,
                    cfg_src=cfg_src, cfg_tar=cfg_src, avg_payload=None, device=device,
                )
                baseline_path = run_dir / f"{base}_baseline.png"
                paired_path = run_dir / f"{base}_paired.png"
                baseline.save(baseline_path)
                make_paired_image(baseline, image).save(paired_path)
                row_meta.update({"baseline": baseline_path.name, "paired": paired_path.name})
            rows_meta.append(row_meta)
            write_metadata(run_dir / args.meta_name, {**common_meta, "dataset": str(dataset_path), "dataset_entry": dataset_entry, "rows": rows_meta, "skipped_existing": skipped})
        write_metadata(run_dir / args.meta_name, {**common_meta, "dataset": str(dataset_path), "dataset_entry": dataset_entry, "rows": rows_meta, "skipped_existing": skipped})
        if skipped:
            print(f"[resume] skipped existing images: {skipped}")
        print(f"[done] run dir: {run_dir}")
        return

    images = {}
    for seed in seeds:
        generated = _generate(
            pipe=pipe, model=args.model, prompt=args.prompt, args=args, seed=seed,
            steering=steering, height=height, width=width, steps=steps,
            cfg_src=cfg_src, cfg_tar=cfg_tar, avg_payload=avg_payload, device=device,
        )
        suffix = f"_{seed}" if len(seeds) > 1 else ""
        images[f"{'steered' if steering else 'baseline'}{suffix}"] = generated
        if steering is not None and args.include_baseline:
            baseline = _generate(
                pipe=pipe, model=args.model, prompt=args.prompt, args=args, seed=seed,
                steering=None, height=height, width=width, steps=steps,
                cfg_src=cfg_src, cfg_tar=cfg_src, avg_payload=None, device=device,
            )
            images[f"baseline{suffix}"] = baseline
            images[f"paired{suffix}"] = make_paired_image(baseline, generated)
    target = steering_target_label(
        args.target_prompt, args.safe_prompt, args.avg_target_embeddings
    )
    default_name = f"{args.prompt}_to_{target}" if steering else f"{args.model}_{args.prompt}"
    run_name = args.run_name or timestamped_run_name(default_name)
    if legacy_cli:
        run_dir = output_root.expanduser().resolve() / args.model / ("steered" if steering else "baseline") / slugify(run_name, max_len=80)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_directory(output_root, model=args.model, workflow="t2i", mode=args.generation_mode, run_name=run_name)
    save_images_and_metadata(run_dir, images, {**common_meta, "prompt": args.prompt, "source_prompt": args.prompt, "seed": default_seed, "seeds": seeds})


if __name__ == "__main__":
    main()
