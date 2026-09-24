"""Compatibility implementation for the historical ``flux.py`` command."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Sequence

from ..config import get_model_checkpoint, load_config
from ..datasets import load_prompt_records
from ..flux_utils import decode_flux_latent_to_pil, load_flux_pipeline
from ..outputs import find_existing_image, prompt_first8words_slug, sanitize_filename_token, write_metadata
from ..runtime import make_generator, resolve_device, resolve_dtype, set_global_seed
from .. import sample_min_transport_flux


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vanilla FLUX generation (compatibility interface).")
    parser.add_argument("--input", required=True, help="Prompt text or CSV path.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--source-column", default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--cfg-src", "--guidance", "--cfg", dest="cfg_src", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--img_path", "--img-path", "--image", dest="image", default=None)
    parser.add_argument("--img_strength", "--img-strength", "--image-strength", dest="image_strength", type=float, default=0.8)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-number", type=int, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--output-root", default=None)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    overrides = {
        "paths.flux_checkpoint": args.checkpoint,
        "paths.output_root": args.output_root,
        "runtime.device": args.device,
        "runtime.device_number": args.device_number,
        "runtime.dtype": args.dtype,
    }
    config = load_config(args.config, overrides)
    checkpoint = get_model_checkpoint(config, "flux")
    output_root = Path(config.get("paths.output_root"))
    output_root.mkdir(parents=True, exist_ok=True)
    height = int(args.height if args.height is not None else config.get("defaults.height", 1024))
    width = int(args.width if args.width is not None else config.get("defaults.width", 1024))
    steps = int(args.num_steps if args.num_steps is not None else config.get("defaults.num_steps", 28))
    seed = int(args.seed if args.seed is not None else config.get("defaults.seed", 42))
    if steps <= 0:
        raise ValueError("--num-steps must be >= 1")
    if not 0.0 < args.image_strength <= 1.0:
        raise ValueError("--img_strength must be in (0, 1]")

    device = resolve_device(config.get("runtime.device", "auto"), config.get("runtime.device_number", 0))
    dtype = resolve_dtype(config.get("runtime.dtype", "auto"), device, model="flux")
    set_global_seed(seed)
    conditioning_image = None
    if args.image:
        from PIL import Image
        image_path = Path(args.image).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Conditioning image not found: {image_path}")
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        conditioning_image = Image.open(image_path).convert("RGB").resize((width, height), resampling)
    pipe = load_flux_pipeline(checkpoint, device, dtype, image_to_image=conditioning_image is not None)

    def generate(prompt: str, row_seed: int):
        latent = sample_min_transport_flux(
            pipe=pipe,
            source_prompt=prompt,
            target_prompt=None,
            height=height,
            width=width,
            num_steps=steps,
            cfg_src=float(args.cfg_src),
            cfg_tar=float(args.cfg_src),
            steering=None,
            generator=make_generator(device, row_seed),
            conditioning_image=conditioning_image,
            image_strength=float(args.image_strength) if conditioning_image is not None else None,
        )
        return decode_flux_latent_to_pil(pipe, latent, device, next(pipe.vae.parameters()).dtype)

    input_path = Path(args.input).expanduser()
    try:
        is_csv = input_path.exists() and input_path.suffix.lower() == ".csv"
    except OSError:
        is_csv = False
    stamp = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    if not is_csv:
        prompt = args.input.strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        image = generate(prompt, seed)
        base = f"{prompt_first8words_slug(prompt)}_cfg_{args.cfg_src}_seed_{seed}_{stamp}"
        output = output_root / f"{base}.png"
        duplicate = 1
        while output.exists():
            output = output_root / f"{base}_{duplicate}.png"
            duplicate += 1
        image.save(output)
        write_metadata(output.with_suffix(".json"), {
            "mode": "single_prompt", "checkpoint": str(checkpoint), "input": args.input,
            "prompt": prompt, "seed": seed, "cfg_src": args.cfg_src, "height": height,
            "width": width, "num_steps": steps, "img_path": args.image,
            "img_strength": args.image_strength if args.image else None,
            "device": str(device), "dtype": str(dtype),
        })
        print(f"[done] saved flux: {output}")
        return

    dataset_config = Path(args.dataset_config).expanduser().resolve() if args.dataset_config else Path(config.get("paths.dataset_config"))
    csv_path, entry, records = load_prompt_records(
        input_path,
        dataset_config=dataset_config,
        source_column=args.source_column,
        base_seed=seed,
        cli_seed_override=args.seed is not None,
        max_rows=args.max_rows,
    )
    started = perf_counter()
    rows_meta: list[dict] = []
    meta_path = output_root / f"meta_{csv_path.stem}_{args.cfg_src}_seed_{records[0].seed}_{stamp}.json"
    metadata = {
        "mode": "csv", "checkpoint": str(checkpoint), "input_csv": str(csv_path),
        "dataset_config": str(dataset_config), "dataset_entry": entry, "cfg_src": args.cfg_src,
        "height": height, "width": width, "num_steps": steps, "img_path": args.image,
        "img_strength": args.image_strength if args.image else None, "device": str(device),
        "dtype": str(dtype), "rows": rows_meta,
    }
    write_metadata(meta_path, metadata)
    search_dirs = [output_root / "final", output_root] if (output_root / "final").exists() else [output_root]
    for record in records:
        name_base = f"{sanitize_filename_token(record.row_id)}_{prompt_first8words_slug(record.prompt)}"
        existing = find_existing_image(name_base, search_dirs, suffixes=(".jpg",))
        if existing:
            rows_meta.append({"csv_row_index": record.row_index, "filename_id": record.row_id, "prompt": record.prompt, "seed": record.seed, "image": existing.name, "skipped_existing": True})
            write_metadata(meta_path, metadata)
            continue
        image = generate(record.prompt, record.seed)
        output = output_root / f"{name_base}.jpg"
        duplicate = 1
        while output.exists():
            output = output_root / f"{name_base}_{duplicate}.jpg"
            duplicate += 1
        image.convert("RGB").save(output, format="JPEG", quality=95)
        rows_meta.append({"csv_row_index": record.row_index, "filename_id": record.row_id, "prompt": record.prompt, "seed": record.seed, "image": output.name, "img_path": args.image})
        write_metadata(meta_path, metadata)
    total = perf_counter() - started
    metadata["total_generation_time"] = f"{int(total // 3600):02d}:{int((total % 3600) // 60):02d}:{int(total % 60):02d}"
    write_metadata(meta_path, metadata)
    print(f"[done] run dir: {output_root}")
