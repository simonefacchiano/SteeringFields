"""Compute Flux average prompt embeddings and optional offline velocity deltas.

Run ``python -m steering_fields.compute_avg_embeddings --help`` for usage.
The ``--only-avg-embedding`` mode preserves the payload construction used by
the original FlowEdit offline steering script.
"""

import argparse
import csv
import datetime as dt
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from diffusers import DiffusionPipeline, FluxPipeline

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from steering_fields.config import get_model_checkpoint, load_config
from steering_fields.utils import calc_v_flux, calculate_shift


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AVG_OUTPUT_ROOT = PROJECT_ROOT / "data" / "average_embeddings"
DEFAULT_DELTA_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "offline_steer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline prompt-pair delta estimation for FLUX.\n"
            "For each timestep, computes v_delta = mean_i[V(z_tar(t), t | c_tar_i)] - mean_i[V(z_src(t), t | c_src_i)]\n"
            "using TWO separate latent trajectories (z_src and z_tar) evolved under their respective mean velocities.\n"
            "This avoids evaluating cat velocities on dog latents (and vice versa).\n"
            "CSV convention:\n"
            "- positive_prompts: UNSAFE distribution\n"
            "- negative_prompts: SAFE distribution\n"
            "Arguments keep the legacy names src/tar, where src=UNSAFE and tar=SAFE."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML configuration file. STEERING_FIELDS_CONFIG is used when omitted.",
    )
    parser.add_argument(
        "--pairs-csv",
        type=str,
        default="FlowEdit/pairs.csv",
        help="CSV containing prompt pairs (default: FlowEdit/pairs.csv).",
    )
    parser.add_argument(
        "--src-col",
        type=str,
        default="positive_prompts",
        help="CSV column name for UNSAFE prompts (v_unsafe). Default: positive_prompts.",
    )
    parser.add_argument(
        "--tar-col",
        type=str,
        default="negative_prompts",
        help="CSV column name for SAFE prompts (v_safe). Default: negative_prompts.",
    )
    parser.add_argument(
        "--max-rows",
        "--num-rows",
        type=int,
        default=50,
        dest="max_rows",
        help="Max number of CSV rows to use (default: 50).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Local Flux checkpoint override. Otherwise read from configuration or FLUX_CHECKPOINT.",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=28)
    parser.add_argument("--cfg-src", type=float, default=1.5)
    parser.add_argument("--cfg-tar", type=float, default=5.5)
    parser.add_argument(
        "--trajectory-mode",
        type=str,
        default="twist",
        choices=("twist", "standard_steering"),
        help=(
            "How offline reference latents are evolved:\n"
            "- twist: per-row dog trajectory z_src_i; evaluate BOTH v_src and v_tar on z_src_i (cat-on-dog states).\n"
            "- standard_steering: shared z_src under POSITIVES and shared z_tar under NEGATIVES; evaluate v_src on z_src, v_tar on z_tar."
        ),
    )
    parser.add_argument(
        "--prompt-batch",
        type=int,
        default=8,
        help="Microbatch size over prompt pairs when evaluating velocities (default: 8).",
    )
    parser.add_argument(
        "--save-components",
        action="store_true",
        help="Also save per-step mean v_src and mean v_tar in delta.pt (in addition to v_delta).",
    )
    parser.add_argument(
        "--only-avg-embedding",
        action="store_true",
        help="Only compute and save average SAFE/UNSAFE embeddings (no velocity computation).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-number", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=(
            "Output directory override. Average embeddings default to "
            "data/average_embeddings; offline deltas default to outputs/offline_steer."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slugify(text: str, max_len: int = 60) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
    out = "_".join(part for part in out.split("_") if part)
    return out[:max_len] if out else "text"


def unique_avg_output_paths(output_root: Path, csv_tag: str) -> tuple[Path, Path]:
    """Return matching payload/metadata paths that cannot overwrite an earlier run."""
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_name = f"{timestamp}_{csv_tag}_avg_embedding"
    suffix = 1
    while True:
        unique_name = base_name if suffix == 1 else f"{base_name}_{suffix}"
        payload_path = output_root / f"{unique_name}.pt"
        metadata_path = output_root / f"{unique_name}.json"
        if not payload_path.exists() and not metadata_path.exists():
            return payload_path, metadata_path
        suffix += 1


def autocast_ctx(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda")
    return nullcontext()


def load_flux_pipeline(checkpoint: Path, device: torch.device, dtype: torch.dtype) -> FluxPipeline:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint}")

    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")
    os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"[info] loading {checkpoint} on {device} ({dtype})")
    pipe = DiffusionPipeline.from_pretrained(
        str(checkpoint),
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if not isinstance(pipe, FluxPipeline):
        raise TypeError(f"Loaded pipeline is not FluxPipeline: {type(pipe)}")
    return pipe.to(device)


def read_pairs(
    csv_path: Path,
    src_col: str,
    tar_col: str,
    max_rows: int,
    *,
    require_src: bool,
    require_tar: bool,
    allow_empty_tar: bool = False,
) -> tuple[list[str], list[str], str | None, str | None]:
    if not csv_path.exists():
        raise FileNotFoundError(f"pairs.csv not found: {csv_path}")

    src_prompts: list[str] = []
    tar_prompts: list[str] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("pairs.csv has no header row (expected CSV with named columns)")

        # Be lenient with column names (some CSVs use different spellings).
        src_candidates = [src_col, "positive_prompts", "positive_prompt"]
        tar_candidates = [tar_col, "negative_prompts", "negative_prompt", "negative_promtps"]

        src_key = next((k for k in src_candidates if k in reader.fieldnames), None)
        tar_key = next((k for k in tar_candidates if k in reader.fieldnames), None)

        if require_src and src_key is None:
            raise ValueError(
                f"Missing required source column in CSV. found={reader.fieldnames} need one of {src_candidates}"
            )
        if require_tar and tar_key is None:
            raise ValueError(
                f"Missing required target column in CSV. found={reader.fieldnames} need one of {tar_candidates}"
            )

        for row in reader:
            if max_rows is not None and max_rows > 0 and len(src_prompts) >= max_rows:
                break
            src = (row.get(src_key) or "").strip() if src_key is not None else ""
            tar = (row.get(tar_key) or "").strip() if tar_key is not None else ""
            if require_src and not src:
                continue
            if require_tar and not tar and not allow_empty_tar:
                continue
            # Keep lengths aligned even when src is optional.
            src_prompts.append(src)
            tar_prompts.append(tar)

    if require_src and not src_prompts:
        raise ValueError("No usable rows found in pairs.csv (empty source prompts after filtering)")
    if require_tar and not tar_prompts:
        raise ValueError("No usable rows found in pairs.csv (empty target prompts after filtering)")
    return src_prompts, tar_prompts, src_key, tar_key


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    if args.num_steps <= 0:
        raise ValueError("--num-steps must be >= 1")
    if args.prompt_batch <= 0:
        raise ValueError("--prompt-batch must be >= 1")

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device_number}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    pairs_csv = Path(args.pairs_csv)
    csv_tag = slugify(pairs_csv.stem)
    config = load_config(
        args.config,
        {"paths.flux_checkpoint": args.checkpoint},
    )
    checkpoint = get_model_checkpoint(config, "flux")
    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve()
    elif args.only_avg_embedding:
        output_root = DEFAULT_AVG_OUTPUT_ROOT
    else:
        output_root = DEFAULT_DELTA_OUTPUT_ROOT

    src_prompts, tar_prompts, src_key, tar_key = read_pairs(
        csv_path=pairs_csv,
        src_col=args.src_col,
        tar_col=args.tar_col,
        max_rows=args.max_rows,
        require_src=not args.only_avg_embedding,
        require_tar=True,
        allow_empty_tar=True,
    )
    num_pairs = len(src_prompts)
    print(f"[info] loaded {num_pairs} prompt pairs from {pairs_csv} (unsafe_col={src_key} safe_col={tar_key})")

    pipe = load_flux_pipeline(checkpoint, device, dtype)
    scheduler = pipe.scheduler

    if args.only_avg_embedding:
        # Encode SAFE prompts (negative_prompts) and, if available, UNSAFE prompts (positive_prompts),
        # then store their mean embeddings for inference-time reuse.
        safe_prompt_embeds, safe_pooled_prompt_embeds, safe_text_ids = pipe.encode_prompt(
            prompt=tar_prompts,  # SAFE
            prompt_2=None,
            device=device,
        )
        unsafe_payload = None
        if any(p.strip() for p in src_prompts):
            unsafe_prompt_embeds, unsafe_pooled_prompt_embeds, unsafe_text_ids = pipe.encode_prompt(
                prompt=src_prompts,  # UNSAFE
                prompt_2=None,
                device=device,
            )
            unsafe_payload = (unsafe_prompt_embeds, unsafe_pooled_prompt_embeds, unsafe_text_ids)
        with torch.no_grad():
            avg_safe_prompt_embeds = safe_prompt_embeds.to(torch.float32).mean(dim=0).to(safe_prompt_embeds.dtype).cpu()
            avg_safe_pooled_prompt_embeds = (
                safe_pooled_prompt_embeds.to(torch.float32).mean(dim=0).to(safe_pooled_prompt_embeds.dtype).cpu()
            )
            avg_safe_text_ids = safe_text_ids.cpu()

            avg_unsafe_prompt_embeds = None
            avg_unsafe_pooled_prompt_embeds = None
            avg_unsafe_text_ids = None
            if unsafe_payload is not None:
                up, upp, uti = unsafe_payload
                avg_unsafe_prompt_embeds = up.to(torch.float32).mean(dim=0).to(up.dtype).cpu()
                avg_unsafe_pooled_prompt_embeds = upp.to(torch.float32).mean(dim=0).to(upp.dtype).cpu()
                avg_unsafe_text_ids = uti.cpu()

        payload_path, metadata_path = unique_avg_output_paths(output_root, csv_tag)
        payload = {
            "src_col": src_key,
            "tar_col": tar_key,
            "num_pairs": num_pairs,
            "avg_safe_prompt_embeds": avg_safe_prompt_embeds,
            "avg_safe_pooled_prompt_embeds": avg_safe_pooled_prompt_embeds,
            "avg_safe_text_ids": avg_safe_text_ids,
            "checkpoint": str(checkpoint),
        }
        if avg_unsafe_prompt_embeds is not None:
            payload.update(
                {
                    "avg_unsafe_prompt_embeds": avg_unsafe_prompt_embeds,
                    "avg_unsafe_pooled_prompt_embeds": avg_unsafe_pooled_prompt_embeds,
                    "avg_unsafe_text_ids": avg_unsafe_text_ids,
                }
            )
        torch.save(payload, payload_path)

        meta = {
            "pairs_csv": str(pairs_csv),
            "src_col": src_key,
            "tar_col": tar_key,
            "num_pairs": num_pairs,
            "max_rows": args.max_rows,
            "only_avg_embedding": True,
            "checkpoint": str(checkpoint),
            "seed": args.seed,
            "device": str(device),
            "dtype": str(dtype),
            "has_unsafe_avg": avg_unsafe_prompt_embeds is not None,
            "output": str(payload_path),
        }
        metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[done] wrote {payload_path}")
        print(f"[done] wrote {metadata_path}")
        print(f"[done] out dir: {output_root}")
        return

    # Create an initial noise latent z in the model's packed format.
    generator = torch.Generator(device=device).manual_seed(args.seed)
    num_channels_latents = pipe.transformer.config.in_channels // 4
    pipe.check_inputs(
        prompt=src_prompts[0],
        prompt_2=None,
        height=args.height,
        width=args.width,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    )
    latents, latent_image_ids = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=num_channels_latents,
        height=args.height,
        width=args.width,
        dtype=next(pipe.transformer.parameters()).dtype,
        device=device,
        generator=generator,
        latents=None,
    )
    # Diffusers >= 0.36 returns packed latents when `latents=None`.
    if latents.ndim == 3:
        z0 = latents
    elif latents.ndim == 4:
        z0 = pipe._pack_latents(latents, 1, num_channels_latents, latents.shape[2], latents.shape[3])
    else:
        raise ValueError(f"Unexpected latents shape from prepare_latents: {tuple(latents.shape)}")

    # Offline reference latents.
    #
    # trajectory_mode="twist" (ROW-WISE):
    #   - keep a separate dog/source trajectory per CSV row: z_src_i(t)
    #   - evolve each z_src_i(t) under its own POSITIVES/source velocity V(z_src_i(t), t | c_src_i)
    #   - evaluate the paired cat/target velocity on the SAME dog state:
    #       V(z_src_i(t), t | c_tar_i)
    #   - average across rows to obtain v_src_mean[t], v_tar_mean[t], and v_delta[t]
    #
    # trajectory_mode="standard_steering":
    #   - keep one shared z_src(t) evolved under mean POSITIVES/source velocity
    #   - keep one shared z_tar(t) evolved under mean NEGATIVES/target velocity
    #   - evaluate v_src on z_src(t) and v_tar on z_tar(t)
    if args.trajectory_mode == "twist":
        z_src_all = z0.expand(num_pairs, -1, -1).clone()
        z_src = None
        z_tar = None
    else:
        z_src_all = None
        z_src = z0.clone()
        z_tar = z0.clone()

    # Prepare timesteps (same pattern used elsewhere in this repo for FLUX).
    sigmas = np.linspace(1.0, 1 / args.num_steps, args.num_steps)
    image_seq_len = z0.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, _ = retrieve_timesteps(
        scheduler,
        args.num_steps,
        device,
        timesteps=None,
        sigmas=sigmas,
        mu=mu,
    )
    pipe._num_timesteps = len(timesteps)

    # Encode all prompts once; we will microbatch only the velocity evaluation.
    src_prompt_embeds, src_pooled_prompt_embeds, src_text_ids = pipe.encode_prompt(
        prompt=src_prompts,
        prompt_2=None,
        device=device,
    )
    tar_prompt_embeds, tar_pooled_prompt_embeds, tar_text_ids = pipe.encode_prompt(
        prompt=tar_prompts,
        prompt_2=None,
        device=device,
    )
    # Average SAFE embeddings once so inference can reuse a single representative "safe prompt"
    # without re-running the text encoders. Also average UNSAFE embeddings for remove-mode.
    #
    # Note: This is a heuristic; V(z,t | mean(embeds)) is not guaranteed to equal mean(V(z,t | embeds_i)).
    with torch.no_grad():
        avg_safe_prompt_embeds = tar_prompt_embeds.to(torch.float32).mean(dim=0).to(tar_prompt_embeds.dtype).cpu()
        avg_safe_pooled_prompt_embeds = (
            tar_pooled_prompt_embeds.to(torch.float32).mean(dim=0).to(tar_pooled_prompt_embeds.dtype).cpu()
        )
        avg_safe_text_ids = tar_text_ids.cpu()

        avg_unsafe_prompt_embeds = src_prompt_embeds.to(torch.float32).mean(dim=0).to(src_prompt_embeds.dtype).cpu()
        avg_unsafe_pooled_prompt_embeds = (
            src_pooled_prompt_embeds.to(torch.float32).mean(dim=0).to(src_pooled_prompt_embeds.dtype).cpu()
        )
        avg_unsafe_text_ids = src_text_ids.cpu()

    if pipe.transformer.config.guidance_embeds:
        src_guidance_all = torch.full((num_pairs,), float(args.cfg_src), device=device)
        tar_guidance_all = torch.full((num_pairs,), float(args.cfg_tar), device=device)
    else:
        src_guidance_all = None
        tar_guidance_all = None

    deltas: list[torch.Tensor] = []
    v_src_means: list[torch.Tensor] = []
    v_tar_means: list[torch.Tensor] = []
    dt_values: list[float] = []

    for step_idx, t in enumerate(timesteps):
        scheduler._init_step_index(t)
        sigma_i = scheduler.sigmas[scheduler.step_index]
        sigma_im1 = scheduler.sigmas[scheduler.step_index + 1]
        dt_sigma = float((sigma_im1 - sigma_i).item())
        dt_values.append(dt_sigma)

        delta_sum = torch.zeros_like(z0, dtype=torch.float32)
        src_sum = torch.zeros_like(z0, dtype=torch.float32)
        tar_sum = torch.zeros_like(z0, dtype=torch.float32)
        total_rows = 0
        microbatches = 0

        for start in range(0, num_pairs, args.prompt_batch):
            end = min(start + args.prompt_batch, num_pairs)
            bsz = end - start

            if args.trajectory_mode == "twist":
                z_src_batch = z_src_all[start:end]
                z_tar_batch = z_src_batch
            else:
                z_src_batch = z_src.expand(bsz, -1, -1).contiguous()
                z_tar_batch = z_tar.expand(bsz, -1, -1).contiguous()

            src_guidance = src_guidance_all[start:end] if src_guidance_all is not None else None
            tar_guidance = tar_guidance_all[start:end] if tar_guidance_all is not None else None

            v_src = calc_v_flux(
                pipe,
                latents=z_src_batch,
                prompt_embeds=src_prompt_embeds[start:end],
                pooled_prompt_embeds=src_pooled_prompt_embeds[start:end],
                guidance=src_guidance,
                text_ids=src_text_ids,
                latent_image_ids=latent_image_ids,
                t=t,
            )
            v_tar = calc_v_flux(
                pipe,
                latents=z_tar_batch,
                prompt_embeds=tar_prompt_embeds[start:end],
                pooled_prompt_embeds=tar_pooled_prompt_embeds[start:end],
                guidance=tar_guidance,
                text_ids=tar_text_ids,
                latent_image_ids=latent_image_ids,
                t=t,
            )

            # EXACTLY the same delta definition used in simone_min_transport_steering.py:
            #   v_delta = v_tar - v_src
            v_delta = (v_tar - v_src).to(torch.float32)
            delta_sum += v_delta.sum(dim=0, keepdim=True)
            src_sum += v_src.to(torch.float32).sum(dim=0, keepdim=True)
            tar_sum += v_tar.to(torch.float32).sum(dim=0, keepdim=True)
            total_rows += bsz
            microbatches += 1

            if args.trajectory_mode == "twist":
                # Row-wise dog trajectory update: z_src_i <- z_src_i + dt * V(z_src_i, t | c_src_i)
                orig_dtype = z_src_batch.dtype
                z_next = z_src_batch.to(torch.float32) + dt_sigma * v_src.to(torch.float32)
                z_src_all[start:end] = z_next.to(orig_dtype)

        delta_avg = delta_sum / float(total_rows)
        deltas.append(delta_avg.squeeze(0).to(z0.dtype))
        if args.save_components:
            v_src_means.append((src_sum / float(total_rows)).squeeze(0).to(z0.dtype))
            v_tar_means.append((tar_sum / float(total_rows)).squeeze(0).to(z0.dtype))
        print(
            f"[step {step_idx:02d}/{len(timesteps)-1:02d}] dt_sigma={dt_sigma:+.6f} "
            f"pairs={num_pairs} microbatches={microbatches}"
        )

        if args.trajectory_mode == "standard_steering":
            # Shared-trajectory update for baseline mode.
            v_src_avg = src_sum / float(total_rows)
            v_tar_avg = tar_sum / float(total_rows)
            src_dtype = z_src.dtype
            tar_dtype = z_tar.dtype
            z_src = z_src.to(torch.float32)
            z_tar = z_tar.to(torch.float32)
            z_src = z_src + dt_sigma * v_src_avg
            z_tar = z_tar + dt_sigma * v_tar_avg
            z_src = z_src.to(src_dtype)
            z_tar = z_tar.to(tar_dtype)

    out_dir = output_root / csv_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "deltas": torch.stack(deltas, dim=0).cpu(),  # (T, num_patches, channels)
        "dt_sigma": torch.tensor(dt_values, dtype=torch.float32),
        "src_col": src_key,
        "tar_col": tar_key,
        "trajectory_mode": args.trajectory_mode,
        "src_prompts": src_prompts,
        "tar_prompts": tar_prompts,
        # SAFE/UNSAFE averages for repulsive-attractive remove-mode.
        "avg_safe_prompt_embeds": avg_safe_prompt_embeds,
        "avg_safe_pooled_prompt_embeds": avg_safe_pooled_prompt_embeds,
        "avg_safe_text_ids": avg_safe_text_ids,
        "avg_unsafe_prompt_embeds": avg_unsafe_prompt_embeds,
        "avg_unsafe_pooled_prompt_embeds": avg_unsafe_pooled_prompt_embeds,
        "avg_unsafe_text_ids": avg_unsafe_text_ids,
        "height": args.height,
        "width": args.width,
        "num_steps": args.num_steps,
        "cfg_src": args.cfg_src,
        "cfg_tar": args.cfg_tar,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
    }
    if args.save_components:
        payload["v_src_mean"] = torch.stack(v_src_means, dim=0).cpu()
        payload["v_tar_mean"] = torch.stack(v_tar_means, dim=0).cpu()
    torch.save(payload, out_dir / "delta.pt")

    meta = {
        "pairs_csv": str(pairs_csv),
        "src_col": src_key,
        "tar_col": tar_key,
        "trajectory_mode": args.trajectory_mode,
        "num_pairs": num_pairs,
        "max_rows": args.max_rows,
        "height": args.height,
        "width": args.width,
        "num_steps": args.num_steps,
        "cfg_src": args.cfg_src,
        "cfg_tar": args.cfg_tar,
        "saved_avg_safe_embedding": True,
        "saved_avg_unsafe_embedding": True,
        "only_avg_embedding": False,
        "prompt_batch": args.prompt_batch,
        "seed": args.seed,
        "device": str(device),
        "dtype": str(dtype),
        "checkpoint": str(checkpoint),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] wrote {out_dir / 'delta.pt'}")
    print(f"[done] wrote {out_dir / 'meta.json'}")
    print(f"[done] out dir: {out_dir}")


if __name__ == "__main__":
    main()
