"""Flux model loading, conditioning, sampling and latent decoding.

This module intentionally has no CLI or output-writing responsibilities. PyTorch,
NumPy and Diffusers are loaded only when a generation function is called so that
configuration inspection and ``--help`` work on login nodes without ML packages.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .runtime import autocast_context


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Flux generation") from exc
    return torch


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for Flux generation") from exc
    return np


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
) -> float:
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * slope + (base_shift - slope * base_seq_len)


def load_flux_pipeline(checkpoint: Path, device, dtype, *, image_to_image: bool = False):
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint}")
    try:
        from diffusers import DiffusionPipeline, FluxPipeline
        if image_to_image:
            from diffusers import FluxImg2ImgPipeline
    except ImportError as exc:
        raise RuntimeError("Diffusers with Flux support is required for Flux generation") from exc

    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")
    os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    pipeline_class = FluxImg2ImgPipeline if image_to_image else DiffusionPipeline
    print(f"[info] loading FLUX{' img2img' if image_to_image else ''} from {checkpoint} on {device} ({dtype})")
    pipe = pipeline_class.from_pretrained(
        str(checkpoint),
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if not image_to_image and not isinstance(pipe, FluxPipeline):
        raise TypeError(f"Loaded pipeline is not FluxPipeline: {type(pipe)}")
    return pipe.to(device)


def load_flux_img2img_pipeline(checkpoint: Path, device, dtype):
    return load_flux_pipeline(checkpoint, device, dtype, image_to_image=True)


def decode_flux_latent_to_pil(pipe, latent, device, vae_dtype):
    torch = _torch()
    with torch.inference_mode():
        latent = latent.to(device=device, dtype=vae_dtype)
        decoded = (latent / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        with autocast_context(device):
            image = pipe.vae.decode(decoded, return_dict=False)[0]
        return pipe.image_processor.postprocess(image)[0]


def calc_v_flux(pipe, latents, prompt_embeds, pooled_prompt_embeds, guidance, text_ids, latent_image_ids, t):
    torch = _torch()
    timestep = t.expand(latents.shape[0])
    with torch.no_grad():
        return pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]


def parse_box(box: str) -> tuple[float, float, float, float]:
    parts = [part.strip() for part in box.split(",")]
    if len(parts) != 4:
        raise ValueError("--box must use x0,y0,x1,y1")
    try:
        x0, y0, x1, y1 = map(float, parts)
    except ValueError as exc:
        raise ValueError("--box must contain four numeric values") from exc
    if x1 <= x0 or y1 <= y0:
        raise ValueError("--box must satisfy x1 > x0 and y1 > y0")
    return x0, y0, x1, y1


def build_patch_roi_mask(
    *,
    latents,
    width: int,
    height: int,
    vae_scale_factor: int,
    box_xyxy: tuple[float, float, float, float] | None,
    feather_px: float,
):
    if box_xyxy is None:
        return None
    torch = _torch()
    grid_h = height // (vae_scale_factor * 2)
    grid_w = width // (vae_scale_factor * 2)
    if grid_h * grid_w != latents.shape[1]:
        raise ValueError(
            f"Cannot map ROI onto latent grid: {grid_w}x{grid_h} does not match {latents.shape[1]} tokens"
        )
    x0, y0, x1, y1 = box_xyxy
    x_centers = (torch.arange(grid_w, device=latents.device, dtype=torch.float32) + 0.5) * (width / grid_w)
    y_centers = (torch.arange(grid_h, device=latents.device, dtype=torch.float32) + 0.5) * (height / grid_h)
    xx, yy = torch.meshgrid(x_centers, y_centers, indexing="xy")
    if feather_px <= 0.0:
        mask = ((xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)).to(torch.float32)
    else:
        inside = torch.minimum(
            torch.minimum(xx - x0, x1 - xx),
            torch.minimum(yy - y0, y1 - yy),
        )
        mask = (inside / feather_px).clamp(0.0, 1.0)
    return mask.reshape(1, -1, 1).to(device=latents.device, dtype=latents.dtype)


def _scheduler_value(config: Any, key: str, fallback: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(key, fallback)
    return getattr(config, key, fallback)


def _prepare_latents(
    pipe,
    *,
    height: int,
    width: int,
    num_steps: int,
    generator,
    initial_noise,
    conditioning_image,
    image_strength: float | None,
):
    torch = _torch()
    np = _numpy()
    try:
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
    except ImportError as exc:
        raise RuntimeError("A compatible Diffusers installation is required") from exc

    device = pipe.device
    dtype = next(pipe.transformer.parameters()).dtype
    scheduler = pipe.scheduler
    channels = pipe.transformer.config.in_channels // 4
    sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
    if conditioning_image is None:
        latents, image_ids = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=channels,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator if initial_noise is None else None,
            latents=initial_noise.to(device=device, dtype=dtype) if initial_noise is not None else None,
        )
        if latents.ndim == 4:
            latents = pipe._pack_latents(latents, 1, channels, latents.shape[2], latents.shape[3])
        if latents.ndim != 3:
            raise ValueError(f"Unexpected Flux latent shape: {tuple(latents.shape)}")
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            _scheduler_value(scheduler.config, "base_image_seq_len", 256),
            _scheduler_value(scheduler.config, "max_image_seq_len", 4096),
            _scheduler_value(scheduler.config, "base_shift", 0.5),
            _scheduler_value(scheduler.config, "max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(scheduler, num_steps, device, timesteps=None, sigmas=sigmas, mu=mu)
        return latents, image_ids, timesteps, False

    strength = float(image_strength if image_strength is not None else 0.8)
    pipe.check_inputs(
        prompt="",
        prompt_2=None,
        strength=strength,
        height=height,
        width=width,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    )
    image_seq_len = (height // pipe.vae_scale_factor // 2) * (width // pipe.vae_scale_factor // 2)
    mu = calculate_shift(
        image_seq_len,
        _scheduler_value(scheduler.config, "base_image_seq_len", 256),
        _scheduler_value(scheduler.config, "max_image_seq_len", 4096),
        _scheduler_value(scheduler.config, "base_shift", 0.5),
        _scheduler_value(scheduler.config, "max_shift", 1.15),
    )
    retrieve_timesteps(scheduler, num_steps, device, timesteps=None, sigmas=sigmas, mu=mu)
    timesteps, effective_steps = pipe.get_timesteps(num_steps, strength, device)
    if effective_steps < 1:
        raise ValueError(
            f"Image strength {strength} leaves no denoising steps; increase --num-steps or --image-strength"
        )
    image_tensor = pipe.image_processor.preprocess(conditioning_image, height=height, width=width)
    latent_timestep = timesteps[:1].repeat(1)
    latents, image_ids = pipe.prepare_latents(
        image_tensor,
        latent_timestep,
        batch_size=1,
        num_channels_latents=channels,
        height=height,
        width=width,
        dtype=dtype,
        device=device,
        generator=generator,
        latents=None,
    )
    return latents, image_ids, timesteps, True


def _average_flux_conditions(payload: dict[str, Any], *, device, dtype) -> dict[str, tuple[Any, Any, Any] | None]:
    safe_keys = (
        ("avg_safe_prompt_embeds", "avg_safe_pooled_prompt_embeds", "avg_safe_text_ids")
        if "avg_safe_prompt_embeds" in payload
        else ("avg_tar_prompt_embeds", "avg_tar_pooled_prompt_embeds", "avg_tar_text_ids")
    )
    if not all(key in payload for key in safe_keys):
        raise ValueError(f"Average embedding payload is missing Flux safe keys: {safe_keys}")

    def convert(keys):
        prompt, pooled, text_ids = (payload[key].to(device=device, dtype=dtype) for key in keys)
        if prompt.ndim != 2 or pooled.ndim != 1 or text_ids.ndim != 2:
            raise ValueError("Flux average embeddings must have shapes (seq,dim), (dim,), and (seq,3)")
        return prompt.unsqueeze(0), pooled.unsqueeze(0), text_ids

    unsafe_keys = ("avg_unsafe_prompt_embeds", "avg_unsafe_pooled_prompt_embeds", "avg_unsafe_text_ids")
    return {
        "safe": convert(safe_keys),
        "unsafe": convert(unsafe_keys) if all(key in payload for key in unsafe_keys) else None,
    }
