"""SD3/SD3.5 loading, conditioning, sampling and latent decoding."""

from __future__ import annotations

import csv
import inspect
import os
from pathlib import Path
from typing import Any

from .common import SteeringParameters
from .embeddings import load_embedding_payload
from .runtime import autocast_context
from .schedules import alpha_at_step, clamp_alpha_value, mu_at_step


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Stable Diffusion generation") from exc
    return torch


def default_cfg_for_model(model: str) -> float:
    normalized = model.lower().replace(".", "")
    if normalized == "sd3":
        return 7.0
    if normalized == "sd35":
        return 3.5
    raise ValueError(f"Unsupported SD model {model!r}; expected sd3 or sd35")


default_cfg_for_version = default_cfg_for_model


def load_sd_pipeline(model: str, checkpoint: Path, device, dtype, *, image_to_image: bool = False):
    normalized = model.lower().replace(".", "")
    if normalized not in {"sd3", "sd35"}:
        raise ValueError(f"Unsupported SD model: {model}")
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint}")
    try:
        if image_to_image:
            from diffusers import StableDiffusion3Img2ImgPipeline as Pipeline
        else:
            from diffusers import StableDiffusion3Pipeline as Pipeline
    except ImportError as exc:
        feature = "StableDiffusion3Img2ImgPipeline" if image_to_image else "StableDiffusion3Pipeline"
        raise RuntimeError(f"Diffusers with {feature} support is required") from exc
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(f"[info] loading {normalized}{' img2img' if image_to_image else ''} from {checkpoint} on {device} ({dtype})")
    return Pipeline.from_pretrained(
        str(checkpoint), torch_dtype=dtype, local_files_only=True
    ).to(device)


def get_cfg_condition(pipe, prompt: str, device) -> dict[str, Any]:
    pos, neg, pos_pool, neg_pool = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt="",
        do_classifier_free_guidance=True,
        device=device,
    )
    return {"pos": pos, "neg": neg, "pos_pool": pos_pool, "neg_pool": neg_pool}


def _ensure_batch_dim(tensor):
    return tensor.unsqueeze(0) if tensor.ndim in (1, 2) else tensor


def condition_from_avg(payload: dict[str, Any], *, kind: str, fallback_neg, fallback_neg_pool, device, dtype):
    if kind == "safe":
        prompt_key = "avg_safe_prompt_embeds" if "avg_safe_prompt_embeds" in payload else "avg_tar_prompt_embeds"
        pooled_key = "avg_safe_pooled_prompt_embeds" if "avg_safe_pooled_prompt_embeds" in payload else "avg_tar_pooled_prompt_embeds"
    elif kind == "unsafe":
        prompt_key = "avg_unsafe_prompt_embeds"
        pooled_key = "avg_unsafe_pooled_prompt_embeds"
    else:
        raise ValueError(f"Unknown condition kind: {kind}")
    if prompt_key not in payload or pooled_key not in payload:
        raise ValueError(f"Average embedding payload is missing required {kind} SD keys")
    return {
        "pos": _ensure_batch_dim(payload[prompt_key]).to(device=device, dtype=dtype),
        "neg": fallback_neg.to(device=device, dtype=dtype),
        "pos_pool": _ensure_batch_dim(payload[pooled_key]).to(device=device, dtype=dtype),
        "neg_pool": fallback_neg_pool.to(device=device, dtype=dtype),
    }


def calc_v_sd3_single(pipe, latents, condition: dict[str, Any], guidance_scale: float, timestep):
    torch = _torch()
    model_input = torch.cat([latents, latents], dim=0)
    prompt_embeds = torch.cat([condition["neg"], condition["pos"]], dim=0)
    pooled = torch.cat([condition["neg_pool"], condition["pos_pool"]], dim=0)
    prediction = pipe.transformer(
        hidden_states=model_input,
        timestep=timestep.expand(model_input.shape[0]),
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]
    unconditioned, conditioned = prediction.chunk(2)
    return unconditioned + float(guidance_scale) * (conditioned - unconditioned)


def blend_velocities(v_src, v_safe, alpha: float, mode: str, *, v_unsafe=None, mu: float = 0.0):
    if mode == "add":
        return v_src + alpha * (v_safe - v_src)
    if mode == "replace":
        if v_unsafe is None:
            raise ValueError("replace mode requires an unsafe velocity")
        denominator = 1.0 + mu - alpha
        if mu < 0.0 or denominator <= 0.0:
            raise ValueError("replace mode requires mu >= 0 and alpha < 1 + mu")
        return (v_src + mu * v_safe - alpha * v_unsafe) / denominator
    raise ValueError(f"Unknown steering mode: {mode}")


def decode_sd3_latent_to_pil(pipe, latent, device, vae_dtype):
    torch = _torch()
    with torch.inference_mode():
        latent = latent.to(device=device, dtype=vae_dtype)
        shift = getattr(pipe.vae.config, "shift_factor", 0.0)
        denormalized = (latent / pipe.vae.config.scaling_factor) + shift
        with autocast_context(device):
            image = pipe.vae.decode(denormalized, return_dict=False)[0]
        return pipe.image_processor.postprocess(image)[0]


def load_avg_target_embeddings(path: str | Path) -> dict[str, Any]:
    return load_embedding_payload(path)


def _build_avg_target_embeddings_from_csv(pipe, csv_path: Path, *, device, dtype) -> dict[str, Any]:
    torch = _torch()
    sums: dict[str, Any] = {"safe": None, "safe_pool": None, "unsafe": None, "unsafe_pool": None}
    counts = {"safe": 0, "unsafe": 0}
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        for row in reader:
            for kind, column in (("safe", "negative_prompts"), ("unsafe", "positive_prompts")):
                prompt = str(row.get(column, "")).strip()
                if not prompt:
                    continue
                condition = get_cfg_condition(pipe, prompt, device)
                value = condition["pos"].detach().to(device="cpu", dtype=torch.float32)
                pooled = condition["pos_pool"].detach().to(device="cpu", dtype=torch.float32)
                sums[kind] = value.clone() if sums[kind] is None else sums[kind] + value
                pool_key = f"{kind}_pool"
                sums[pool_key] = pooled.clone() if sums[pool_key] is None else sums[pool_key] + pooled
                counts[kind] += 1
    if not counts["safe"] or not counts["unsafe"]:
        raise ValueError(
            f"{csv_path} must contain safe prompts in negative_prompts and unsafe prompts in positive_prompts"
        )
    return {
        "avg_safe_prompt_embeds": (sums["safe"] / counts["safe"]).squeeze(0).to(dtype=dtype),
        "avg_safe_pooled_prompt_embeds": (sums["safe_pool"] / counts["safe"]).squeeze(0).to(dtype=dtype),
        "avg_unsafe_prompt_embeds": (sums["unsafe"] / counts["unsafe"]).squeeze(0).to(dtype=dtype),
        "avg_unsafe_pooled_prompt_embeds": (sums["unsafe_pool"] / counts["unsafe"]).squeeze(0).to(dtype=dtype),
        "avg_source_csv": str(csv_path),
        "avg_counts": counts,
    }


def build_avg_target_embeddings_from_csv(pipe, csv_path: Path, *, device, dtype) -> dict[str, Any]:
    torch = _torch()
    with torch.inference_mode():
        return _build_avg_target_embeddings_from_csv(
            pipe, csv_path, device=device, dtype=dtype
        )


def load_or_build_avg_target_embeddings(path: str | Path, *, pipe, device, dtype) -> dict[str, Any]:
    raw = str(path)
    if raw == "data/average_embeddings/naked":
        candidate = Path(__file__).resolve().parent.parent / raw
    else:
        candidate = Path(path).expanduser()
    if candidate.is_dir() or candidate.suffix.lower() == ".pt":
        return load_embedding_payload(candidate)
    if candidate.suffix.lower() == ".csv":
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Average-embedding CSV not found: {candidate}")
        print(f"[info] building average target embeddings from CSV: {candidate}")
        return build_avg_target_embeddings_from_csv(pipe, candidate, device=device, dtype=dtype)
    raise ValueError(f"Average embeddings must be a .pt file, a directory, or a .csv file: {path}")


def _prepare_sd_latents(
    pipe,
    *,
    height: int,
    width: int,
    num_steps: int,
    generator,
    conditioning_image,
    image_strength: float | None,
):
    try:
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
    except ImportError as exc:
        raise RuntimeError("A compatible Diffusers installation is required") from exc
    device = pipe.device
    dtype = next(pipe.transformer.parameters()).dtype
    scheduler = pipe.scheduler
    timesteps, _ = retrieve_timesteps(scheduler, num_steps, device, timesteps=None)
    if conditioning_image is None:
        latents = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=pipe.transformer.config.in_channels,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=None,
        )
        return latents, timesteps, False

    strength = float(image_strength if image_strength is not None else 0.8)
    timesteps, effective_steps = pipe.get_timesteps(num_steps, strength, device)
    if effective_steps < 1:
        raise ValueError(
            f"Image strength {strength} leaves no denoising steps; increase --num-steps or --image-strength"
        )
    image = pipe.image_processor.preprocess(conditioning_image, height=height, width=width)
    latent_timestep = timesteps[:1].repeat(1)
    available = inspect.signature(pipe.prepare_latents).parameters
    values = {
        "image": image,
        "timestep": latent_timestep,
        "batch_size": 1,
        "num_images_per_prompt": 1,
        "dtype": dtype,
        "device": device,
        "generator": generator,
        "add_noise": True,
    }
    call_kwargs = {key: value for key, value in values.items() if key in available}
    latents = pipe.prepare_latents(**call_kwargs)
    if isinstance(latents, tuple):
        latents = latents[0]
    return latents, timesteps, True


def sample_sd(
    *,
    pipe,
    model: str,
    source_prompt: str,
    target_prompt: str | None,
    height: int,
    width: int,
    num_steps: int,
    cfg_src: float,
    cfg_tar: float,
    steering: SteeringParameters | None,
    generator,
    avg_target_embeddings: dict[str, Any] | None = None,
    safe_prompt: str | None = None,
    unsafe_prompt: str | None = None,
    conditioning_image=None,
    image_strength: float | None = None,
):
    """Run baseline or steering for SD3/SD3.5, optionally from an encoded image."""
    del model  # Both versions share this implementation; the pipeline owns differences.
    torch = _torch()
    with torch.inference_mode():
        device = pipe.device
        dtype = next(pipe.transformer.parameters()).dtype
        pipe.check_inputs(
            prompt=source_prompt,
            prompt_2=None,
            prompt_3=None,
            height=height,
            width=width,
            negative_prompt=None,
            negative_prompt_2=None,
            negative_prompt_3=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            callback_on_step_end_tensor_inputs=None,
            max_sequence_length=256,
            **({"strength": image_strength} if conditioning_image is not None else {}),
        )
        latents, timesteps, is_i2i = _prepare_sd_latents(
            pipe,
            height=height,
            width=width,
            num_steps=num_steps,
            generator=generator,
            conditioning_image=conditioning_image,
            image_strength=image_strength,
        )
        source = {key: value.to(device=device, dtype=dtype) for key, value in get_cfg_condition(pipe, source_prompt, device).items()}
        safe = unsafe = None
        if steering is not None:
            if safe_prompt is not None:
                safe = get_cfg_condition(pipe, safe_prompt, device)
            elif avg_target_embeddings is not None:
                safe = condition_from_avg(
                    avg_target_embeddings,
                    kind="safe",
                    fallback_neg=source["neg"],
                    fallback_neg_pool=source["neg_pool"],
                    device=device,
                    dtype=dtype,
                )
            elif target_prompt is not None:
                safe = get_cfg_condition(pipe, target_prompt, device)
            if safe is None:
                raise ValueError("Steering requires --target-prompt, --safe-prompt, or average safe embeddings")
            safe = {key: value.to(device=device, dtype=dtype) for key, value in safe.items()}
            if steering.mode == "replace":
                if unsafe_prompt is not None:
                    unsafe = get_cfg_condition(pipe, unsafe_prompt, device)
                elif avg_target_embeddings is not None:
                    unsafe = condition_from_avg(
                        avg_target_embeddings,
                        kind="unsafe",
                        fallback_neg=source["neg"],
                        fallback_neg_pool=source["neg_pool"],
                        device=device,
                        dtype=dtype,
                    )
                if unsafe is None:
                    raise ValueError("Replace steering requires --unsafe-prompt or average unsafe embeddings")
                unsafe = {key: value.to(device=device, dtype=dtype) for key, value in unsafe.items()}

        scheduler = pipe.scheduler
        for index, timestep in enumerate(timesteps):
            v_src = calc_v_sd3_single(pipe, latents, source, cfg_src, timestep)
            if steering is None:
                velocity = v_src
            else:
                a = alpha_at_step(
                    index, len(timesteps), steering.alpha_schedule,
                    steering.alpha, steering.alpha_start, steering.alpha_end,
                )
                mu = max(0.0, mu_at_step(
                    index, len(timesteps), steering.mu_schedule,
                    steering.mu, steering.mu_start, steering.mu_end,
                ))
                a = clamp_alpha_value(a, steering.mode, mu)
                v_safe = calc_v_sd3_single(pipe, latents, safe, cfg_tar, timestep)
                v_unsafe = calc_v_sd3_single(pipe, latents, unsafe, cfg_tar, timestep) if unsafe is not None else None
                velocity = blend_velocities(v_src, v_safe, a, steering.mode, v_unsafe=v_unsafe, mu=mu)
            if is_i2i:
                original_dtype = latents.dtype
                latents = scheduler.step(velocity, timestep, latents, return_dict=False)[0]
                if latents.dtype != original_dtype and torch.backends.mps.is_available():
                    latents = latents.to(original_dtype)
            else:
                scheduler._init_step_index(timestep)
                dt = scheduler.sigmas[scheduler.step_index + 1] - scheduler.sigmas[scheduler.step_index]
                latents = (latents.to(torch.float32) + dt * velocity).to(velocity.dtype)
        return latents


def sample_min_transport_sd(pipe, prompt_src: str, prompt_tar: str | None, **kwargs):
    """Compatibility adapter for the former ``sd.sample_min_transport_sd`` API."""
    steering = SteeringParameters(
        mode=kwargs.pop("mode"),
        alpha_schedule=kwargs.pop("alpha_schedule"),
        alpha=kwargs.pop("alpha"),
        alpha_start=kwargs.pop("alpha_start"),
        alpha_end=kwargs.pop("alpha_end"),
        mu_schedule=kwargs.pop("mu_schedule"),
        mu=kwargs.pop("mu_remove"),
        mu_start=kwargs.pop("mu_start"),
        mu_end=kwargs.pop("mu_end"),
    )
    return sample_sd(
        pipe=pipe,
        model=kwargs.pop("model", "sd3"),
        source_prompt=prompt_src,
        target_prompt=prompt_tar,
        steering=steering,
        safe_prompt=kwargs.pop("safe_prompt_override", None),
        unsafe_prompt=kwargs.pop("unsafe_prompt_override", None),
        **kwargs,
    )
