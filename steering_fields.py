"""Core Steering Fields algorithm.

This is the best starting point for understanding the method. Model loading,
conditioning and latent preparation live in the backend utility modules; the
steering equation and the FLUX integration loop live here.
"""

from __future__ import annotations

from typing import Any

from .common import SteeringParameters
from .flux_utils import (
    _average_flux_conditions,
    _prepare_latents,
    _torch,
    build_patch_roi_mask,
    calc_v_flux,
)
from .schedules import alpha_at_step, clamp_alpha_value, mu_at_step

__all__ = ["blend_velocities", "sample_min_transport_flux"]


def blend_velocities(
    v_src,
    v_safe,
    alpha: float,
    mode: str,
    *,
    v_unsafe=None,
    mu: float = 0.0,
):
    """Construct the steered vector field from source and concept velocities.

    ``add`` interpolates from the source field toward the safe field::

        v = v_src + alpha * (v_safe - v_src)

    ``replace`` removes the unsafe component while adding the safe component::

        v = (v_src + mu * v_safe - alpha * v_unsafe) / (1 + mu - alpha)

    The denominator in replace mode keeps the affine weights normalized.
    """
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


def sample_min_transport_flux(
    *,
    pipe,
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
    initial_noise=None,
    conditioning_image=None,
    image_strength: float | None = None,
    timestep_start: int = 1,
    box_xyxy: tuple[float, float, float, float] | None = None,
    box_feather: float = 0.0,
):
    """Integrate the source or minimal-transport FLUX vector field.

    Passing ``steering=None`` follows the unmodified source field and produces
    the baseline.  Otherwise, each Euler step evaluates the source and concept
    fields at the same latent, blends them with :func:`blend_velocities`, and
    advances the latent with the resulting field.  The same implementation is
    used by text-to-image and image-to-image commands.
    """
    torch = _torch()
    with torch.inference_mode():
        device = pipe.device
        dtype = next(pipe.transformer.parameters()).dtype
        pipe.check_inputs(
            prompt=source_prompt,
            prompt_2=None,
            height=height,
            width=width,
            **({"strength": image_strength} if conditioning_image is not None else {}),
            callback_on_step_end_tensor_inputs=None,
            max_sequence_length=512,
        )

        latents, latent_image_ids, timesteps, is_i2i = _prepare_latents(
            pipe,
            height=height,
            width=width,
            num_steps=num_steps,
            generator=generator,
            initial_noise=initial_noise,
            conditioning_image=conditioning_image,
            image_strength=image_strength,
        )
        pipe._num_timesteps = len(timesteps)
        source_condition = pipe.encode_prompt(
            prompt=source_prompt, prompt_2=None, device=device
        )

        guidance_supported = bool(pipe.transformer.config.guidance_embeds)
        source_guidance = (
            torch.tensor([cfg_src], device=device).expand(latents.shape[0])
            if guidance_supported
            else None
        )
        target_guidance = (
            torch.tensor([cfg_tar], device=device).expand(latents.shape[0])
            if guidance_supported
            else None
        )

        safe_condition = unsafe_condition = None
        if steering is not None:
            if avg_target_embeddings is not None:
                averages = _average_flux_conditions(
                    avg_target_embeddings, device=device, dtype=dtype
                )
                safe_condition = averages["safe"]
                unsafe_condition = averages["unsafe"]
            if safe_prompt is not None:
                safe_condition = pipe.encode_prompt(
                    prompt=safe_prompt, prompt_2=None, device=device
                )
            elif safe_condition is None and target_prompt is not None:
                safe_condition = pipe.encode_prompt(
                    prompt=target_prompt, prompt_2=None, device=device
                )
            if unsafe_prompt is not None:
                unsafe_condition = pipe.encode_prompt(
                    prompt=unsafe_prompt, prompt_2=None, device=device
                )
            if safe_condition is None:
                raise ValueError(
                    "Steering requires --target-prompt, --safe-prompt, or "
                    "average safe embeddings"
                )
            if steering.mode == "replace" and unsafe_condition is None:
                raise ValueError(
                    "Replace steering requires --unsafe-prompt or average "
                    "unsafe embeddings"
                )

        roi_mask = (
            build_patch_roi_mask(
                latents=latents,
                width=width,
                height=height,
                vae_scale_factor=pipe.vae_scale_factor,
                box_xyxy=box_xyxy,
                feather_px=box_feather,
            )
            if is_i2i
            else None
        )

        scheduler = pipe.scheduler
        for step_index, timestep in enumerate(timesteps):
            source_velocity = calc_v_flux(
                pipe,
                latents,
                source_condition[0],
                source_condition[1],
                source_guidance,
                source_condition[2],
                latent_image_ids,
                timestep,
            )

            if steering is None or step_index + 1 < timestep_start:
                velocity = source_velocity
            else:
                alpha = alpha_at_step(
                    step_index,
                    len(timesteps),
                    steering.alpha_schedule,
                    steering.alpha,
                    steering.alpha_start,
                    steering.alpha_end,
                )
                mu = max(
                    0.0,
                    mu_at_step(
                        step_index,
                        len(timesteps),
                        steering.mu_schedule,
                        steering.mu,
                        steering.mu_start,
                        steering.mu_end,
                    ),
                )
                alpha = clamp_alpha_value(alpha, steering.mode, mu)

                safe_velocity = calc_v_flux(
                    pipe,
                    latents,
                    safe_condition[0],
                    safe_condition[1],
                    target_guidance,
                    safe_condition[2],
                    latent_image_ids,
                    timestep,
                )
                unsafe_velocity = None
                if steering.mode == "replace":
                    unsafe_velocity = calc_v_flux(
                        pipe,
                        latents,
                        unsafe_condition[0],
                        unsafe_condition[1],
                        target_guidance,
                        unsafe_condition[2],
                        latent_image_ids,
                        timestep,
                    )
                velocity = blend_velocities(
                    source_velocity,
                    safe_velocity,
                    alpha,
                    steering.mode,
                    v_unsafe=unsafe_velocity,
                    mu=mu,
                )

            if roi_mask is not None:
                velocity = source_velocity + roi_mask * (velocity - source_velocity)

            if is_i2i:
                original_dtype = latents.dtype
                latents = scheduler.step(
                    velocity, timestep, latents, return_dict=False
                )[0]
                if latents.dtype != original_dtype and torch.backends.mps.is_available():
                    latents = latents.to(original_dtype)
            else:
                scheduler._init_step_index(timestep)
                dt = (
                    scheduler.sigmas[scheduler.step_index + 1]
                    - scheduler.sigmas[scheduler.step_index]
                )
                latents = (
                    latents.to(torch.float32) + dt * velocity
                ).to(velocity.dtype)

        return pipe._unpack_latents(
            latents, height, width, pipe.vae_scale_factor
        )
