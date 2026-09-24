"""Shared legacy-compatible utilities for SteeringFields workflows."""

from typing import Optional, Tuple, Union
import datetime as dt
import importlib.util
import json
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from diffusers import DiffusionPipeline, FluxPipeline, FlowMatchEulerDiscreteScheduler
import diffusers.pipelines.pipeline_utils as pipeline_utils
try:
    from diffusers import FluxImg2ImgPipeline
except ImportError:
    FluxImg2ImgPipeline = None
from PIL import Image
from tqdm import tqdm

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slugify(text: str, max_len: int = 40) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
    out = "_".join(part for part in out.split("_") if part)
    return out[:max_len] if out else "text"


def save_outputs(output_root: Path, images: dict, meta: dict, name: str) -> Path:
    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{stamp}_{slugify(name)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for key, img in images.items():
        img.save(run_dir / f"{key}.png")
        print(f"[done] saved {key}: {run_dir / f'{key}.png'}")

    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] run dir: {run_dir}")
    return run_dir


def make_paired_image(left: Image.Image, right: Image.Image) -> Image.Image:
    if left.mode != right.mode:
        right = right.convert(left.mode)
    paired = Image.new(left.mode, (left.width + right.width, max(left.height, right.height)))
    paired.paste(left, (0, 0))
    paired.paste(right, (left.width, 0))
    return paired


def clamp_alpha_value(alpha: float, mode: str, mu_t: float, eps: float = 1e-6) -> float:
    """Clamp alpha to a numerically safe range for the selected mode."""
    if mode == "add":
        a = float(max(0.0, min(1.0, alpha)))
        if a <= eps:
            return 0.0
        if a >= 1.0 - eps:
            return 1.0
        return a
    if mode == "replace":
        max_a = (1.0 + float(mu_t)) - float(eps)
        if max_a <= 0.0:
            return 0.0
        a = float(max(0.0, min(max_a, alpha)))
        if a <= eps:
            return 0.0
        return a
    raise ValueError(f"Unknown mode={mode}")


def autocast_ctx(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda")
    return nullcontext()


@torch.inference_mode()
def decode_flux_latent_to_pil(
    pipe: FluxPipeline, x0: torch.Tensor, device: torch.device, vae_dtype: torch.dtype
) -> Image.Image:
    x0 = x0.to(device=device, dtype=vae_dtype)
    x0_denorm = (x0 / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    with autocast_ctx(device):
        image = pipe.vae.decode(x0_denorm, return_dict=False)[0]
    return pipe.image_processor.postprocess(image)[0]


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


def load_flux_img2img_pipeline(checkpoint: Path, device: torch.device, dtype: torch.dtype):
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint}")
    if FluxImg2ImgPipeline is None:
        raise ImportError(
            "This diffusers installation does not provide FluxImg2ImgPipeline. "
            "Upgrade diffusers or use an environment with FLUX img2img support."
        )

    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")
    os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"[info] loading FLUX img2img pipeline {checkpoint} on {device} ({dtype})")
    pipe = FluxImg2ImgPipeline.from_pretrained(
        str(checkpoint),
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    return pipe.to(device)


def load_flux_rf_inversion_pipeline(
    checkpoint: Path,
    device: torch.device,
    dtype: torch.dtype,
    rf_root: Optional[Path] = None,
):
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint}")

    rf_root = rf_root or (Path(__file__).resolve().parent.parent / "RF_Inversion")
    if not rf_root.exists():
        raise FileNotFoundError(f"RF inversion directory not found: {rf_root}")

    pipeline_file = rf_root / "pipeline_flux_rf_inversion_local.py"
    rf_sde_file = rf_root / "pipeline_rf_inversion_sde.py"
    if not pipeline_file.exists():
        raise FileNotFoundError(f"Missing RF pipeline file: {pipeline_file}")
    if not rf_sde_file.exists():
        raise FileNotFoundError(f"Missing RF SDE pipeline file: {rf_sde_file}")

    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")
    os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if str(rf_root) not in sys.path:
        sys.path.insert(0, str(rf_root))

    original_get_custom_pipeline_class = pipeline_utils._get_custom_pipeline_class

    def _patched_get_custom_pipeline_class(custom_pipeline, *args, **kwargs):
        if custom_pipeline == "pipeline_flux_rf_inversion":
            custom_pipeline = str(pipeline_file)
        return original_get_custom_pipeline_class(custom_pipeline, *args, **kwargs)

    pipeline_utils._get_custom_pipeline_class = _patched_get_custom_pipeline_class

    spec = importlib.util.spec_from_file_location("rf_inversion_sde_local", str(rf_sde_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import RF SDE module: {rf_sde_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    RFInversionFluxPipelineSDE = mod.RFInversionFluxPipelineSDE

    print(f"[info] loading FLUX RF-inversion pipeline {checkpoint} on {device} ({dtype})")
    pipe = RFInversionFluxPipelineSDE.from_pretrained(
        str(checkpoint),
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    return pipe.to(device)


def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Foward process in flow-matching

    Args:
        sample (`torch.FloatTensor`):
            The input sample.
        timestep (`int`, *optional*):
            The current timestep in the diffusion chain.

    Returns:
        `torch.FloatTensor`:
            A scaled input sample.
    """
    # if scheduler.step_index is None:
    scheduler._init_step_index(timestep)

    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample

    return sample


# for flux
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu



def calc_v_sd3(pipe, src_tar_latent_model_input, src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(src_tar_latent_model_input.shape[0])
    # joint_attention_kwargs = {}
    # # add timestep to joint_attention_kwargs
    # joint_attention_kwargs["timestep"] = timestep[0]
    # joint_attention_kwargs["timestep_idx"] = i


    with torch.no_grad():
        # # predict the noise for the source prompt
        noise_pred_src_tar = pipe.transformer(
            hidden_states=src_tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds,
            pooled_projections=src_tar_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        # perform guidance source
        if pipe.do_classifier_free_guidance:
            src_noise_pred_uncond, src_noise_pred_text, tar_noise_pred_uncond, tar_noise_pred_text = noise_pred_src_tar.chunk(4)
            noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)
            noise_pred_tar = tar_noise_pred_uncond + tar_guidance_scale * (tar_noise_pred_text - tar_noise_pred_uncond)

    return noise_pred_src, noise_pred_tar



def calc_v_flux(pipe, latents, prompt_embeds, pooled_prompt_embeds, guidance, text_ids, latent_image_ids, t):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(latents.shape[0])
    # joint_attention_kwargs = {}
    # # add timestep to joint_attention_kwargs
    # joint_attention_kwargs["timestep"] = timestep[0]
    # joint_attention_kwargs["timestep_idx"] = i


    with torch.no_grad():
        # # predict the noise for the source prompt
        noise_pred = pipe.transformer(
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

    return noise_pred



@torch.no_grad()
def FlowEditSD3(pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 15,):
    
    device = x_src.device

    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)

    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)
    pipe._guidance_scale = src_guidance_scale
    
    # src prompts
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    # tar prompts
    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )
 
    # CFG prep
    src_tar_prompt_embeds = torch.cat([src_negative_prompt_embeds, src_prompt_embeds, tar_negative_prompt_embeds, tar_prompt_embeds], dim=0)
    src_tar_pooled_prompt_embeds = torch.cat([src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0)
    
    # initialize our ODE Zt_edit_1=x_src
    zt_edit = x_src.clone()

    for i, t in tqdm(enumerate(timesteps)):
        
        if T_steps - i > n_max:
            continue
        
        t_i = t/1000
        if i+1 < len(timesteps): 
            t_im1 = (timesteps[i+1])/1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)
        
        if T_steps - i > n_min:

            # Calculate the average of the V predictions
            V_delta_avg = torch.zeros_like(x_src)
            for k in range(n_avg):

                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                
                zt_src = (1-t_i)*x_src + (t_i)*fwd_noise

                zt_tar = zt_edit + zt_src - x_src

                src_tar_latent_model_input = torch.cat([zt_src, zt_src, zt_tar, zt_tar]) if pipe.do_classifier_free_guidance else (zt_src, zt_tar) 

                Vt_src, Vt_tar = calc_v_sd3(pipe, src_tar_latent_model_input,src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t)

                V_delta_avg += (1/n_avg) * (Vt_tar - Vt_src) # - (hfg-1)*( x_src))

            # propagate direct ODE
            zt_edit = zt_edit.to(torch.float32)

            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else: # i >= T_steps-n_min # regular sampling for last n_min steps

            if i == T_steps-n_min:
                # initialize SDEDIT-style generation phase
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src
                
            src_tar_latent_model_input = torch.cat([xt_tar, xt_tar, xt_tar, xt_tar]) if pipe.do_classifier_free_guidance else (xt_src, xt_tar)

            _, Vt_tar = calc_v_sd3(pipe, src_tar_latent_model_input,src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t)

            xt_tar = xt_tar.to(torch.float32)

            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)

            prev_sample = prev_sample.to(noise_pred_tar.dtype)

            xt_tar = prev_sample
        
    return zt_edit if n_min == 0 else xt_tar



@torch.no_grad()
def FlowEditFLUX(pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 28,
    n_avg: int = 1,
    src_guidance_scale: float = 1.5,
    tar_guidance_scale: float = 5.5,
    n_min: int = 0,
    n_max: int = 24,):

    device = x_src.device
    # Map latent spatial size back to image size expected by the current Flux pipeline.
    # This must be consistent with FluxPipeline.prepare_latents/_unpack_latents.
    orig_height, orig_width = x_src.shape[2] * pipe.vae_scale_factor, x_src.shape[3] * pipe.vae_scale_factor
    num_channels_latents = pipe.transformer.config.in_channels // 4

    pipe.check_inputs(
        prompt=src_prompt,
        prompt_2=None,
        height=orig_height,
        width=orig_width,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    )

    x_src, latent_src_image_ids = pipe.prepare_latents(batch_size= x_src.shape[0], num_channels_latents=num_channels_latents, height=orig_height, width=orig_width, dtype=x_src.dtype, device=x_src.device, generator=None,latents=x_src)
    x_src_packed = pipe._pack_latents(x_src, x_src.shape[0], num_channels_latents, x_src.shape[2], x_src.shape[3])
    latent_tar_image_ids = latent_src_image_ids

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    image_seq_len = x_src_packed.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, T_steps = retrieve_timesteps(
        scheduler,
        T_steps,
        device,
        timesteps=None,
        sigmas=sigmas,
        mu=mu,
        )
    
    num_warmup_steps = max(len(timesteps) - T_steps * pipe.scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    
    # src prompts
    (
        src_prompt_embeds,
        src_pooled_prompt_embeds,
        src_text_ids,

    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        device=device,
    )

    # tar prompts
    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_text_ids,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        device=device,
    )

    # handle guidance
    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=device)
        src_guidance = src_guidance.expand(x_src_packed.shape[0])
        tar_guidance = torch.tensor([tar_guidance_scale], device=device)
        tar_guidance = tar_guidance.expand(x_src_packed.shape[0])
    else:
        src_guidance = None
        tar_guidance = None

    # initialize our ODE Zt_edit_1=x_src
    zt_edit = x_src_packed.clone()

    for i, t in tqdm(enumerate(timesteps)):
        
        if T_steps - i > n_max:
            continue
        
        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        if i < len(timesteps):
            t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        else:
            t_im1 = t_i
        
        if T_steps - i > n_min:

            # Calculate the average of the V predictions
            V_delta_avg = torch.zeros_like(x_src_packed)

            for k in range(n_avg):
                                    

                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
                
                zt_src = (1-t_i)*x_src_packed + (t_i)*fwd_noise

                zt_tar = zt_edit + zt_src - x_src_packed

                # Merge in the future to avoid double computation
                Vt_src = calc_v_flux(pipe,
                                                    latents=zt_src,
                                                    prompt_embeds=src_prompt_embeds, 
                                                    pooled_prompt_embeds=src_pooled_prompt_embeds, 
                                                    guidance=src_guidance,
                                                    text_ids=src_text_ids, 
                                                    latent_image_ids=latent_src_image_ids, 
                                                    t=t)
                
                Vt_tar = calc_v_flux(pipe,
                                                    latents=zt_tar,
                                                    prompt_embeds=tar_prompt_embeds, 
                                                    pooled_prompt_embeds=tar_pooled_prompt_embeds, 
                                                    guidance=tar_guidance,
                                                    text_ids=tar_text_ids, 
                                                    latent_image_ids=latent_tar_image_ids, 
                                                    t=t)

                V_delta_avg += (1/n_avg) * (Vt_tar - Vt_src) # - (hfg-1)*( x_src))

            # propagate direct ODE
            zt_edit = zt_edit.to(torch.float32)

            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg

            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else: # i >= T_steps-n_min # regular sampling last n_min steps

            if i == T_steps-n_min:
                # initialize SDEDIT-style generation phase
                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
                xt_src = scale_noise(scheduler, x_src_packed, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src_packed
                
            Vt_tar = calc_v_flux(pipe,
                                    latents=xt_tar,
                                    prompt_embeds=tar_prompt_embeds, 
                                    pooled_prompt_embeds=tar_pooled_prompt_embeds, 
                                    guidance=tar_guidance,
                                    text_ids=tar_text_ids, 
                                    latent_image_ids=latent_tar_image_ids, 
                                    t=t)


            xt_tar = xt_tar.to(torch.float32)

            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)

            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample
    out = zt_edit if n_min == 0 else xt_tar
    unpacked_out = pipe._unpack_latents(out, orig_height, orig_width, pipe.vae_scale_factor)
    return unpacked_out
