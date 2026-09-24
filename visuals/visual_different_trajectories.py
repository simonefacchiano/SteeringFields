"""
visual_different_trajectories.py
================================

What this file does
-------------------
This script visualizes FLUX latent trajectories in 2D for quick qualitative analysis.
It projects high-dimensional latent states with PCA, normalizes coordinates in [0, 1],
and plots trajectories with clear labels/styling.

It supports three workflows:

1) Single trajectory (`default`)
   - Runs one prompt and shows the trajectory.
   - Colors represent projected vector-field strength (Low -> High, magma colorbar).

2) Prompt comparison (`--compare-prompts`)
   - Runs multiple prompts from the same initial noise (same seed).
   - Overlays trajectories to compare how prompt conditioning changes dynamics.

3) Steering comparison (`--compare-steering`)
   - Compares:
     - baseline (source prompt),
     - baseline (alternate prompt),
     - add mode (towards safe prompt),
     - replace mode (safe/unsafe blend).
   - All trajectories start from the exact same initial noise.

Important notes
---------------
- Axes are PCA projections, not physical coordinates.
- Coordinates are normalized to [0, 1] only for visualization.
- This is an analysis/visualization utility; it does not save generated images from FLUX,
  only plots trajectory geometry.

How to use
----------
Run with Python from the project environment:

1) Single trajectory:
   python3 /leonardo_work/IscrC_VUnl/usr/simone/FlowEdit/visuals/visual_different_trajectories.py

2) Compare multiple prompts:
   python3 /leonardo_work/IscrC_VUnl/usr/simone/FlowEdit/visuals/visual_different_trajectories.py --compare-prompts

3) Compare baseline vs steering:
   python3 /leonardo_work/IscrC_VUnl/usr/simone/FlowEdit/visuals/visual_different_trajectories.py --compare-steering

Optional useful args
--------------------
- --seed
- --num-steps
- --cfg / --cfg-src / --cfg-tar
- --prompt / --prompts
- --steer-source-prompt / --steer-safe-prompt / --baseline-alt-prompt
- --out
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

plt.style.use("default")
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
    }
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from steering_fields.utils import calc_v_flux, calculate_shift, load_flux_pipeline, set_seed


DEFAULT_CHECKPOINT = "/leonardo_scratch/fast/IscrC_VUnl/flux1"
DEFAULT_PROMPT = "a tiger walks in a forest"
DEFAULT_SEED = 42
DEFAULT_COMPARE_PROMPTS = [
    "a cat playing in the snow",
    "a dog playing in the snow",
    "a camel walks in the desert",
    "a woman plays tennis in a court",
    "a nike advertisement campaign"
]
DEFAULT_STEER_SOURCE = "a dog playing in the snow"
DEFAULT_STEER_SAFE = "a cat"
DEFAULT_BASELINE_ALT = "a cat playing on the snow"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize FLUX latent trajectory and vector field in 2D.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help="Prompt list used in --compare-prompts mode.",
    )
    parser.add_argument(
        "--compare-prompts",
        action="store_true",
        help="Overlay trajectories for multiple prompts from the same initial noise.",
    )
    parser.add_argument(
        "--compare-steering",
        action="store_true",
        help="Compare baseline vs add vs replace trajectories from the same initial noise.",
    )
    parser.add_argument(
        "--compare-source-target",
        action="store_true",
        help="Plot only source and target trajectories from the same initial noise.",
    )
    parser.add_argument(
        "--compare-source-target-ours",
        action="store_true",
        help="Plot source, target, and ours trajectories from the same initial noise.",
    )
    parser.add_argument("--steer-source-prompt", type=str, default=DEFAULT_STEER_SOURCE)
    parser.add_argument("--steer-safe-prompt", type=str, default=DEFAULT_STEER_SAFE)
    parser.add_argument("--baseline-alt-prompt", type=str, default=DEFAULT_BASELINE_ALT)
    parser.add_argument(
        "--steering-title-mode",
        type=str,
        choices=["blending", "replacing"],
        default="blending",
        help="Title shown in compare-steering mode.",
    )
    parser.add_argument("--cfg-src", type=float, default=2.0)
    parser.add_argument("--cfg-tar", type=float, default=5.5)
    parser.add_argument("--add-alpha-start", type=float, default=0.4)
    parser.add_argument("--add-alpha-end", type=float, default=0.0)
    parser.add_argument("--replace-mu-start", type=float, default=0.4)
    parser.add_argument("--replace-mu-end", type=float, default=0.4)
    parser.add_argument("--replace-alpha-start", type=float, default=0.3)
    parser.add_argument("--replace-alpha-end", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-steps", type=int, default=28)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--device-number", type=int, default=0)
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "tiger_flux_field_seed42.jpeg"),
    )
    return parser.parse_args()


def pca2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x.mean(axis=0, keepdims=True)
    xc = x - x_mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[:2].T
    y = xc @ basis
    return y, basis, x_mean


def normalize_2d(y: np.ndarray, pad: float = 0.08) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = y.min(axis=0)
    maxs = y.max(axis=0)
    span = np.maximum(maxs - mins, 1e-8)
    yn = (y - mins) / span
    yn = yn * (1.0 - 2.0 * pad) + pad
    return yn, mins, span


def linear_at_step(i: int, total: int, start: float, end: float) -> float:
    if total <= 1:
        return float(end)
    frac = float(i) / float(total - 1)
    return float(start + frac * (end - start))


def normalize_output_path(path: Path) -> Path:
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        return path.with_suffix(".jpeg")
    return path


def save_plot(fig: plt.Figure, out_path: Path) -> Path:
    out_path = normalize_output_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=128, facecolor="white")
    return out_path


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(f"cuda:{args.device_number}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    pipe = load_flux_pipeline(Path(args.checkpoint), device=device, dtype=dtype)
    scheduler = pipe.scheduler

    gen = torch.Generator(device=device).manual_seed(args.seed)
    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, latent_image_ids = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=num_channels_latents,
        height=args.height,
        width=args.width,
        dtype=dtype,
        device=device,
        generator=gen,
        latents=None,
    )
    if latents.ndim == 3:
        z = latents
    elif latents.ndim == 4:
        z = pipe._pack_latents(latents, 1, num_channels_latents, latents.shape[2], latents.shape[3])
    else:
        raise ValueError(f"Unexpected latent shape: {tuple(latents.shape)}")

    sigmas = np.linspace(1.0, 1.0 / args.num_steps, args.num_steps)
    image_seq_len = z.shape[1]
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

    z0 = z.clone()

    def run_trajectory(prompt: str) -> tuple[np.ndarray, np.ndarray]:
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
        )
        if pipe.transformer.config.guidance_embeds:
            guidance = torch.tensor([args.cfg], device=device)
        else:
            guidance = None

        z_loc = z0.clone()
        z_hist = []
        v_hist = []
        for t in timesteps:
            scheduler._init_step_index(t)
            sigma_i = scheduler.sigmas[scheduler.step_index]
            sigma_im1 = scheduler.sigmas[scheduler.step_index + 1]
            dt = sigma_im1 - sigma_i

            v = calc_v_flux(
                pipe,
                z_loc,
                prompt_embeds,
                pooled_prompt_embeds,
                guidance,
                text_ids,
                latent_image_ids,
                t,
            )
            z_hist.append(z_loc.detach().to(torch.float32).flatten().cpu().numpy())
            v_hist.append(v.detach().to(torch.float32).flatten().cpu().numpy())
            z_loc = (z_loc.to(torch.float32) + dt * v.to(torch.float32)).to(dtype)

        return np.stack(z_hist, axis=0), np.stack(v_hist, axis=0)

    def run_trajectory_steered(
        mode: str,
        *,
        src_prompt: str,
        safe_prompt: str,
        cfg_src_val: float,
        cfg_tar_val: float,
        alpha_start: float,
        alpha_end: float,
        mu_start: float = 0.0,
        mu_end: float = 0.0,
    ) -> np.ndarray:
        src_pe, src_pp, src_ids = pipe.encode_prompt(prompt=src_prompt, prompt_2=None, device=device)
        safe_pe, safe_pp, safe_ids = pipe.encode_prompt(prompt=safe_prompt, prompt_2=None, device=device)
        # replace-mode unsafe anchor = source prompt
        unsafe_pe, unsafe_pp, unsafe_ids = pipe.encode_prompt(prompt=src_prompt, prompt_2=None, device=device)

        if pipe.transformer.config.guidance_embeds:
            g_src = torch.tensor([cfg_src_val], device=device)
            g_tar = torch.tensor([cfg_tar_val], device=device)
        else:
            g_src = None
            g_tar = None

        z_loc = z0.clone()
        z_hist = []
        for i, t in enumerate(timesteps):
            scheduler._init_step_index(t)
            sigma_i = scheduler.sigmas[scheduler.step_index]
            sigma_im1 = scheduler.sigmas[scheduler.step_index + 1]
            dt = sigma_im1 - sigma_i

            v_src = calc_v_flux(pipe, z_loc, src_pe, src_pp, g_src, src_ids, latent_image_ids, t)
            z_hist.append(z_loc.detach().to(torch.float32).flatten().cpu().numpy())

            if mode == "baseline":
                v = v_src
            elif mode == "add":
                a = linear_at_step(i, len(timesteps), alpha_start, alpha_end)
                a = min(max(a, 0.0), 1.0)
                v_safe = calc_v_flux(pipe, z_loc, safe_pe, safe_pp, g_tar, safe_ids, latent_image_ids, t)
                v = (1.0 - a) * v_src + a * v_safe
            elif mode == "replace":
                a = linear_at_step(i, len(timesteps), alpha_start, alpha_end)
                mu_t = max(0.0, linear_at_step(i, len(timesteps), mu_start, mu_end))
                if a >= 1.0 + mu_t:
                    a = (1.0 + mu_t) - 1e-6
                v_safe = calc_v_flux(pipe, z_loc, safe_pe, safe_pp, g_tar, safe_ids, latent_image_ids, t)
                v_unsafe = calc_v_flux(pipe, z_loc, unsafe_pe, unsafe_pp, g_tar, unsafe_ids, latent_image_ids, t)
                v = (v_src + mu_t * v_safe - a * v_unsafe) / (1.0 + mu_t - a)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            z_loc = (z_loc.to(torch.float32) + dt * v.to(torch.float32)).to(dtype)

        return np.stack(z_hist, axis=0)

    if args.compare_steering:
        x_source = run_trajectory_steered(
            "baseline",
            src_prompt=args.steer_source_prompt,
            safe_prompt=args.steer_safe_prompt,
            cfg_src_val=args.cfg_src,
            cfg_tar_val=args.cfg_tar,
            alpha_start=0.0,
            alpha_end=0.0,
        )
        x_target = run_trajectory_steered(
            "baseline",
            src_prompt=args.steer_safe_prompt,
            safe_prompt=args.steer_safe_prompt,
            cfg_src_val=args.cfg_src,
            cfg_tar_val=args.cfg_tar,
            alpha_start=0.0,
            alpha_end=0.0,
        )
        x_away = run_trajectory_steered(
            "baseline",
            src_prompt=args.baseline_alt_prompt,
            safe_prompt=args.steer_safe_prompt,
            cfg_src_val=args.cfg_src,
            cfg_tar_val=args.cfg_tar,
            alpha_start=0.0,
            alpha_end=0.0,
        )
        x_ours = run_trajectory_steered(
            "add" if args.steering_title_mode == "blending" else "replace",
            src_prompt=args.steer_source_prompt,
            safe_prompt=args.steer_safe_prompt,
            cfg_src_val=args.cfg_src,
            cfg_tar_val=args.cfg_tar,
            alpha_start=args.add_alpha_start if args.steering_title_mode == "blending" else args.replace_alpha_start,
            alpha_end=args.add_alpha_end if args.steering_title_mode == "blending" else args.replace_alpha_end,
            mu_start=args.replace_mu_start if args.steering_title_mode == "replacing" else 0.0,
            mu_end=args.replace_mu_end if args.steering_title_mode == "replacing" else 0.0,
        )

        trajs = [
            {"label": "source trajectory", "x": x_source, "color": "#2563eb"},
            {"label": "target trajectory", "x": x_target, "color": "#16a34a"},
            {"label": "away trajectory", "x": x_away, "color": "#f59e0b"},
            {"label": "ours", "x": x_ours, "color": "#dc2626"},
        ]

        x_all = np.concatenate([t["x"] for t in trajs], axis=0)
        y_all_raw, _, _ = pca2(x_all)
        y_all_norm, _, _ = normalize_2d(y_all_raw)

        fig = plt.figure(figsize=(8, 8), dpi=128)
        ax = fig.add_subplot(111)
        idx0 = 0
        for tr in trajs:
            n = tr["x"].shape[0]
            y = y_all_norm[idx0:idx0 + n]
            idx0 += n
            c = tr["color"]
            ax.plot(y[:, 0], y[:, 1], color=c, linewidth=3.0, alpha=0.95, zorder=3)
            ax.scatter(y[:, 0], y[:, 1], color=c, s=15, alpha=0.75, zorder=4)
            ax.scatter(y[-1, 0], y[-1, 1], s=140, c=c, marker="X", edgecolors="white", linewidths=1.0, zorder=6)
            ax.text(
                min(0.98, y[-1, 0] + 0.015),
                min(0.98, y[-1, 1] + 0.012),
                tr["label"],
                color=c,
                fontsize=10,
                weight="semibold",
                ha="left",
                va="bottom",
                zorder=7,
            )

        # all share the same start point
        y0 = y_all_norm[0]
        ax.scatter(y0[0], y0[1], s=220, c="#374151", marker="o", edgecolors="white", linewidths=1.2, zorder=7)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("PCA Axis 1", color="#111111")
        ax.set_ylabel("PCA Axis 2", color="#111111")
        ax.tick_params(axis="both", colors="#111111", width=1.2)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        ax.grid(True, color="#94a3b8", alpha=0.25, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("#111111")
            spine.set_linewidth(1.8)

        fig.suptitle("Replacing Trajectories", fontsize=16, color="black", y=0.985)
        subtitle = (
            f"source trajectory: {args.steer_source_prompt}\n"
            f"target trajectory: {args.steer_safe_prompt}\n"
            f"away trajectory: {args.baseline_alt_prompt}"
        )
        fig.text(0.5, 0.905, subtitle, ha="center", va="top", color="black", fontsize=10)
        fig.subplots_adjust(top=0.80)

        out_path = save_plot(fig, Path(args.out))
        plt.close(fig)
        print(f"[done] saved: {out_path}")
        return

    if args.compare_source_target:
        x_source = run_trajectory_steered(
            "baseline",
            src_prompt=args.steer_source_prompt,
            safe_prompt=args.steer_safe_prompt,
            cfg_src_val=args.cfg_src,
            cfg_tar_val=args.cfg_tar,
            alpha_start=0.0,
            alpha_end=0.0,
        )
        x_target = run_trajectory_steered(
            "baseline",
            src_prompt=args.steer_safe_prompt,
            safe_prompt=args.steer_safe_prompt,
            cfg_src_val=args.cfg_src,
            cfg_tar_val=args.cfg_tar,
            alpha_start=0.0,
            alpha_end=0.0,
        )

        trajs = [
            {"label": "source trajectory", "x": x_source, "color": "#2563eb"},
            {"label": "target trajectory", "x": x_target, "color": "#16a34a"},
        ]

        x_all = np.concatenate([t["x"] for t in trajs], axis=0)
        y_all_raw, _, _ = pca2(x_all)
        y_all_norm, _, _ = normalize_2d(y_all_raw)

        fig = plt.figure(figsize=(8, 8), dpi=128)
        ax = fig.add_subplot(111)
        idx0 = 0
        for tr in trajs:
            n = tr["x"].shape[0]
            y = y_all_norm[idx0:idx0 + n]
            idx0 += n
            c = tr["color"]
            ax.plot(y[:, 0], y[:, 1], color=c, linewidth=3.0, alpha=0.95, zorder=3)
            ax.scatter(y[:, 0], y[:, 1], color=c, s=15, alpha=0.75, zorder=4)
            ax.scatter(y[-1, 0], y[-1, 1], s=140, c=c, marker="X", edgecolors="white", linewidths=1.0, zorder=6)
            ax.text(
                min(0.98, y[-1, 0] + 0.015),
                min(0.98, y[-1, 1] + 0.012),
                tr["label"],
                color=c,
                fontsize=10,
                weight="semibold",
                ha="left",
                va="bottom",
                zorder=7,
            )

        y0 = y_all_norm[0]
        ax.scatter(y0[0], y0[1], s=220, c="#374151", marker="o", edgecolors="white", linewidths=1.2, zorder=7)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("PCA Axis 1", color="#111111")
        ax.set_ylabel("PCA Axis 2", color="#111111")
        ax.tick_params(axis="both", colors="#111111", width=1.2)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        ax.grid(True, color="#94a3b8", alpha=0.25, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("#111111")
            spine.set_linewidth(1.8)

        fig.suptitle("Source vs Target Trajectories", fontsize=16, color="black", y=0.985)
        subtitle = (
            f"source trajectory: {args.steer_source_prompt}\n"
            f"target trajectory: {args.steer_safe_prompt}"
        )
        fig.text(0.5, 0.905, subtitle, ha="center", va="top", color="black", fontsize=10)
        fig.subplots_adjust(top=0.80)

        out_path = save_plot(fig, Path(args.out))
        plt.close(fig)
        print(f"[done] saved: {out_path}")
        return

    if args.compare_source_target_ours:
        x_source = run_trajectory_steered(
            "baseline",
            src_prompt=args.steer_source_prompt,
            safe_prompt=args.steer_safe_prompt,
            cfg_src_val=args.cfg_src,
            cfg_tar_val=args.cfg_tar,
            alpha_start=0.0,
            alpha_end=0.0,
        )
        x_target = run_trajectory_steered(
            "baseline",
            src_prompt=args.steer_safe_prompt,
            safe_prompt=args.steer_safe_prompt,
            cfg_src_val=args.cfg_src,
            cfg_tar_val=args.cfg_tar,
            alpha_start=0.0,
            alpha_end=0.0,
        )
        x_ours = run_trajectory_steered(
            "add" if args.steering_title_mode == "blending" else "replace",
            src_prompt=args.steer_source_prompt,
            safe_prompt=args.steer_safe_prompt,
            cfg_src_val=args.cfg_src,
            cfg_tar_val=args.cfg_tar,
            alpha_start=args.add_alpha_start if args.steering_title_mode == "blending" else args.replace_alpha_start,
            alpha_end=args.add_alpha_end if args.steering_title_mode == "blending" else args.replace_alpha_end,
            mu_start=args.replace_mu_start if args.steering_title_mode == "replacing" else 0.0,
            mu_end=args.replace_mu_end if args.steering_title_mode == "replacing" else 0.0,
        )

        trajs = [
            {"label": "source trajectory", "x": x_source, "color": "#2563eb"},
            {"label": "target trajectory", "x": x_target, "color": "#16a34a"},
            {"label": "ours", "x": x_ours, "color": "#dc2626"},
        ]

        x_all = np.concatenate([t["x"] for t in trajs], axis=0)
        y_all_raw, _, _ = pca2(x_all)
        y_all_norm, _, _ = normalize_2d(y_all_raw)

        fig = plt.figure(figsize=(8, 8), dpi=128)
        ax = fig.add_subplot(111)
        idx0 = 0
        for tr in trajs:
            n = tr["x"].shape[0]
            y = y_all_norm[idx0:idx0 + n]
            idx0 += n
            c = tr["color"]
            ax.plot(y[:, 0], y[:, 1], color=c, linewidth=3.0, alpha=0.95, zorder=3)
            ax.scatter(y[:, 0], y[:, 1], color=c, s=15, alpha=0.75, zorder=4)
            ax.scatter(y[-1, 0], y[-1, 1], s=140, c=c, marker="X", edgecolors="white", linewidths=1.0, zorder=6)
            ax.text(
                min(0.98, y[-1, 0] + 0.015),
                min(0.98, y[-1, 1] + 0.012),
                tr["label"],
                color=c,
                fontsize=10,
                weight="semibold",
                ha="left",
                va="bottom",
                zorder=7,
            )

        y0 = y_all_norm[0]
        ax.scatter(y0[0], y0[1], s=220, c="#374151", marker="o", edgecolors="white", linewidths=1.2, zorder=7)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("PCA Axis 1", color="#111111")
        ax.set_ylabel("PCA Axis 2", color="#111111")
        ax.tick_params(axis="both", colors="#111111", width=1.2)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        ax.grid(True, color="#94a3b8", alpha=0.25, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("#111111")
            spine.set_linewidth(1.8)

        fig.suptitle("Blending Trajectories", fontsize=16, color="black", y=0.985)
        subtitle = (
            f"source trajectory: {args.steer_source_prompt}\n"
            f"target trajectory: {args.steer_safe_prompt}\n"
            f"ours trajectory: {args.steer_source_prompt}"
        )
        fig.text(0.5, 0.905, subtitle, ha="center", va="top", color="black", fontsize=10)
        fig.subplots_adjust(top=0.80)

        out_path = save_plot(fig, Path(args.out))
        plt.close(fig)
        print(f"[done] saved: {out_path}")
        return

    if args.compare_prompts:
        prompts = args.prompts if args.prompts else DEFAULT_COMPARE_PROMPTS
        trajs = []
        for p in prompts:
            x_p, _u_p = run_trajectory(p)
            trajs.append({"prompt": p, "x": x_p})

        x_all = np.concatenate([t["x"] for t in trajs], axis=0)
        y_all_raw, basis, x_mean = pca2(x_all)
        y_all_norm, _, _ = normalize_2d(y_all_raw)

        fig = plt.figure(figsize=(8, 8), dpi=128)
        ax = fig.add_subplot(111)

        colors = ["#fb7185", "#38bdf8", "#a78bfa", "#f59e0b", "#34d399", "#f472b6"]
        idx0 = 0
        for i, tr in enumerate(trajs):
            n = tr["x"].shape[0]
            y = y_all_norm[idx0:idx0 + n]
            idx0 += n
            c = colors[i % len(colors)]
            ax.plot(y[:, 0], y[:, 1], color=c, linewidth=2.8, alpha=0.95, zorder=3)
            ax.scatter(y[:, 0], y[:, 1], color=c, s=18, alpha=0.85, zorder=4)
            ax.scatter(y[0, 0], y[0, 1], s=120, c="#374151", marker="o", edgecolors="white", linewidths=1.0, zorder=6)
            ax.scatter(y[-1, 0], y[-1, 1], s=130, c=c, marker="X", edgecolors="white", linewidths=1.0, zorder=6)
            ax.text(
                min(0.98, y[-1, 0] + 0.015),
                min(0.98, y[-1, 1] + 0.015),
                tr["prompt"],
                color=c,
                fontsize=9,
                weight="semibold",
                ha="left",
                va="bottom",
                zorder=7,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("PCA Axis 1 (normalized latent projection)", color="#111111")
        ax.set_ylabel("PCA Axis 2 (normalized latent projection)", color="#111111")
        ax.tick_params(axis="both", colors="#111111", width=1.2)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        ax.grid(True, color="#94a3b8", alpha=0.25, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("#111111")
            spine.set_linewidth(1.8)

        ax.set_title("FLUX Latent Trajectories from Same Noise", fontsize=14, color="#111111", pad=14)
        subtitle = f"Shared initial noise (seed={args.seed})   Steps={args.num_steps}   CFG={args.cfg}"
        ax.text(0.5, 1.01, subtitle, ha="center", va="bottom", transform=ax.transAxes, color="#111111", fontsize=10)

        out_path = save_plot(fig, Path(args.out))
        plt.close(fig)
        print(f"[done] saved: {out_path}")
        return

    x, u = run_trajectory(args.prompt)

    y, basis, x_mean = pca2(x)
    eps = 0.02
    y2 = ((x + eps * u) - x_mean) @ basis
    d = y2 - y

    # Vector-field strength in projected 2D space (before arrow normalization).
    strength = np.linalg.norm(d, axis=1)
    dn = d / (strength[:, None] + 1e-8)
    d_scaled = dn * 0.045

    y_all = np.concatenate([y, y + d_scaled], axis=0)
    y_norm, mins, span = normalize_2d(y_all)
    y_main = y_norm[: len(y)]
    y_arrow_tip = y_norm[len(y):2 * len(y)]
    d_main = y_arrow_tip - y_main

    sval = strength
    segs = np.concatenate([y_main[:-1, None, :], y_main[1:, None, :]], axis=1)

    fig = plt.figure(figsize=(8, 8), dpi=128)
    ax = fig.add_subplot(111)

    segs_line = np.concatenate([y_main[:-1, None, :], y_main[1:, None, :]], axis=1)
    lc = LineCollection(segs_line, cmap="magma", linewidths=3.0, alpha=0.95, zorder=3)
    lc.set_array(sval[:-1])
    ax.add_collection(lc)

    ax.scatter(
        y_main[:, 0],
        y_main[:, 1],
        c=sval,
        cmap="magma",
        s=34,
        edgecolors="white",
        linewidths=0.35,
        alpha=0.98,
        zorder=4,
    )
    ax.scatter(y_main[0, 0], y_main[0, 1], s=210, c="#374151", marker="o", edgecolors="white", linewidths=1.2, zorder=6)
    ax.scatter(y_main[-1, 0], y_main[-1, 1], s=240, c="#ef4444", marker="X", edgecolors="white", linewidths=1.2, zorder=6)

    smin = float(np.min(sval))
    smax = float(np.max(sval))
    if abs(smax - smin) < 1e-12:
        smax = smin + 1e-12
    norm = Normalize(vmin=smin, vmax=smax)
    lc.set_norm(norm)
    sm = ScalarMappable(norm=norm, cmap="magma")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.038, pad=0.02)
    cbar.set_label("Vector Field Strength (projected magnitude)", fontsize=11)
    cbar.set_ticks([smin, smax])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.yaxis.label.set_color("#111111")
    cbar.ax.tick_params(colors="#111111", length=0, width=1.2)
    cbar.outline.set_visible(False)
    cbar.ax.set_facecolor("none")
    if getattr(cbar, "solids", None) is not None:
        cbar.solids.set_edgecolor("face")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("PCA Axis 1", color="#111111")
    ax.set_ylabel("PCA Axis 2", color="#111111")
    ax.tick_params(axis="both", colors="#111111", width=1.2)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(True, color="#94a3b8", alpha=0.15, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#111111")
        spine.set_linewidth(1.8)

    ax.set_title("FLUX Latent Flow: Trajectory + Normalized Vector Field", fontsize=14, color="#111111", pad=14)
    subtitle = f'Prompt: "{args.prompt}"   Seed: {args.seed}   Steps: {args.num_steps}'
    ax.text(0.5, 1.01, subtitle, ha="center", va="bottom", transform=ax.transAxes, color="#111111", fontsize=10)
    ax.text(y_main[0, 0], y_main[0, 1] + 0.03, "start", color="#86efac", fontsize=9, ha="center")
    ax.text(y_main[-1, 0], y_main[-1, 1] - 0.035, "end", color="#fca5a5", fontsize=9, ha="center")

    out_path = save_plot(fig, Path(args.out))
    plt.close(fig)
    print(f"[done] saved: {out_path}")


if __name__ == "__main__":
    main()
