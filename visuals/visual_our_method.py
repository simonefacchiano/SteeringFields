"""
visual_our_method.py
====================

What this script does
---------------------
This script visualizes latent trajectories for our steering method in FLUX, in 2D.
It projects high-dimensional latent states with PCA and normalizes coordinates in [0, 1].

Modes:
- --mode add:
  Plots source trajectory, target trajectory, and one blended trajectory per alpha schedule.
  You can pass multiple schedules with:
    --alpha-pairs "0.4:0.0,0.3:0.1,..."
  Blended colors are assigned automatically by proximity:
  closer to source -> bluer, closer to target -> greener.

- --mode replace:
  Plots final replace trajectory plus source/safe/unsafe trajectories.
  You can pass multiple replace schedules with:
    --replace-pairs "0.4:0.4:0.3:0.3,0.5:0.3:0.4:0.2"
  Format is:
    mu_start:mu_end:alpha_start:alpha_end

Important defaults
------------------
- seed: 42
- cfg-src: 2.0
- cfg-tar: 5.5
- add default alpha: 0.4 -> 0.0
- replace default: mu 0.4 -> 0.4, alpha 0.3 -> 0.3

Output
------
The output filename automatically gets a mode suffix:
- .../name.png + --mode add -> .../name_add.png
- .../name.png + --mode replace -> .../name_replace.png

Examples
--------
1) Add mode with one schedule:
   python3 visual_our_method.py --mode add --alpha-start 0.4 --alpha-end 0.0

2) Add mode with multiple schedules:
   python3 visual_our_method.py --mode add --alpha-pairs "1.0:0.0,0.8:0.4,0.4:0.1"

3) Replace mode:
   python3 visual_our_method.py --mode replace
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from steering_fields.utils import calc_v_flux, calculate_shift, load_flux_pipeline, set_seed


DEFAULT_CHECKPOINT = "/leonardo_scratch/fast/IscrC_VUnl/flux1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize our method trajectories in add/replace/remove mode.")
    p.add_argument("--mode", choices=("add", "replace", "remove"), required=True)
    p.add_argument("--source-prompt", type=str, default="a dog playing in the snow")
    p.add_argument("--target-prompt", type=str, default="a cat")
    p.add_argument("--safe-prompt", type=str, default="a cat")
    p.add_argument("--unsafe-prompt", type=str, default=None, help="Default: same as source prompt.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-steps", type=int, default=28)
    p.add_argument("--cfg-src", type=float, default=2.0)
    p.add_argument("--cfg-tar", type=float, default=5.5)
    p.add_argument("--alpha-start", type=float, default=None)
    p.add_argument("--alpha-end", type=float, default=None)
    p.add_argument(
        "--alpha-pairs",
        type=str,
        default=None,
        help='Only for --mode add. Comma-separated start:end pairs, e.g. "0.4:0.0,0.3:0.1".',
    )
    p.add_argument("--mu-start", type=float, default=None)
    p.add_argument("--mu-end", type=float, default=None)
    p.add_argument(
        "--replace-pairs",
        type=str,
        default=None,
        help='Only for --mode replace. Comma-separated ms:me:as:ae tuples, e.g. "0.4:0.4:0.3:0.3,0.5:0.3:0.4:0.2".',
    )
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument("--device-number", type=int, default=0)
    p.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "visual_our_method.png"),
    )
    return p.parse_args()


def linear_at_step(i: int, total: int, start: float, end: float) -> float:
    if total <= 1:
        return float(end)
    frac = float(i) / float(total - 1)
    return float(start + frac * (end - start))


def pca2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x.mean(axis=0, keepdims=True)
    xc = x - x_mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[:2].T
    y = xc @ basis
    return y, basis, x_mean


def normalize_0_1(y: np.ndarray, pad: float = 0.08) -> np.ndarray:
    mins = y.min(axis=0)
    maxs = y.max(axis=0)
    span = np.maximum(maxs - mins, 1e-8)
    yn = (y - mins) / span
    yn = yn * (1.0 - 2.0 * pad) + pad
    return yn


def parse_alpha_pairs(spec: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for token in spec.split(","):
        t = token.strip()
        if not t:
            continue
        if ":" not in t:
            raise ValueError(f'Invalid alpha pair "{t}". Expected start:end')
        a0, a1 = t.split(":", 1)
        pairs.append((float(a0), float(a1)))
    if not pairs:
        raise ValueError("No valid alpha pairs parsed from --alpha-pairs")
    return pairs


def parse_replace_pairs(spec: str) -> list[tuple[float, float, float, float]]:
    pairs: list[tuple[float, float, float, float]] = []
    for token in spec.split(","):
        t = token.strip()
        if not t:
            continue
        chunks = t.split(":")
        if len(chunks) != 4:
            raise ValueError(
                f'Invalid replace pair "{t}". Expected ms:me:as:ae'
            )
        ms, me, a_s, a_e = map(float, chunks)
        pairs.append((ms, me, a_s, a_e))
    if not pairs:
        raise ValueError("No valid tuples parsed from --replace-pairs")
    return pairs


def fmt_num(v: float) -> str:
    s = f"{float(v):.2f}".rstrip("0").rstrip(".")
    if s.startswith("0.") and s != "0":
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return s


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.mode == "remove":
        unsafe_prompt = args.unsafe_prompt if args.unsafe_prompt is not None else args.target_prompt
    else:
        unsafe_prompt = args.unsafe_prompt if args.unsafe_prompt is not None else args.source_prompt

    if args.mode == "add":
        alpha_start = 0.4 if args.alpha_start is None else args.alpha_start
        alpha_end = 0.0 if args.alpha_end is None else args.alpha_end
        mu_start = 0.4 if args.mu_start is None else args.mu_start
        mu_end = 0.4 if args.mu_end is None else args.mu_end
    elif args.mode == "remove":
        alpha_start = 0.4 if args.alpha_start is None else args.alpha_start
        alpha_end = 0.0 if args.alpha_end is None else args.alpha_end
        mu_start = 0.0 if args.mu_start is None else args.mu_start
        mu_end = 0.0 if args.mu_end is None else args.mu_end
    else:
        alpha_start = 0.3 if args.alpha_start is None else args.alpha_start
        alpha_end = 0.3 if args.alpha_end is None else args.alpha_end
        mu_start = 0.4 if args.mu_start is None else args.mu_start
        mu_end = 0.4 if args.mu_end is None else args.mu_end

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
        z0 = latents
    elif latents.ndim == 4:
        z0 = pipe._pack_latents(latents, 1, num_channels_latents, latents.shape[2], latents.shape[3])
    else:
        raise ValueError(f"Unexpected latent shape: {tuple(latents.shape)}")

    sigmas = np.linspace(1.0, 1.0 / args.num_steps, args.num_steps)
    image_seq_len = z0.shape[1]
    mu_shift = calculate_shift(
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
        mu=mu_shift,
    )
    pipe._num_timesteps = len(timesteps)

    src_pe, src_pp, src_ids = pipe.encode_prompt(prompt=args.source_prompt, prompt_2=None, device=device)
    tar_pe, tar_pp, tar_ids = pipe.encode_prompt(prompt=args.target_prompt, prompt_2=None, device=device)
    safe_pe, safe_pp, safe_ids = pipe.encode_prompt(prompt=args.safe_prompt, prompt_2=None, device=device)
    unsafe_pe, unsafe_pp, unsafe_ids = pipe.encode_prompt(prompt=unsafe_prompt, prompt_2=None, device=device)

    if pipe.transformer.config.guidance_embeds:
        g_src = torch.tensor([args.cfg_src], device=device)
        g_tar = torch.tensor([args.cfg_tar], device=device)
    else:
        g_src = None
        g_tar = None

    def integrate_single(prompt_embeds, pooled, text_ids, guidance) -> np.ndarray:
        z = z0.clone()
        z_hist = []
        for t in timesteps:
            scheduler._init_step_index(t)
            sigma_i = scheduler.sigmas[scheduler.step_index]
            sigma_im1 = scheduler.sigmas[scheduler.step_index + 1]
            dt = sigma_im1 - sigma_i
            v = calc_v_flux(pipe, z, prompt_embeds, pooled, guidance, text_ids, latent_image_ids, t)
            z_hist.append(z.detach().to(torch.float32).flatten().cpu().numpy())
            z = (z.to(torch.float32) + dt * v.to(torch.float32)).to(dtype)
        return np.stack(z_hist, axis=0)

    src_traj = integrate_single(src_pe, src_pp, src_ids, g_src)

    if args.mode == "add":
        tar_traj = integrate_single(tar_pe, tar_pp, tar_ids, g_tar)
        def integrate_blend(alpha_s: float, alpha_e: float) -> np.ndarray:
            z = z0.clone()
            blend_hist = []
            for i, t in enumerate(timesteps):
                scheduler._init_step_index(t)
                sigma_i = scheduler.sigmas[scheduler.step_index]
                sigma_im1 = scheduler.sigmas[scheduler.step_index + 1]
                dt = sigma_im1 - sigma_i
                a = linear_at_step(i, len(timesteps), alpha_s, alpha_e)
                a = min(max(a, 0.0), 1.0)
                v_src = calc_v_flux(pipe, z, src_pe, src_pp, g_src, src_ids, latent_image_ids, t)
                v_tar = calc_v_flux(pipe, z, tar_pe, tar_pp, g_tar, tar_ids, latent_image_ids, t)
                v = (1.0 - a) * v_src + a * v_tar
                blend_hist.append(z.detach().to(torch.float32).flatten().cpu().numpy())
                z = (z.to(torch.float32) + dt * v.to(torch.float32)).to(dtype)
            return np.stack(blend_hist, axis=0)

        alpha_pairs = [(alpha_start, alpha_end)]
        if args.alpha_pairs:
            alpha_pairs = parse_alpha_pairs(args.alpha_pairs)

        trajs = [
            {"label": "src trajectory", "legend_label": "Source Trajectory", "x": src_traj, "color": "#2563eb", "lw": 4.2, "alpha": 1.0, "annotate": False, "legend": True},
            {"label": "tar trajectory", "legend_label": "Target Trajectory", "x": tar_traj, "color": "#fa8072", "lw": 4.2, "alpha": 1.0, "annotate": False, "legend": True},
        ]
        blend_colors = [
            "#dc2626", "#ea580c", "#d97706", "#ca8a04", "#65a30d",
            "#0d9488", "#0284c7", "#7c3aed", "#c026d3", "#be123c",
        ]
        for i, (a_s, a_e) in enumerate(alpha_pairs):
            trajs.append(
                {
                    "label": f"blend (α {fmt_num(a_s)} → {fmt_num(a_e)})",
                    "x": integrate_blend(a_s, a_e),
                    "color": blend_colors[i % len(blend_colors)],
                    "lw": 2.4,
                    "alpha": 0.62,
                    "annotate": True,
                    "legend": False,
                }
            )
        title = "Add Paradigm"
        subtitle = f'Source="{args.source_prompt}"  Target="{args.target_prompt}"'
    elif args.mode == "remove":
        unsafe_traj = integrate_single(unsafe_pe, unsafe_pp, unsafe_ids, g_tar)

        def integrate_remove(alpha_s: float, alpha_e: float) -> np.ndarray:
            z = z0.clone()
            remove_hist = []
            for i, t in enumerate(timesteps):
                scheduler._init_step_index(t)
                sigma_i = scheduler.sigmas[scheduler.step_index]
                sigma_im1 = scheduler.sigmas[scheduler.step_index + 1]
                dt = sigma_im1 - sigma_i
                a = linear_at_step(i, len(timesteps), alpha_s, alpha_e)
                a = min(max(a, 0.0), 0.999999)
                v_src = calc_v_flux(pipe, z, src_pe, src_pp, g_src, src_ids, latent_image_ids, t)
                v_unsafe = calc_v_flux(pipe, z, unsafe_pe, unsafe_pp, g_tar, unsafe_ids, latent_image_ids, t)
                v = (v_src - a * v_unsafe) / (1.0 - a)
                remove_hist.append(z.detach().to(torch.float32).flatten().cpu().numpy())
                z = (z.to(torch.float32) + dt * v.to(torch.float32)).to(dtype)
            return np.stack(remove_hist, axis=0)

        alpha_pairs = [(alpha_start, alpha_end)]
        if args.alpha_pairs:
            alpha_pairs = parse_alpha_pairs(args.alpha_pairs)

        trajs = [
            {"label": "src trajectory", "legend_label": "Source Trajectory", "x": src_traj, "color": "#2563eb", "lw": 4.2, "alpha": 1.0, "annotate": False, "legend": True},
            {"label": "unsafe trajectory", "legend_label": "Unsafe Trajectory", "x": unsafe_traj, "color": "#fa8072", "lw": 4.2, "alpha": 1.0, "annotate": False, "legend": True},
        ]
        for a_s, a_e in alpha_pairs:
            trajs.append(
                {
                    "label": f"removed (α {fmt_num(a_s)} → {fmt_num(a_e)})",
                    "x": integrate_remove(a_s, a_e),
                    "color": "#64748b",
                    "lw": 2.4,
                    "alpha": 0.62,
                    "annotate": True,
                    "legend": False,
                }
            )
        title = "Replace Paradigm"
        subtitle = f'Source="{args.source_prompt}"  Unsafe="{unsafe_prompt}"'
    else:
        safe_traj = integrate_single(safe_pe, safe_pp, safe_ids, g_tar)
        unsafe_traj = integrate_single(unsafe_pe, unsafe_pp, unsafe_ids, g_tar)
        def integrate_replace(ms: float, me: float, a_s: float, a_e: float) -> np.ndarray:
            z = z0.clone()
            final_hist = []
            for i, t in enumerate(timesteps):
                scheduler._init_step_index(t)
                sigma_i = scheduler.sigmas[scheduler.step_index]
                sigma_im1 = scheduler.sigmas[scheduler.step_index + 1]
                dt = sigma_im1 - sigma_i
                a = linear_at_step(i, len(timesteps), a_s, a_e)
                mu_t = max(0.0, linear_at_step(i, len(timesteps), ms, me))
                if a >= 1.0 + mu_t:
                    a = (1.0 + mu_t) - 1e-6
                v_src = calc_v_flux(pipe, z, src_pe, src_pp, g_src, src_ids, latent_image_ids, t)
                v_safe = calc_v_flux(pipe, z, safe_pe, safe_pp, g_tar, safe_ids, latent_image_ids, t)
                v_unsafe = calc_v_flux(pipe, z, unsafe_pe, unsafe_pp, g_tar, unsafe_ids, latent_image_ids, t)
                v = (v_src + mu_t * v_safe - a * v_unsafe) / (1.0 + mu_t - a)
                final_hist.append(z.detach().to(torch.float32).flatten().cpu().numpy())
                z = (z.to(torch.float32) + dt * v.to(torch.float32)).to(dtype)
            return np.stack(final_hist, axis=0)

        replace_pairs = [(mu_start, mu_end, alpha_start, alpha_end)]
        if args.replace_pairs:
            replace_pairs = parse_replace_pairs(args.replace_pairs)

        trajs = [
            {"label": "src trajectory", "legend_label": "Source Trajectory", "x": src_traj, "color": "#2563eb", "lw": 4.2, "alpha": 1.0, "annotate": False, "legend": True},
            {"label": "unsafe trajectory", "legend_label": "Unsafe Trajectory", "x": unsafe_traj, "color": "#fa8072", "lw": 4.2, "alpha": 1.0, "annotate": False, "legend": True},
            {"label": "safe trajectory", "legend_label": "Safe Trajectory", "x": safe_traj, "color": "#16a34a", "lw": 4.2, "alpha": 1.0, "annotate": False, "legend": True},
        ]
        replace_colors = [
            "#fa8072", "#fb7185", "#f97316", "#f59e0b", "#84cc16",
            "#14b8a6", "#22d3ee", "#a78bfa", "#e879f9", "#f43f5e",
        ]
        for i, (ms, me, a_s, a_e) in enumerate(replace_pairs):
            trajs.append(
                {
                    "label": f"replace (μ {fmt_num(ms)} → {fmt_num(me)}, α {fmt_num(a_s)} → {fmt_num(a_e)})",
                    "x": integrate_replace(ms, me, a_s, a_e),
                    "color": replace_colors[i % len(replace_colors)],
                    "lw": 2.4,
                    "alpha": 0.62,
                    "annotate": True,
                    "legend": False,
                }
            )
        title = "Replace Paradigm"
        subtitle = f'Source="{args.source_prompt}"  Safe="{args.safe_prompt}"  Unsafe="{unsafe_prompt}"'


    x_all = np.concatenate([t["x"] for t in trajs], axis=0)
    y_all, _, _ = pca2(x_all)
    y_all = normalize_0_1(y_all)

    idx = 0
    for tr in trajs:
        n = tr["x"].shape[0]
        tr["y"] = y_all[idx:idx + n]
        idx += n

    if args.mode in ("add", "remove") and len(trajs) >= 3:
        src_y = trajs[0]["y"]
        tar_y = trajs[1]["y"]
        blue = np.array(mcolors.to_rgb("#2563eb"))
        salmon = np.array(mcolors.to_rgb("#fa8072"))
        for tr in trajs[2:]:
            yb = tr["y"]
            d_src = np.linalg.norm(yb[:, None, :] - src_y[None, :, :], axis=2).min(axis=1).mean()
            d_tar = np.linalg.norm(yb[:, None, :] - tar_y[None, :, :], axis=2).min(axis=1).mean()
            denom = max(float(d_src + d_tar), 1e-8)
            w_tar = float(d_src / denom)
            rgb = (1.0 - w_tar) * blue + w_tar * salmon
            tr["color"] = mcolors.to_hex(np.clip(rgb, 0.0, 1.0))
    elif args.mode == "replace" and len(trajs) >= 4:
        src_ref = next(t for t in trajs if t.get("legend_label") == "Source Trajectory")
        safe_ref = next(t for t in trajs if t.get("legend_label") == "Safe Trajectory")
        unsafe_ref = next(t for t in trajs if t.get("legend_label") == "Unsafe Trajectory")
        ref_defs = [
            ("src", src_ref["y"], np.array(mcolors.to_rgb("#2563eb"))),
            ("safe", safe_ref["y"], np.array(mcolors.to_rgb("#16a34a"))),
            ("unsafe", unsafe_ref["y"], np.array(mcolors.to_rgb("#fa8072"))),
        ]
        for tr in trajs[3:]:
            yb = tr["y"]
            best_dist = float("inf")
            best_rgb = ref_defs[0][2]
            for name, yref, rgb in ref_defs:
                d = np.linalg.norm(yb[:, None, :] - yref[None, :, :], axis=2).min(axis=1).mean()
                if d < best_dist:
                    best_dist = float(d)
                    best_rgb = rgb
            tr["color"] = mcolors.to_hex(np.clip(best_rgb, 0.0, 1.0))

    fig = plt.figure(figsize=(10, 10), dpi=160)
    ax = fig.add_subplot(111)
    legend_handles = []
    legend_labels = []
    for tr in trajs:
        y = tr["y"]
        c = tr["color"]
        lw = tr.get("lw", 3.0)
        a = tr.get("alpha", 0.95)
        line_label = tr.get("legend_label", tr["label"]) if tr.get("legend", False) else "_nolegend_"
        line, = ax.plot(y[:, 0], y[:, 1], color=c, linewidth=lw, alpha=a, zorder=3, label=line_label)
        if tr.get("legend", False):
            legend_handles.append(line)
            legend_labels.append(line_label)
        ax.scatter(y[:, 0], y[:, 1], color=c, s=14, alpha=min(0.9, a), zorder=4)
        ax.scatter(y[-1, 0], y[-1, 1], s=140, c=c, marker="X", edgecolors="white", linewidths=1.0, zorder=6)
        if tr.get("annotate", True):
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

    # common start
    y0 = y_all[0]
    ax.scatter(y0[0], y0[1], s=230, c="#374151", marker="o", edgecolors="white", linewidths=1.2, zorder=7)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("PCA Axis 1", color="#111111")
    ax.set_ylabel("PCA Axis 2", color="#111111")
    ax.tick_params(axis="both", colors="#111111", width=1.2)
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#f8fafc")
    ax.grid(True, color="#94a3b8", alpha=0.35, linewidth=0.9, linestyle=":")
    for spine in ax.spines.values():
        spine.set_color("#111111")
        spine.set_linewidth(1.8)

    ax.set_title(title, fontsize=14, color="#111111", pad=14)
    ax.text(0.5, 1.005, subtitle, ha="center", va="bottom", transform=ax.transAxes, color="#111111", fontsize=10)
    if legend_handles:
        leg = ax.legend(
            legend_handles,
            legend_labels,
            loc="lower right",
            frameon=True,
            framealpha=0.92,
            fontsize=10,
        )
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("#cbd5e1")

    out_path = Path(args.out)
    suffix = f"_{args.mode}"
    if out_path.stem.endswith(suffix):
        out_with_mode = out_path
    else:
        out_with_mode = out_path.with_name(f"{out_path.stem}{suffix}{out_path.suffix}")
    out_with_mode.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_with_mode, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[done] saved: {out_with_mode}")


if __name__ == "__main__":
    main()
