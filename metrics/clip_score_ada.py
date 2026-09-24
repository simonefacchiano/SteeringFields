import argparse, os, json, random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

# --- your framework imports ---
from pcir.core.utils import ensure_dir  # we only need this from utils
from pcir.models.sdxl import SDXLBackend
from pcir.models.sd21 import SD21Backend
from pcir.models.sd35 import SD35Backend
from pcir.models.flux import FluxBackend

cache_dir = "/scratch/inf0/user/agoerguen/Models"

# Point every HF cache to your scratch dir
os.environ["HF_HOME"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
os.environ["DIFFUSERS_CACHE"]   = os.path.join(cache_dir, "diffusers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# -----------------------------
# Backend builder
# -----------------------------
def build_backend(name: str, cache_dir: Optional[str]):
    """Mirror defaults in run_pcir.py (model id, steps, guidance, do_cfg)."""
    if name == "sdxl":
        return SDXLBackend("stabilityai/stable-diffusion-xl-base-1.0", cache_dir=cache_dir), 40, 5.0, True
    if name == "sd21":
        return SD21Backend("stabilityai/stable-diffusion-2-1-base", cache_dir=cache_dir), 50, 7.5, True
    if name == "sd35":
        return SD35Backend("stabilityai/stable-diffusion-3.5-large", cache_dir=cache_dir), 50, 7.0, True
    if name == "flux":
        return FluxBackend("black-forest-labs/FLUX.1-dev", cache_dir=cache_dir), 25, 3.5, False
    raise ValueError(f"Unknown backend: {name}")


# -----------------------------
# Config + prompt utilities
# -----------------------------
def load_cfg(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def _load_prompts_base_ablated(
    cfg_path: Path,
    category: str,
    aspect: str,
    concept_key: Optional[str],
) -> Tuple[str, str, Set[str]]:
    """
    Returns (base_prompt, ablated_prompt, concept_strings) from {sample}.json.
    Tries common key variants for 'ablated' prompts.
    """
    cfg = load_cfg(cfg_path)
    node = cfg.get(category, {})
    if isinstance(node, dict):
        node = node.get(aspect, {})
    if not isinstance(node, dict):
        return "", "", set()

    concepts = node.get("concepts", [])
    base_prompt = node.get("base_prompt", "")
    ablated_prompts = node.get("ablated_prompts", [])

    if concept_key:
        for i in range(len(ablated_prompts)):
            if concept_key in ablated_prompts[i]:
                if isinstance(base_prompt, list):
                    return base_prompt[i], ablated_prompts[i], set([concepts[i]])
                else:
                    return base_prompt, ablated_prompts[i], set([concepts[i]])

    # fallback (shouldn't really be used for a single concept)
    return base_prompt, ablated_prompts, set(concepts)


# -----------------------------
# Tau stats + seeds utilities
# -----------------------------
def nearest_t(val: float, timesteps) -> Tuple[float, int]:
    """
    Snap a scalar 'val' to the nearest actual timestep in the scheduler grid.
    'timesteps' can be a list, numpy array, or torch.Tensor.
    Returns (timestep_value, index).
    """
    import torch

    if isinstance(timesteps, torch.Tensor):
        ts_tensor = timesteps.to("cpu").float()
    else:
        # list / np.ndarray → tensor
        ts_tensor = torch.as_tensor(timesteps, dtype=torch.float32)

    idx = (ts_tensor - float(val)).abs().argmin().item()
    return float(ts_tensor[idx].item()), int(idx)


def load_cis_tau_stats(
    base_dir: Path,
    model: str,
    category: str,
    aspect: str,
) -> Dict[str, Optional[float]]:
    """
    Load tau*_mean values from:
      results_scores/<model>/<category>/<aspect>/cis_tau_stats.txt
    """
    stats_path = base_dir / "results_scores" / model / category / aspect / "cis_tau_stats.txt"
    if not stats_path.exists():
        raise FileNotFoundError(f"Could not find CIS stats file at: {stats_path}")
    with open(stats_path, "r") as f:
        stats = json.load(f)
    return stats

def read_seeds_from_txt(
    base_dir: Path,
    model: str,
    sample: str,
    category: str,
    aspect: str,
    concept: str,
) -> Tuple[List[int], Optional[Dict[int, Dict[str, str]]]]:
    """
    Try to read seeds from:
      results/<model>/<sample>/<category>/<aspect>/<concept>/seeds.txt

    If seeds.txt does not exist, fall back to:
      results/<model>/<sample>/<category>/<aspect>/<concept>/balanced_seeds.json

    Return:
        seeds: list[int]
        seed_prompt_map: optional dict[seed] -> {
            "target_prompt": str,
            "ablated_prompt": str
        }
    """
    base_path = (
        base_dir
        / "results"
        / model
        / sample
        / category
        / aspect
        / concept
    )
    seeds_path = base_path / "seeds.txt"
    json_path  = base_path / "balanced_seeds.json"

    # --- 1) Prefer seeds.txt if present ---
    if seeds_path.exists():
        seeds: List[int] = []
        with open(seeds_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    seeds.append(int(line))
                except ValueError:
                    continue
        if not seeds:
            raise RuntimeError(f"No valid seeds found in {seeds_path}")
        # no per-seed prompt info in txt mode
        return seeds, None

    # --- 2) Fallback: balanced_seeds.json ---
    if not json_path.exists():
        raise FileNotFoundError(
            f"Could not find seeds file at {seeds_path} "
            f"or balanced_seeds.json at {json_path}"
        )

    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise RuntimeError(f"balanced_seeds.json at {json_path} must be a list of objects")

    seeds: List[int] = []
    seed_prompt_map: Dict[int, Dict[str, str]] = {}

    for entry in data:
        if not isinstance(entry, dict):
            continue
        if "seed" not in entry:
            continue
        try:
            s = int(entry["seed"])
        except (TypeError, ValueError):
            continue

        seeds.append(s)
        # store prompts if provided (fall back to empty strings if missing)
        tgt = entry.get("target_prompt", "")
        abl = entry.get("ablated_prompt", "")
        seed_prompt_map[s] = {
            "target_prompt": tgt,
            "ablated_prompt": abl,
        }

    if not seeds:
        raise RuntimeError(f"No valid seeds found in {json_path}")

    return seeds, seed_prompt_map



def read_seeds_from_txt_v1(
    base_dir: Path,
    model: str,
    sample: str,
    category: str,
    aspect: str,
    concept: str,
) -> List[int]:
    """
    Reads seeds from:
      results/<model>/<sample>/<category>/<aspect>/<concept>/seeds.txt
    """
    seeds_path = (
        base_dir
        / "results"
        / model
        / sample
        / category
        / aspect
        / concept
        / "seeds.txt"
    )
    if not seeds_path.exists():
        raise FileNotFoundError(f"Could not find seeds file at: {seeds_path}")

    seeds: List[int] = []
    with open(seeds_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seeds.append(int(line))
            except ValueError:
                continue
    if not seeds:
        raise RuntimeError(f"No valid seeds found in {seeds_path}")
    return seeds


# -----------------------------
# Main generation logic
# -----------------------------
def generate_for_concept_v1(
    base_dir: Path,
    cache_dir: str,
    model_id: str,
    sample: str,
    category: str,
    aspect: str,
    concept: str,
    num_seeds: int,
    tau_scale: float = 1000.0,
    seed_selection_seed: int = 123,
):
    # --- Backend ---
    backend, default_steps, guidance, do_cfg = build_backend(model_id, cache_dir=cache_dir)
    backend.load()
    backend.set_guidance(guidance)
    
    # Load CLIP once
    clip_model, clip_processor = load_clip(device=device)

    cfg_dir = base_dir / "configs"
    cfg_path = cfg_dir / f"{sample}.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find config at {cfg_path}")

    base_ps, ablated_ps, concept_strs = _load_prompts_base_ablated(
        cfg_path, category, aspect, concept
    )

    print(f"Using base prompt:    {base_ps}")
    print(f"Using ablated prompt: {ablated_ps}")
    print(f"Using concept strings: {concept_strs}")

    # --- pre-encode contexts once ---
    base_ctx = backend.encode(base_ps, negative_prompt=None, do_cfg=do_cfg)
    concept_ctx = backend.encode(ablated_ps, negative_prompt=None, do_cfg=do_cfg)

    # --- set timesteps + get scheduler grid ---
    num_steps = default_steps
    backend.set_timesteps(num_steps)
    timesteps_raw = backend.timesteps()  # might be list or tensor
    # make a tensor view for math, but keep raw for passing into backend if needed
    timesteps = torch.as_tensor(timesteps_raw, dtype=torch.float32)

    # --- load tau stats (normalized in [0,1]) and convert to timesteps ---
    tau_stats = load_cis_tau_stats(base_dir, model_id, category, aspect)

    tau_keys_and_labels = [
        ("tau30_mean", "tau30"),
        ("tau50_mean", "tau50"),
        ("tau60_mean", "tau60"),
        ("tau70_mean", "tau70"),
        ("tau90_mean", "tau90"),
    ]
    tau_timesteps: Dict[str, int] = {}

    for key, label in tau_keys_and_labels:
        v = tau_stats.get(key, None)
        if v is None:
            continue
        # v is in [0,1]; convert to absolute timestep via tau_scale (usually 1000)
        t_raw = v * tau_scale
        t_snap, _ = nearest_t(t_raw, timesteps)
        tau_timesteps[label] = int(t_snap)

    print("Using tau-based timesteps:", tau_timesteps)

    # --- read seeds and sample N of them ---
    all_seeds = read_seeds_from_txt(base_dir, model_id, sample, category, aspect, concept)
    print(f"Found {len(all_seeds)} seeds in seeds.txt")

    if num_seeds > len(all_seeds):
        print(f"Requested {num_seeds} seeds but only {len(all_seeds)} available; using all.")
        num_seeds = len(all_seeds)

    random.seed(seed_selection_seed)
    chosen_seeds = random.sample(all_seeds, num_seeds)
    print(f"Chosen seeds: {chosen_seeds}")

    # --- per-seed generation ---
    for seed in chosen_seeds:
        print(f"\n=== Processing seed {seed} ===")
        out_dir = (
            base_dir
            / "image2image"
            / model_id
            / sample
            / category
            / aspect
            / concept
            / f"seed_{seed}"
        )
        ensure_dir(out_dir)

        # -------------------------
        # 1) Generate original.png
        # -------------------------
        backend.set_timesteps(num_steps)
        lat0 = backend.prepare_latents(seed)

        # Use smallest timestep as "end" (fully denoised)
        lat_end, idx_end = backend.run_until(backend.timesteps()[-1], lat0.clone(), base_ctx)
        img_orig = backend.reconstruct_to_end(idx_end, lat_end, base_ctx)
        img_orig.save(out_dir / "original.png")
        print(f"Saved original image to {out_dir / 'original.png'}")

        # -------------------------
        # 2) Generate tau images
        # -------------------------
        for label in ["tau30", "tau50", "tau60", "tau70", "tau90"]:
            if label not in tau_timesteps:
                print(f"Skipping {label}: no tau value found.")
                continue
            t_val = tau_timesteps[label]
            print(f"Generating image at {label} (t ≈ {t_val})")

            backend.set_timesteps(num_steps)
            lat0 = backend.prepare_latents(seed)

            # base → down to t_val
            lat_t, next_idx = backend.run_until(int(t_val), lat0.clone(), base_ctx)
            # switch to concept_ctx and denoise to the end
            img = backend.reconstruct_to_end(next_idx, lat_t, concept_ctx)

            out_path = out_dir / f"{label}.png"
            img.save(out_path)
            print(f"Saved {out_path}")
            
            scores = clip_metrics(
                original_img=img_orig,
                edited_img=img,
                base_prompt=base_ps,
                edit_prompt=ablated_ps,
                clip_model=clip_model,
                clip_processor=clip_processor,
                device=device
            )
            
            #save scores
            scores_path = out_dir / f"{label}_clip_scores.json"
            with open(scores_path, "w") as f:
                json.dump(scores, f, indent=4)
            print(f"Saved CLIP scores to {scores_path}")
            
            
def generate_for_concept(
    base_dir: Path,
    cache_dir: str,
    model_id: str,
    sample: str,
    category: str,
    aspect: str,
    concept: str,
    num_seeds: int,
    tau_scale: float = 1000.0,
    seed_selection_seed: int = 123,
):
    # --- Backend ---
    backend, default_steps, guidance, do_cfg = build_backend(model_id, cache_dir=cache_dir)
    backend.load()
    backend.set_guidance(guidance)
    
    # Load CLIP once
    clip_model, clip_processor = load_clip(device=device)

    cfg_dir = base_dir / "configs"
    cfg_path = cfg_dir / f"{sample}.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find config at {cfg_path}")

    # Default prompts from config (used if we don't override from JSON)
    base_ps_default, ablated_ps_default, concept_strs = _load_prompts_base_ablated(
        cfg_path, category, aspect, concept
    )

    print(f"[Config] base prompt:    {base_ps_default}")
    print(f"[Config] ablated prompt: {ablated_ps_default}")
    print(f"Using concept strings: {concept_strs}")

    # --- set timesteps + get scheduler grid ---
    num_steps = default_steps
    backend.set_timesteps(num_steps)
    timesteps_raw = backend.timesteps()  # might be list or tensor
    timesteps = torch.as_tensor(timesteps_raw, dtype=torch.float32)

    # --- load tau stats (normalized in [0,1]) and convert to timesteps ---
    tau_stats = load_cis_tau_stats(base_dir, model_id, category, aspect)

    tau_keys_and_labels = [
        ("tau30_mean", "tau30"),
        ("tau50_mean", "tau50"),
        ("tau60_mean", "tau60"),
        ("tau70_mean", "tau70"),
        ("tau90_mean", "tau90"),
    ]
    tau_timesteps: Dict[str, int] = {}

    for key, label in tau_keys_and_labels:
        v = tau_stats.get(key, None)
        if v is None:
            continue
        # v is in [0,1]; convert to absolute timestep via tau_scale (usually 1000)
        t_raw = v * tau_scale
        t_snap, _ = nearest_t(t_raw, timesteps)
        tau_timesteps[label] = int(t_snap)

    print("Using tau-based timesteps:", tau_timesteps)

    # --- read seeds (with optional per-seed prompts) and sample N of them ---
    all_seeds, seed_prompt_map = read_seeds_from_txt(
        base_dir, model_id, sample, category, aspect, concept
    )
    print(f"Found {len(all_seeds)} seeds.")

    if num_seeds > len(all_seeds):
        print(f"Requested {num_seeds} seeds but only {len(all_seeds)} available; using all.")
        num_seeds = len(all_seeds)

    if seed_selection_seed is not None:
        random.seed(seed_selection_seed)
    chosen_seeds = random.sample(all_seeds, num_seeds)
    print(f"Chosen seeds: {chosen_seeds}")

    # --- per-seed generation ---
    for seed in chosen_seeds:
        print(f"\n=== Processing seed {seed} ===")

        # --- per-seed prompts: override from JSON if available ---
        if seed_prompt_map is not None and seed in seed_prompt_map:
            base_ps_seed = seed_prompt_map[seed]["target_prompt"] or base_ps_default
            ablated_ps_seed = seed_prompt_map[seed]["ablated_prompt"] or ablated_ps_default
            print(f"[Seed {seed}] Using prompts from balanced_seeds.json:")
        else:
            base_ps_seed = base_ps_default
            ablated_ps_seed = ablated_ps_default
            print(f"[Seed {seed}] Using prompts from config:")
        print(f"  base_ps_seed   = {base_ps_seed}")
        print(f"  ablated_ps_seed= {ablated_ps_seed}")

        # encode contexts for THIS seed's prompts
        base_ctx = backend.encode(base_ps_seed, negative_prompt=None, do_cfg=do_cfg)
        concept_ctx = backend.encode(ablated_ps_seed, negative_prompt=None, do_cfg=do_cfg)

        out_dir = (
            base_dir
            / "image2image"
            / model_id
            / sample
            / category
            / aspect
            / concept
            / f"seed_{seed}"
        )
        ensure_dir(out_dir)

        # -------------------------
        # 1) Generate original.png
        # -------------------------
        backend.set_timesteps(num_steps)
        lat0 = backend.prepare_latents(seed)

        # Use smallest timestep as "end" (fully denoised)
        lat_end, idx_end = backend.run_until(backend.timesteps()[-1], lat0.clone(), base_ctx)
        img_orig = backend.reconstruct_to_end(idx_end, lat_end, base_ctx)
        img_orig.save(out_dir / "original.png")
        print(f"Saved original image to {out_dir / 'original.png'}")

        # -------------------------
        # 2) Generate tau images
        # -------------------------
        for label in ["tau30", "tau50", "tau60", "tau70", "tau90"]:
            if label not in tau_timesteps:
                print(f"Skipping {label}: no tau value found.")
                continue
            t_val = tau_timesteps[label]
            print(f"Generating image at {label} (t ≈ {t_val})")

            backend.set_timesteps(num_steps)
            lat0 = backend.prepare_latents(seed)

            # base → down to t_val
            lat_t, next_idx = backend.run_until(int(t_val), lat0.clone(), base_ctx)
            # switch to concept_ctx and denoise to the end
            img = backend.reconstruct_to_end(next_idx, lat_t, concept_ctx)

            out_path = out_dir / f"{label}.png"
            img.save(out_path)
            print(f"Saved {out_path}")
            
            scores = clip_metrics(
                original_img=img_orig,
                edited_img=img,
                base_prompt=base_ps_seed,      # per-seed base
                edit_prompt=ablated_ps_seed,   # per-seed ablated
                clip_model=clip_model,
                clip_processor=clip_processor,
                device=device
            )
            
            # save scores
            scores_path = out_dir / f"{label}_clip_scores.json"
            with open(scores_path, "w") as f:
                json.dump(scores, f, indent=4)
            print(f"Saved CLIP scores to {scores_path}")



def load_clip(model_id: str = "openai/clip-vit-large-patch14", device: torch.device = torch.device("cpu")):
    model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir).to(device)
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model.eval()
    return model, processor

@torch.no_grad()
def _clip_image_features(model, processor, images, device):
    if not isinstance(images, (list, tuple)):
        images = [images]
    inputs = processor(images=images, return_tensors="pt").to(device)
    feats = model.get_image_features(**inputs)  # (B, D)
    feats = F.normalize(feats, dim=-1)
    return feats

@torch.no_grad()
def _clip_text_features(model, processor, texts, device):
    if isinstance(texts, str):
        texts = [texts]
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    feats = model.get_text_features(**inputs)  # (B, D)
    feats = F.normalize(feats, dim=-1)
    return feats

def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.view(1, -1); b = b.view(1, -1)
    return float(F.cosine_similarity(a, b, dim=-1).item())

@torch.no_grad()
def clip_metrics(original_img: Image.Image,
                 edited_img: Image.Image,
                 base_prompt: str,
                 edit_prompt: str,
                 clip_model,
                 clip_processor,
                 device) -> dict:
    # normalized embeddings
    i_base = _clip_image_features(clip_model, clip_processor, original_img, device=device)[0]
    i_edit = _clip_image_features(clip_model, clip_processor, edited_img, device=device)[0]
    t_base = _clip_text_features(clip_model, clip_processor, base_prompt, device=device)[0]
    t_edit = _clip_text_features(clip_model, clip_processor, edit_prompt, device=device)[0]

    clip_img = _cos(i_base, i_edit)       # (1) image-image
    clip_txt = _cos(i_edit, t_edit)       # (2) image-text
    d_img   = F.normalize(i_edit - i_base, dim=-1)
    d_txt   = F.normalize(t_edit - t_base, dim=-1)
    clip_dir = _cos(d_img, d_txt)         # (3) direction

    return {"CLIPimg": clip_img, "CLIPtxt": clip_txt, "CLIPdir": clip_dir}

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate image2image visualizations at tau30/50/60/70/90 for a concept."
    )
    p.add_argument("--model", type=str, default="sdxl", choices=["sdxl", "sd21", "sd35", "flux"])
    p.add_argument("--sample", type=str, default="sample_1", help="Sample id, e.g. sample_7")
    p.add_argument("--category", type=str, default="objects")
    p.add_argument("--aspect", type=str, default="animals")
    p.add_argument("--concept", type=str, default="dog")
    p.add_argument(
        "--num_seeds",
        "-N",
        type=int,
        default=5,
        help="Number of seeds to randomly select from seeds.txt",
    )
    p.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base project directory (where configs/, results/, results_scores/ live). "
             "Default: directory of this script.",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default="/scratch/inf0/user/agoerguen/Models/",
        help="HF cache directory for the models.",
    )
    p.add_argument(
        "--tau-scale",
        type=float,
        default=1000.0,
        help="Scale factor to convert normalized tau in [0,1] to timestep, e.g. 1000.",
    )
    p.add_argument(
        "--seed-selection-seed",
        type=int,
        default=None,
        help="Random seed for selecting seeds from seeds.txt",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.base_dir is None:
        BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    else:
        BASE_DIR = Path(args.base_dir).resolve()

    generate_for_concept(
        base_dir=BASE_DIR,
        cache_dir=args.cache_dir,
        model_id=args.model,
        sample=args.sample,
        category=args.category,
        aspect=args.aspect,
        concept=args.concept,
        num_seeds=args.num_seeds,
        tau_scale=args.tau_scale,
        seed_selection_seed=args.seed_selection_seed,
    )
