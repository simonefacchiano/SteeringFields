#!/usr/bin/env python3
"""
HOW TO RUN
==========

1) Activate environment:
   source /leonardo_scratch/fast/IscrC_VUnl/envs/steering_fields/bin/activate

2) Run on your PIE-Bench change-object setup:
   python /leonardo_work/IscrC_VUnl/usr/simone/FlowEdit/metrics/clip_txt_img_dir.py \
     --csv /leonardo_work/IscrC_VUnl/usr/simone/FlowEdit/experiments/edit/pie_bench_change_object.csv \
     --baseline-dir /leonardo_scratch/fast/IscrC_NNID/simone/steering_fields/pie_bench_pp/images/1_change_object_80 \
     --edited-dir /leonardo_scratch/fast/IscrC_NNID/simone/steering_fields/results/pie_bench_pp/change/flux_0p7_alpha_0p4_mu_0p4-0p3_alphaend_0p6 \
     --model flux1 \
     --output-csv /leonardo_scratch/fast/IscrC_NNID/simone/steering_fields/results/pie_bench_pp/change/flux_clip_metrics.csv \
     --summary-json /leonardo_scratch/fast/IscrC_NNID/simone/steering_fields/results/pie_bench_pp/change/flux_clip_metrics_summary.json

Notes:
- By default, the script matches files by filename taken from CSV column `image_path`.
- You can provide `--edited-image-col` if edited filenames differ from baseline filenames.
- Required CSV columns (defaults): `image_path`, `source_prompt`, `new_prompt`.
- Default CLIP model is local L/14. You can switch to local B/32 with `--clip-model b32`.

Compute CLIP-based editing metrics from:
1) baseline images dir
2) edited images dir
3) CSV with source/new prompts and image identifiers

Metrics per sample:
- CLIPimg = cos( baseline_image, edited_image )
- CLIPtxt = cos( edited_image, new_prompt )
- CLIPdir = cos( (edited_image-baseline_image), (new_prompt-source_prompt) )
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


MODEL_ALIASES = {
    "sd35": "SD3.5",
    "flux1": "FLUX1",
}

DEFAULT_CLIP_MODEL_ID = "/leonardo_scratch/fast/IscrC_NNID/simone/steering_fields/clip-vit-large-patch14"
B32_CLIP_MODEL_ID = "/leonardo_scratch/fast/EUHPC_D25_097/clip_vit_b32"
# CLIP presets:
# - l14: /leonardo_scratch/fast/IscrC_NNID/simone/steering_fields/clip-vit-large-patch14
# - b32: /leonardo_scratch/fast/EUHPC_D25_097/clip_vit_b32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute CLIPimg/CLIPtxt/CLIPdir from CSV + baseline/edited folders.")
    p.add_argument("--csv", required=True, help="CSV with source_prompt/new_prompt/image_path columns.")
    p.add_argument("--baseline-dir", required=True, help="Directory with baseline images.")
    p.add_argument("--edited-dir", required=True, help="Directory with edited images.")
    p.add_argument("--output-csv", required=True, help="Path to write per-sample metrics CSV.")
    p.add_argument("--summary-json", default=None, help="Optional path to write aggregate means.")
    p.add_argument("--model", required=True, choices=sorted(MODEL_ALIASES.keys()), help="Tag run as sd35 or flux1.")
    p.add_argument(
        "--clip-model",
        default="l14",
        choices=["l14", "b32"],
        help="CLIP preset. Default: l14.",
    )
    p.add_argument(
        "--clip-model-id",
        default=None,
        help="Optional explicit HF/local CLIP model path. Overrides --clip-model.",
    )
    p.add_argument("--cache-dir", default=None, help="Optional HF cache dir.")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for CLIP encoding.")
    p.add_argument("--device", default=None, help="cuda/cpu. Default: cuda if available else cpu.")
    p.add_argument("--image-col", default="image_path", help="CSV column containing image path/name.")
    p.add_argument(
        "--edited-image-col",
        default=None,
        help="Optional CSV column containing edited image path/name. If not set, uses --image-col.",
    )
    p.add_argument("--source-col", default="source_prompt", help="CSV source prompt column.")
    p.add_argument("--target-col", default="new_prompt", help="CSV edited prompt column.")
    return p.parse_args()


def _set_cache_env(cache_dir: str) -> None:
    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")


def load_rows(
    csv_path: Path,
    image_col: str,
    source_col: str,
    target_col: str,
    edited_image_col: str | None,
) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        required = {image_col, source_col, target_col}
        if edited_image_col:
            required.add(edited_image_col)
        missing = required - cols
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        return list(reader)


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def resolve_edited_path(edited_dir: Path, requested_name: str) -> Path:
    """Resolve an edited image path when the edited filename differs from the CSV.

    PIE-Bench CSVs usually point to baseline JPEG names such as `123.jpg`, while
    some edited runs save files like `123_rf_inversion.png`. We first try the
    exact requested name, then fall back to files whose stem is either the same
    or prefixed by `<stem>_`.
    """
    exact = edited_dir / requested_name
    if exact.exists():
        return exact

    requested = Path(requested_name)
    stem = requested.stem
    candidates = sorted(
        p for p in edited_dir.iterdir()
        if p.is_file() and (p.stem == stem or p.stem.startswith(f"{stem}_"))
    )
    if candidates:
        return candidates[0]
    return exact


@torch.no_grad()
def encode_images(model: CLIPModel, processor: CLIPProcessor, images: List[Image.Image], device: torch.device) -> torch.Tensor:
    # Use the same embedding path as CLIPModel forward/clip_score.py.
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    vision_outputs = model.vision_model(pixel_values=pixel_values)
    if hasattr(vision_outputs, "pooler_output") and torch.is_tensor(vision_outputs.pooler_output):
        pooled = vision_outputs.pooler_output
    elif isinstance(vision_outputs, (tuple, list)) and len(vision_outputs) > 1 and torch.is_tensor(vision_outputs[1]):
        pooled = vision_outputs[1]
    else:
        raise TypeError(f"Unsupported vision output type: {type(vision_outputs)}")
    feats = model.visual_projection(pooled)
    return F.normalize(feats, dim=-1)


@torch.no_grad()
def encode_texts(model: CLIPModel, processor: CLIPProcessor, texts: List[str], device: torch.device) -> torch.Tensor:
    # Use the same embedding path as CLIPModel forward/clip_score.py.
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    text_kwargs = {
        "input_ids": inputs["input_ids"].to(device),
    }
    if "attention_mask" in inputs:
        text_kwargs["attention_mask"] = inputs["attention_mask"].to(device)
    text_outputs = model.text_model(**text_kwargs)
    if hasattr(text_outputs, "pooler_output") and torch.is_tensor(text_outputs.pooler_output):
        pooled = text_outputs.pooler_output
    elif isinstance(text_outputs, (tuple, list)) and len(text_outputs) > 1 and torch.is_tensor(text_outputs[1]):
        pooled = text_outputs[1]
    else:
        raise TypeError(f"Unsupported text output type: {type(text_outputs)}")
    feats = model.text_projection(pooled)
    return F.normalize(feats, dim=-1)


def cosine_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=-1)


def compute_metrics_batch(
    model: CLIPModel,
    processor: CLIPProcessor,
    baseline_paths: List[Path],
    edited_paths: List[Path],
    source_prompts: List[str],
    target_prompts: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    baseline_images = [_open_rgb(p) for p in baseline_paths]
    edited_images = [_open_rgb(p) for p in edited_paths]

    i_base = encode_images(model, processor, baseline_images, device=device)
    i_edit = encode_images(model, processor, edited_images, device=device)
    t_base = encode_texts(model, processor, source_prompts, device=device)
    t_edit = encode_texts(model, processor, target_prompts, device=device)

    clip_img = cosine_batch(i_base, i_edit)
    clip_txt = cosine_batch(i_edit, t_edit)

    d_img = F.normalize(i_edit - i_base, dim=-1)
    d_txt = F.normalize(t_edit - t_base, dim=-1)
    clip_dir = cosine_batch(d_img, d_txt)
    return clip_img, clip_txt, clip_dir


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv).resolve()
    baseline_dir = Path(args.baseline_dir).resolve()
    edited_dir = Path(args.edited_dir).resolve()
    output_csv = Path(args.output_csv).resolve()
    summary_json = Path(args.summary_json).resolve() if args.summary_json else None

    if args.cache_dir:
        _set_cache_env(args.cache_dir)
        cache_dir = args.cache_dir
    else:
        cache_dir = None

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = load_rows(csv_path, args.image_col, args.source_col, args.target_col, args.edited_image_col)

    if args.clip_model_id:
        clip_model_id = args.clip_model_id
    else:
        clip_model_id = DEFAULT_CLIP_MODEL_ID if args.clip_model == "l14" else B32_CLIP_MODEL_ID

    model = CLIPModel.from_pretrained(clip_model_id, cache_dir=cache_dir).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_id, cache_dir=cache_dir)
    model.eval()

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    out_rows: List[Dict[str, object]] = []
    valid_items: List[Tuple[int, Dict[str, str], Path, Path, str]] = []

    for i, row in enumerate(rows):
        image_name = Path(row[args.image_col]).name
        edited_name = Path(row[args.edited_image_col]).name if args.edited_image_col else image_name
        bpath = baseline_dir / image_name
        epath = resolve_edited_path(edited_dir, edited_name)
        if not bpath.exists() or not epath.exists():
            out_rows.append(
                {
                    "index": i,
                    "image_name": f"{image_name}|{edited_name}",
                    "baseline_path": str(bpath),
                    "edited_path": str(epath),
                    "clip_img": "",
                    "clip_txt": "",
                    "clip_dir": "",
                    "status": "missing_file",
                    "error": f"baseline_exists={bpath.exists()}, edited_exists={epath.exists()}",
                    "model": MODEL_ALIASES[args.model],
                }
            )
            continue
        valid_items.append((i, row, bpath, epath, image_name))

    bs = max(1, args.batch_size)
    for start in range(0, len(valid_items), bs):
        chunk = valid_items[start : start + bs]
        idxs = [x[0] for x in chunk]
        rws = [x[1] for x in chunk]
        bps = [x[2] for x in chunk]
        eps = [x[3] for x in chunk]
        names = [x[4] for x in chunk]

        source_prompts = [r[args.source_col] for r in rws]
        target_prompts = [r[args.target_col] for r in rws]

        try:
            clip_img, clip_txt, clip_dir = compute_metrics_batch(
                model=model,
                processor=processor,
                baseline_paths=bps,
                edited_paths=eps,
                source_prompts=source_prompts,
                target_prompts=target_prompts,
                device=device,
            )
            for j in range(len(chunk)):
                out_rows.append(
                    {
                        "index": idxs[j],
                        "image_name": names[j],
                        "baseline_path": str(bps[j]),
                        "edited_path": str(eps[j]),
                        "clip_img": float(clip_img[j].item()),
                        "clip_txt": float(clip_txt[j].item()),
                        "clip_dir": float(clip_dir[j].item()),
                        "status": "ok",
                        "error": "",
                        "model": MODEL_ALIASES[args.model],
                    }
                )
        except Exception as exc:
            for j in range(len(chunk)):
                out_rows.append(
                    {
                        "index": idxs[j],
                        "image_name": names[j],
                        "baseline_path": str(bps[j]),
                        "edited_path": str(eps[j]),
                        "clip_img": "",
                        "clip_txt": "",
                        "clip_dir": "",
                        "status": "error",
                        "error": str(exc),
                        "model": MODEL_ALIASES[args.model],
                    }
                )

    out_rows.sort(key=lambda x: int(x["index"]))

    fields = [
        "index",
        "image_name",
        "baseline_path",
        "edited_path",
        "clip_img",
        "clip_txt",
        "clip_dir",
        "status",
        "error",
        "model",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    ok = [r for r in out_rows if r["status"] == "ok"]
    if ok:
        mean_clip_img = float(sum(float(r["clip_img"]) for r in ok) / len(ok))
        mean_clip_txt = float(sum(float(r["clip_txt"]) for r in ok) / len(ok))
        mean_clip_dir = float(sum(float(r["clip_dir"]) for r in ok) / len(ok))
    else:
        mean_clip_img = mean_clip_txt = mean_clip_dir = float("nan")

    summary = {
        "model": MODEL_ALIASES[args.model],
        "n_total": len(out_rows),
        "n_ok": len(ok),
        "n_failed": len(out_rows) - len(ok),
        "mean_clip_img": mean_clip_img,
        "mean_clip_txt": mean_clip_txt,
        "mean_clip_dir": mean_clip_dir,
        "output_csv": str(output_csv),
    }

    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        with summary_json.open("w") as f:
            json.dump(summary, f, indent=2)

    print(json.dumps(summary))


if __name__ == "__main__":
    main()
