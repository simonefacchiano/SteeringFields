#!/usr/bin/env python3
import argparse
import csv
import datetime
import os
import re
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor

# How to execute:
"""""
python /leonardo/home/userexternal/sfacchia/work/usr/simone/hycoclip_small/PAOLO/new_scripts/metrics/clip_score.py \
   --images-dir /leonardo_scratch/fast/EUHPC_D26_044/HyperbolicSteering/coco_generation_flux1 \
   --csv /leonardo/home/userexternal/sfacchia/work/usr/simone/hycoclip_small/PAOLO/new_scripts/metrics/coco/coco_captions_val2017_1000.csv
"""""

DEFAULT_MODEL_PATH = "/leonardo_scratch/fast/IscrC_NNID/simone/steering_fields/clip-vit-large-patch14"

def load_pairs(csv_path: Path):
    pairs = []
    sensitive_prompts = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV must contain a header row.")
        has_legacy = "id" in reader.fieldnames and "prompt" in reader.fieldnames
        has_coco = "image_id" in reader.fieldnames and "caption" in reader.fieldnames
        has_sensitive = "sensitive prompt" in reader.fieldnames
        if not has_legacy and not has_coco and not has_sensitive:
            raise ValueError(
                "CSV must contain 'image_id,caption', 'id,prompt', "
                "or 'sensitive prompt' columns."
            )
        for row in reader:
            if has_coco:
                pairs.append((row["image_id"], row["caption"]))
            elif has_sensitive and not has_legacy:
                sensitive_prompts.append(row["sensitive prompt"])
            else:
                pairs.append((row["id"], row["prompt"]))
    if has_coco:
        return "id_prompt", pairs
    if has_sensitive and not has_legacy:
        return "sensitive_prompt", sensitive_prompts
    return "id_prompt", pairs


@torch.no_grad()
def clip_score(model, processor, device, image: Image.Image, prompt: str) -> float:
    inputs = processor(text=[prompt], images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    img_emb = outputs.image_embeds
    txt_emb = outputs.text_embeds
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    return (img_emb @ txt_emb.T).item()


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean CLIP score for image-prompt pairs."
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Folder containing images named as <id>.(png|jpg|jpeg)",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="CSV with columns: id,prompt",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path or HF id for CLIP model.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use (e.g., cuda, cuda:0, cpu).",
    )
    args = parser.parse_args()

    print("Computing...")
    images_dir = Path(args.images_dir)
    csv_path = Path(args.csv)
    results_dir = Path(
        os.environ.get(
            "CLIP_SCORE_RESULTS_DIR",
            "/leonardo_scratch/fast/IscrC_TBSP/simone/clip_score_results",
        )
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    processor = CLIPProcessor.from_pretrained(args.model_path, use_fast=True)
    model = CLIPModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()

    mode, data = load_pairs(csv_path)
    if not data:
        raise SystemExit("No rows found in CSV.")

    total = 0.0
    count = 0
    if mode == "sensitive_prompt":
        prompts = data
        pattern = re.compile(r"^(\d+)_.*\.(png|jpe?g)$")
        image_paths = sorted(
            list(images_dir.glob("*.png")) +
            list(images_dir.glob("*.jpg")) +
            list(images_dir.glob("*.jpeg"))
        )
        for image_path in image_paths:
            match = pattern.match(image_path.name)
            if not match:
                print(f"Skipping image without index prefix: {image_path}")
                continue
            idx = int(match.group(1))
            if idx < 0 or idx >= len(prompts):
                print(f"Missing prompt for index {idx} (image {image_path.name})")
                continue
            prompt = prompts[idx]
            image = Image.open(image_path).convert("RGB")
            score = clip_score(model, processor, device, image, prompt)
            total += score
            count += 1
    else:
        pairs = data
        missing_images = []
        for img_id, prompt in pairs:
            image_path = None
            for ext in (".png", ".jpg", ".jpeg"):
                candidate = images_dir / f"{img_id}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is None:
                missing_images.append(img_id)
                continue
            try:
                image = Image.open(image_path).convert("RGB")
            except UnidentifiedImageError:
                print(f"Skipping unreadable image: {image_path}")
                continue
            score = clip_score(model, processor, device, image, prompt)
            total += score
            count += 1

    if count == 0 and mode == "id_prompt":
        print("No valid image-prompt pairs found; trying underscore-prefix fallback.")
        prefix_map = {}
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for image_path in sorted(images_dir.glob(ext)):
                match = re.match(r"^(\d+)_", image_path.name)
                if not match:
                    continue
                prefix = match.group(1)
                if prefix not in prefix_map:
                    prefix_map[prefix] = image_path
        missing_images = []
        for img_id, prompt in pairs:
            image_path = prefix_map.get(str(img_id))
            if image_path is None:
                missing_images.append(img_id)
                continue
            try:
                image = Image.open(image_path).convert("RGB")
            except UnidentifiedImageError:
                print(f"Skipping unreadable image: {image_path}")
                continue
            score = clip_score(model, processor, device, image, prompt)
            total += score
            count += 1

    if count == 0:
        if mode == "id_prompt" and missing_images:
            for missing in missing_images:
                print(f"Missing image: {missing}")
        raise SystemExit("No valid image-prompt pairs found.")
    mean_score = total / count
    mean_score_rounded = round(mean_score, 4)
    print(
        f">>> Mean CLIP score for the image folder {images_dir.name}: "
        f"{mean_score_rounded} (N={count})"
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = results_dir / f"clip_score_{timestamp}.txt"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"images_dir: {images_dir}\n")
        handle.write(f"csv: {csv_path}\n")
        handle.write(f"mean_clip_score: {mean_score_rounded}\n")
        handle.write(f"count: {count}\n")


if __name__ == "__main__":
    main()
