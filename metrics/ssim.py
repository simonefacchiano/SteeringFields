#!/usr/bin/env python3
import argparse
import csv
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision import transforms

# How to execute:
# python /leonardo_work/IscrC_VUnl/usr/simone/FlowEdit/metrics/ssim.py \
#  --dir0 /path/to/imagesA \
#  --dir1 /path/to/imagesB \


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def prompt_to_stem(prompt: str, max_words: int = 8) -> str:
    words = prompt.split()
    stem = "_".join(words[:max_words]) if words else "empty_prompt"
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("._-")
    return stem or "empty_prompt"


def normalize_image_id(value: str):
    token = str(value).strip()
    if not token:
        return None
    if token.isdigit():
        return str(int(token))
    return token


def canonical_stem(stem: str):
    return str(stem).strip().lower()


def make_bucket_by_numeric_id(folder: Path):
    buckets = defaultdict(list)
    for p in list_images(folder):
        key = _normalized_numeric_key(p)
        if key is not None:
            buckets[key].append(p)
    return buckets


def make_bucket_by_stem(folder: Path):
    buckets = defaultdict(list)
    for p in list_images(folder):
        buckets[canonical_stem(p.stem)].append(p)
    return buckets


def _pop_first(bucket, key, used):
    if key is None:
        return None
    items = bucket.get(key)
    if not items:
        return None
    while items and items[0] in used:
        items.pop(0)
    if not items:
        return None
    picked = items.pop(0)
    used.add(picked)
    return picked


def _pop_first_any(bucket, keys, used):
    for k in keys:
        picked = _pop_first(bucket, k, used)
        if picked is not None:
            return picked
    return None


def _load_prompt_rows(csv_path: Path, prompt_column: str):
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows
        if prompt_column not in reader.fieldnames:
            return rows
        seq_idx = 0
        stem_counts = {}
        for row_idx, row in enumerate(reader, start=1):
            prompt = str(row.get(prompt_column, "") or "").strip()
            if not prompt:
                continue
            seq_idx += 1
            base = prompt_to_stem(prompt, max_words=8)
            cnt = stem_counts.get(base, 0) + 1
            stem_counts[base] = cnt
            stem = base if cnt == 1 else f"{base}_{cnt}"
            rows.append(
                {
                    "row_idx": row_idx,
                    "seq_idx": seq_idx,
                    "prompt_stem": canonical_stem(stem),
                    "image_id": normalize_image_id(row.get("image_id", "")),
                }
            )
    return rows


def _index_candidates(entry):
    keys = [entry["seq_idx"], entry["row_idx"], entry["seq_idx"] - 1, entry["row_idx"] - 1]
    out = []
    seen = set()
    for v in keys:
        if isinstance(v, int) and v >= 0:
            s = str(v)
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def match_pairs_csv_guided(dir0: Path, dir1: Path, prompts_csv: Path, dataset: str, prompt_column: str):
    if not prompts_csv or not prompts_csv.is_file():
        return []

    rows = _load_prompt_rows(prompts_csv, prompt_column)
    if not rows:
        return []

    num0 = make_bucket_by_numeric_id(dir0)
    num1 = make_bucket_by_numeric_id(dir1)
    stem0 = make_bucket_by_stem(dir0)
    stem1 = make_bucket_by_stem(dir1)

    def run_non_coco(num_bucket, stem_bucket, numeric_first=True):
        used_num = set()
        used_stem = set()
        pairs = []
        for entry in rows:
            idx_candidates = _index_candidates(entry)
            stem_key = entry["prompt_stem"]
            if numeric_first:
                a = _pop_first_any(num_bucket, idx_candidates, used_num)
                b = _pop_first(stem_bucket, stem_key, used_stem)
            else:
                a = _pop_first(stem_bucket, stem_key, used_stem)
                b = _pop_first_any(num_bucket, idx_candidates, used_num)
            if a is None or b is None:
                continue
            pairs.append((a, b))
        return pairs

    def run_coco():
        used0 = set()
        used1 = set()
        pairs_0idx_1img = []
        for entry in rows:
            img_key = entry["image_id"]
            if img_key is None:
                continue
            base_idx = _pop_first_any(num0, _index_candidates(entry), used0)
            gen_img = _pop_first(num1, img_key, used1)
            if base_idx is None or gen_img is None:
                continue
            pairs_0idx_1img.append((base_idx, gen_img))

        used0 = set()
        used1 = set()
        pairs_0img_1idx = []
        for entry in rows:
            img_key = entry["image_id"]
            if img_key is None:
                continue
            gen_img = _pop_first(num0, img_key, used0)
            base_idx = _pop_first_any(num1, _index_candidates(entry), used1)
            if gen_img is None or base_idx is None:
                continue
            pairs_0img_1idx.append((gen_img, base_idx))

        return pairs_0idx_1img if len(pairs_0idx_1img) >= len(pairs_0img_1idx) else pairs_0img_1idx

    if dataset == "coco":
        return run_coco()

    orient_0num_1stem = run_non_coco(num0, stem1, numeric_first=True)
    orient_0stem_1num = run_non_coco(num1, stem0, numeric_first=False)
    if len(orient_0num_1stem) >= len(orient_0stem_1num):
        return orient_0num_1stem
    return orient_0stem_1num


def match_pairs(dir0: Path, dir1: Path):
    files0 = {p.name: p for p in list_images(dir0)}
    files1 = {p.name: p for p in list_images(dir1)}
    shared = sorted(set(files0.keys()) & set(files1.keys()))
    return [(files0[name], files1[name]) for name in shared]


def match_pairs_stem(dir0: Path, dir1: Path):
    files0 = {p.stem: p for p in list_images(dir0)}
    files1 = {p.stem: p for p in list_images(dir1)}
    shared = sorted(set(files0.keys()) & set(files1.keys()))
    return [(files0[stem], files1[stem]) for stem in shared]


def match_pairs_prefix(dir0: Path, dir1: Path):
    files0 = list_images(dir0)
    files1 = list_images(dir1)
    prefix0 = {}
    for p in files0:
        prefix = p.stem.split("_", 1)[0]
        if prefix and prefix not in prefix0:
            prefix0[prefix] = p
    prefix1 = {}
    for p in files1:
        prefix = p.stem.split("_", 1)[0]
        if prefix and prefix not in prefix1:
            prefix1[prefix] = p
    shared = sorted(set(prefix0.keys()) & set(prefix1.keys()))
    return [(prefix0[prefix], prefix1[prefix]) for prefix in shared]


def _normalized_numeric_key(path: Path):
    token = path.stem.split("_", 1)[0]
    if not token:
        return None
    if not token.isdigit():
        return None
    return str(int(token))


def match_pairs_numeric_id(dir0: Path, dir1: Path):
    files0 = {}
    for p in list_images(dir0):
        key = _normalized_numeric_key(p)
        if key is not None and key not in files0:
            files0[key] = p

    files1 = {}
    for p in list_images(dir1):
        key = _normalized_numeric_key(p)
        if key is not None and key not in files1:
            files1[key] = p

    shared = sorted(set(files0.keys()) & set(files1.keys()), key=int)
    return [(files0[key], files1[key]) for key in shared]


def _resample_lanczos():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def resize_larger_to_smaller(img0: Image.Image, img1: Image.Image):
    w0, h0 = img0.size
    w1, h1 = img1.size
    if (w0, h0) == (w1, h1):
        return img0, img1

    area0 = w0 * h0
    area1 = w1 * h1
    resample = _resample_lanczos()
    if area0 > area1:
        img0 = img0.resize((w1, h1), resample=resample)
    else:
        img1 = img1.resize((w0, h0), resample=resample)
    return img0, img1


def main():
    os.environ.setdefault(
        "TORCH_HOME",
        "/leonardo_scratch/fast/EUHPC_D26_044/HyperbolicSteering/torch_cache",
    )
    parser = argparse.ArgumentParser(
        description="Compute mean SSIM between paired images in two folders."
    )
    parser.add_argument("--dir0", required=True, help="First images folder.")
    parser.add_argument("--dir1", required=True, help="Second images folder.")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to use (e.g., cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Optional resize to SxS before SSIM (e.g., 512).",
    )
    parser.add_argument(
        "--pairwise-resize",
        default="larger_to_smaller",
        choices=["none", "larger_to_smaller"],
        help=(
            "How to handle size mismatch per pair before transforms. "
            "'larger_to_smaller' resizes only the larger image to the smaller one."
        ),
    )
    parser.add_argument(
        "--prompts-csv",
        default=None,
        help="Optional prompts CSV used for robust filename pairing across different naming conventions.",
    )
    parser.add_argument(
        "--dataset",
        default="",
        choices=["", "ring", "p4d", "coco"],
        help="Dataset identifier used by CSV-guided pairing logic.",
    )
    parser.add_argument(
        "--prompt-column",
        default="caption",
        help="Prompt column name for CSV-guided pairing (caption for coco, sensitive prompt for ring/p4d).",
    )
    args = parser.parse_args()

    dir0 = Path(args.dir0)
    dir1 = Path(args.dir1)
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    pair_candidates = []

    exact_pairs = match_pairs(dir0, dir1)
    if exact_pairs:
        pair_candidates.append(("exact-filename", exact_pairs))

    if args.prompts_csv:
        csv_pairs = match_pairs_csv_guided(
            dir0,
            dir1,
            Path(args.prompts_csv),
            args.dataset,
            args.prompt_column,
        )
        if csv_pairs:
            pair_candidates.append(("csv-guided", csv_pairs))

    stem_pairs = match_pairs_stem(dir0, dir1)
    if stem_pairs:
        pair_candidates.append(("stem", stem_pairs))

    prefix_pairs = match_pairs_prefix(dir0, dir1)
    if prefix_pairs:
        pair_candidates.append(("prefix", prefix_pairs))

    numeric_pairs = match_pairs_numeric_id(dir0, dir1)
    if numeric_pairs:
        pair_candidates.append(("numeric-id", numeric_pairs))

    if not pair_candidates:
        raise SystemExit("No matching image filenames found between folders.")

    priority = {
        "exact-filename": 5,
        "numeric-id": 4,
        "csv-guided": 3,
        "stem": 2,
        "prefix": 1,
    }
    best_method, pairs = max(pair_candidates, key=lambda x: (len(x[1]), priority.get(x[0], 0)))
    print(f"SSIM pairing method: {best_method} (N={len(pairs)})")

    transform_ops = []
    if args.size is not None:
        transform_ops.append(transforms.Resize((args.size, args.size)))
    transform_ops.append(transforms.ToTensor())
    transform = transforms.Compose(transform_ops)
    metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    total = 0.0
    count = 0
    for p0, p1 in pairs:
        pil0 = Image.open(p0).convert("RGB")
        pil1 = Image.open(p1).convert("RGB")
        if args.pairwise_resize == "larger_to_smaller":
            pil0, pil1 = resize_larger_to_smaller(pil0, pil1)
        img0 = transform(pil0).unsqueeze(0).to(device)
        img1 = transform(pil1).unsqueeze(0).to(device)
        metric.reset()
        d = metric(img0, img1)
        total += d.item()
        count += 1

    mean = total / count
    print(f"Mean SSIM: {mean} (N={count})")


if __name__ == "__main__":
    main()
