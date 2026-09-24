#!/usr/bin/env python3
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

# How to execute:

# 1)
# python /leonardo/home/userexternal/sfacchia/work/usr/simone/hycoclip_small/PAOLO/new_scripts/metrics/fid_score.py \
#  /path/to/dataset /path/to/output_stats.npz --save-stats --device cuda:0

# 2) Directly from the terminal (but you will likely receive an error):
# python -m pytorch_fid /leonardo_scratch/fast/EUHPC_D26_044/HyperbolicSteering/coco_original_1000 /leonardo_scratch/fast/EUHPC_D26_044/HyperbolicSteering/coco_generation_sdxl

# By default compute the resizing to 299x299

def build_cmd(args):
    cmd = [sys.executable, "-m", "pytorch_fid"]
    if args.save_stats:
        cmd.append("--save-stats")
    if args.device:
        cmd.extend(["--device", args.device])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.dims is not None:
        cmd.extend(["--dims", str(args.dims)])
    if args.num_workers is not None:
        cmd.extend(["--num-workers", str(args.num_workers)])
    cmd.extend([args.path1, args.path2])
    return cmd


def is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in {".png", ".jpg", ".jpeg"}


def image_key_from_name(name: str) -> str | None:
    # Expected generated format: "<row_index>_<prompt stem>.png".
    # Use the prompt stem when present because some folders are 0-based and
    # others are 1-based.
    stem = Path(name).stem
    m = re.match(r"^\d+_(.+)$", stem)
    if m:
        return m.group(1).strip().lower()
    # Also support plain numeric filenames, e.g. "111000000001.jpg".
    if re.fullmatch(r"\d+", stem):
        return stem
    # Fallback: use full stem so paired filtering works for non-numeric but matching names.
    if stem:
        return stem
    return None


def list_images_by_key(folder: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(glob.glob(os.path.join(folder, "*"))):
        if not (os.path.isfile(p) and is_image_file(p)):
            continue
        key = image_key_from_name(os.path.basename(p))
        if key is None:
            continue
        # Keep first deterministic occurrence if duplicates exist.
        out.setdefault(key, p)
    return out


def list_image_files(folder: str) -> list[str]:
    return [
        p
        for p in sorted(glob.glob(os.path.join(folder, "*")))
        if os.path.isfile(p) and is_image_file(p)
    ]


def natural_key(value: str):
    return (0, int(value)) if value.isdigit() else (1, value)


def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def prepare_filtered_pair_dirs(
    path1: str, path2: str, min_pair_overlap: int = 50
) -> tuple[str, str, str | None]:
    """
    If both paths are directories, keep only filename-matched pairs where both images are valid.
    Returns (new_path1, new_path2, tmp_root_or_none).
    """
    if not (os.path.isdir(path1) and os.path.isdir(path2)):
        return path1, path2, None

    imgs1 = list_images_by_key(path1)
    imgs2 = list_images_by_key(path2)
    common = sorted(set(imgs1.keys()) & set(imgs2.keys()), key=natural_key)
    all_imgs1 = list_image_files(path1)
    all_imgs2 = list_image_files(path2)
    min_files = min(len(all_imgs1), len(all_imgs2))
    pair_mode = len(common) >= min_pair_overlap and len(common) >= int(0.9 * min_files)

    tmp_root = tempfile.mkdtemp(prefix="fid_pairs_")
    filt1 = os.path.join(tmp_root, "path1")
    filt2 = os.path.join(tmp_root, "path2")
    os.makedirs(filt1, exist_ok=True)
    os.makedirs(filt2, exist_ok=True)

    if pair_mode:
        kept = 0
        dropped = 0
        for key in common:
            p1 = imgs1[key]
            p2 = imgs2[key]
            if not (is_valid_image(p1) and is_valid_image(p2)):
                dropped += 1
                continue
            ext1 = Path(p1).suffix.lower() or ".png"
            ext2 = Path(p2).suffix.lower() or ".png"
            os.symlink(p1, os.path.join(filt1, f"{key}{ext1}"))
            os.symlink(p2, os.path.join(filt2, f"{key}{ext2}"))
            kept += 1

        print(
            f"[fid_score] mode=paired matched_pairs={len(common)} kept={kept} dropped_corrupted={dropped}",
            file=sys.stderr,
        )
        if kept == 0:
            raise RuntimeError(
                "No valid paired images left after filtering corrupted/missing pairs."
            )
    else:
        # Fallback for datasets with different naming conventions across folders
        # or repeated prompt stems. FID is distributional, so pairing is not
        # required; keep every valid image instead of collapsing duplicates.
        kept1 = dropped1 = 0
        for p1 in all_imgs1:
            if not is_valid_image(p1):
                dropped1 += 1
                continue
            os.symlink(p1, os.path.join(filt1, os.path.basename(p1)))
            kept1 += 1

        kept2 = dropped2 = 0
        for p2 in all_imgs2:
            if not is_valid_image(p2):
                dropped2 += 1
                continue
            os.symlink(p2, os.path.join(filt2, os.path.basename(p2)))
            kept2 += 1

        print(
            f"[fid_score] mode=unpaired matched_pairs={len(common)} kept_path1={kept1} dropped_path1={dropped1} kept_path2={kept2} dropped_path2={dropped2}",
            file=sys.stderr,
        )
        if kept1 == 0 or kept2 == 0:
            raise RuntimeError(
                "No valid images left after filtering in one or both folders."
            )

    return filt1, filt2, tmp_root


def main():
    os.environ.setdefault(
        "TORCH_HOME",
        "/leonardo_scratch/fast/IscrC_VUnl/steering_fields/torch_cache",
    )
    parser = argparse.ArgumentParser(
        description="Wrapper for pytorch-fid to compute FID or save stats."
    )
    parser.add_argument("path1", help="Path to dataset (or .npz stats).")
    parser.add_argument(
        "path2",
        help="Path to dataset, or output .npz when using --save-stats.",
    )
    parser.add_argument("--device", help="Device to use (e.g., cuda:0, cpu).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1).",
    )
    parser.add_argument("--dims", type=int, help="Feature dimensions.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Data loader workers (default: 0).",
    )
    parser.add_argument(
        "--save-stats",
        action="store_true",
        help="Save stats to path2 instead of computing FID.",
    )
    args = parser.parse_args()

    tmp_root = None
    try:
        if not args.save_stats:
            # Compare mode: ignore corrupted images by removing invalid matched pairs.
            args.path1, args.path2, tmp_root = prepare_filtered_pair_dirs(
                args.path1, args.path2
            )

        cmd = build_cmd(args)
        result = subprocess.run(cmd)
        raise SystemExit(result.returncode)
    finally:
        if tmp_root and os.path.isdir(tmp_root):
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
