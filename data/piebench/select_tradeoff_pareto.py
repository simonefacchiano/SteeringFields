#!/usr/bin/env python3
"""
Pareto-based tradeoff selection for PIE-Bench metrics.

Row selection logic:
1) Read rows from metrics_summary.csv.
2) Keep only rows with:
   - clip_score_delta > min_delta (default min_delta=0.0, so strictly positive delta),
   - finite numeric values for clip_img, clip_dir, fid, vqascore_edit_gain, clip_score_delta.
3) Build the Pareto front with these objectives:
   - maximize clip_img,
   - maximize clip_dir,
   - maximize vqascore_edit_gain,
   - minimize fid.
   A row is removed if another row is >= on all objectives and strictly better on at least one.
4) Save only non-dominated rows to output CSV.
5) Add compromise_score only to sort/display Pareto rows:
   - normalize each metric to [0,1],
   - invert fid contribution (lower fid -> higher contribution),
   - average the four normalized terms with equal weight.

Important: compromise_score does NOT determine Pareto membership.
It is only a helper to order Pareto-optimal rows.
"""

import argparse
import csv
import math
from pathlib import Path


def dominates(a: dict, b: dict) -> bool:
    # Maximize: clip_img, clip_dir, vqascore_edit_gain
    # Minimize: fid
    better_or_equal = (
        a["clip_img"] >= b["clip_img"]
        and a["clip_dir"] >= b["clip_dir"]
        and a["vqascore_edit_gain"] >= b["vqascore_edit_gain"]
        and a["fid"] <= b["fid"]
    )
    strictly_better = (
        a["clip_img"] > b["clip_img"]
        or a["clip_dir"] > b["clip_dir"]
        or a["vqascore_edit_gain"] > b["vqascore_edit_gain"]
        or a["fid"] < b["fid"]
    )
    return better_or_equal and strictly_better


def norm(v: float, lo: float, hi: float) -> float:
    return 0.0 if hi <= lo else (v - lo) / (hi - lo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(
            "/leonardo_scratch/fast/IscrC_NNID/simone/steering_fields/results/pie_bench_pp_gridsearch/change/metrics_summary.csv"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "/leonardo_scratch/fast/IscrC_NNID/simone/steering_fields/results/pie_bench_pp_gridsearch/change/metrics_summary_pareto.csv"
        ),
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Keep rows with clip_score_delta > min_delta",
    )
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    rows = []
    dropped_non_finite = 0

    with args.input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = {
                    "subfolder": row["subfolder"],
                    "clip_score_delta": float(row["clip_score_delta"]),
                    "clip_img": float(row["clip_img"]),
                    "clip_dir": float(row["clip_dir"]),
                    "fid": float(row["fid"]),
                    "vqascore_edit_gain": float(row["vqascore_edit_gain"]),
                }
            except Exception:
                continue

            if x["clip_score_delta"] <= args.min_delta:
                continue

            if any(
                not math.isfinite(x[k])
                for k in ["clip_score_delta", "clip_img", "clip_dir", "fid", "vqascore_edit_gain"]
            ):
                dropped_non_finite += 1
                continue

            rows.append(x)

    if not rows:
        raise ValueError("No valid rows after filtering. Check --input-csv and --min-delta.")

    pareto = []
    for i, a in enumerate(rows):
        is_dominated = False
        for j, b in enumerate(rows):
            if i != j and dominates(b, a):
                is_dominated = True
                break
        if not is_dominated:
            pareto.append(a)

    mins = {k: min(r[k] for r in pareto) for k in ["clip_img", "clip_dir", "fid", "vqascore_edit_gain"]}
    maxs = {k: max(r[k] for r in pareto) for k in ["clip_img", "clip_dir", "fid", "vqascore_edit_gain"]}

    for r in pareto:
        r["compromise_score"] = (
            norm(r["clip_img"], mins["clip_img"], maxs["clip_img"])
            + norm(r["clip_dir"], mins["clip_dir"], maxs["clip_dir"])
            + norm(r["vqascore_edit_gain"], mins["vqascore_edit_gain"], maxs["vqascore_edit_gain"])
            + (1.0 - norm(r["fid"], mins["fid"], maxs["fid"]))
        ) / 4.0

    pareto.sort(key=lambda r: r["compromise_score"], reverse=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "subfolder",
        "clip_score_delta",
        "clip_img",
        "clip_dir",
        "fid",
        "vqascore_edit_gain",
        "compromise_score",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pareto)

    print(f"input rows kept: {len(rows)}")
    print(f"non-finite dropped: {dropped_non_finite}")
    print(f"pareto size: {len(pareto)}")
    print(f"saved: {args.output_csv}")
    print("\nTop Pareto rows:")
    for i, r in enumerate(pareto[: args.top_k], 1):
        print(
            f"{i:2d}. {r['subfolder']} | score={r['compromise_score']:.4f} | "
            f"clip_img={r['clip_img']:.6f} clip_dir={r['clip_dir']:.6f} "
            f"fid={r['fid']:.3f} vqa_gain={r['vqascore_edit_gain']:.6f} "
            f"delta={r['clip_score_delta']:.6f}"
        )


if __name__ == "__main__":
    main()
