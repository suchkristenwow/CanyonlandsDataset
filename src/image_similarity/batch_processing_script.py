#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch driver for Canyonlands matching.

- Walks /media/kristen/easystore2/RestorebotData/benchmarking_ex/matches_fused
- Finds every May fused PNG
- Runs run_matches.py on each (4 in parallel)
- Aggregates all per-May metrics_all.csv files
- Writes a global top-100 by `final_weighted`

Usage (example):
  python batch_run_matches.py \
    --idx /media/kristen/easystore2/RestorebotData/analysis_shape/nov_index.npz \
    --may-results-base /media/kristen/easystore2/RestorebotData/benchmarking_ex/May2022/1conmod \
    --root /media/kristen/easystore2/RestorebotData/benchmarking_ex/matches_fused \
    --workers 4 \
    --use-clip --use-dino --use-sift \
    --out /media/kristen/easystore2/RestorebotData/benchmarking_ex/global_top100.csv
"""

import os, sys, argparse, subprocess, concurrent.futures, csv, glob, time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re

# ---------- config defaults (override via CLI) ----------
DEFAULT_ROOT = "/media/kristen/easystore2/RestorebotData/benchmarking_ex/matches_fused"
DEFAULT_OUT  = "/media/kristen/easystore2/RestorebotData/benchmarking_ex/global_top100.csv"
# Set these or pass via CLI:
DEFAULT_IDX  = "/path/to/nov_index.npz"
DEFAULT_MAY_RESULTS_BASE = "/path/to/May2022/1conmod"


def run_one(may_img: str, args) -> Tuple[str, int]:
    import shutil
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Prefer stdbuf if available (Linux); otherwise run directly
    use_stdbuf = shutil.which("stdbuf") is not None
    base = [sys.executable, "-u", "batchable_image_comparison.py"]
    if use_stdbuf:
        # -oL/-eL = line-buffer stdout/stderr from C stdio as well
        base = ["stdbuf", "-oL", "-eL"] + base

    cmd = base + [
        "--idx", args.idx,
        "--may", may_img,
        "--may-results-base", args.may_results_base,
        "--topk", str(args.topk),
    ]

    if args.use_clip: cmd.append("--use-clip")
    if args.use_dino: cmd.append("--use-dino")
    if args.use_sift: cmd.append("--use-sift")

    def add_w(flag, val): cmd.extend([flag, str(val)])
    add_w("--w-clip", args.w_clip)
    add_w("--w-dino", args.w_dino)
    add_w("--w-hog",  args.w_hog)
    add_w("--w-sift", args.w_sift)
    add_w("--w-ssim", args.w_ssim)
    add_w("--w-chamfer", args.w_chamfer)

    print(f"[RUN] {may_img}", flush=True)
    try:
        proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=False, env=env)
        return may_img, proc.returncode
    except Exception as e:
        print(f"[ERR] Subprocess failed for {may_img}: {e}", file=sys.stderr, flush=True)
        return may_img, -1

def read_metrics_csv(may_img: str) -> List[Dict[str, Any]]:
    """
    Load the per-May metrics CSV written by run_matches.py: <MayDir>/metrics_all.csv
    Returns list of dict rows; adds 'may_img' column.
    """
    may_dir = os.path.dirname(may_img)
    csv_path = os.path.join(may_dir, "metrics_all.csv")
    if not os.path.isfile(csv_path):
        # The run may have failed before CSV export
        print(f"[WARN] Missing metrics CSV: {csv_path}")
        return []

    rows = []
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = dict(row)
                row["may_img"] = may_img
                rows.append(row)
        return rows
    except Exception as e:
        print(f"[WARN] Failed reading {csv_path}: {e}")
        return []

def parse_float_safe(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default

MAY_FUSED_RE = re.compile(r"^may_cluster\d{4}_part\d{2}\.png$", re.IGNORECASE)

def find_may_images(root):
    """
    Return only May fused frames like may_cluster0000_part02.png
    under .../**/May/*.png (skips gps_plot.png, top grids, etc).
    """
    pattern = os.path.join(root, "**", "May", "*.png")
    all_pngs = sorted(glob.glob(pattern, recursive=True))

    out, skipped = [], 0
    for p in all_pngs:
        name = os.path.basename(p)
        if MAY_FUSED_RE.match(name):
            out.append(p)
        else:
            #print("[WARN] skipping name: ",name)
            skipped += 1
    print(f"[INFO] Found {len(out)} valid May frames; skipped {skipped} non-matching PNGs.")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT, help="Root dir containing .../ID/{May,Nov}")
    ap.add_argument("--idx", default=DEFAULT_IDX, help="Path to nov_index.npz")
    ap.add_argument("--may-results-base", default=DEFAULT_MAY_RESULTS_BASE,
                    help="Base dir containing processed_results/{covariance_matrices,frustrum_corners.csv}")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output CSV path for global top-100")
    ap.add_argument("--workers", type=int, default=4, help="Parallel May images to process")
    ap.add_argument("--topk", type=int, default=4, help="--topk passed to run_matches.py (viz only)")
    # component toggles
    ap.add_argument("--use-clip", action="store_true")
    ap.add_argument("--use-dino", action="store_true")
    ap.add_argument("--use-sift", action="store_true")
    # weights (keep consistent with run_matches.py defaults)
    ap.add_argument("--w-clip", type=float, default=0.22)
    ap.add_argument("--w-dino", type=float, default=0.22)
    ap.add_argument("--w-hog",  type=float, default=0.12)
    ap.add_argument("--w-sift", type=float, default=0.20)
    ap.add_argument("--w-ssim", type=float, default=0.18)
    ap.add_argument("--w-chamfer", type=float, default=0.14)

    args = ap.parse_args()

    # Discover May images
    may_list = find_may_images(args.root)
    if not may_list:
        print(f"[ERR] No May images found under {args.root}", file=sys.stderr)
        sys.exit(2)

    #filter filenames 

    print(f"[INFO] Found {len(may_list)} May images. Processing with {args.workers} workers...")

    # Run in parallel (up to N workers)
    start = time.time()
    results: List[Tuple[str,int]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        try:
            futs = [ex.submit(run_one, m, args) for m in may_list]
            for fut in concurrent.futures.as_completed(futs):
                may_img, rc = fut.result()
                results.append((may_img, rc))
                status = "OK" if rc == 0 else f"FAIL({rc})"
                print(f"[DONE] {status} :: {may_img}")

        finally:
            ex.shutdown(wait=True, cancel_futures=True)

    # with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
    #     futs = [ex.submit(run_one, m, args) for m in may_list]
    #     for fut in concurrent.futures.as_completed(futs):
    #         may_img, rc = fut.result()
    #         results.append((may_img, rc))
    #         status = "OK" if rc == 0 else f"FAIL({rc})"
    #         print(f"[DONE] {status} :: {may_img}")

    print(f"[INFO] All runs finished in {time.time()-start:.1f}s. Aggregating metrics...")

    # Aggregate all metrics
    all_rows: List[Dict[str, Any]] = []
    for may_img, rc in results:
        if rc == 0:
            all_rows.extend(read_metrics_csv(may_img))

    if not all_rows:
        print("[ERR] No metrics to aggregate. Exiting.", file=sys.stderr)
        sys.exit(3)

    # Sort globally by final_weighted desc and keep top-100
    # Preserve columns as they appear; ensure may_img included.
    fieldnames = list(all_rows[0].keys())
    if "final_weighted" not in fieldnames:
        print("[ERR] 'final_weighted' column not found in metrics CSVs.", file=sys.stderr)
        # still write a combined file for debugging
        fieldnames = sorted(set().union(*[row.keys() for row in all_rows]))
    # sort
    all_rows.sort(key=lambda r: parse_float_safe(r.get("final_weighted", "nan")), reverse=True)
    topN = all_rows[:100]

    # Ensure output dir exists
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)

    # Write CSV
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(topN)

    print(f"[OK] Wrote global top-100 to: {args.out}")

if __name__ == "__main__":
    main()
