import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

#import os, sys, gc, glob, pickle, argparse
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple, Iterable, Any
import os, sys, argparse, subprocess, concurrent.futures, csv, glob, time, pickle
import re

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches as mpatches
from matplotlib.patches import Polygon as MplPolygon, Ellipse
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from shapely.ops import unary_union
from numpy.linalg import norm 

# ---------- config defaults (override via CLI) ----------
DEFAULT_ROOT = "/media/kristen/easystore2/RestorebotData/benchmarking_ex/matches_fused"
DEFAULT_OUT  = "/media/kristen/easystore2/RestorebotData/benchmarking_ex/global_top100.csv"
# Set these or pass via CLI:
DEFAULT_IDX  = "/path/to/nov_index.npz"
DEFAULT_MAY_RESULTS_BASE = "/path/to/May2022/1conmod"
_WORKER = {} 

# ---- your package utils ----
from seasonal_comparison.gps_utils import (
    precompute_timestamps,
    load_covariance_matrix,
    scale_covariance_to_degrees,
)
from seasonal_comparison.image_stitching_utils import (
    create_polygons,
    reorder_corners,
)
from seasonal_comparison.plot_figs import (
    fix_invalid_corners,
    find_largest_overlap_subset,
)
from seasonal_comparison.general_utils import find_closest_index
from image_similarity.image_similarity_utils import (
    set_axes_border,
)
from image_similarity.palette_generation import gen_distinct_palette, color_for_rank
from image_similarity.shape_retrieval_helpers import (
    get_top_matches,
    get_top_match_metrics,
    get_inside_match_metrics,
    get_ellipse_extrema,
    add_scalebar_1m_bottom_left,
    get_inside_match_metrics_multi,
    get_top_matches_multi,
    export_csv_for_may,
    load_pano_pickle_for_image,
    parse_cluster_number,
    _maybe_swap_easystore, 
    RunArgs,
    CovarPaths,
    Paths 
)

from image_similarity.embedding_helpers import ensure_embed_sidecars 

from image_similarity.enforce_viewpoint_uniqueness import (
    diverse_rerank,
    rebuild_rank_and_metrics, 
    renumber_selected
)

from image_similarity.monitor import (
    ResourceMonitor,
    ProgressETAThreaded, 
    _kill_child_processes
) 

PALETTE = gen_distinct_palette(n_colors=32)  # stable color pool

# ------------------- HELPERS -------------------

# import the child module + helper types
import batchable_image_comparison as bic
from image_similarity.shape_retrieval_helpers import RunArgs, CovarPaths, Paths
from image_similarity.embedding_helpers import ensure_embed_sidecars

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

# --- worker globals (live inside each process) ---


def _init_worker(idx_npz: str, may_results_base: str, weights: dict,
                 use_clip: bool, use_dino: bool, use_sift: bool, topk: int):
    """
    Runs ONCE per worker process. Prepares objects that will be reused by all tasks handled by this worker.
    """
    # build cov paths once
    cov_dir = os.path.join(may_results_base, "processed_results", "covariance_matrices")
    results_dir = os.path.join(may_results_base, "processed_results")
    covp = CovarPaths(cov_dir=cov_dir, results_dir=results_dir)

    # Memory-map the index to avoid copying per worker
    data = np.load(idx_npz, allow_pickle=True, mmap_mode='r')
    paths = data["paths"]
    feats = data["feats"]  # HOG features, shape [N, D] ?
   
    # Precompute HOG norms once for fast cosine
    _WORKER["paths"] = paths
    _WORKER["feats"] = feats
    _WORKER["feats_norm"] = np.maximum(norm(feats, axis=1), 1e-8)

    # Build path -> row index once
    path_to_row = {str(p): i for i, p in enumerate(paths)}
    _WORKER["path_to_row"] = path_to_row

    # Load CLIP/DINO sidecars once (no ensure_embed_sidecars here; do that in parent)
    try:
        clip_mat, dino_mat = load_embed_sidecars(idx_npz)
        # Unit-normalize rows once so cosine is just a dot
        clip_norms = np.maximum(norm(clip_mat, axis=1), 1e-8)
        dino_norms = np.maximum(norm(dino_mat, axis=1), 1e-8)
        clip_mat = clip_mat / clip_norms[:, None]
        dino_mat = dino_mat / dino_norms[:, None]
        _WORKER["clip_mat"] = clip_mat
        _WORKER["dino_mat"] = dino_mat
    except Exception:
        _WORKER["clip_mat"] = None
        _WORKER["dino_mat"] = None

    # store in global dict
    _WORKER["idx_npz"] = idx_npz
    _WORKER["covp"] = covp
    _WORKER["weights"] = weights
    _WORKER["flags"] = dict(use_clip=use_clip, use_dino=use_dino, use_sift=use_sift)
    _WORKER["topk"] = topk

    # *** critical: warm the per-process LRU cache ***
    # The first call in this worker will populate bic.load_index(...) (LRU) and keep it for all tasks.
    # We call it here so the cost is paid once per worker:
    try:
        bic.load_index(idx_npz)  # uses @lru_cache in the module
        # If the NPZ is big, mmap helps reduce RSS (optional):
        # paths_arr, feats_arr, idx_blob = np.load(idx_npz, allow_pickle=True, mmap_mode='r')  # if you change load_index
    except Exception as e:
        print(f"[WORKER-INIT] Failed to warm index cache: {e}", file=sys.stderr, flush=True)

def _process_one(may_img: str) -> Tuple[str, int]:
    """
    Runs inside worker. Reuses _WORKER state and calls the module function directly.
    """
    try:
        # infer Nov dir + ts
        nov_dir = bic.infer_nov_dir(may_img)
        if not os.path.isdir(nov_dir):
            print(f"[WARN] Inferred Nov dir doesn't exist: {nov_dir}", file=sys.stderr, flush=True)
            return may_img, 2

        ts_ns = bic.infer_timestamp_ns(may_img)

        rargs = RunArgs(
            topk_show=_WORKER["topk"],
            timestamp_ns=ts_ns,
            weights=_WORKER["weights"],
            use_clip=_WORKER["flags"]["use_clip"],
            use_dino=_WORKER["flags"]["use_dino"],
            use_sift=_WORKER["flags"]["use_sift"],
        )
        out_dir = os.path.dirname(may_img)
        pth = Paths(idx_npz=_WORKER["idx_npz"], may_img=may_img, nov_dir=nov_dir, out_dir=out_dir)

        # runs the heavy routine; inside, bic.load_index(...) is cached per worker
        print(f"[RUN] {may_img}", flush=True)
        bic.process_may_frame(pth, _WORKER["covp"], rargs)
        return may_img, 0
    except Exception as e:
        print(f"[ERR] Failed on {may_img}: {e}", file=sys.stderr, flush=True)
        return may_img, 1

def main():
    """
    --root /media/kristen/easystore2/RestorebotData/benchmarking_ex/matches_fused --idx /media/kristen/easystore2/RestorebotData/analysis_shape/nov_index.npz --may-results-base /media/kristen/easystore2/RestorebotData/benchmarking_ex/May2022/1conmod --out /
edia/kristen/easystore2/RestorebotData/benchmarking_ex/top100.csv
    """
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
    ap.add_argument("--monitor", action="store_true", help="Log CPU/memory/thread usage periodically.")
    ap.add_argument("--monitor-interval", type=float, default=5.0, help="Seconds between resource logs.")

    args = ap.parse_args()

    monitor = None
    if args.monitor:
        monitor = ResourceMonitor(interval_sec=args.monitor_interval)
        monitor.start()

    # 1) Make sidecars ONCE (parent)
    if args.use_clip or args.use_dino:
        print("[INFO] Ensuring embed sidecars once in parent...", flush=True)
        try:
            ensure_embed_sidecars(args.idx)
        except Exception as e:
            print(f"[WARN] ensure_embed_sidecars skipped/failed: {e}", file=sys.stderr, flush=True)

    # 2) Find May images (your filtered finder)
    may_list = find_may_images(args.root)
    if not may_list:
        print(f"[ERR] No May images found under {args.root}", file=sys.stderr)
        sys.exit(2)

    tracker = ProgressETAThreaded(total=len(may_list), interval=120.0, label="ETA")
    tracker.start()

    print(f"[INFO] Found {len(may_list)} May images. Processing with {args.workers} workers...", flush=True)

    # 3) Build weights dict once (same as before)
    weights = dict(
        clip=args.w_clip, dino=args.w_dino, hog=args.w_hog,
        sift=args.w_sift, ssim=args.w_ssim, chamfer=args.w_chamfer
    )

    start = time.time()
    results: List[Tuple[str, int]] = []

    # 4) ProcessPool with initializer to preload per-worker caches
    # with ProcessPoolExecutor(max_workers=args.workers,
    #                          initializer=_init_worker,
    #                          initargs=(args.idx, args.may_results_base, weights,
    #                                    args.use_clip, args.use_dino, args.use_sift, args.topk)) as ex:
    #     futs = [ex.submit(_process_one, m) for m in may_list]
    #     for fut in as_completed(futs):
    #         may_img, rc = fut.result()
    #         results.append((may_img, rc))
    #         status = "OK" if rc == 0 else f"FAIL({rc})"
    #         print(f"[DONE] {status} :: {may_img}", flush=True)

    ex = ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.idx, args.may_results_base, weights,
                  args.use_clip, args.use_dino, args.use_sift, args.topk)
    )

    first_done_logged = False
    futs = []
    try:
        futs = [ex.submit(_process_one, m) for m in may_list]

        for fut in as_completed(futs):
            try:
                may_img, rc = fut.result()  # will raise if worker crashed
            except Exception as e:
                # Treat worker exceptions as FAIL for that item; abort.
                print(f"[ERR] Future raised exception: {e}", file=sys.stderr, flush=True)
                # hard abort path
                for f in futs:
                    f.cancel()
                try:
                    ex.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                _kill_child_processes(grace_sec=2.0)
                raise  # bubble up to outer except

            results.append((may_img, rc))
            status = "OK" if rc == 0 else f"FAIL({rc})"
            print(f"[DONE] {status} :: {may_img}", flush=True)

            if 'tracker' in locals() and tracker:
                tracker.hit()

            # sanity: confirm we're actually getting completions
            if not first_done_logged:
                print("[DEBUG] First completion recorded; ETA tracker should now reflect progress.", flush=True)
                first_done_logged = True

            if rc != 0:
                err_msg = f"[ABORT] Processing failed for {may_img} with rc={rc}"
                print(err_msg, file=sys.stderr, flush=True)

                # cancel outstanding tasks
                for f in futs:
                    f.cancel()
                try:
                    ex.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                _kill_child_processes(grace_sec=2.0)
                raise RuntimeError(err_msg)

    except KeyboardInterrupt:
        # Ctrl+C pressed: cancel and bail quickly
        print("\n[INTERRUPT] Ctrl+C received — cancelling futures and shutting down pool...", flush=True)

        # Cancel anything not started / still pending
        for f in futs:
            f.cancel()

        # Ask the executor to stop scheduling and not wait for tasks
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

        # As an extra safety net, kill any workers still hanging
        _kill_child_processes(grace_sec=2.0)

        # Optional: return a non-zero/CTRL-C exit code
        sys.exit(130)

    except Exception as e:
        print(f"[ERR] Unhandled exception in pool: {e}", file=sys.stderr, flush=True)
        # Ensure we tear things down on unexpected errors, too
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        _kill_child_processes(grace_sec=2.0)
        raise

    finally:
        if 'tracker' in locals() and tracker:
            tracker.stop(force_summary=True)

        # Normal shutdown path (if not already shut down)
        try:
            ex.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass 

    print(f"[INFO] All runs finished in {time.time()-start:.1f}s. Aggregating metrics...", flush=True)

    if monitor:
        monitor.stop()

    # 5) Aggregate metrics (same as your code)
    all_rows: List[Dict[str, Any]] = []
    for may_img, rc in results:
        if rc == 0:
            all_rows.extend(read_metrics_csv(may_img))

    if not all_rows:
        print("[ERR] No metrics to aggregate. Exiting.", file=sys.stderr)
        sys.exit(3)

    fieldnames = list(all_rows[0].keys())
    if "final_weighted" not in fieldnames:
        print("[ERR] 'final_weighted' column not found in metrics CSVs.", file=sys.stderr)
        fieldnames = sorted(set().union(*[row.keys() for row in all_rows]))

    all_rows.sort(key=lambda r: parse_float_safe(r.get("final_weighted", "nan")), reverse=True)
    topN = all_rows[:100]
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(topN)
    print(f"[OK] Wrote global top-100 to: {args.out}", flush=True)
        
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # safe across platforms
    main()
