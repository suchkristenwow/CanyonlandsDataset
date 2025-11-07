#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, gc, glob, pickle, argparse
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple, Iterable

import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches as mpatches
from matplotlib.patches import Polygon as MplPolygon, Ellipse
from shapely.ops import unary_union
import time 

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

PALETTE = gen_distinct_palette(n_colors=32)  # stable color pool

# ------------------- HELPERS -------------------
# image_similarity/shape_retrieval_helpers.py
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

def infer_nov_dir(may_img: str) -> str:
    # Example: .../1650825680196263936/May/may_cluster0000_part02.png
    # → .../1650825680196263936/Nov
    d = os.path.dirname(may_img)      # .../May/
    parent = os.path.dirname(d)       # .../1650825680196263936
    return os.path.join(parent, "Nov")

def infer_timestamp_ns(may_img: str) -> int:
    parts = [p for p in Path(may_img).parts if p.isdigit()]
    if not parts:
        raise ValueError(f"No timestamp found in {may_img}")
    return int(max(parts, key=len))

@lru_cache(maxsize=1)
def load_index(idx_npz: str):
    data = np.load(idx_npz, allow_pickle=True)
    return data["paths"], data["feats"], data  # keep data to pass into inside metrics

def normalized_key(img_path: str, pano_dict: Dict) -> str:
    """Return a key present in pano_dict, trying easystore flip if needed."""
    if img_path in pano_dict:
        return img_path
    alt = _maybe_swap_easystore(img_path)
    if alt in pano_dict:
        return alt
    raise FileNotFoundError(f"Key not found in pano: {img_path} or {alt}")

def polygons_from_pano(pano_dict: Dict, key: str):
    polys = create_polygons(pano_dict[key])
    return find_largest_overlap_subset(polys)

def compute_covariance_and_center(covp: CovarPaths, timestamp_ns: int) -> Tuple[np.ndarray, Tuple[float, float], np.ndarray]:
    """
    Returns: (scaled_cov_2x2, center(lon,lat), raw_cov_2x2)
    """
    may_cov_timestamps = precompute_timestamps(covp.cov_dir)

    may_data = np.genfromtxt(os.path.join(covp.results_dir, "frustrum_corners.csv"),
                             delimiter=",", skip_header=1)
    may_timestamps = may_data[:, 0]
    cam_lat, cam_lon = may_data[:, 15], may_data[:, 16]

    cov_matrix = load_covariance_matrix(covp.cov_dir, may_cov_timestamps, timestamp_ns)
    scaled_cov = scale_covariance_to_degrees(cov_matrix)

    i_may = find_closest_index(may_timestamps, timestamp_ns * 1e-9, max_delta_t=0.3)
    center = (float(cam_lon[i_may]), float(cam_lat[i_may]))  # (lon, lat)
    return scaled_cov, center, cov_matrix

def ellipse_from_cov(scaled_cov: np.ndarray, center: Tuple[float, float]) -> Ellipse:
    vals, vecs = np.linalg.eigh(scaled_cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(vals)  # 1-sigma
    return Ellipse(xy=center, width=width, height=height, angle=theta,
                   edgecolor='blue', facecolor='none', linestyle='dotted', label="Covariance Ellipse")

def save_top_grid(grid_title_png: str, numbered_rows: List[Tuple[int, str, float, float, float, float]]):
    if not numbered_rows:
        return
    fig = plt.figure(figsize=(10, 8), dpi=150)
    R, C = 2, 2
    gs = gridspec.GridSpec(R, C, wspace=0.06, hspace=0.12)

    for idx, (rank, nov_path, coarse, gssim_val, ch_val, final) in enumerate(numbered_rows[:4]):
        r, c = divmod(idx, C)
        axr = fig.add_subplot(gs[r, c])
        nimg = cv.imread(nov_path, cv.IMREAD_COLOR)
        if nimg is None:
            axr.text(0.5, 0.5, f"Missing:\n{os.path.basename(nov_path)}",
                     ha="center", va="center", fontsize=8)
            axr.axis("off")
            continue
        nimg = cv.cvtColor(nimg, cv.COLOR_BGR2RGB)
        axr.imshow(nimg); axr.axis("off")
        axr.set_title(f"{rank}.  final={final:.3f}  SSIM={gssim_val:.3f}", fontsize=9)
        set_axes_border(axr, color_for_rank(rank, PALETTE), lw=2)

    #plt.tight_layout()
    fig.savefig(grid_title_png, dpi=220, pil_kwargs={"dpi": (220, 220)}) 
    plt.close(fig)

def draw_polygons(ax, shp_polys, edge_color, face_color=None, alpha=0.25, lw=1.0):
    for poly in shp_polys:
        coords = list(poly.exterior.coords)[:-1]
        oc = fix_invalid_corners(reorder_corners(coords))
        ax.add_patch(MplPolygon(oc, fill=True,
                                facecolor=face_color if face_color is not None else edge_color,
                                alpha=alpha, edgecolor=edge_color, linewidth=lw))

def autoscale_from_patches(ax):
    # Ensure patches extend the data limits before autoscale
    ax.relim()
    for p in ax.patches:
        path = p.get_path().transformed(p.get_transform())
        ax.dataLim.update_from_data_xy(path.vertices, ignore=False)
    ax.autoscale_view()

# ------------------- CORE WORKFLOW -------------------
def process_may_frame(paths: Paths, covp: CovarPaths, args: RunArgs):
    #print("processing may frame. This is covp: ",covp) 

    may_dir = os.path.dirname(paths.may_img)
    out_dir = paths.out_dir or may_dir
    os.makedirs(out_dir, exist_ok=True)

    gps_plot_png = os.path.join(out_dir, "gps_plot.png")
    top4_png = os.path.join(out_dir, "top4_overall_2x2.png")
    ellipse_top4_png = os.path.join(out_dir, "top4_inside_ellipse_2x2.png")

    # ---- index + matches
    print(f"[INFO] Loading index: {paths.idx_npz}")
    start_time = time.perf_counter()
    paths_arr, feats_arr, idx_blob = load_index(paths.idx_npz)
    end_time = time.perf_counter() 
    #print(f"Loading index took: {end_time-start_time} s")


    #print(f"[INFO] Finding top-{args.topk_show} overall matches for:\n  {paths.may_img}")
    _WORKER = {}
    # build cov paths once
    # cov_dir = os.path.join(may_dir, "processed_results", "covariance_matrices")
    # results_dir = os.path.join(may_dir, "processed_results")
    # covp = CovarPaths(cov_dir=cov_dir, results_dir=results_dir)

    # Memory-map the index to avoid copying per worker
    data = np.load(paths.idx_npz, allow_pickle=True, mmap_mode='r')

    data_paths = data["paths"]
    feats = data["feats"]  # HOG features, shape [N, D] ?

    # Precompute HOG norms once for fast cosine
    _WORKER["paths"] = data_paths
    _WORKER["feats"] = feats
    _WORKER["feats_norm"] = np.maximum(np.linalg.norm(feats, axis=1), 1e-8)

    # Build path -> row index once
    path_to_row = {str(p): i for i, p in enumerate(data_paths)}
    _WORKER["path_to_row"] = path_to_row

    # Load CLIP/DINO sidecars once (no ensure_embed_sidecars here; do that in parent)
    try:
        clip_mat, dino_mat = load_embed_sidecars(paths.idx_npz) #idx_npz 
        # Unit-normalize rows once so cosine is just a dot
        clip_norms = np.maximum(np.linalg.norm(clip_mat, axis=1), 1e-8)
        dino_norms = np.maximum(np.linalg.norm(dino_mat, axis=1), 1e-8)
        clip_mat = clip_mat / clip_norms[:, None]
        dino_mat = dino_mat / dino_norms[:, None]
        _WORKER["clip_mat"] = clip_mat
        _WORKER["dino_mat"] = dino_mat
    except Exception:
        _WORKER["clip_mat"] = None
        _WORKER["dino_mat"] = None

    # store in global dict
    _WORKER["idx_npz"] = paths.idx_npz
    _WORKER["covp"] = covp
    _WORKER["weights"] = args.weights
    _WORKER["flags"] = dict(use_clip=args.use_clip, use_dino=args.use_dino, use_sift=args.use_sift)
    _WORKER["topk"] = args.topk_show 

    print("Finding top matches ...")
    start_time = time.perf_counter()
    top_rows = get_top_matches_multi(
        paths.may_img, 48, _WORKER, weights=args.weights
    )
    end_time = time.perf_counter() 
    print(f"Finding top matches took: {end_time-start_time:.2f} s")

    start_time = time.perf_counter()
    top_numbered_all, rank_top_all, metrics_top_all = get_top_match_metrics(top_rows)
    end_time = time.perf_counter() 
    print(f"Finding top match metrics took: {end_time-start_time:.2f} s")
    
    start_time = time.perf_counter()
    top_numbered_sel = diverse_rerank(
        top_numbered_all,
        k=args.topk_show,
        paths_arr=paths_arr,
        feats_arr=feats_arr,
        max_per_cluster=1,         # change to 2 if you want some repeats per viewpoint
        hash_hamming_thresh=6,
        max_cosine_sim=0.985
    ) 

    top_numbered = renumber_selected(top_numbered_sel)
    rank_top, metrics_top = rebuild_rank_and_metrics(top_numbered, metrics_top_all)
    end_time = time.perf_counter() 

    print(f"rebuilding/ranking metrics took: {end_time-start_time:.2f} s") 

    # ---- May polygons[]
    print("[INFO] Loading May pano and polygons...")

    start_time = time.perf_counter() 
    may_pano = load_pano_pickle_for_image(paths.may_img, which="may")
    try:
        may_key = normalized_key(paths.may_img, may_pano)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"May pano entry not found: {e}")

    may_polys_all = create_polygons(may_pano[may_key])
    may_overlap_subset = find_largest_overlap_subset(may_polys_all)
    if not may_overlap_subset:
        raise RuntimeError("Empty/invalid May polygons.")
    end_time = time.perf_counter() 
    print(f"Loading may/nov polygons took {np.round(end_time-start_time,2)} s")

    # ---- Covariance + center
    print("[INFO] Computing covariance ellipse & center...")
    scaled_cov, center, _ = compute_covariance_and_center(covp, args.timestamp_ns)
    cov_ellipse = ellipse_from_cov(scaled_cov, center)
    ellipse_min_x, ellipse_max_x, ellipse_min_y, ellipse_max_y = get_ellipse_extrema(cov_ellipse)

    # ---- Gather inside-ellipse Nov frames
    nov_inside_frames = sorted(
        os.path.join(paths.nov_dir, x) for x in os.listdir(paths.nov_dir) if x.lower().endswith(".png")
    )
    
    start_time = time.perf_counter()
    inside_rows, inside_numbered, rank_inside, metrics_inside = get_inside_match_metrics_multi(
        idx_blob, paths.may_img, nov_inside_frames,_WORKER, weights=args.weights
    )
    end_time = time.perf_counter()
    print(f"Getting inside match metrics took: {end_time - start_time}") 

    inside_top_set = {nv_path for (rank, nv_path, *_rest) in inside_numbered}

    scaled_cov, center, _ = compute_covariance_and_center(covp, args.timestamp_ns)
    cov_ellipse = ellipse_from_cov(scaled_cov, center)

    # export CSV
    top_match_paths = [x[1] for x in top_numbered] 

    metrics_dir = os.path.join(out_dir,"metrics") 
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir, exist_ok=True) 

    csv_path = os.path.join(metrics_dir,f"{os.path.basename(paths.may_img)[:-4]}.csv") 
    export_csv_for_may(paths, top_match_paths, args, idx_blob, cov_ellipse, center, csv_path)
    print(f"batchable image comparison: [OK] Wrote CSV: {csv_path}")

    # ------------------- GPS plot -------------------
    print("[INFO] Rendering GPS-only plot ...")
    start_time = time.perf_counter()  

    fig_gps = plt.figure(figsize=(10.5, 6), dpi=150, constrained_layout=False)
    gs = fig_gps.add_gridspec(1, 2, width_ratios=[1.0, 0.45], wspace=0.03)
    ax_gps = fig_gps.add_subplot(gs[0, 0])
    ax_leg = fig_gps.add_subplot(gs[0, 1])
    ax_leg.axis("off")

    legend_handles = []

    # May (reference) polygons in blue
    draw_polygons(ax_gps, may_overlap_subset, edge_color="blue", face_color="none", alpha=1.0, lw=1.0)
    legend_handles.append(mpatches.Patch(facecolor='none', edgecolor="blue",
                                         label=os.path.basename(paths.may_img)))

    # Covariance ellipse
    ax_gps.add_patch(cov_ellipse)
    legend_handles.append(cov_ellipse)

    # Draw polygons from INSIDE ellipse candidates
    path_color: Dict[str, Tuple[float, float, float]] = {}
    inside_polygons = []
    master_polygon_list = []

    for fused_nov_img in nov_inside_frames:
        # Load pano dict for this Nov image, normalize key, get top-overlap subset
        nov_pano = load_pano_pickle_for_image(fused_nov_img, which="nov")
        try:
            nov_key = normalized_key(fused_nov_img, nov_pano)
        except FileNotFoundError:
            # skip silently
            continue

        top_subset = find_largest_overlap_subset(create_polygons(nov_pano[nov_key]))
        if not top_subset:
            continue

        c = PALETTE[abs(hash(nov_key)) % len(PALETTE)]
        path_color[nov_key] = c
        draw_polygons(ax_gps, top_subset, edge_color=c, face_color=c, alpha=0.25, lw=0.8)
        inside_polygons.extend(top_subset)
        master_polygon_list.extend(top_subset)

    # Legend entries for ranked inside frames
    for nv_path in inside_top_set:
        norm = nv_path if nv_path in path_color else _maybe_swap_easystore(nv_path)
        c = path_color.get(norm, PALETTE[abs(hash(norm)) % len(PALETTE)])
        fin_val, ssim_val = metrics_inside[nv_path]
        legend_handles.append(
            mpatches.Patch(facecolor=c, edgecolor=c, alpha=0.25,
                           label=f"Inside rank {rank_inside[nv_path]}  final={fin_val:.3f}  SSIM={ssim_val:.3f}")
        )

    # Draw BEST overall matches (top-N)
    ccount = 0
    for rank, nov_path, coarse, gssim_val, ch_val, final in top_numbered:
        c = PALETTE[ccount % len(PALETTE)]; ccount += 1
        nov_pano = load_pano_pickle_for_image(nov_path, which="nov")
        nov_key = nov_path if nov_path in nov_pano else _maybe_swap_easystore(nov_path)
        top_subset = find_largest_overlap_subset(create_polygons(nov_pano[nov_key]))
        draw_polygons(ax_gps, top_subset, edge_color=c, face_color=c, alpha=0.25, lw=1.0)
        master_polygon_list.extend(top_subset)
        legend_handles.append(
            mpatches.Patch(facecolor=c, edgecolor=c, alpha=0.25,
                           label=f"{rank}.  final={metrics_top[nov_path][0]:.3f}  SSIM={metrics_top[nov_path][1]:.3f}")
        )

    # Bounds from everything drawn + covariance extents
    may_merged   = unary_union(may_polys_all) if may_polys_all else None
    inside_merged = unary_union(inside_polygons) if inside_polygons else None
    top_merged   = unary_union(master_polygon_list) if master_polygon_list else None

    if may_merged is None:
        raise RuntimeError("May polygon list unexpectedly empty.")

    min_lon, min_lat, max_lon, max_lat = may_merged.bounds
    for merged in (inside_merged, top_merged):
        if merged is not None:
            a, b, c, d = merged.bounds
            min_lon, min_lat = min(min_lon, a), min(min_lat, b)
            max_lon, max_lat = max(max_lon, c), max(max_lat, d)

    min_lon = min(min_lon, ellipse_min_x); max_lon = max(max_lon, ellipse_max_x)
    min_lat = min(min_lat, ellipse_min_y); max_lat = max(max_lat, ellipse_max_y)
    
    autoscale_from_patches(ax_gps)

    dx = max((max_lon - min_lon) * 0.10, 1e-6)
    dy = max((max_lat - min_lat) * 0.10, 1e-6)
    ax_gps.set_xlim(min_lon - dx, max_lon + dx)
    ax_gps.set_ylim(min_lat - dy, max_lat + dy)
    mid_lat = 0.5 * (min_lat + max_lat)
    ax_gps.set_aspect(np.cos(np.deg2rad(mid_lat)), adjustable='box')

    add_scalebar_1m_bottom_left(ax_gps)

    ax_leg.legend(handles=legend_handles, loc="center", frameon=False, fontsize=7, handlelength=2.5)
    #plt.savefig(gps_plot_png, dpi=220) #, dpi=220, pil_kwargs={"dpi": (220, 220)}
    fig_gps.savefig(gps_plot_png) 
    plt.close(fig_gps)

    end_time = time.perf_counter()
    print(f"GPS plot took: {end_time-start_time} s")
    print(f"[OK] Wrote GPS plot: {gps_plot_png}")

    # ------------------- Grids -------------------
    t0 = time.perf_counter()
    #print("[INFO] Saving top-4 overall grid ...")
    save_top_grid(top4_png, top_numbered)
    print(f"[OK] Wrote: {top4_png}")

    if inside_numbered:
        #print("[INFO] Saving top-4 inside-ellipse grid ...")
        save_top_grid(ellipse_top4_png, inside_numbered)
        print(f"[OK] Wrote: {ellipse_top4_png}")
    else:
        print("[WARN] No inside-ellipse candidates with successful similarity computation.")
    tf = time.perf_counter()
    print(f"Making grids took: {tf-t0}")

    gc.collect()

# ------------------- CLI -------------------
def expand_may_inputs(may_args: Iterable[str]) -> List[str]:
    out = []
    for m in may_args:
        if any(ch in m for ch in "*?[]"):
            out.extend(glob.glob(m, recursive=True))
        else:
            out.append(m)
    # stable order
    return sorted(list(dict.fromkeys(out)))

def main():
    p = argparse.ArgumentParser(description="Batchable Canyonlands matching & plotting")
    p.add_argument("--idx", required=True, help="Path to nov_index.npz")
    p.add_argument("--may", required=True, nargs="+", help="One or more May fused images (paths or globs)")
    p.add_argument("--topk", type=int, default=4, help="Top-K overall to show (default: 4)")
    p.add_argument("--out-dir", default=None, help="Output directory (default: May image directory)")
    # May covariance resources
    p.add_argument("--may-results-base", required=True) #/media/kristen/easystore2/RestorebotData/benchmarking_ex/May2022/1conmod 
    # p.add_argument("--may-cov-dir", required=True, help="Dir with May covariance_matrices")
    # p.add_argument("--may-results-dir", required=True, help="Dir with May frustrum_corners.csv")
    p.add_argument("--use-clip", action="store_true", help="Include CLIP embeddings if available")
    p.add_argument("--use-dino", action="store_true", help="Include DINO embeddings if available")
    p.add_argument("--use-sift", action="store_true", help="Include SIFT+RANSAC inlier ratio")
    # weights (defaults tuned for cross-season grass)
    p.add_argument("--w-clip", type=float, default=0.22)
    p.add_argument("--w-dino", type=float, default=0.22)
    p.add_argument("--w-hog",  type=float, default=0.12)
    p.add_argument("--w-sift", type=float, default=0.20)
    p.add_argument("--w-ssim", type=float, default=0.18)
    p.add_argument("--w-chamfer", type=float, default=0.14)

    args_ns = p.parse_args()

    may_list = expand_may_inputs(args_ns.may)
    if not may_list:
        print("[ERR] No May images found from provided inputs.", file=sys.stderr)
        sys.exit(2)

    may_cov_dir = os.path.join(args_ns.may_results_base, "processed_results/covariance_matrices") 
    may_results_dir = os.path.join(args_ns.may_results_base, "processed_results")
    covp = CovarPaths(cov_dir=may_cov_dir, results_dir=may_results_dir) 

    # load/embed once (sidecars are created only if missing & libs available)
    if args_ns.use_clip or args_ns.use_dino:
        try:
            ensure_embed_sidecars(args_ns.idx)
        except Exception as e:
            print(f"[WARN] Embedding sidecars not built ({e}). Will fall back if needed.", file=sys.stderr)

    weights = dict(
        clip=args_ns.w_clip, dino=args_ns.w_dino, hog=args_ns.w_hog,
        sift=args_ns.w_sift, ssim=args_ns.w_ssim, chamfer=args_ns.w_chamfer
    )

    #print(f"[batchable_image_comparison] there are {len(may_list)} may images")
    for may_img in may_list:
        start_time = time.perf_counter()
        
        nov_dir = infer_nov_dir(may_img)

        if not os.path.isdir(nov_dir):
            print(f"[WARN] Inferred Nov dir doesn't exist: {nov_dir}", file=sys.stderr)
            continue 

        ts_ns = infer_timestamp_ns(may_img)
        #rargs = RunArgs(topk_show=args_ns.topk, timestamp_ns=ts_ns) 
        rargs = RunArgs(
            topk_show=args_ns.topk,
            timestamp_ns=ts_ns,
            weights=weights,
            use_clip=args_ns.use_clip,
            use_dino=args_ns.use_dino,
            use_sift=args_ns.use_sift,
        )
        out_dir = args_ns.out_dir or os.path.dirname(may_img)
        pth = Paths(idx_npz=args_ns.idx, may_img=may_img, nov_dir=nov_dir, out_dir=out_dir)

        try:
            print("[INFO] processing may frame ...")
            process_may_frame(pth, covp, rargs)
            print("Woo! Successfully processed the frame!")

        except Exception as e:
            print(f"[ERR] Failed on {may_img}: {e}", file=sys.stderr)

        end_time = time.perf_counter()
        print(f"Done with 1 May image: That took {np.round(end_time - start_time, 2)} s")

if __name__ == "__main__":
    main()
