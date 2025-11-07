from matplotlib.patches import Polygon as MplPolygon, Ellipse
import numpy as np 
import os, numpy as np, cv2 as cv, pickle, gc
from pyproj import Geod 
from dataclasses import dataclass 
import csv 
from shapely.ops import unary_union 

from image_similarity.image_similarity_utils import (
    compute_similarity_metrics,
    load_gray_eq, 
    hog_vec,
    cos_sims,
    grad,
    ecc_align,
    ssim,
    chamfer,
    mahalanobis_inside,
    set_axes_border,
    sift_ransac_inlier_ratio
)

from image_similarity.embedding_helpers import (
    ensure_embed_sidecars,
    load_embed_sidecars, 
    embed_clip,
    embed_dino
) 

from seasonal_comparison.plot_figs import (
    find_largest_overlap_subset
)

from seasonal_comparison.image_stitching_utils import (
    create_polygons,
    reorder_corners,
)

from shapely.geometry import Point as ShapelyPoint
from shapely.affinity import scale as shp_scale, rotate as shp_rotate, translate as shp_translate

GEOD = Geod(ellps="WGS84")
_WORKER = {} 

@dataclass
class Paths:
    idx_npz: str
    may_img: str
    nov_dir: str  # directory containing Nov fused PNGs for this timestamp
    out_dir: str  # where to save plots (default: same as may_img dir)

@dataclass
class CovarPaths:
    cov_dir: str       # dir with covariance_matrices (May)
    results_dir: str   # dir with frustrum_corners.csv (May)

@dataclass
class RunArgs:
    topk_show: int
    timestamp_ns: int
    weights: dict
    use_clip: bool
    use_dino: bool
    use_sift: bool 

def safe_imread(path):
    img = cv.imread(path, cv.IMREAD_COLOR)
    if img is not None:
        return img, path
    # try easystore swap if read failed
    print("swapping easystore ...")
    alt = _maybe_swap_easystore(path)
    if alt != path:
        img2 = cv.imread(alt, cv.IMREAD_COLOR)
        if img2 is not None:
            return img2, alt
    return None, path

def safe_load_gray_eq(path):
    rgb, key_used = safe_imread(path)
    if rgb is None:
        return None
    g = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)
    g = load_gray_eq(path) if g is None else g  # keep your eq pipeline
    return g

def _maybe_swap_easystore(path: str) -> str:
    """Try easystore2/easystore3 flip if a key is missing in a pickle."""
    return path.replace("easystore2", "easystore3") if "easystore2" in path else path.replace("easystore3", "easystore2")

def parse_cluster_number(p: str, prefix: str) -> int:
    """
    Extract cluster number from filename like may_cluster0000_part02.png / nov_cluster0007_part01.png
    prefix: 'may' or 'nov'
    """
    fn = os.path.basename(p)
    i1 = [i for i, ch in enumerate(fn) if ch == "_"][1]
    i0 = fn.index("cluster") + 8
    return int(fn[i0:i1])

def load_pano_pickle_for_image(img_path: str, which: str = "may"): 
    """
    Load the pano dict that corresponds to a fused image.
    which in {'may','nov'} controls file prefix.
    """
    cluster = parse_cluster_number(img_path, which)
    d = os.path.dirname(img_path)
    pp = os.path.join(d, f"{which}_panos_{cluster}.pickle")
    with open(pp, "rb") as f:
        return pickle.load(f)

def get_inside_match_metrics(data,MAY,nov_inside_candidates):
    paths, feats = data["paths"], data["feats"]
    mg = load_gray_eq(MAY)
    assert mg is not None and mg.ndim == 2, f"bad image at {MAY}"
    q = hog_vec(mg)
    sims = cos_sims(q, feats)

    mgG = grad(mg)
    eM = cv.Canny(mgG, 50, 150)


    path_to_coarse = {str(paths[i]): float(sims[i]) for i in range(len(paths))}
    inside_rows = []  # (nov_path, coarse, gssim, ch, final)
    for npth in sorted(nov_inside_candidates):
        coarse_sim = path_to_coarse.get(str(npth), 0.0)
        ng = load_gray_eq(npth)
        if ng is None:
            continue
        
        ng_al = ecc_align(mg, ng, "affine")
        gssim = ssim(grad(mg), grad(ng_al))
        eN = cv.Canny(grad(ng_al), 50, 150)
        H, W = mg.shape[:2]
        ch = chamfer(eM, eN) / max(H, W)
        final = 0.7 * gssim - 0.3 * ch
        inside_rows.append((npth, float(coarse_sim), float(gssim), float(ch), float(final)))
    
    inside_top = sorted(inside_rows, key=lambda x: -x[4])
    inside_numbered = []    # (rank, nov_path, coarse, gssim, ch, final)
    rank_inside = {}
    metrics_inside = {}
    for offs, (nov_path, coarse, gssim, ch, final) in enumerate(inside_top):
        inside_numbered.append((offs, nov_path, coarse, gssim, ch, final))
        rank_inside[nov_path] = offs
        metrics_inside[nov_path] = (final, gssim)
    inside_top_set = set(rank_inside.keys())

    return inside_rows, inside_numbered, rank_inside, metrics_inside  
    
def get_top_match_metrics(top_rows):
    top_numbered = []          # (rank, nov_path, coarse, gssim, ch, final)
    rank_top = {}
    metrics_top = {}
    for rank, (nov_path, coarse, gssim, ch, final) in enumerate(top_rows):
        top_numbered.append((rank, nov_path, coarse, gssim, ch, final))
        rank_top[nov_path] = rank
        metrics_top[nov_path] = (final, gssim) 
    return top_numbered, rank_top, metrics_top 

def get_top_matches(IDX,MAY,TOPK):
    data = np.load(IDX, allow_pickle=True)
    paths, feats = data["paths"], data["feats"]

    mg = load_gray_eq(MAY)
    assert mg is not None and mg.ndim == 2, f"bad image at {MAY}"
    q = hog_vec(mg)

    sims = cos_sims(q, feats)
    top = np.argsort(-sims)[:TOPK]

    mgG = grad(mg)
    eM = cv.Canny(mgG, 50, 150)

    # Compute similarities for top-K retrieved (global best matches)
    print("computing ECC/SSIM/Chamfer for top candidates ...")
    rows = []  # (nov_path, coarse_sim, gssim, chamfer_norm, final)
    for i in top:
        npth = str(paths[i])
        ng = load_gray_eq(npth)
        if ng is None:
            continue
        #SIFT/RANSAC
        sift_ir = sift_ransac_inlier_ratio(mg, ng, model="affine") 

        ng_al = ecc_align(mg, ng, "affine")
        gssim = ssim(grad(mg), grad(ng_al))
        eN = cv.Canny(grad(ng_al), 50, 150)
        H, W = mg.shape[:2]
        ch = chamfer(eM, eN) / max(H, W)
        final = 0.7 * gssim - 0.3 * ch
        rows.append((npth, float(sims[i]), float(gssim), float(ch), float(final)))
        print(f"sims: {sims[i]:.5f}  SSIM: {gssim:.5f}  Chamfer: {ch:.5f}  final: {final:.5f}")

    return rows 

def get_ellipse_extrema(ellipse_patch):
        cx, cy = ellipse_patch.center
        width = ellipse_patch.width
        height = ellipse_patch.height
        angle_deg = ellipse_patch.angle

        a = width / 2
        b = height / 2
        angle_rad = np.deg2rad(angle_deg)

        # Generate many points on the ellipse
        t = np.linspace(0, 2 * np.pi, 1000)
        x_coords = cx + a * np.cos(t) * np.cos(angle_rad) - b * np.sin(t) * np.sin(angle_rad)
        y_coords = cy + a * np.cos(t) * np.sin(angle_rad) + b * np.sin(t) * np.cos(angle_rad)

        min_x = np.min(x_coords)
        max_x = np.max(x_coords)
        min_y = np.min(y_coords)
        max_y = np.max(y_coords)

        return (min_x, max_x, min_y, max_y)

def add_scalebar_1m_bottom_left(ax, label="1 m", pad_frac=0.02, lw=2):
    """
    Draw a 1 m east–west scalebar in the bottom-left of a lon/lat plot.
    - pad_frac: padding from the plot edges as a fraction of axis span
    """
    geod = Geod(ellps="WGS84")

    # axis bounds (x=lon, y=lat)
    lon_min, lon_max = ax.get_xlim()
    lat_min, lat_max = ax.get_ylim()
    # ensure ascending for padding computation
    if lon_min > lon_max: lon_min, lon_max = lon_max, lon_min
    if lat_min > lat_max: lat_min, lat_max = lat_max, lat_min

    dx = (lon_max - lon_min)
    dy = (lat_max - lat_min)

    # start point a bit inside bottom-left corner
    lon0 = lon_min + pad_frac * dx
    lat0 = lat_min + pad_frac * dy

    # end point = 1 m due east of (lon0,lat0)
    lon1, lat1, _ = geod.fwd(lon0, lat0, 90.0, 1.0)

    # draw bar
    ax.plot([lon0, lon1], [lat0, lat1], color="k", lw=lw, solid_capstyle="butt")

    # label centered above the bar with a small *northward* offset (~0.6 m)
    cx = (lon0 + lon1) / 2.0
    cy = (lat0 + lat1) / 2.0
    cx_off, cy_off, _ = geod.fwd(cx, cy, 0.0, 0.05)  

    ax.text(
        cx_off, cy_off,
        label,
        ha="center", va="bottom", fontsize=8,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.2)
    )

def get_top_matches_multi(MAY, TOPK, _WORKER, weights=None, geom_topk=32, downscale=0.5):
    # weights
    print("get_top_matches_multi ...")
    W = {'clip':0.22,'dino':0.22,'hog':0.12,'sift':0.20,'ssim':0.18,'chamfer':0.14}
    if weights: W.update(weights)

    if "paths" not in _WORKER:
        raise OSError 

    paths        = _WORKER["paths"]
    feats        = _WORKER["feats"]
    feats_norm   = _WORKER["feats_norm"]
    clip_mat     = _WORKER["clip_mat"]   # rows are unit-norm if present
    dino_mat     = _WORKER["dino_mat"]   # rows are unit-norm if present

    # --- query prep (May) ---
    mg = load_gray_eq(MAY); assert mg is not None and mg.ndim == 2
    q_hog = hog_vec(mg)
    q_hog = q_hog / (np.linalg.norm(q_hog) + 1e-8)

    # HOG cosine for all in one GEMV
    hog_sims = (feats @ q_hog) / feats_norm  # shape [N]

    # Optional CLIP/DINO (vectorized)
    rgbM = cv.cvtColor(cv.imread(MAY, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
    clip_sims = None
    if clip_mat is not None:
        v_clip = embed_clip(rgbM)
        v_clip = v_clip / (np.linalg.norm(v_clip) + 1e-8)
        clip_sims = clip_mat @ v_clip  # since rows are unit-norm

    dino_sims = None
    if dino_mat is not None:
        v_dino = embed_dino(rgbM)
        v_dino = v_dino / (np.linalg.norm(v_dino) + 1e-8)
        dino_sims = dino_mat @ v_dino

    # Normalize per channel to [0,1] (cheap)
    def nzminmax(x):
        if x is None: return None
        lo, hi = np.nanmin(x), np.nanmax(x)
        return (x - lo) / (hi - lo + 1e-8)

    hog_n  = nzminmax(hog_sims)
    clip_n = nzminmax(clip_sims) if clip_sims is not None else None
    dino_n = nzminmax(dino_sims) if dino_sims is not None else None

    # Blend (no loops)
    coarse = 0.0
    denom = 0.0
    if hog_n  is not None: coarse += W['hog']  * hog_n;  denom += W['hog']
    if clip_n is not None: coarse += W['clip'] * clip_n; denom += W['clip']
    if dino_n is not None: coarse += W['dino'] * dino_n; denom += W['dino']
    coarse = coarse / max(denom, 1e-8)

    # Top-K via argpartition (much faster than full sort)
    # Expand K for geometric rerank, then compress to final TOPK
    K1 = max(TOPK, geom_topk)
    idx_part = np.argpartition(-coarse, K1)[:K1]
    # Order those by score
    top_idx = idx_part[np.argsort(-coarse[idx_part])]

    # --- cheap precompute for geometry ---
    mg_small = cv.resize(mg, None, fx=downscale, fy=downscale, interpolation=cv.INTER_AREA)
    gM = grad(mg_small); eM = cv.Canny(gM, 50, 150)
    Hs, Ws = mg_small.shape[:2]

    rows = []
    # Only geometric rerank on this small shortlist
    for i in top_idx:
        npth = str(paths[i])
        ng = load_gray_eq(npth)
        if ng is None:
            continue

        # Downscale target too
        ngs = cv.resize(ng, (Ws, Hs), interpolation=cv.INTER_AREA)

        # ECC align (fewer iters/pyr levels)
        ng_al = ecc_align(mg_small, ngs, "affine")  # make sure your ECC uses small iters

        gssim = ssim(grad(mg_small), grad(ng_al))
        eN = cv.Canny(grad(ng_al), 50, 150)
        ch = chamfer(eM, eN) / max(Hs, Ws)

        # SIFT on downscaled images (or ORB if still too slow)
        sift_ir = sift_ransac_inlier_ratio(mg_small, ngs, model="affine")

        final = (W['hog']*float(hog_n[i]) +
                 (W['clip']*float(clip_n[i]) if clip_n is not None else 0.0) +
                 (W['dino']*float(dino_n[i]) if dino_n is not None else 0.0) +
                 W['sift']*float(sift_ir) +
                 W['ssim']*float(gssim) -
                 W['chamfer']*float(ch))

        rows.append((npth, float(coarse[i]), float(gssim), float(ch), float(final)))

    # Final selection to requested TOPK
    rows.sort(key=lambda r: -r[4])
    return rows[:TOPK]

def get_top_matches_multi_OLD(MAY, TOPK, weights=None, idx_npz_path=None):
    print("getting top matches multi ...")
    # weights: dict with keys 'clip','dino','hog','sift','ssim','chamfer'
    W = {'clip':0.22,'dino':0.22,'hog':0.12,'sift':0.20,'ssim':0.18,'chamfer':0.14}
    if weights: W.update(weights)
    print("updated the weights" )

    data = np.load(idx_npz_path, allow_pickle=True) 

    print("parsing data ...")
    print(data.keys()) 

    paths, feats = data["paths"], data["feats"]
    mg = load_gray_eq(MAY); assert mg is not None and mg.ndim==2
    q_hog = hog_vec(mg)
    hog_sims_all = cos_sims(q_hog, feats)
    path_to_hog = {str(paths[i]): float(hog_sims_all[i]) for i in range(len(paths))}
    
    print("extracting embed sidecars")
    # Optional: only touch sidecars if we actually want CLIP/DINO from them
    clip_mat = dino_mat = None
    if idx_npz_path: #and (use_clip or use_dino):
        try:
            ensure_embed_sidecars(idx_npz_path)
            clip_mat, dino_mat = load_embed_sidecars(idx_npz_path)
        except Exception:
            clip_mat = dino_mat = None

    # # --- global shortlist via (HOG + CLIP + DINO)
    mg = load_gray_eq(MAY); assert mg is not None and mg.ndim==2
    q_hog = hog_vec(mg)
    hog_sims = cos_sims(q_hog, feats)

    rgbM = cv.cvtColor(cv.imread(MAY, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
    v_clip = embed_clip(rgbM) if clip_mat is not None else None
    v_dino = embed_dino(rgbM) if dino_mat is not None else None

    clip_sims = None
    dino_sims = None
    if v_clip is not None and clip_mat is not None:
        # cosine row-wise
        v = v_clip / (np.linalg.norm(v_clip)+1e-8)
        clip_sims = (clip_mat @ v) / (np.linalg.norm(clip_mat, axis=1)+1e-8)

    if v_dino is not None and dino_mat is not None:
        v = v_dino / (np.linalg.norm(v_dino)+1e-8)
        dino_sims = (dino_mat @ v) / (np.linalg.norm(dino_mat, axis=1)+1e-8)

    # Normalize each sim to [0,1] for blending
    def nzminmax(x):
        if x is None: return None
        lo, hi = np.nanmin(x), np.nanmax(x)
        return (x - lo) / (hi - lo + 1e-8)

    print("normalize sim for blending")
    hog_n = nzminmax(hog_sims)
    clip_n = nzminmax(clip_sims) if clip_sims is not None else None
    dino_n = nzminmax(dino_sims) if dino_sims is not None else None

    # blended coarse: fallback gracefully if some are None
    coarse = 0.0
    denom = 0.0
    if hog_n is not None:   coarse += W['hog'] * hog_n;   denom += W['hog']
    if clip_n is not None:  coarse += W['clip'] * clip_n; denom += W['clip']
    if dino_n is not None:  coarse += W['dino'] * dino_n; denom += W['dino']
    coarse = coarse / max(denom, 1e-8)

    # shortlist
    top_idx = np.argsort(-coarse)[:TOPK]

    # --- re-rank with geometry + structure
    mgG = grad(mg); eM = cv.Canny(mgG, 50, 150)

    rows = []  # (nov_path, coarse_sim_blend, gssim, chamfer_norm, final)
    for i in top_idx:
        npth = str(paths[i])
        ng = load_gray_eq(npth)
        if ng is None: continue

        # ECC align for SSIM/Chamfer
        ng_al = ecc_align(mg, ng, "affine")
        gssim = ssim(grad(mg), grad(ng_al))
        eN = cv.Canny(grad(ng_al), 50, 150)
        H, Wd = mg.shape[:2]
        ch = chamfer(eM, eN) / max(H, Wd)

        # SIFT+RANSAC (raw images)
        sift_ir = sift_ransac_inlier_ratio(mg, ng, model="affine")

        # final score
        final = (W['hog']*float(hog_n[i] if hog_n is not None else 0.0) +
                 W['clip']*float(clip_n[i] if clip_n is not None else 0.0) +
                 W['dino']*float(dino_n[i] if dino_n is not None else 0.0) +
                 W['sift']*float(sift_ir) +
                 W['ssim']*float(gssim) -
                 W['chamfer']*float(ch))
        rows.append((npth, float(coarse[i]), float(gssim), float(ch), float(final)))
    
    print("returning rows ...")
    return rows

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.dot(a, b) /
                ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))
                
def get_inside_match_metrics_multi(data, MAY, nov_inside_candidates, _WORKER, weights=None, downscale=0.5):
    print("get_inside_match_metrics ...")
    W = {'clip':0.22,'dino':0.22,'hog':0.12,'sift':0.20,'ssim':0.18,'chamfer':0.14}
    if weights: W.update(weights)

    if "paths" not in _WORKER:
        raise OSError 

    paths        = _WORKER["paths"]
    feats        = _WORKER["feats"]
    feats_norm   = _WORKER["feats_norm"]
    path_to_row  = _WORKER["path_to_row"]
    clip_mat     = _WORKER["clip_mat"]   # unit-norm rows or None
    dino_mat     = _WORKER["dino_mat"]

    if len(nov_inside_candidates) == 0:
        raise OSError("No inside candidates.")

    mg = load_gray_eq(MAY); assert mg is not None and mg.ndim == 2
    q_hog = hog_vec(mg)
    q_hog = q_hog / (np.linalg.norm(q_hog) + 1e-8)

    # HOG sims for all
    hog_sims_all = (feats @ q_hog) / feats_norm
    path_to_hog = {str(paths[i]): float(hog_sims_all[i]) for i in range(len(paths))}

    # Precompute May embeddings once
    rgbM = cv.cvtColor(cv.imread(MAY, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
    v_clip = v_dino = None
    if clip_mat is not None:
        v_clip = embed_clip(rgbM)
        v_clip = v_clip / (np.linalg.norm(v_clip) + 1e-8)
    if dino_mat is not None:
        v_dino = embed_dino(rgbM)
        v_dino = v_dino / (np.linalg.norm(v_dino) + 1e-8)

    # Cheap precompute for geometry on downscaled images
    mg_small = cv.resize(mg, None, fx=downscale, fy=downscale, interpolation=cv.INTER_AREA)
    gM = grad(mg_small); eM = cv.Canny(gM, 50, 150)
    Hs, Ws = mg_small.shape[:2]

    inside_rows = []

    for npth in sorted(nov_inside_candidates):
        row = path_to_row.get(str(npth), None)
        if row is None:
            continue

        ng = load_gray_eq(npth)
        if ng is None:
            continue

        # HOG (lookup)
        coarse_hog = path_to_hog.get(str(npth), 0.0)

        # CLIP/DINO similarities via sidecar rows (no per-candidate embedding)
        s_clip = 0.0
        s_dino = 0.0
        if v_clip is not None and clip_mat is not None:
            s_clip = float(clip_mat[row].dot(v_clip))
        if v_dino is not None and dino_mat is not None:
            s_dino = float(dino_mat[row].dot(v_dino))

        # Downscale candidate
        ngs = cv.resize(ng, (Ws, Hs), interpolation=cv.INTER_AREA)

        # Geometry on downscaled
        ng_al = ecc_align(mg_small, ngs, "affine")
        gssim = ssim(grad(mg_small), grad(ng_al))
        eN = cv.Canny(grad(ng_al), 50, 150)
        ch = chamfer(eM, eN) / max(Hs, Ws)

        # SIFT downscaled (or switch to ORB if needed)
        sift_ir = sift_ransac_inlier_ratio(mg_small, ngs, model="affine")

        # Blend: normalize trio locally
        trio = np.array([coarse_hog, s_clip, s_dino], dtype=np.float32)
        lo, hi = float(np.min(trio)), float(np.max(trio))
        trio_n = (trio - lo) / (hi - lo + 1e-8)
        hog_n, clip_n, dino_n = trio_n.tolist()

        final = (W['hog']*hog_n + W['clip']*clip_n + W['dino']*dino_n +
                 W['sift']*float(sift_ir) + W['ssim']*float(gssim) -
                 W['chamfer']*float(ch))

        inside_rows.append((npth, float(coarse_hog), float(gssim), float(ch), float(final)))

    inside_rows.sort(key=lambda x: -x[4])
    inside_numbered, rank_inside, metrics_inside = [], {}, {}
    for offs, (nov_path, coarse, gssim, ch, final) in enumerate(inside_rows):
        inside_numbered.append((offs, nov_path, coarse, gssim, ch, final))
        rank_inside[nov_path] = offs
        metrics_inside[nov_path] = (final, gssim)

    return inside_rows, inside_numbered, rank_inside, metrics_inside

def get_inside_match_metrics_multi_OLD(data, MAY, nov_inside_candidates, weights=None, idx_npz_path=None):
    print("getting inside match metrics multi ...")
    W = {'clip':0.22,'dino':0.22,'hog':0.12,'sift':0.20,'ssim':0.18,'chamfer':0.14}
    if weights: W.update(weights)

    paths, feats = data["paths"], data["feats"]

    mg = load_gray_eq(MAY); assert mg is not None and mg.ndim==2
    q_hog = hog_vec(mg)
    #print("q_hog: ",q_hog)
    hog_sims_all = cos_sims(q_hog, feats)
    #print("hog_sims_all: ",hog_sims_all)
    path_to_hog = {str(paths[i]): float(hog_sims_all[i]) for i in range(len(paths))}

    clip_mat = dino_mat = None
    ensure_embed_sidecars(idx_npz_path) #this part is TAKING 5 EVER
    print("loading embed sidecars ...")
    clip_mat, dino_mat = load_embed_sidecars(idx_npz_path)
    print("successfully loaded embed sidecars") 

    #Precompute May embeddings
    rgbM = cv.cvtColor(cv.imread(MAY, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
    v_clip = embed_clip(rgbM)
    v_dino = embed_dino(rgbM)
    print("Finished precomputing May embeddings!") 

    inside_rows = []
    mgG = grad(mg); eM = cv.Canny(mgG, 50, 150)
    H, Wd = mg.shape[:2]

    if len(nov_inside_candidates) == 0:
        print("no inside candidates!")
        raise OSError 

    print("iterating over nov_inside_candidates ...")
    for npth in sorted(nov_inside_candidates):
        ng = load_gray_eq(npth)
        if ng is None:
            print("ng None!")
            raise OSError
            continue

        # HOG against prebuilt feats
        coarse_hog = path_to_hog.get(str(npth), 0.0)

        # CLIP/DINO on-the-fly for candidate if sidecars not wired here
        rgbN = cv.cvtColor(cv.imread(npth, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)

        #print("calling cos sims on clip")
        s_clip = cos_sim(v_clip, embed_clip(rgbN)) if v_clip is not None else 0.0
        #print("calling cos sims on dino")
        s_dino = cos_sim(v_dino, embed_dino(rgbN)) if v_dino is not None else 0.0

        # ECC for SSIM/Chamfer
        print("computing ecc for SSIM/Chamfer")
        ng_al = ecc_align(mg, ng, "affine")
        gssim = ssim(grad(mg), grad(ng_al))
        eN = cv.Canny(grad(ng_al), 50, 150)
        ch = chamfer(eM, eN) / max(H, Wd)

        # SIFT+RANSAC (raw)
        print("computing ecc for SIFT/RANSAC")
        sift_ir = sift_ransac_inlier_ratio(mg, ng, model="affine")

        # blend HOG + embeddings
        # First normalize three terms to [0,1] per candidate set (quick & local)
        trio = np.array([coarse_hog, s_clip, s_dino], dtype=np.float32)
        lo, hi = float(np.min(trio)), float(np.max(trio))
        trio_n = (trio - lo) / (hi - lo + 1e-8)
        hog_n, clip_n, dino_n = trio_n.tolist()

        final = (W['hog']*hog_n + W['clip']*clip_n + W['dino']*dino_n +
                 W['sift']*float(sift_ir) + W['ssim']*float(gssim) -
                 W['chamfer']*float(ch))

        inside_rows.append((npth, float(coarse_hog), float(gssim), float(ch), float(final)))

    inside_top = sorted(inside_rows, key=lambda x: -x[4])
    inside_numbered, rank_inside, metrics_inside = [], {}, {}
    for offs, (nov_path, coarse, gssim, ch, final) in enumerate(inside_top):
        inside_numbered.append((offs, nov_path, coarse, gssim, ch, final))
        rank_inside[nov_path] = offs
        metrics_inside[nov_path] = (final, gssim)
    
    #print(f"inside_rows: {inside_rows}, inside_numbered: {inside_numbered}, rank_inside: {rank_inside}, metrics_inside: {metrics_inside}")
    return inside_rows, inside_numbered, rank_inside, metrics_inside

# --- make query-time HOG match the index builder exactly ---
import cv2 as cv
import numpy as np
from skimage.feature import hog

TARGET_HW = (512, 512)

def _letterbox(gray: np.ndarray, target_hw=TARGET_HW) -> np.ndarray:
    th, tw = target_hw
    h, w = gray.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    r = cv.resize(gray, (nw, nh), interpolation=cv.INTER_AREA if s < 1 else cv.INTER_LINEAR)
    canvas = np.zeros((th, tw), dtype=r.dtype)
    top = (th - nh) // 2
    left = (tw - nw) // 2
    canvas[top:top + nh, left:left + nw] = r
    return canvas

def _gradient_mag(img_u8: np.ndarray) -> np.ndarray:
    gx = cv.Scharr(img_u8, cv.CV_32F, 1, 0)
    gy = cv.Scharr(img_u8, cv.CV_32F, 0, 1)
    mag = cv.magnitude(gx, gy)
    return cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

def hog_vec_index_style(gray_u8: np.ndarray) -> np.ndarray:
    # match index: CLAHE/blur *optional* — but the index already did it
    if gray_u8.ndim != 2:
        raise ValueError(f"expected HxW grayscale, got {gray_u8.shape}")
    g = _letterbox(gray_u8, TARGET_HW)
    g = _gradient_mag(g)
    v = hog(
        g,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)
    n = float(np.linalg.norm(v)) + 1e-9
    return v / n

def ecc_align_to_ref(ref_u8: np.ndarray, mov_u8: np.ndarray, model="affine") -> np.ndarray:
    """Align mov_u8 to ref_u8 and guarantee identical HxW."""
    aligned = ecc_align(ref_u8, mov_u8, model)  # your existing function
    H, W = ref_u8.shape[:2]
    if aligned.shape[:2] != (H, W):
        aligned = cv.resize(aligned, (W, H), interpolation=cv.INTER_LINEAR)
    return aligned

def robust_read_gray_color(pth: str):
    # return (gray, color, used_path)
    c = cv.imread(pth, cv.IMREAD_COLOR)
    used = pth
    if c is None:
        alt = _maybe_swap_easystore(pth)
        if alt != pth:
            c2 = cv.imread(alt, cv.IMREAD_COLOR)
            if c2 is not None:
                c, used = c2, alt
    if c is None:
        return None, None, pth
    g = cv.cvtColor(c, cv.COLOR_BGR2GRAY)
    return g, c, used


import hashlib, os

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()



def export_csv_for_may(paths: Paths, top_match_paths, args: RunArgs, idx_blob, cov_ellipse, center_lonlat, out_csv_path: str):
    # All Nov candidates in the folder
    nov_candidates = list(dict.fromkeys(
        sorted(os.path.join(paths.nov_dir, x)
            for x in os.listdir(paths.nov_dir)
            if x.lower().endswith(".png")
        ) + list(top_match_paths)
    ))

    # Build HOG lookup from the index once
    np_paths, feats, _ = idx_blob["paths"], idx_blob["feats"], idx_blob
    path_to_hog = {}

    mg = safe_load_gray_eq(paths.may_img)  # keep your existing equalization
    #print("mg stats:", type(mg), mg.dtype, mg.shape, float(mg.min()), float(mg.max()), float(mg.std()))  
    q_hog = hog_vec_index_style(mg)
    #print("q_hog shape:", getattr(q_hog, "shape", None), "norm:", float(np.linalg.norm(q_hog))) 

    # sanity guard: match index D
    N, D_idx = feats.shape
    if q_hog.shape[0] != D_idx:
        print(f"[WARN] HOG dim mismatch at query: q={q_hog.shape[0]} vs index={D_idx}. Forcing letterbox.")
        # (This shouldn’t trigger if hog_vec_index_style is used.)
    hog_sims_all = cos_sims(q_hog, feats)

    for i in range(len(np_paths)):
        path_to_hog[str(np_paths[i])] = float(hog_sims_all[i])

    # Sidecars for CLIP/DINO (fast path)
    clip_mat = dino_mat = None
    path_index = {}
    try:
        clip_mat, dino_mat = load_embed_sidecars(paths.idx_npz)
        path_index = {str(p): i for i, p in enumerate(np_paths)}
    except Exception:
        clip_mat = dino_mat = None

    # Query embeddings (May)
    q_rgb, _ = safe_imread(paths.may_img)
    if q_rgb is None:
        raise OSError 

    #print("calling embed clip from export csv")
    v_clip_q = embed_clip(q_rgb) #if (args.use_clip and q_rgb is not None) else None

    #print("calling embed dino from export csv")
    v_dino_q = embed_dino(q_rgb) #if (args.use_dino and q_rgb is not None) else None

    def sim_from_sidecar(mat, q_vec, cand_path):
        if mat is None or q_vec is None:
            return None
        i = path_index.get(cand_path) or path_index.get(_maybe_swap_easystore(cand_path))
        if i is None or i >= mat.shape[0]:
            return None
        v = mat[i].astype(np.float32, copy=False)
        if not np.isfinite(v).all() or np.linalg.norm(v) < 1e-8:
            return None  # force fallback on zeros/NaNs
        return float(np.dot(v, q_vec) /
                    ((np.linalg.norm(v)+1e-8)*(np.linalg.norm(q_vec)+1e-8)))


    ell_poly = ellipse_polygon(center_lonlat, cov_ellipse.width, cov_ellipse.height, cov_ellipse.angle)
    #print("successfuly made it past ellipse_polygon") 

    header = [
        "may_image", "nov_image", "inside_ellipse", "distance_m",
        "hog_cosine", "clip_cosine", "dino_cosine", "sift_inlier_ratio",
        "ssim_grad", "chamfer_norm", "final_weighted"
    ]
    with open(out_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        mgG = grad(mg); eM = cv.Canny(mgG, 50, 150)
        H, Wd = mg.shape[:2]

        #print("nov_candidates:",nov_candidates) 

        for npth in nov_candidates:
            #print("npth for nov_candidates: ",npth) 
            ng, rgbN, used_path = robust_read_gray_color(npth) 

            # Geometry block
            merged_poly, centroid = nov_subset_geometry_and_centroid(used_path) 

            #print("got merged poly and centroid!") 

            if merged_poly is None or centroid is None:
                inside = False
                dist_m = np.nan
            else:
                inside = ell_poly.intersects(merged_poly)
                dist_m = geodesic_m(center_lonlat, centroid)

            #print("this is npth: ",npth) 

            # HOG (from index; fallback 0 if not found)
            hog_sim = path_to_hog.get(npth)
            
            if hog_sim is None:
                print("[WARN] hog_sim is NONE")
                hog_sim = path_to_hog.get(_maybe_swap_easystore(npth), 0.0)
            #print("got hog sim") 

            # Read Nov image & gray
            #rgbN, _ = safe_imread(npth)
            

            #ng = cv.cvtColor(rgbN, cv.COLOR_BGR2GRAY) if rgbN is not None else None
            if ng is None:
                print("[WARN] ng is NONE") 
                # still write a row with whatever we can
                w.writerow([
                    paths.may_img, npth, int(inside),
                    f"{dist_m:.3f}" if np.isfinite(dist_m) else "",
                    f"{hog_sim:.6f}",
                    "", "", "",  # clip/dino/sift not available
                    "", "", ""   # ssim/chamfer/final not available
                ])
                continue
                #input("Press Enter to Acknowledge")
                
            # CLIP/DINO sims (sidecar fast path; if missing, on-the-fly fallback)
            clip_sim = sim_from_sidecar(clip_mat, v_clip_q, used_path)
            if clip_sim is None:
                clip_sim = cos_sim(v_clip_q, embed_clip(cv.cvtColor(rgbN, cv.COLOR_BGR2RGB)))

            dino_sim = sim_from_sidecar(dino_mat, v_dino_q, used_path)
            if dino_sim is None:
                dino_sim = cos_sim(v_dino_q, embed_dino(cv.cvtColor(rgbN, cv.COLOR_BGR2RGB)))

            # Alignment + structure metrics
            ng_al = ecc_align_to_ref(mg, ng, "affine")
            gssim = ssim(grad(mg), grad(ng_al))
            eN = cv.Canny(grad(ng_al), 50, 150)
            chamfer_norm = chamfer(eM, eN) / max(H, Wd)

            # SIFT
            if args.use_sift:
                sift_ir = sift_ransac_inlier_ratio(mg, ng, model="affine")
            else:
                sift_ir = None

            # Recreate your weighted final (same weights you pass in args)
            W = args.weights
            # normalize “coarse” terms locally to keep scale sane
            trio = np.array([
                hog_sim,
                clip_sim if clip_sim is not None else 0.0,
                dino_sim if dino_sim is not None else 0.0
            ], dtype=np.float32)
            lo, hi = float(np.min(trio)), float(np.max(trio))
            trio_n = (trio - lo) / (hi - lo + 1e-8)
            hog_n, clip_n, dino_n = trio_n.tolist()

            # print("ROW: ", npth)
            # print("  used_path:", used_path)
            # print("  file_size:", os.path.getsize(used_path) if os.path.exists(used_path) else -1)
            # print("  gray_sum/std:", float(ng.sum()), float(ng.std()))
            # print("  hog/clip/dino:", hog_sim, clip_sim, dino_sim)

            # print("  inode/dev:", os.stat(used_path).st_ino, os.stat(used_path).st_dev)
            # print("  sha256  :", sha256_of(used_path))

            final_w = (W['hog']*hog_n +
                       W['clip']*(clip_n if clip_sim is not None else 0.0) +
                       W['dino']*(dino_n if dino_sim is not None else 0.0) +
                       W['sift']*(float(sift_ir) if sift_ir is not None else 0.0) +
                       W['ssim']*float(gssim) -
                       W['chamfer']*float(chamfer_norm))

            w.writerow([
                paths.may_img, npth, int(inside),
                f"{dist_m:.3f}" if np.isfinite(dist_m) else "",
                f"{hog_sim:.6f}",
                f"{clip_sim:.6f}" if clip_sim is not None else "",
                f"{dino_sim:.6f}" if dino_sim is not None else "",
                f"{sift_ir:.6f}" if sift_ir is not None else "",
                f"{gssim:.6f}", f"{chamfer_norm:.6f}", f"{final_w:.6f}"
            ])

def ellipse_polygon(center_lonlat, width, height, angle_deg, res=256):
    # build shapely ellipse from Matplotlib ellipse params (degrees)
    cx, cy = center_lonlat
    circ = ShapelyPoint(0, 0).buffer(1.0, resolution=res)
    ell = shp_scale(circ, xfact=width/2.0, yfact=height/2.0, origin=(0, 0))
    ell = shp_rotate(ell, angle_deg, origin=(0, 0), use_radians=False)
    ell = shp_translate(ell, xoff=cx, yoff=cy)
    return ell

def nov_subset_geometry_and_centroid(nov_path: str):
    nov_pano = load_pano_pickle_for_image(nov_path, which="nov")
    nov_key = nov_path if nov_path in nov_pano else _maybe_swap_easystore(nov_path)
    subset = find_largest_overlap_subset(create_polygons(nov_pano[nov_key]))
    if not subset:
        return None, None
    merged = unary_union(subset)
    c = merged.centroid
    return merged, (float(c.x), float(c.y))

def geodesic_m(p0_lonlat, p1_lonlat):
    lon0, lat0 = p0_lonlat
    lon1, lat1 = p1_lonlat
    _, _, dist_m = GEOD.inv(lon0, lat0, lon1, lat1)
    return float(dist_m)
