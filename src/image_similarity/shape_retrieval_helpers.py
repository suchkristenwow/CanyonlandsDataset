from matplotlib.patches import Polygon as MplPolygon, Ellipse
import numpy as np 
import os, numpy as np, cv2 as cv, pickle, gc
from pyproj import Geod 

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
    set_axes_border
)

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
