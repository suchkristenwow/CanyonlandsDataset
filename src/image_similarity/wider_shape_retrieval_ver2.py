import os, numpy as np, cv2 as cv, pickle, gc
from pathlib import Path
from skimage.feature import hog
from scipy.spatial import KDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches as mpatches
from matplotlib.patches import Polygon as MplPolygon, Ellipse
from shapely.ops import unary_union
from shapely.geometry import Point as ShapelyPoint
from pyproj import Geod
from seasonal_comparison.gps_utils import (
    precompute_timestamps, 
    load_covariance_matrix,
    scale_covariance_to_degrees
)
# ---- your package utils ----
from seasonal_comparison.image_stitching_utils import (
    create_polygons,
    reorder_corners,
)
from seasonal_comparison.plot_figs import (
    fix_invalid_corners,
    find_largest_overlap_subset,
)
from seasonal_comparison.general_utils import (
    find_closest_index
)

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
from image_similarity.palette_generation import (
    gen_distinct_palette
)
# ------------------- CONFIG -------------------
IDX = "/media/kristen/easystore2/RestorebotData/analysis_shape/nov_index.npz"
MAY = "/media/kristen/easystore2/RestorebotData/benchmarking_ex/matches_fused/1650825680196263936/May/may_cluster0000_part02.png"
nov_dir = "/media/kristen/easystore2/RestorebotData/benchmarking_ex/matches_fused/1650825680196263936/Nov"

TARGET_HW = (512, 512)
TOPK_RETRIEVE = 50
TOPK_SHOW = 4     # now show 4; 2×2 grid
timestamp = 1650825680196263936  # nanoseconds

PALETTE = gen_distinct_palette(n_colors=8)

def color_by_index(k: int):
    return PALETTE[k % len(PALETTE)]

def color_for_rank(rank: int):
    # rank is 1..8 for your labeled items
    return color_by_index(rank - 1)

# ------------------- HELPERS -------------------
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

# ------------------- MAIN -------------------
print("loading index and building candidates ...")
data = np.load(IDX, allow_pickle=True)
paths, feats = data["paths"], data["feats"]

mg = load_gray_eq(MAY)
assert mg is not None and mg.ndim == 2, f"bad image at {MAY}"
q = hog_vec(mg)

sims = cos_sims(q, feats)
top = np.argsort(-sims)[:TOPK_RETRIEVE]

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

print(f"Compared against {len(rows)} November images")
top_rows = sorted(rows, key=lambda x: -x[4])[:TOPK_SHOW]

# Number top-4 as 1..4
top_numbered = []          # (rank, nov_path, coarse, gssim, ch, final)
rank_top = {}
metrics_top = {}
for rank, (nov_path, coarse, gssim, ch, final) in enumerate(top_rows[:4], start=1):
    top_numbered.append((rank, nov_path, coarse, gssim, ch, final))
    rank_top[nov_path] = rank
    metrics_top[nov_path] = (final, gssim)

print("preparing GPS geometry and covariance ellipse ...")
may_dir = os.path.dirname(MAY)
gps_plot_png = os.path.join(may_dir, "gps_plot.png")
top4_png = os.path.join(may_dir, "top4_overall_2x2.png")
ellipse_top4_png = os.path.join(may_dir, "top4_inside_ellipse_2x2.png")

# Load May polygons
may_filename = os.path.basename(MAY)
idx1 = [i for i, ch in enumerate(may_filename) if ch == "_"][1]
idx0 = may_filename.index("cluster") + 8
may_cluster_no = int(may_filename[idx0:idx1])
may_pano_pickle = os.path.join(may_dir, f"may_panos_{may_cluster_no}.pickle")
with open(may_pano_pickle, "rb") as f:
    may_pano = pickle.load(f)

if MAY not in may_pano:
    MAY_try = MAY.replace("easystore2", "easystore3")
    if MAY_try not in may_pano:
        raise FileNotFoundError(f"May pano entry not found for key {MAY} or {MAY_try}")
    MAY = MAY_try

may_polygon_list = create_polygons(may_pano[MAY])
may_overlap_subset = find_largest_overlap_subset(may_polygon_list)

# Covariance + center
may_cov_dir = "/media/kristen/easystore2/RestorebotData/benchmarking_ex/May2022/1conmod/processed_results/covariance_matrices"
may_cov_timestamps = precompute_timestamps(may_cov_dir)
may_results_dir = "/media/kristen/easystore2/RestorebotData/benchmarking_ex/May2022/1conmod/processed_results"
may_data = np.genfromtxt(os.path.join(may_results_dir, "frustrum_corners.csv"), delimiter=",", skip_header=1)
may_timestamps = may_data[:, 0]
cam_lat, cam_lon = may_data[:, 15], may_data[:, 16]

cov_matrix = load_covariance_matrix(may_cov_dir, may_cov_timestamps, timestamp)
scaled_cov = scale_covariance_to_degrees(cov_matrix)

i_may = find_closest_index(may_timestamps, timestamp * 1e-9, max_delta_t=0.3)
center = (cam_lon[i_may], cam_lat[i_may])  # (lon, lat)

# eig-decomp for drawing ellipse
vals, vecs = np.linalg.eigh(scaled_cov)
order = vals.argsort()[::-1]
vals = vals[order]
vecs = vecs[:, order]
theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
width, height = 2 * np.sqrt(vals)  # 1-sigma

# ---------- Scan Nov pickles to find inside-ellipse candidates ----------
nov_inside_candidates = set()
inside_polygons = []                 # for bounds
inside_polys_by_path = {}            # path -> [polygons]

for f in os.listdir(nov_dir):
    if not f.endswith(".pickle"):
        continue
    pkl_path = os.path.join(nov_dir, f)
    with open(pkl_path, "rb") as handle:
        pano_dict = pickle.load(handle)  # {nov_stitched_img_path: [polygons]}
    for nov_stitched_path, polygons_k in pano_dict.items():
        nov_polygon_list = create_polygons(polygons_k)
        kept = []
        for shp_poly in nov_polygon_list:
            cx, cy = shp_poly.centroid.x, shp_poly.centroid.y
            if mahalanobis_inside(cx, cy, center, scaled_cov, k2=1.0):
                kept.append(shp_poly)
        if kept:
            nov_inside_candidates.add(nov_stitched_path)
            inside_polys_by_path.setdefault(nov_stitched_path, []).extend(kept)
            inside_polygons.extend(kept)

# ---------- Compute inside-ellipse similarity metrics and number 5..8 ----------
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

inside_top = sorted(inside_rows, key=lambda x: -x[4])[:TOPK_SHOW]
inside_numbered = []    # (rank, nov_path, coarse, gssim, ch, final)
rank_inside = {}
metrics_inside = {}
for offs, (nov_path, coarse, gssim, ch, final) in enumerate(inside_top, start=5):
    inside_numbered.append((offs, nov_path, coarse, gssim, ch, final))
    rank_inside[nov_path] = offs
    metrics_inside[nov_path] = (final, gssim)
inside_top_set = set(rank_inside.keys())

# ------------------- GPS plot -------------------
print("saving GPS-only plot ...")
fig_gps = plt.figure(figsize=(8, 6), dpi=150)
ax_gps = fig_gps.add_subplot(111)

# May polygons (overlap subset) in blue
for poly in may_overlap_subset:
    coords = list(poly.exterior.coords)[:-1]
    oc = fix_invalid_corners(reorder_corners(coords))
    ax_gps.add_patch(MplPolygon(oc, fill=True, facecolor='none', edgecolor="blue", linewidth=1.0))

# Covariance ellipse
covar_ellipse = Ellipse(xy=center, width=width, height=height, angle=theta,
                         edgecolor='blue', facecolor='none', linestyle='dotted')
ellipse_min_x, ellipse_max_x, ellipse_min_y, ellipse_max_y = get_ellipse_extrema(covar_ellipse)
ax_gps.add_patch(covar_ellipse)

# Draw inside-ellipse polygons
legend_handles = []
other_idx = 8  # after ranks 1..8, start coloring the unranked at index 8

inside_paths_sorted = sorted(nov_inside_candidates)
for npth in inside_paths_sorted:
    # use the *rank color* if it's in your top-4 inside (5..8),
    # otherwise assign a new color from 8 upwards
    if npth in inside_top_set:
        c = color_for_rank(rank_inside[npth])  # ranks 5..8
    else:
        c = color_by_index(other_idx)
        other_idx += 1

    for shp_poly in inside_polys_by_path[npth]:
        coords = list(shp_poly.exterior.coords)[:-1]
        oc = fix_invalid_corners(reorder_corners(coords))
        ax_gps.add_patch(MplPolygon(oc, fill=True, facecolor=c, alpha=0.25,
                                    edgecolor=c, linewidth=0.8))

    # Legend entries only for the ranked inside (5..8)
    if npth in inside_top_set:
        fin_val, ssim_val = metrics_inside[npth]
        legend_handles.append(
            mpatches.Patch(facecolor=c, edgecolor=c, alpha=0.7,
                           label=f"Inside {rank_inside[npth]}.  final={fin_val:.3f}  SSIM={ssim_val:.3f}")
        )

# Draw TOP-4 best matches (often outside ellipse) with distinct colors and numeric labels 1..4
#TOP4_COLORS = [plt.cm.tab10(i) for i in [0, 1, 2, 3]]
master_polygon_list = []
for idx, (rank, nov_path, coarse, gssim_val, ch_val, final) in enumerate(top_numbered):
    c = color_for_rank(rank)
    nov_filename = os.path.basename(nov_path)
    nov_path_dir = os.path.dirname(nov_path)
    nidx1 = [k for k, ch in enumerate(nov_filename) if ch == "_"][1]
    nidx0 = nov_filename.index("cluster") + 8
    nov_cluster_no = int(nov_filename[nidx0:nidx1])
    nov_pano_pickle = os.path.join(nov_path_dir, f"nov_panos_{nov_cluster_no}.pickle")
    with open(nov_pano_pickle, "rb") as f:
        nov_pano = pickle.load(f)
    key = nov_path if nov_path in nov_pano else nov_path.replace("easystore2", "easystore3")
    nov_polygon_list = create_polygons(nov_pano[key])
    top_subset = find_largest_overlap_subset(nov_polygon_list)
    master_polygon_list.extend(top_subset)
    for poly in top_subset:
        coords = list(poly.exterior.coords)[:-1]
        oc = fix_invalid_corners(reorder_corners(coords))
        ax_gps.add_patch(MplPolygon(oc, fill=True, facecolor=c, alpha=0.25,
                                    edgecolor=c, linewidth=1.0))
    legend_handles.append(
        mpatches.Patch(facecolor=c, edgecolor=c, alpha=0.7,
                       label=f"{rank}.  final={metrics_top[nov_path][0]:.3f}  SSIM={metrics_top[nov_path][1]:.3f}")
    )

# Bounds from everything drawn + small padding
may_merged   = unary_union(may_polygon_list) if may_polygon_list else None
inside_merged = unary_union(inside_polygons) if inside_polygons else None
top4_merged   = unary_union(master_polygon_list) if master_polygon_list else None

if may_merged is None:
    raise RuntimeError("Empty May polygon list.")

min_lon, min_lat, max_lon, max_lat = may_merged.bounds
for merged in [inside_merged, top4_merged]:
    if merged is not None:
        a, b, c, d = merged.bounds
        min_lon, min_lat = min(min_lon, a), min(min_lat, b)
        max_lon, max_lat = max(max_lon, c), max(max_lat, d)

min_lon = min(min_lon, ellipse_min_x); max_lon = max(max_lon,ellipse_max_x) 
min_lat = min(min_lat, ellipse_min_y); max_lat = max(max_lat,ellipse_max_y) 

dx = (max_lon - min_lon) * 0.10
dy = (max_lat - min_lat) * 0.10

ax_gps.set_xlim(min_lon - dx, max_lon + dx)
ax_gps.set_ylim(min_lat - dy, max_lat + dy)

aspect_correction = np.cos(np.deg2rad((min_lat + max_lat) / 2))
ax_gps.set_aspect(aspect_correction)
ax_gps.set_xlabel("Longitude"); ax_gps.set_ylabel("Latitude")
ax_gps.set_title("Down-facing camera frames with 1σ covariance ellipse")

if legend_handles:
    ax_gps.legend(
        handles=legend_handles,
        loc="center left",              # place legend on the left-center of bbox
        bbox_to_anchor=(1.02, 0.5),     # push it just outside the right edge
        borderaxespad=0,
        fontsize=7,
        frameon=False
    )

add_scalebar_1m_bottom_left(ax_gps)

plt.tight_layout()
plt.savefig(gps_plot_png, dpi=220)
plt.close(fig_gps)
print(f"Wrote GPS plot: {gps_plot_png}")

# ------------------- Save 2×2 grid of top-4 overall matches (ranks 1..4) -------------------
print("saving 2x2 grid for top overall matches ...")
fig_overall = plt.figure(figsize=(10, 8), dpi=150)
R, C = 2, 2
gs_overall = gridspec.GridSpec(R, C, wspace=0.06, hspace=0.12)
for idx, (rank, nov_path, coarse, gssim_val, ch_val, final) in enumerate(top_numbered):
    r, c = divmod(idx, C)
    axr = fig_overall.add_subplot(gs_overall[r, c])
    nimg = cv.imread(nov_path, cv.IMREAD_COLOR)
    if nimg is None:
        axr.text(0.5, 0.5, f"Missing:\n{os.path.basename(nov_path)}", ha="center", va="center", fontsize=8)
        axr.axis("off"); continue
    nimg = cv.cvtColor(nimg, cv.COLOR_BGR2RGB)
    axr.imshow(nimg); axr.axis("off")
    axr.set_title(f"{rank}.  final={final:.3f}  SSIM={gssim_val:.3f}", fontsize=9)
    set_axes_border(axr, color_for_rank(rank), lw=2) 
plt.tight_layout()
plt.savefig(top4_png, dpi=220)
plt.close(fig_overall)
print(f"Wrote top-4 overall grid: {top4_png}")

# ------------------- Save 2×2 grid for top-4 inside ellipse (ranks 5..8) -------------------
if len(inside_numbered) > 0:
    print("saving 2x2 grid for top inside-ellipse matches ...")
    fig_inside = plt.figure(figsize=(10, 8), dpi=150)
    gs_inside = gridspec.GridSpec(2, 2, wspace=0.06, hspace=0.12)
    for idx, (rank, nov_path, coarse, gssim_val, ch_val, final) in enumerate(inside_numbered):
        r, c = divmod(idx, 2)
        axr = fig_inside.add_subplot(gs_inside[r, c])
        nimg = cv.imread(nov_path, cv.IMREAD_COLOR)
        if nimg is None:
            axr.text(0.5, 0.5, f"Missing:\n{os.path.basename(nov_path)}", ha="center", va="center", fontsize=8)
            axr.axis("off"); continue
        nimg = cv.cvtColor(nimg, cv.COLOR_BGR2RGB)
        axr.imshow(nimg); axr.axis("off")
        axr.set_title(f"{rank}.  final={final:.3f}  SSIM={gssim_val:.3f}", fontsize=9)
        set_axes_border(axr, color_for_rank(rank), lw=2)
    plt.tight_layout()
    plt.savefig(ellipse_top4_png, dpi=220)
    plt.close(fig_inside)
    print(f"Wrote top-4 inside-ellipse grid: {ellipse_top4_png}")
else:
    print("No inside-ellipse candidates with successful similarity computation. Skipping ellipse grid.")

gc.collect()
print("Done.")