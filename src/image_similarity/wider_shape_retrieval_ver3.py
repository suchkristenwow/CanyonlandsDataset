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
    gen_distinct_palette,
    color_for_rank
)
from image_similarity.shape_retrieval_helpers import (
    get_top_matches,
    get_top_match_metrics,
    get_inside_match_metrics,
    get_ellipse_extrema,
    add_scalebar_1m_bottom_left
)

# ------------------- CONFIG -------------------
IDX = "/media/kristen/easystore2/RestorebotData/analysis_shape/nov_index.npz"
MAY = "/media/kristen/easystore2/RestorebotData/benchmarking_ex/matches_fused/1650825680196263936/May/may_cluster0000_part02.png"
nov_dir = "/media/kristen/easystore2/RestorebotData/benchmarking_ex/matches_fused/1650825680196263936/Nov"

TARGET_HW = (512, 512)
TOPK_SHOW = 4     # now show 4; 2×2 grid
timestamp = 1650825680196263936  # nanoseconds

PALETTE = gen_distinct_palette(n_colors=24)
ccount = 0

# ------------------- MAIN -------------------
print("loading index and building candidates ...")
data = np.load(IDX, allow_pickle=True)
paths, feats = data["paths"], data["feats"]

top_rows = get_top_matches(IDX,MAY,TOPK_SHOW)
top_numbered, rank_top, metrics_top = get_top_match_metrics(top_rows) 

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

# ------------------- GPS plot -------------------
print("saving GPS-only plot ...")

fig_gps = plt.figure(figsize=(10.5, 6), dpi=150, constrained_layout=False)
gs = fig_gps.add_gridspec(1, 2, width_ratios=[1.0, 0.45], wspace=0.03)
ax_gps = fig_gps.add_subplot(gs[0, 0])
ax_leg = fig_gps.add_subplot(gs[0, 1])
ax_leg.axis("off")

legend_handles = []

max_lat = -1e9; min_lat = 1e9
max_lon = -1e9; min_lon = 1e9

# May polygons (overlap subset) in blue
for poly in may_overlap_subset:
    coords = list(poly.exterior.coords)[:-1]
    oc = fix_invalid_corners(reorder_corners(coords)) #(list of (lon, lat)) 
    min_lat = min(min_lat,min([x[1] for x in oc]))
    max_lat = max(max_lat,max([x[1] for x in oc])) 
    min_lon = min(min_lon,min([x[0] for x in oc]))
    max_lon = max(max_lon,max([x[0] for x in oc])) 
    print("Adding may patch")
    ax_gps.add_patch(MplPolygon(oc, fill=True, facecolor='none', edgecolor="blue", linewidth=1.0))
legend_handles.append(
        mpatches.Patch(facecolor='none', edgecolor="blue",
                        label="may_cluster0000_part02.png")
    ) 

# Covariance ellipse
covar_ellipse = Ellipse(xy=center, width=width, height=height, angle=theta,
                         edgecolor='blue', facecolor='none', linestyle='dotted',label="Covariance Ellipse")
ellipse_min_x, ellipse_max_x, ellipse_min_y, ellipse_max_y = get_ellipse_extrema(covar_ellipse)

ax_gps.add_patch(covar_ellipse)
legend_handles.append(
    covar_ellipse
)

# Draw Frames Inside the Covariance Ellipsoid
nov_inside_frames = [os.path.join(nov_dir,x) for x in os.listdir(nov_dir) if x[-3:] == "png"]

# Ensure required containers exist for later bounds code
inside_polys_by_path = {}
inside_polygons = []          # all polygons from inside frames (for bounds)
master_polygon_list = []      # all polygons drawn (for bounds)
path_color = {}               # stable color per fused image path
# Rank/metrics for the provided inside set
#data,MAY,sims,nov_inside_candidates 
inside_rows, inside_numbered, rank_inside, metrics_inside = get_inside_match_metrics(data, MAY, nov_inside_frames)

inside_top_set = {nv_path for (rank, nv_path, *_rest) in inside_numbered}

# Draw each "inside" fused image footprint
for fused_nov_img in sorted(nov_inside_frames):
    # Derive the matching pano pickle for this fused image
    nov_filename = os.path.basename(fused_nov_img)
    nov_path_dir = os.path.dirname(fused_nov_img)
    # pickles live alongside fused image; adjust path if needed
    nidx1 = [k for k, ch in enumerate(nov_filename) if ch == "_"][1]
    nidx0 = nov_filename.index("cluster") + 8
    nov_cluster_no = int(nov_filename[nidx0:nidx1])
    nov_pano_pickle = os.path.join(nov_path_dir, f"nov_panos_{nov_cluster_no}.pickle")

    # Load pano dict
    with open(nov_pano_pickle, "rb") as f:
        nov_pano_dict = pickle.load(f)

    # Handle easystore2/easystore3 path flip if needed
    key = fused_nov_img if fused_nov_img in nov_pano_dict else fused_nov_img.replace("easystore2", "easystore3")
    if key not in nov_pano_dict:
        # Skip silently if missing (or add a warning print)
        continue

    # Normalize polygons and select the largest-overlap subset for drawing
    nov_polygon_list = create_polygons(nov_pano_dict[key])
    top_subset = find_largest_overlap_subset(nov_polygon_list)
    if not top_subset:
        continue

    # Stable color per path
    c = PALETTE[abs(hash(key)) % len(PALETTE)]
    path_color[key] = c

    # Draw subset polygons
    for shp_poly in top_subset:
        coords = list(shp_poly.exterior.coords)[:-1]
        oc = fix_invalid_corners(reorder_corners(coords))
        min_lat = min(min_lat,min([x[1] for x in oc]))
        max_lat = max(max_lat,max([x[1] for x in oc])) 
        min_lon = min(min_lon,min([x[0] for x in oc]))
        max_lon = max(max_lon,max([x[0] for x in oc]))
        print("adding polygon inside the covariance ellipsoid")
        ax_gps.add_patch(MplPolygon(oc, fill=True, facecolor=c, alpha=0.25,
                                    edgecolor=c, linewidth=0.8))
        master_polygon_list.append(shp_poly)
        inside_polygons.append(shp_poly)

    inside_polys_by_path[key] = top_subset

# Add legend entries for ranked inside frames (e.g., ranks 5..8)
for nv_path in inside_top_set:
    norm = nv_path if nv_path in path_color else nv_path.replace("easystore2", "easystore3")
    c = path_color.get(norm, PALETTE[abs(hash(norm)) % len(PALETTE)])
    fin_val, ssim_val = metrics_inside[nv_path]
    legend_handles.append(
        mpatches.Patch(facecolor=c, edgecolor=c, alpha=0.25,
                       label=f"Inside rank {rank_inside[nv_path]}  final={fin_val:.3f}  SSIM={ssim_val:.3f}")
    )

# Draw Best Match Frames 
for idx, (rank, nov_path, coarse, gssim_val, ch_val, final) in enumerate(top_numbered):
    c = PALETTE[ccount]; ccount += 1 
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

    for poly in top_subset:
        coords = list(poly.exterior.coords)[:-1]
        oc = fix_invalid_corners(reorder_corners(coords))
        min_lat = min(min_lat,min([x[1] for x in oc]))
        max_lat = max(max_lat,max([x[1] for x in oc])) 
        min_lon = min(min_lon,min([x[0] for x in oc]))
        max_lon = max(max_lon,max([x[0] for x in oc]))
        print("Adding patch for best match image")
        ax_gps.add_patch(MplPolygon(oc, fill=True, facecolor=c, alpha=0.25,
                                    edgecolor=c, linewidth=1.0))
        master_polygon_list.append(poly)

    legend_handles.append(
        mpatches.Patch(facecolor=c, edgecolor=c, alpha=0.25,
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

add_scalebar_1m_bottom_left(ax_gps)

# Ensure Matplotlib knows about all patch vertices for autoscaling
ax_gps.relim()  # handles lines/collections

# Make patches (MplPolygon/Ellipse) extend dataLim
for p in ax_gps.patches:
    path = p.get_path().transformed(p.get_transform())
    ax_gps.dataLim.update_from_data_xy(path.vertices, ignore=False)

ax_gps.autoscale_view()

dx = max((max_lon - min_lon) * 0.10, 1e-6)
dy = max((max_lat - min_lat) * 0.10, 1e-6)
ax_gps.set_xlim(min_lon - dx, max_lon + dx)
ax_gps.set_ylim(min_lat - dy, max_lat + dy)


# Geographic aspect (use mid-latitude)
mid_lat = 0.5 * (min_lat + max_lat)
ax_gps.set_aspect(np.cos(np.deg2rad(mid_lat)), adjustable='box')

# Legend (keep it outside)
# if legend_handles:
#     ax_gps.legend(
#         handles=legend_handles,
#         loc="center left",
#         bbox_to_anchor=(1.02, 0.5),
#         borderaxespad=0,
#         fontsize=7,
#         frameon=False,
#         handlelength=2.5
#     )

# add_scalebar_1m_bottom_left(ax_gps)

# AFTER you create fig_gps/ax_gps and draw everything (limits/aspect final):
# fig_gps.subplots_adjust(right=0.78)  # reserve ~22% for legend

# leg = fig_gps.legend(
#     handles=legend_handles,
#     loc="center left",
#     bbox_to_anchor=(0.80, 0.5),   # in figure coords
#     borderaxespad=0,
#     fontsize=7,
#     frameon=False,
#     handlelength=2.5,
# )

# aspect set, scalebar placed
ax_leg.legend(
    handles=legend_handles,
    loc="center",
    frameon=False,
    fontsize=7,
    handlelength=2.5,
)

plt.savefig(gps_plot_png, dpi=220)


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
    set_axes_border(axr, color_for_rank(rank,PALETTE), lw=2) 
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
        set_axes_border(axr, color_for_rank(rank,PALETTE), lw=2)
    plt.tight_layout()
    plt.savefig(ellipse_top4_png, dpi=220)
    plt.close(fig_inside)
    print(f"Wrote top-4 inside-ellipse grid: {ellipse_top4_png}")
else:
    print("No inside-ellipse candidates with successful similarity computation. Skipping ellipse grid.")

gc.collect()
print("Done.")