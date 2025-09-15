import os, numpy as np, cv2 as cv, pickle, gc
from pathlib import Path
from skimage.feature import hog
from scipy.spatial import KDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Ellipse, Rectangle
from shapely.ops import unary_union

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

# ------------------- CONFIG -------------------
IDX = "/media/kristen/easystore2/RestorebotData/analysis_shape/nov_index.npz"
MAY = "/media/kristen/easystore2/RestorebotData/benchmarking_ex/matches_fused/1650825687996141824/May/may_cluster0000_part02.png"
nov_dir = "/media/kristen/easystore2/RestorebotData/benchmarking_ex/matches_fused/1650825687996141824/Nov"
TARGET_HW = (512, 512)
TOPK_RETRIEVE = 50
TOPK_SHOW = 6  # will show 6 in a 2×3 grid
timestamp = 1650825687996141824 

# 6 distinct, colorblind-friendly colors
TOP6_COLORS = [plt.cm.tab10(i) for i in [0, 1, 2, 3, 4, 5]]

# ------------------- HELPERS -------------------
def letterbox(im, hw=TARGET_HW):
    th, tw = hw
    h, w = im.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    r = cv.resize(im, (nw, nh), interpolation=cv.INTER_AREA if s < 1 else cv.INTER_LINEAR)
    can = np.zeros((th, tw), dtype=r.dtype)
    top = (th - nh) // 2
    left = (tw - nw) // 2
    can[top:top + nh, left:left + nw] = r
    return can

def load_gray_eq(p):
    im = cv.imread(p, cv.IMREAD_GRAYSCALE)
    if im is None:
        return None
    im = cv.GaussianBlur(im, (3, 3), 0)
    im = cv.createCLAHE(2.0, (8, 8)).apply(im)
    return letterbox(im)

def grad(im):
    gx = cv.Scharr(im, cv.CV_32F, 1, 0)
    gy = cv.Scharr(im, cv.CV_32F, 0, 1)
    mag = cv.magnitude(gx, gy)
    return cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

def hog_vec(im):
    v = hog(
        im, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
        block_norm="L2-Hys", visualize=False, feature_vector=True, channel_axis=None
    )
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)

def hog_vec_on_grad(im):
    return hog_vec(grad(im))

def cos_sims(q, M):
    q = q / (np.linalg.norm(q) + 1e-9)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return (M @ q)

def ssim(a, b):
    a = a.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0
    mu1 = cv.GaussianBlur(a, (11, 11), 1.5)
    mu2 = cv.GaussianBlur(b, (11, 11), 1.5)
    mu1sq, mu2sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = cv.GaussianBlur(a * a, (11, 11), 1.5) - mu1sq
    s2 = cv.GaussianBlur(b * b, (11, 11), 1.5) - mu2sq
    s12 = cv.GaussianBlur(a * b, (11, 11), 1.5) - mu12
    C1, C2 = 0.01**2, 0.03**2
    num = (2 * mu12 + C1) * (2 * s12 + C2)
    den = (mu1sq + mu2sq + C1) * (s1 + s2 + C2) + 1e-12
    return float((num / den).mean())

def chamfer(e1, e2):
    pts1 = np.argwhere(e1 > 0)
    pts2 = np.argwhere(e2 > 0)
    if len(pts1) == 0 or len(pts2) == 0:
        return float("inf")
    tree1 = KDTree(pts1)
    tree2 = KDTree(pts2)
    d12, _ = tree2.query(pts1)
    d21, _ = tree1.query(pts2)
    return float(d12.sum() + d21.sum())

def ecc_align(ref, mov, mode="affine"):
    warp = cv.MOTION_AFFINE if mode == "affine" else cv.MOTION_HOMOGRAPHY
    W = np.eye(2, 3) if warp == cv.MOTION_AFFINE else np.eye(3)
    try:
        _, W = cv.findTransformECC(ref, mov, W, warp, (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 100, 1e-5))
        if warp == cv.MOTION_AFFINE:
            return cv.warpAffine(mov, W, (ref.shape[1], ref.shape[0]), flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP)
        else:
            return cv.warpPerspective(mov, W, (ref.shape[1], ref.shape[0]), flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP)
    except cv.error:
        return mov

def set_axes_border(ax, color, lw=2):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(lw)
        sp.set_edgecolor(color)

# ------------------- MAIN -------------------
# Load index and build candidates
print("loading index and building candidates ...")
data = np.load(IDX, allow_pickle=True)
paths, feats = data["paths"], data["feats"]

mg = load_gray_eq(MAY)
assert mg is not None and mg.ndim == 2, f"bad image at {MAY}"
q = hog_vec(mg)  # (or: hog_vec_on_grad(mg))

sims = cos_sims(q, feats)
top = np.argsort(-sims)[:TOPK_RETRIEVE]

mgG = grad(mg)
eM = cv.Canny(mgG, 50, 150)
H, W = mg.shape[:2]

print("appending ecc_align, ssim, chamfer etc")
rows = []  # (nov_path, coarse_sim, grad_ssim, chamfer_norm, final)
for i in top:
    npth = str(paths[i])
    ng = load_gray_eq(npth)
    if ng is None:
        continue
    ng_al = ecc_align(mg, ng, "affine")
    gssim = ssim(grad(mg), grad(ng_al))
    eN = cv.Canny(grad(ng_al), 50, 150)
    ch = chamfer(eM, eN) / max(H, W)
    final = 0.7 * gssim - 0.3 * ch
    print(f"sims[i]: {sims[i]}, gssim: {gssim}, ch: {ch}, final: {final}")
    rows.append((npth, float(sims[i]), gssim, ch, final))

# Take top-6 for display
top_rows = sorted(rows, key=lambda x: -x[4])[:TOPK_SHOW]
print("Found the top rows!") 

# ------------------- FIGURE -------------------
print("Doing the visualization ....")
fig = plt.figure(figsize=(16, 6), dpi=150)
gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 1.1, 1.4], wspace=0.06)

# Left: May fused image
ax0 = fig.add_subplot(gs[0])
may_img = cv.cvtColor(cv.imread(MAY), cv.COLOR_BGR2RGB)
ax0.imshow(may_img); ax0.axis("off")
ax0.set_title(f"May fused image\n{os.path.basename(MAY)}", fontsize=10)

# Center: GPS plot (draw polygons)
ax1 = fig.add_subplot(gs[1])

may_dir = os.path.dirname(MAY)
out_png = os.path.join(may_dir, "best_match_comparison_fig.png")

# Load May polygons
may_filename = os.path.basename(MAY)
idx1 = [i for i, ch in enumerate(may_filename) if ch == "_"][1]
idx0 = may_filename.index("cluster") + 8
may_cluster_no = int(may_filename[idx0:idx1])
may_pano_pickle = os.path.join(may_dir, f"may_panos_{may_cluster_no}.pickle")

with open(may_pano_pickle, "rb") as f:
    may_pano = pickle.load(f)


if MAY not in may_pano:  
    print("MAY:", MAY)  
    MAY = MAY.replace("easystore2", "easystore3")
    if MAY not in may_pano:
        print("MAY:",MAY)
        print("may_pano:",may_pano)
        raise FileNotFoundError(f"May pano entry not found for key {MAY}")

may_polygon_list = create_polygons(may_pano[MAY])
may_overlap_subset = find_largest_overlap_subset(may_polygon_list)

for poly in may_overlap_subset:
    coords = list(poly.exterior.coords)[:-1]
    oc = fix_invalid_corners(reorder_corners(coords))
    ax0.add_patch(MplPolygon(oc, fill=True, facecolor="blue", alpha=0.25, edgecolor="blue", linewidth=1.0))
    ax1.add_patch(MplPolygon(oc, fill=True, facecolor="blue", alpha=0.25, edgecolor="blue", linewidth=1.0))

# Draw top-6 Nov polygons, colored 1..6
print("drawing the Nov polygons...")
legend_handles = []
master_polygon_list = []
for idx, (nov_path, coarse, gssim_val, ch_val, final) in enumerate(top_rows):
    color = TOP6_COLORS[idx]
    nov_filename = os.path.basename(nov_path)
    nov_dir = os.path.dirname(nov_path)
    nidx1 = [k for k, ch in enumerate(nov_filename) if ch == "_"][1]
    nidx0 = nov_filename.index("cluster") + 8
    nov_cluster_no = int(nov_filename[nidx0:nidx1])
    nov_pano_pickle = os.path.join(nov_dir, f"nov_panos_{nov_cluster_no}.pickle")

    with open(nov_pano_pickle, "rb") as f:
        nov_pano = pickle.load(f)
    nov_polygon_list = create_polygons(nov_pano[nov_path])
    master_polygon_list.extend(nov_polygon_list)

    nov_subset = find_largest_overlap_subset(nov_polygon_list)
    for poly in nov_subset:
        coords = list(poly.exterior.coords)[:-1]
        oc = fix_invalid_corners(reorder_corners(coords))
        ax0.add_patch(MplPolygon(oc, fill=True, facecolor=color, alpha=0.25, edgecolor=color, linewidth=1.0))
        ax1.add_patch(MplPolygon(oc, fill=True, facecolor=color, alpha=0.25, edgecolor=color, linewidth=1.0))

    legend_handles.append(mpatches.Patch(facecolor=color, edgecolor=color, alpha=0.7,
                                         label=f"{idx+1}. {os.path.basename(nov_path)}"))

may_cov_dir = "/media/kristen/easystore2/RestorebotData/benchmarking_ex/May2022/1conmod/processed_results/covariance_matrices"
may_cov_timestamps = precompute_timestamps(may_cov_dir)
may_results_dir = "/media/kristen/easystore2/RestorebotData/benchmarking_ex/May2022/1conmod/processed_results"
may_data = np.genfromtxt(
            os.path.join(may_results_dir, "frustrum_corners.csv"),
            delimiter=",",
            skip_header=1)
may_timestamps = may_data[:, 0] 
cam_lat, cam_lon = may_data[:, 15], may_data[:, 16]
#want to add the covariance ellipse 
cov_matrix = load_covariance_matrix(may_cov_dir, may_cov_timestamps, timestamp)
scaled_cov = scale_covariance_to_degrees(cov_matrix) 
#convert timestamp into nanoseconds 
i_may = find_closest_index(may_timestamps, timestamp*10**(-9), max_delta_t=0.3)
center = (cam_lon[i_may], cam_lat[i_may])  

vals, vecs = np.linalg.eigh(scaled_cov)
order = vals.argsort()[::-1]
vals = vals[order]
vecs = vecs[:, order]
# Calculate angle of ellipse rotation (in degrees)
theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

# Width and height of the ellipse (2*stddevs)
width, height = 2 * np.sqrt(vals)

ellipse = Ellipse(
    xy=center,
    width=width,
    height=height,
    #angle=90-theta,
    angle=theta,
    edgecolor='blue',
    facecolor='none',
    linestyle='dotted'  # or use '--' for dashed
) 

ax1.add_patch(ellipse)

#add fused November polygons that fall in the covariance ellipse 
for f in os.listdir(nov_dir):
    if "pickle" == f[-5:]:
        with open(f,"rb") as handle:
            pano_dict = pickle.load(handle)
        #this pano dict contains the filenames that were fused to make the stitched image  
        for k in pano_dict:
            polygons_k = pano_dict[k] 
            for nov_polygon in polygons_k:
                x, y = nov_polygon.exterior.xy  # get boundary coordinates
                ax1.fill(x, y, color='red', alpha=0.1)  

#TODO want to find the image similarity of the frames actually in the ellipse

# GPS bounds/aspect
if may_overlap_subset:
    may_merged = unary_union(may_polygon_list)
else:
    raise RuntimeError("Empty May overlap subset.")
if master_polygon_list:
    nov_merged = unary_union(master_polygon_list)
else:
    raise RuntimeError("Empty Nov polygon list.")

min_lon = min(may_merged.bounds[0], nov_merged.bounds[0])
min_lat = min(may_merged.bounds[1], nov_merged.bounds[1])
max_lon = max(may_merged.bounds[2], nov_merged.bounds[2])
max_lat = max(may_merged.bounds[3], nov_merged.bounds[3])

ax1.set_xlim(min_lon, max_lon); ax1.set_ylim(min_lat, max_lat)
aspect_correction = np.cos(np.deg2rad((min_lat + max_lat) / 2))
ax1.set_aspect(aspect_correction)
ax1.set_xlabel("Longitude"); ax1.set_ylabel("Latitude")
ax1.set_title("Down-facing camera frames", fontsize=10)
ax1.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.25),
           ncol=2, fontsize=7, frameon=False)

# Right: 2×3 grid of top-6 Nov thumbnails, with matching colored borders
ax_right_container = fig.add_subplot(gs[2]); ax_right_container.axis("off")
R, C = 3, 2
right_gs = gridspec.GridSpecFromSubplotSpec(R, C, subplot_spec=gs[2], wspace=0.06, hspace=0.08)

for idx, (nov_path, coarse, gssim_val, ch_val, final) in enumerate(top_rows):
    r, c = divmod(idx, C)
    axr = fig.add_subplot(right_gs[r, c])
    nimg = cv.imread(nov_path, cv.IMREAD_COLOR)
    if nimg is None:
        axr.text(0.5, 0.5, f"Missing:\n{os.path.basename(nov_path)}", ha="center", va="center", fontsize=8)
        axr.axis("off")
        continue
    nimg = cv.cvtColor(nimg, cv.COLOR_BGR2RGB)
    axr.imshow(nimg); axr.axis("off")
    color = TOP6_COLORS[idx]
    axr.set_title(f"{idx+1}. final={final:.3f}  SSIM={gssim_val:.3f}", fontsize=8)
    # colored border to match GPS polygons
    for sp in axr.spines.values():
        sp.set_visible(True); sp.set_linewidth(2); sp.set_edgecolor(color)

plt.tight_layout(rect=[0, 0, 1, 0.95])
print(f"Writing {out_png}")
plt.savefig(out_png, dpi=220)
plt.close(fig); gc.collect()
