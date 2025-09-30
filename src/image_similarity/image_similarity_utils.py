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

TARGET_HW = (512, 512)
TOPK_RETRIEVE = 50
TOPK_SHOW = 4     # now show 4; 2×2 grid
timestamp = 1650825687996141824  # nanoseconds

# Distinct, colorblind-friendly colors (we only need 4)
GRID_COLORS = [plt.cm.tab10(i) for i in [0, 1, 2, 3]]


def sift_ransac_inlier_ratio(imgA_gray, imgB_gray, ratio_test=0.75, model="affine"):
    # Detect + describe
    sift = cv.SIFT_create()
    kA, dA = sift.detectAndCompute(imgA_gray, None)
    kB, dB = sift.detectAndCompute(imgB_gray, None)
    if dA is None or dB is None or len(kA)<4 or len(kB)<4:
        return 0.0

    # Match with ratio test
    matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    knn = matcher.knnMatch(dA, dB, k=2)
    good = []
    for m,n in knn:
        if m.distance < ratio_test * n.distance:
            good.append(m)
    if len(good) < 4:
        return 0.0

    # Build coords
    ptsA = np.float32([kA[m.queryIdx].pt for m in good])
    ptsB = np.float32([kB[m.trainIdx].pt for m in good])

    # Fit model
    if model == "homography":
        M, mask = cv.findHomography(ptsA, ptsB, cv.RANSAC, ransacReprojThreshold=3.0)
    else:  # affine (least-squares + RANSAC)
        M, mask = cv.estimateAffine2D(ptsA, ptsB, method=cv.RANSAC, ransacReprojThreshold=3.0)
    if mask is None:
        return 0.0
    inliers = int(mask.sum())
    return inliers / max(len(good), 1)
    
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
        print("load_gray_eq!")
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

import numpy as np

def cos_sims(q: np.ndarray, feats: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarities between a single query vector q (D,)
    or (1, D) and a matrix feats (N, D). Returns (N,) sims.
    """
    q = np.asarray(q, dtype=np.float32)
    F = np.asarray(feats, dtype=np.float32)

    # Ensure shapes are (D,) and (N, D)
    if q.ndim == 2 and q.shape[0] == 1:
        q = q[0]
    elif q.ndim != 1:
        raise ValueError(f"q must be (D,) or (1, D), got {q.shape}")

    if F.ndim != 2:
        raise ValueError(f"feats must be (N, D), got {F.shape}")

    # L2-normalize
    q = q / (np.linalg.norm(q) + 1e-8)
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)

    # Cosine = dot of unit vectors
    return F @ q  # (N,)

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

def mahalanobis_inside(pt_lon, pt_lat, center_lonlat, cov_deg, k2=1.0):
    """
    Return True if (lon,lat) is inside the k-sigma ellipse defined by cov_deg at center.
    k2 is k^2 (1.0 => 1-sigma).
    """
    delta = np.array([pt_lon - center_lonlat[0], pt_lat - center_lonlat[1]], dtype=np.float64)
    try:
        cov_inv = np.linalg.inv(cov_deg)
    except np.linalg.LinAlgError:
        return False
    m2 = float(delta.T @ cov_inv @ delta)
    return m2 <= k2 + 1e-12

def compute_similarity_metrics(may_gray, may_edges, nov_path):
    ng = load_gray_eq(nov_path)
    if ng is None:
        return None
    ng_al = ecc_align(may_gray, ng, "affine")
    gssim = ssim(grad(may_gray), grad(ng_al))
    eN = cv.Canny(grad(ng_al), 50, 150)
    H, W = may_gray.shape[:2]
    ch = chamfer(may_edges, eN) / max(H, W)
    final = 0.7 * gssim - 0.3 * ch
    return (float(gssim), float(ch), float(final))

def parse_cluster_no_by_filename(img_path):
    """
    Extract cluster number from a filename like .../nov_cluster0000_part02.png
    """
    base = os.path.basename(img_path)
    idx1_list = [i for i, ch in enumerate(base) if ch == "_"]
    if len(idx1_list) < 2:
        return None
    idx1 = idx1_list[1]
    idx0 = base.index("cluster") + 8
    try:
        return int(base[idx0:idx1])
    except Exception:
        return None

def set_axes_border(ax, color, lw=2):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(lw)
        sp.set_edgecolor(color)
