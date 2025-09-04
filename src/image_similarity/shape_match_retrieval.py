#!/usr/bin/env python3
import os, csv, math, glob, argparse
from pathlib import Path
import numpy as np
import cv2 as cv
from skimage.feature import hog

TARGET_HW = (640,640)

# ---------- utils ----------
def letterbox(img,hw_target):
    """Resize with aspect ratio preserved and pad to target size (H, W)."""
    th, tw = hw_target
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv.resize(img, (nw, nh), interpolation=cv.INTER_AREA if scale < 1 else cv.INTER_LINEAR)
    canvas = np.zeros((th, tw), dtype=resized.dtype)  # black pad
    top  = (th - nh) // 2
    left = (tw - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas

def load_gray_eq(p):
    img = cv.imread(p, cv.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv.GaussianBlur(img, (3,3), 0)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = letterbox(img, TARGET_HW)  # << ensure fixed size here
    return img

def gradient_mag(img):
    gx = cv.Scharr(img, cv.CV_32F, 1, 0)
    gy = cv.Scharr(img, cv.CV_32F, 0, 1)
    mag = cv.magnitude(gx, gy)
    mag = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    return mag

def ssim_simple(a, b):
    # grayscale SSIM (vectorized, Gaussian window approx)
    a = a.astype(np.float32) / 255.0; b = b.astype(np.float32) / 255.0
    ksize, sigma = (11,11), 1.5
    mu1 = cv.GaussianBlur(a, ksize, sigma); mu2 = cv.GaussianBlur(b, ksize, sigma)
    mu1_sq, mu2_sq, mu12 = mu1*mu1, mu2*mu2, mu1*mu2
    sigma1_sq = cv.GaussianBlur(a*a, ksize, sigma) - mu1_sq
    sigma2_sq = cv.GaussianBlur(b*b, ksize, sigma) - mu2_sq
    sigma12   = cv.GaussianBlur(a*b, ksize, sigma) - mu12
    C1, C2 = 0.01**2, 0.03**2
    num = (2*mu12 + C1)*(2*sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2) + 1e-12
    return float((num/den).mean())

def chamfer_distance(a_edges, b_edges):
    # distance from A edges to B edges and vice versa → symmetric
    da = cv.distanceTransform(255 - a_edges, cv.DIST_L2, 3)
    db = cv.distanceTransform(255 - b_edges, cv.DIST_L2, 3)
    a_to_b = da[a_edges==255].mean() if np.any(a_edges==255) else 1e6
    b_to_a = db[b_edges==255].mean() if np.any(b_edges==255) else 1e6
    return float((a_to_b + b_to_a) * 0.5)

def ecc_align(ref, mov, motion="affine", iters=100, eps=1e-5):
    warp_mode = cv.MOTION_AFFINE if motion=="affine" else cv.MOTION_HOMOGRAPHY
    W = np.eye(2,3, dtype=np.float32) if warp_mode==cv.MOTION_AFFINE else np.eye(3, dtype=np.float32)
    try:
        _, W = cv.findTransformECC(ref, mov, W, warp_mode,
                                   (cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT, iters, eps))
        if warp_mode==cv.MOTION_AFFINE:
            out = cv.warpAffine(mov, W, (ref.shape[1], ref.shape[0]),
                                flags=cv.INTER_LINEAR|cv.WARP_INVERSE_MAP)
        else:
            out = cv.warpPerspective(mov, W, (ref.shape[1], ref.shape[0]),
                                     flags=cv.INTER_LINEAR|cv.WARP_INVERSE_MAP)
        return out
    except cv.error:
        return mov.copy()

def hog_vec(img):
    # img should already be letterboxed to TARGET_HW
    g = gradient_mag(img)
    vec = hog(
        g, orientations=9,
        pixels_per_cell=(16,16),
        cells_per_block=(2,2),
        block_norm="L2-Hys",
        feature_vector=True
    )
    n = np.linalg.norm(vec) + 1e-9
    return (vec / n).astype(np.float32)

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a)+1e-9) / (np.linalg.norm(b)+1e-9))

# ---------- main ----------
def process_timestamp_dir(ts_dir: Path, out_csv, down=0.5, motion="affine", topk=5):
    print("processing dir: ",ts_dir) 

    may_dir = ts_dir / "May"
    nov_dir = ts_dir / "Nov"
    mpaths = sorted(glob.glob(str(may_dir / "*.png")))
    npaths = sorted(glob.glob(str(nov_dir / "*.png")))
    if not mpaths or not npaths:
        return [], []

    # Precompute HOG for Nov
    nov_items = []
    for p in npaths:
        g = load_gray_eq(p)
        if g is None: continue
        v = hog_vec(g)
        nov_items.append((p, g, v))
    if not nov_items: return [], []

    rows = []
    final_scores = []
    npaths = len(mpaths) 
    
    for i,mp in enumerate(mpaths):
        print(f"Processing {i} of {npaths} paths...")
        mg = load_gray_eq(mp)
        if mg is None: continue
        mv = hog_vec(mg)

        # coarse search
        sims = [(cosine_sim(mv, v), p, g) for (p, g, v) in nov_items]
        sims.sort(reverse=True, key=lambda x: x[0])
        cand = sims[:topk]

        # re-rank with alignment + grad-SSIM + chamfer
        best = None
        best_score = -1e9
        best_metrics = (0.0, 0.0, 0.0)
        mg_aln_ref = mg  # reference

        m_edges = cv.Canny(gradient_mag(mg), 50, 150)
        for sim, npth, ng in cand:
            ng_al = ecc_align(mg, ng, motion=motion)     # both H×W
            gssim = ssim_simple(gradient_mag(mg), gradient_mag(ng_al))
            n_edges = cv.Canny(gradient_mag(ng_al), 50, 150)
            ch = chamfer_distance(m_edges, n_edges)
            # normalize chamfer relative to image size
            h, w = mg.shape[:2]
            ch_norm = ch / max(h, w)
            score = 0.7*gssim - 0.3*ch_norm
            if score > best_score:
                best_score = score
                best = (npth, gssim, ch_norm, sim)

        if best is None:
            continue
        npth, gssim, ch_norm, coarse = best
        rows.append({
            "timestamp_dir": ts_dir.name,
            "may": mp,
            "nov_best": npth,
            "grad_ssim": gssim,
            "chamfer_norm": ch_norm,
            "coarse_hog_cosine": coarse,
            "final_score": best_score
        })
        final_scores.append(best_score)

    # write/append
    writer = csv.DictWriter(out_csv, fieldnames=list(rows[0].keys()) if rows else
                            ["timestamp_dir","may","nov_best","grad_ssim","chamfer_norm","coarse_hog_cosine","final_score"])
    if out_csv.tell() == 0:
        writer.writeheader()
    for r in rows:
        print(f"row: {r}")
        writer.writerow(r)

    return [r["grad_ssim"] for r in rows], final_scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help=".../benchmarking_ex/matches_fused")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--down", type=float, default=0.5)
    ap.add_argument("--motion", choices=["affine","homography"], default="affine")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "best_matches.csv")
    hist1 = os.path.join(args.out_dir, "hist_grad_ssim.png")
    hist2 = os.path.join(args.out_dir, "hist_final.png")

    grad_vals, final_vals = [], []
    with open(csv_path, "w", newline="") as fcsv:
        for ts in sorted(Path(args.root).iterdir()):
            print("ts:",ts)
            if not ts.is_dir(): continue
            if not (ts / "May").exists() or not (ts / "Nov").exists(): continue
            g, s = process_timestamp_dir(ts, fcsv, down=args.down, motion=args.motion, topk=args.topk)
            grad_vals += g; final_vals += s

    # histograms
    if grad_vals:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(); plt.hist(grad_vals, bins=50); plt.xlabel("Grad-SSIM"); plt.ylabel("count"); plt.tight_layout(); plt.savefig(hist1, dpi=200); plt.close()
    if final_vals:
        import matplotlib.pyplot as plt
        plt.figure(); plt.hist(final_vals, bins=50); plt.xlabel("Final score (0.7*grad-SSIM - 0.3*Chamfer)"); plt.ylabel("count"); plt.tight_layout(); plt.savefig(hist2, dpi=200); plt.close()

    print(f"[OK] wrote {csv_path}")
    if grad_vals: print(f"[OK] wrote {hist1}")
    if final_vals: print(f"[OK] wrote {hist2}")

if __name__ == "__main__":
    main()
