#!/usr/bin/env python3
import os, csv, math, argparse
import h5py
import numpy as np

# headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2 as cv
cv.ocl.setUseOpenCL(False)
cv.setNumThreads(1)

def find_pair_groups(fm: h5py.File):
    """
    Recursively find groups that look like per-pair matches.
    We detect by presence of one of: 'matches0', 'matches', or ('keypoints0' and 'keypoints1').
    Returns list of (group_path, group_obj).
    """
    pairs = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Group):
            keys = obj.keys()
            if ("matches0" in keys) or ("matches" in keys) or ({"keypoints0","keypoints1"} <= set(keys)):
                pairs.append((name, obj))
    fm.visititems(visitor)
    return pairs

def get_images_for_group(name: str, g: h5py.Group):
    # Prefer attrs set by HLOC
    img0 = g.attrs.get("image0")
    img1 = g.attrs.get("image1")
    if isinstance(img0, bytes): img0 = img0.decode()
    if isinstance(img1, bytes): img1 = img1.decode()
    # Fallbacks: try parsing the group name "img0 img1"
    if not img0 or not img1:
        if " " in name:
            a, b = name.split(" ", 1); img0 = img0 or a; img1 = img1 or b
        elif "," in name:
            a, b = name.split(",", 1); img0 = img0 or a; img1 = img1 or b
    return str(img0 or ""), str(img1 or "")

def load_keypoints(ff: h5py.File, key: str):
    grp = ff.get(key)
    if grp is None or "keypoints" not in grp:
        return None
    return np.asarray(grp["keypoints"], dtype=np.float32)

def build_correspondences(g: h5py.Group, feats: h5py.File, img0: str, img1: str):
    """
    Return pts0, pts1, scores (or None if unavailable).
    Supports:
      - SuperGlue: matches0 (len N0), matching_scores0; keypoints via features or in-group
      - LoFTR: matches (Nx2) + keypoints0/1 (+ optional scores)
      - Direct coords: keypoints0/1 (+ optional scores), no indices
    """
    if "matches0" in g:
        m0 = np.asarray(g["matches0"], dtype=np.int64)
        ok = m0 > -1
        if not np.any(ok):
            # zero correspondences
            # still try to load kpts for consistent shapes
            k0 = np.asarray(g["keypoints0"], dtype=np.float32) if "keypoints0" in g else load_keypoints(feats, img0)
            return (np.zeros((0,2), np.float32), np.zeros((0,2), np.float32),
                    np.zeros((0,), np.float32))
        idx0 = np.where(ok)[0]
        idx1 = m0[ok]
        k0 = np.asarray(g["keypoints0"], dtype=np.float32) if "keypoints0" in g else load_keypoints(feats, img0)
        k1 = np.asarray(g["keypoints1"], dtype=np.float32) if "keypoints1" in g else load_keypoints(feats, img1)
        if k0 is None or k1 is None:  # cannot resolve
            return None, None, None
        pts0 = k0[idx0]; pts1 = k1[idx1]
        scores = None
        if "matching_scores0" in g:
            s0 = np.asarray(g["matching_scores0"], dtype=np.float32)
            scores = s0[ok]
        return pts0, pts1, scores

    if "matches" in g:
        M = np.asarray(g["matches"])
        if M.ndim == 2 and M.shape[1] == 2:
            # index-based correspondences (LoFTR, etc.)
            k0 = np.asarray(g["keypoints0"], dtype=np.float32) if "keypoints0" in g else load_keypoints(feats, img0)
            k1 = np.asarray(g["keypoints1"], dtype=np.float32) if "keypoints1" in g else load_keypoints(feats, img1)
            if k0 is None or k1 is None or len(M) == 0:
                return (np.zeros((0,2), np.float32), np.zeros((0,2), np.float32),
                        np.zeros((0,), np.float32))
            pts0 = k0[M[:,0]]; pts1 = k1[M[:,1]]
        else:
            # sometimes stored directly as coords in keypoints0/1 with 1:1 rows
            k0 = np.asarray(g.get("keypoints0", []), dtype=np.float32)
            k1 = np.asarray(g.get("keypoints1", []), dtype=np.float32)
            n = min(len(k0), len(k1))
            pts0 = k0[:n]; pts1 = k1[:n]
        scores = np.asarray(g["scores"], dtype=np.float32).reshape(-1)[:len(pts0)] if "scores" in g else None
        return pts0, pts1, scores

    # pure keypoints0/1 with implicit 1:1
    if "keypoints0" in g and "keypoints1" in g:
        k0 = np.asarray(g["keypoints0"], dtype=np.float32)
        k1 = np.asarray(g["keypoints1"], dtype=np.float32)
        n = min(len(k0), len(k1))
        return k0[:n], k1[:n], np.asarray(g.get("scores", []), dtype=np.float32).reshape(-1)[:n] if "scores" in g else None

    return None, None, None

def geo_verify(pts0, pts1, model="fundamental", reproj=1.0, conf=0.999):
    if pts0 is None or pts1 is None or len(pts0) < 8:
        return np.zeros((0,), dtype=bool)
    if model == "homography":
        H, mask = cv.findHomography(pts0, pts1, cv.RANSAC, ransacReprojThreshold=reproj, confidence=conf)
    else:
        F, mask = cv.findFundamentalMat(pts0, pts1, cv.FM_RANSAC, ransacReprojThreshold=reproj, confidence=conf)
    if mask is None:
        return np.zeros((len(pts0),), dtype=bool)
    return mask.astype(bool).reshape(-1)

def save_hist(data, out_dir, fname, xlabel, title="", bins=50):
    data = np.asarray(data)
    if data.size == 0: return
    plt.figure()
    plt.hist(data[~np.isnan(data)], bins=bins)
    plt.xlabel(xlabel); plt.ylabel("count")
    if title: plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--matches",  required=True)
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--ransac",   choices=["fundamental", "homography"], default="fundamental")
    ap.add_argument("--include_zero", action="store_true",
                    help="include pairs with 0 matches in the CSV (n_matches=0)")
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    with h5py.File(args.features, "r") as ff, h5py.File(args.matches, "r") as fm:
        pair_groups = find_pair_groups(fm)
        print(f"[INFO] found {len(pair_groups)} pair groups in matches H5")

        for grp_name, g in pair_groups:
            img0, img1 = get_images_for_group(grp_name, g)
            pts0, pts1, scores = build_correspondences(g, ff, img0, img1)

            if pts0 is None:
                continue

            n = len(pts0)
            if n == 0 and not args.include_zero:
                continue

            inliers_mask = geo_verify(pts0, pts1, model=args.ransac)
            n_in = int(inliers_mask.sum()) if inliers_mask.size else 0
            inlier_ratio = (n_in / n) if n > 0 else 0.0
            mean_conf = (float(np.mean(scores)) if scores is not None and len(scores) else math.nan)

            rows.append({
                "pair_group": grp_name,
                "img0": img0, "img1": img1,
                "n_matches": n,
                "n_inliers": n_in,
                "inlier_ratio": inlier_ratio,
                "mean_conf": mean_conf,
            })

    # Always write a CSV header even if empty, for visibility.
    csv_path = os.path.join(args.out_dir, "pair_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        headers = ["pair_group","img0","img1","n_matches","n_inliers","inlier_ratio","mean_conf"]
        w = csv.DictWriter(f, fieldnames=headers); w.writeheader()
        for r in rows: w.writerow(r)

    # Histograms
    save_hist([r["n_matches"] for r in rows], args.out_dir, "hist_n_matches.png", "# correspondences", args.title)
    save_hist([r["n_inliers"] for r in rows], args.out_dir, "hist_n_inliers.png", "# inliers (RANSAC)", args.title)
    save_hist([r["inlier_ratio"] for r in rows], args.out_dir, "hist_inlier_ratio.png", "inlier ratio", args.title)
    if any(not math.isnan(r["mean_conf"]) for r in rows):
        save_hist([r["mean_conf"] for r in rows], args.out_dir, "hist_mean_conf.png", "mean match confidence", args.title)

    print(f"[OK] wrote {csv_path} with {len(rows)} rows; hist_*.png in {args.out_dir}")

if __name__ == "__main__":
    main()
