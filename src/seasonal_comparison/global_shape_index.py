# global_shape_index.py
import os, json, glob, argparse
from pathlib import Path
import numpy as np, cv2 as cv
from skimage.feature import hog

TARGET_HW = (512, 512)

def letterbox(img, target_hw=TARGET_HW):
    th, tw = target_hw
    h, w = img.shape[:2]
    s = min(tw/w, th/h)
    nw, nh = int(round(w*s)), int(round(h*s))
    r = cv.resize(img, (nw, nh), interpolation=cv.INTER_AREA if s<1 else cv.INTER_LINEAR)
    canvas = np.zeros((th, tw), dtype=r.dtype)
    top = (th-nh)//2; left = (tw-nw)//2
    canvas[top:top+nh, left:left+nw] = r
    return canvas

def load_gray_eq(p):
    im = cv.imread(p, cv.IMREAD_GRAYSCALE)
    if im is None: return None
    im = cv.GaussianBlur(im,(3,3),0)
    clahe = cv.createCLAHE(2.0,(8,8))
    im = clahe.apply(im)
    return letterbox(im)

def gradient_mag(img):
    gx = cv.Scharr(img, cv.CV_32F,1,0)
    gy = cv.Scharr(img, cv.CV_32F,0,1)
    mag = cv.magnitude(gx, gy)
    return cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

def hog_vec(img):
    g = gradient_mag(img)
    v = hog(g, orientations=9, pixels_per_cell=(16,16),
            cells_per_block=(2,2), block_norm="L2-Hys", feature_vector=True)
    n = np.linalg.norm(v) + 1e-9
    return (v/n).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help=".../matches_fused")
    ap.add_argument("--out_npz", required=True)
    args = ap.parse_args()

    nov_paths, vecs = [], []
    root = Path(args.root)
    for ts in sorted(root.iterdir()):
        npdir = ts/"Nov"
        if not npdir.is_dir(): continue
        for p in glob.glob(str(npdir/"*.png")):
            g = load_gray_eq(p)
            if g is None: continue
            nov_paths.append(p)
            vecs.append(hog_vec(g))
    arr = np.stack(vecs, axis=0) if vecs else np.zeros((0,),np.float32)
    np.savez_compressed(args.out_npz, paths=np.array(nov_paths), feats=arr)
    print(f"[OK] indexed {len(nov_paths)} Nov images into {args.out_npz}")

if __name__ == "__main__":
    main()
