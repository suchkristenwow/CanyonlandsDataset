# query_best_match.py
import argparse, numpy as np, cv2 as cv
from skimage.feature import hog
from pathlib import Path

TARGET_HW=(512,512)

def letterbox(img, target_hw=TARGET_HW):
    th, tw = target_hw
    h, w = img.shape[:2]
    s = min(tw/w, th/h)
    nw, nh = int(round(w*s)), int(round(h*s))
    r = cv.resize(img, (nw, nh), interpolation=cv.INTER_AREA if s<1 else cv.INTER_LINEAR)
    canvas = np.zeros((th, tw), dtype=r.dtype); top=(th-nh)//2; left=(tw-nw)//2
    canvas[top:top+nh, left:left+nw]=r; return canvas

def load_gray_eq(p):
    im = cv.imread(p, cv.IMREAD_GRAYSCALE)
    if im is None: return None
    im = cv.GaussianBlur(im,(3,3),0)
    clahe = cv.createCLAHE(2.0,(8,8))
    im = clahe.apply(im)
    return letterbox(im)

def gradient_mag(img):
    gx = cv.Scharr(img, cv.CV_32F,1,0); gy = cv.Scharr(img, cv.CV_32F,0,1)
    mag = cv.magnitude(gx, gy)
    return cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

def hog_vec(img):
    g = gradient_mag(img)
    v = hog(g, orientations=9, pixels_per_cell=(16,16),
            cells_per_block=(2,2), block_norm="L2-Hys", feature_vector=True)
    n = np.linalg.norm(v)+1e-9
    return (v/n).astype(np.float32)

def cos_sims(q, M):  # q: (D,), M: (N,D)
    qn = q / (np.linalg.norm(q)+1e-9)
    Mn = M / (np.linalg.norm(M,axis=1,keepdims=True)+1e-9)
    return (Mn @ qn)

def ssim_simple(a,b):
    a = a.astype(np.float32)/255.; b = b.astype(np.float32)/255.
    mu1 = cv.GaussianBlur(a,(11,11),1.5); mu2 = cv.GaussianBlur(b,(11,11),1.5)
    mu1_sq, mu2_sq, mu12 = mu1*mu1, mu2*mu2, mu1*mu2
    sigma1_sq = cv.GaussianBlur(a*a,(11,11),1.5)-mu1_sq
    sigma2_sq = cv.GaussianBlur(b*b,(11,11),1.5)-mu2_sq
    sigma12   = cv.GaussianBlur(a*b,(11,11),1.5)-mu12
    C1,C2 = 0.01**2, 0.03**2
    num=(2*mu12+C1)*(2*sigma12+C2); den=(mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2)+1e-12
    return float((num/den).mean())

def chamfer(a_edges, b_edges):
    da = cv.distanceTransform(255-a_edges, cv.DIST_L2, 3)
    db = cv.distanceTransform(255-b_edges, cv.DIST_L2, 3)
    a2b = da[a_edges==255].mean() if np.any(a_edges==255) else 1e6
    b2a = db[b_edges==255].mean() if np.any(b_edges==255) else 1e6
    return float(0.5*(a2b+b2a))

def ecc_align(ref, mov, motion="affine"):
    warp = cv.MOTION_AFFINE if motion=="affine" else cv.MOTION_HOMOGRAPHY
    W = np.eye(2,3) if warp==cv.MOTION_AFFINE else np.eye(3)
    try:
        _, W = cv.findTransformECC(ref, mov, W, warp, (cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT, 100, 1e-5))
        if warp==cv.MOTION_AFFINE:
            return cv.warpAffine(mov, W, (ref.shape[1],ref.shape[0]), flags=cv.INTER_LINEAR|cv.WARP_INVERSE_MAP)
        else:
            return cv.warpPerspective(mov, W, (ref.shape[1],ref.shape[0]), flags=cv.INTER_LINEAR|cv.WARP_INVERSE_MAP)
    except cv.error:
        return mov

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_npz", required=True)
    ap.add_argument("--may_image", required=True)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--motion", choices=["affine","homography"], default="affine")
    args = ap.parse_args()

    data = np.load(args.index_npz, allow_pickle=True)
    paths = data["paths"]; feats = data["feats"]  # feats: (N,D)
    mg = load_gray_eq(args.may_image); mv = hog_vec(mg)

    sims = cos_sims(mv, feats)  # (N,)
    top = np.argsort(-sims)[:args.topk]

    # re-rank with alignment + grad SSIM + chamfer
    mg_grad = gradient_mag(mg); m_edges = cv.Canny(mg_grad,50,150)
    best = None; best_score = -1e9
    H, W = mg.shape[:2]
    for idx in top:
        npth = str(paths[idx])
        ng = load_gray_eq(npth)
        ng_al = ecc_align(mg, ng, motion=args.motion)
        gssim = ssim_simple(gradient_mag(mg), gradient_mag(ng_al))
        n_edges = cv.Canny(gradient_mag(ng_al), 50, 150)
        ch = chamfer(m_edges, n_edges) / max(H,W)
        final = 0.7*gssim - 0.3*ch
        if final > best_score:
            best_score = final; best = (npth, float(sims[idx]), gssim, ch, final)

    if best is None:
        print("No candidate could be evaluated.")
        return
    npth, coarse, gssim, ch, final = best
    print("Best match:")
    print("  May:", args.may_image)
    print("  Nov:", npth)
    print(f"  coarse_cosine={coarse:.3f}  grad_ssim={gssim:.3f}  chamfer_norm={ch:.3f}  final={final:.3f}")

if __name__ == "__main__":
    main()
