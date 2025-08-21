import cv2 as cv, numpy as np, pandas as pd
from skimage.metrics import structural_similarity as ssim
from pathlib import Path

def gradient_mag(gray):
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    return cv.magnitude(gx, gy)

def sift_pair_score(imgA, imgB):
    # 1) keypoints + matches
    sift = cv.SIFT_create()                 # OR: cv.ORB_create(nfeatures=2000)
    kA, dA = sift.detectAndCompute(imgA, None)
    kB, dB = sift.detectAndCompute(imgB, None)
    if dA is None or dB is None: return None

    matcher = cv.BFMatcher(cv.NORM_L2)      # NORM_HAMMING if ORB
    raw = matcher.knnMatch(dA, dB, k=2)
    good = [m for m,n in raw if m.distance < 0.75*n.distance]
    if len(good) < 8: return None

    ptsA = np.float32([kA[m.queryIdx].pt for m in good])
    ptsB = np.float32([kB[m.trainIdx].pt for m in good])
    H, inliers = cv.findHomography(ptsB, ptsA, cv.RANSAC, 3.0)
    if H is None: return None
    inliers = inliers.ravel().astype(bool)
    inlier_ratio = inliers.mean()

    # reprojection error
    ptsB1 = cv.convertPointsToHomogeneous(ptsB[:,None,:])[:,0,:]
    proj = (H @ ptsB1.T); proj = (proj[:2]/proj[2]).T
    reproj = np.linalg.norm(proj - ptsA, axis=1)
    mean_reproj = float(np.mean(reproj[inliers])) if inliers.any() else np.inf

    # 2) warp + structural SSIM on gradients
    h,w = imgA.shape[:2]
    warpedB = cv.warpPerspective(imgB, H, (w,h))
    GA = gradient_mag(cv.cvtColor(imgA, cv.COLOR_BGR2GRAY))
    GB = gradient_mag(cv.cvtColor(warpedB, cv.COLOR_BGR2GRAY))

    # define overlapping mask (valid warp region)
    mask = (cv.warpPerspective(np.ones((h,w), np.uint8)*255, H, (w,h))>0)
    mask &= (GA>0) & (GB>0)
    if mask.sum() < 0.3*h*w: return None   # overlap gate ~30%

    # SSIM on gradient images (normalize)
    GA_n = (GA - GA.min())/(GA.ptp()+1e-6)
    GB_n = (GB - GB.min())/(GB.ptp()+1e-6)
    ssim_struct = ssim(GA_n, GB_n, data_range=1.0, gaussian_weights=True, use_sample_covariance=False)

    # 3) composite score
    norm_reproj = min(1.0, mean_reproj/5.0)  # 0 px→0, ≥5 px→1
    score = 0.4*inlier_ratio + 0.2*(1 - norm_reproj) + 0.4*ssim_struct

    return dict(n_matches=len(raw), n_inliers=int(inliers.sum()),
                inlier_ratio=float(inlier_ratio),
                mean_reproj_err_px=float(mean_reproj),
                overlap_pct=float(mask.mean()),
                SSIM_struct=float(ssim_struct),
                score=float(score))
