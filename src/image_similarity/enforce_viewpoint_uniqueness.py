from typing import Optional, Set
from functools import lru_cache
import numpy as np
import cv2 as cv
from pathlib import Path

# --- helpers for diversity ---
def _cosine(u, v):
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu == 0 or nv == 0: return 0.0
    return float(np.dot(u, v) / (nu * nv))

@lru_cache(maxsize=4096)
def _ahash(path: str) -> int:
    """Simple average hash (8x8) → 64-bit int."""
    im = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if im is None:
        return -1
    im = cv.resize(im, (8, 8), interpolation=cv.INTER_AREA)
    avg = im.mean()
    bits = (im > avg).astype(np.uint8).flatten()
    # pack to 64-bit
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h

def _hamming64(a: int, b: int) -> int:
    if a < 0 or b < 0: return 64
    x = a ^ b
    # builtin popcount if available via Python 3.8+ (int.bit_count)
    return x.bit_count()

def build_feat_map(paths_arr: np.ndarray, feats_arr: np.ndarray) -> dict:
    """Map nov_path -> feature vector for cosine comparisons."""
    # paths_arr is typically dtype=object of strings from your npz index
    return {str(p): feats_arr[i] for i, p in enumerate(paths_arr)}

def diverse_rerank(
    top_rows: list,
    k: int,
    paths_arr: Optional[np.ndarray] = None,
    feats_arr: Optional[np.ndarray] = None,
    max_per_cluster: int = 1,
    hash_hamming_thresh: int = 6,          # <=6 usually means visually identical/near-dup
    max_cosine_sim: float = 0.985,         # skip if too similar to any already selected
) -> list:
    """
    top_rows: [(rank, nov_path, coarse, ssim, chamfer, final), ...] sorted best->worst
    returns: top-K with enforced diversity.
    """
    selected = []
    seen_clusters: dict[int,int] = {}
    seen_hashes: list[int] = []
    feat_map = None
    if paths_arr is not None and feats_arr is not None:
        feat_map = build_feat_map(paths_arr, feats_arr)

    for row in top_rows:
        if len(selected) >= k: break
        _, nov_path, *_ = row

        # 1) cluster diversity
        try:
            cid = parse_cluster_number(nov_path)
        except Exception:
            cid = None
        if cid is not None and seen_clusters.get(cid, 0) >= max_per_cluster:
            continue

        # 2) perceptual hash near-dup check
        h = _ahash(nov_path)
        is_near_dup = any(_hamming64(h, h2) <= hash_hamming_thresh for h2 in seen_hashes)
        if is_near_dup:
            continue

        # 3) embedding diversity check (cosine)
        if feat_map is not None:
            v = feat_map.get(nov_path)
            if v is not None and selected:
                too_similar = False
                for (_, psel, *_) in selected:
                    vv = feat_map.get(psel)
                    if vv is None: 
                        continue
                    #print("_cosine(v,vv): ",_cosine(v,vv))
                    if _cosine(v, vv) >= max_cosine_sim:
                        too_similar = True
                        break
                if too_similar:
                    continue
        else:
            print("[WARN] feat_map is NONE") 
            
        # keep it
        selected.append(row)
        if cid is not None:
            seen_clusters[cid] = seen_clusters.get(cid, 0) + 1
        if h >= 0:
            seen_hashes.append(h)

    # if we didn’t fill K (e.g., too strict), top off without diversity checks
    if len(selected) < k:
        need = k - len(selected)
        fill = [r for r in top_rows if r not in selected][:need]
        selected.extend(fill)
    return selected

def renumber_selected(selected_rows):
    """
    Input rows look like: (old_rank, nov_path, coarse, gssim, chamfer, final)
    Output rows are the same but with ranks reset to 1..K in current order.
    """
    out = []
    for i, (_old_rank, nov_path, coarse, gssim, chamfer, final) in enumerate(selected_rows, start=1):
        out.append((i, nov_path, coarse, gssim, chamfer, final))
    return out

def rebuild_rank_and_metrics(selected_rows, metrics_all):
    """
    Build fresh rank_top and metrics_top maps for the *selected* list.
    - rank_top: {nov_path: new_rank}
    - metrics_top: {nov_path: (final, ssim)}
    Falls back to values inside selected_rows if metrics_all lacks an entry.
    """
    rank_top_new = {}
    metrics_top_new = {}
    for i, (_old_rank, nov_path, coarse, gssim, chamfer, final) in enumerate(selected_rows, start=1):
        rank_top_new[nov_path] = i
        if metrics_all and (nov_path in metrics_all):
            metrics_top_new[nov_path] = metrics_all[nov_path]
        else:
            metrics_top_new[nov_path] = (final, gssim)
    return rank_top_new, metrics_top_new
