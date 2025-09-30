import torch
import numpy as np
from functools import lru_cache
from pathlib import Path
import cv2 as cv

from numpy.typing import NDArray
from typing import cast

from PIL import Image

import timm
import open_clip 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@lru_cache(maxsize=1)
def _clip_backend():
    # if open_clip is None:
    #     return None
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai"
    )
    model.eval().to(DEVICE)
    return model, preprocess

@lru_cache(maxsize=1)
def _dino_backend():
    # if timm is None:
    #     return None
    model = timm.create_model("vit_base_patch16_224.dino", pretrained=True, num_classes=0)
    model.eval().to(DEVICE)
    # simple torchvision-like preprocess
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])
    return model, preprocess


def _ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Return HxWx3 uint8 RGB ndarray."""
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected numpy ndarray, got {type(img)}")
    if img.ndim == 2:
        g = np.clip(img, 0, 255).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)
    if img.ndim == 3 and img.shape[2] == 3:
        # handle float [0,1] or [0,255]
        if img.dtype != np.uint8:
            arr = img
            if arr.dtype.kind in ("f",):  # float -> scale if likely [0,1]
                if arr.max() <= 1.0 + 1e-6:
                    arr = (arr * 255.0).round()
            return np.clip(arr, 0, 255).astype(np.uint8, copy=False)
        return img
    raise ValueError(f"Unexpected input shape {img.shape} (need HxW or HxWx3)")

def _to_pil_rgb(img: np.ndarray) -> Image.Image:
    """Convert ndarray to PIL RGB."""
    arr = np.ascontiguousarray(_ensure_rgb_uint8(img))
    return Image.fromarray(arr, mode="RGB")

  # --- CLIP: always feed PIL, return 1-D vector
@torch.no_grad()
def embed_clip(img_rgb_uint8: np.ndarray) -> np.ndarray:
    print("[embed_clip] start")
    try:
        model, preprocess = _clip_backend()

        # Ensure proper input type/shape
        if not isinstance(img_rgb_uint8, np.ndarray):
            raise TypeError(f"embed_clip expected np.ndarray, got {type(img_rgb_uint8)}")
        #print("[embed_clip] in np:", img_rgb_uint8.shape, img_rgb_uint8.dtype)

        # ndarray -> PIL RGB
        pil_img = _to_pil_rgb(img_rgb_uint8)
        #print("[embed_clip] PIL size:", pil_img.size, pil_img.mode)

        # preprocess -> torch tensor [C,H,W]
        t = preprocess(pil_img)
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"preprocess returned {type(t)}, expected torch.Tensor")
        #print("[embed_clip] after preprocess:", tuple(t.shape), t.dtype)

        # add batch, move to device
        t = t.unsqueeze(0).to(DEVICE, non_blocking=True)
        #print("[embed_clip] to device:", t.device, tuple(t.shape))

        # forward
        with torch.cuda.amp.autocast(False):
            feats = model.encode_image(t)   # [1, D]
        #print("[embed_clip] feats:", tuple(feats.shape), feats.dtype)

        # L2 normalize and return 1-D np.float32
        feats = torch.nn.functional.normalize(feats, dim=-1)
        out = feats.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        #print("[embed_clip] out:", out.shape, out.dtype)
        return out

    except Exception as e:
        # Print full error context once, then re-raise so caller can handle
        print("[embed_clip] ERROR:", type(e).__name__, e)
        raise

# --- DINO: pool tokens -> vector, then L2 normalize
@torch.no_grad()
def embed_dino(img_rgb_uint8: np.ndarray) -> np.ndarray:
    print("embed dino ...")
    model, preprocess = _dino_backend()
    t = preprocess(_ensure_rgb_uint8(img_rgb_uint8)).unsqueeze(0).to(DEVICE)
    feats = model.forward_features(t)
    # timm ViT variants return dict OR tensor
    if isinstance(feats, dict):
        if feats.get("x_norm_clstoken") is not None:         # (B, C)
            v = feats["x_norm_clstoken"]
        elif feats.get("pooled") is not None:                 # (B, C)
            v = feats["pooled"]
        elif feats.get("x") is not None:                      # (B, N, C)
            v = feats["x"][:, 0, :]                           # CLS -> (B, C)
        else:
            v = model.forward_head(feats, pre_logits=True)    # (B, C) if supported
    else:
        # tensor path: (B, N, C) or (B, C)
        if feats.ndim == 3:                                   # (B, N, C)
            v = feats[:, 0, :]                                # CLS -> (B, C)
        else:
            v = feats                                         # (B, C)
    v = torch.nn.functional.normalize(v, dim=-1)
    return v.squeeze(0).float().cpu().numpy()                 # (C,)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def _load_rgb(path: str) -> np.ndarray | None:
    im = cv.imread(path, cv.IMREAD_COLOR)
    if im is None: return None
    return cv.cvtColor(im, cv.COLOR_BGR2RGB)

def _to_row_vec(feat: np.ndarray | None) -> np.ndarray | None:
    if feat is None:
        return None
    f = np.asarray(feat, dtype=np.float32)
    if f.ndim == 1:
        return f
    if f.ndim == 2:
        # tokens x dim  -> mean-pool tokens to a single vector
        return f.mean(axis=0, dtype=np.float32)
    # rare fallbacks: flatten
    return f.reshape(-1)

def _stack_sidecar(feats_list: list[np.ndarray | None]) -> np.ndarray:
    rows: list[np.ndarray | None] = [_to_row_vec(f) for f in feats_list]
    dim = max((r.shape[-1] for r in rows if r is not None), default=0)
    M = np.zeros((len(rows), dim), dtype=np.float32)
    for i, r in enumerate(rows):
        if r is None:
            continue
        L = min(dim, r.shape[-1])
        M[i, :L] = r[:L]
    return M

def ensure_embed_sidecars(idx_npz_path: str):
    print("ensuring embed sidecars ...")

    base = Path(idx_npz_path)
    root = base.parent

    data = np.load(str(base), allow_pickle=True)
    paths = data["paths"]

    side_clip = root / (base.stem + "_clip.npy")
    side_dino = root / (base.stem + "_dino.npy")

    need_clip = (side_clip.exists() is False) and (_clip_backend() is not None)
    need_dino = (side_dino.exists() is False) and (_dino_backend() is not None)

    clip_feats = [] if need_clip else None
    dino_feats = [] if need_dino else None

    for p in paths:
        p = str(p)
        rgb = _load_rgb(p)
        if rgb is None:
            if need_clip: clip_feats.append(None)
            if need_dino: dino_feats.append(None)
            continue
        if need_clip:
            f = embed_clip(rgb)
            print("here!")
            clip_feats.append(f if f is not None else None)
            print("successfuly appended clip feat")
        if need_dino:
            f = embed_dino(rgb)
            dino_feats.append(f if f is not None else None)
            print("successfuly appended dino feat")

    # if need_clip: 
    #     # pad Nones with zeros of max length seen
    #     dim = max((len(x) for x in clip_feats if x is not None), default=0)
    #     arr = np.zeros((len(paths), dim), dtype=np.float32)
    #     for i,f in enumerate(clip_feats):
    #         if f is not None: arr[i,:] = f 
    #     print("saving side_clip")
    #     np.save(side_clip, arr)

    # if need_dino:
    #     dim = max((len(x) for x in dino_feats if x is not None), default=0)
    #     arr = np.zeros((len(paths), dim), dtype=np.float32)
    #     for i,f in enumerate(dino_feats):
    #         if f is not None: arr[i,:] = f
    #     print("saving side_dino")
    #     np.save(side_dino, arr)

    if need_clip:
        #print("clip example shapes:", [None if f is None else np.asarray(f).shape for f in clip_feats[:3]])
        arr = _stack_sidecar(clip_feats)    # (len(paths), D_clip)
        print("saving side_clip", arr.shape)
        np.save(side_clip, arr)

    if need_dino:
        #print("dino example shapes:", [None if f is None else np.asarray(f).shape for f in dino_feats[:3]]) 
        arr = _stack_sidecar(dino_feats)    # (len(paths), D_dino)
        print("saving side_dino", arr.shape)
        np.save(side_dino, arr)

    print("successfuly embedded sidecars")

def load_embed_sidecars(idx_npz_path: str):
    base = Path(idx_npz_path)
    root = base.parent
    clip_p = root / (base.stem + "_clip.npy")
    dino_p = root / (base.stem + "_dino.npy")
    clip = np.load(clip_p) if clip_p.exists() else None
    dino = np.load(dino_p) if dino_p.exists() else None
    return clip, dino
