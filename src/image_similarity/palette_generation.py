import numpy as _np

def color_by_index(k, PALETTE):
    return PALETTE[k % len(PALETTE)]

def color_for_rank(rank, PALETTE):
    # rank is 1..8 for your labeled items
    return color_by_index(rank - 1, PALETTE)

def _srgb_to_linear(u):
    u = _np.asarray(u)
    return _np.where(u <= 0.04045, u/12.92, ((u+0.055)/1.055)**2.4)

def _rgb_to_xyz(rgb):  # rgb in [0,1], sRGB D65
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    r_lin = _srgb_to_linear(r); g_lin = _srgb_to_linear(g); b_lin = _srgb_to_linear(b)
    # sRGB -> XYZ (D65)
    X = 0.4124564*r_lin + 0.3575761*g_lin + 0.1804375*b_lin
    Y = 0.2126729*r_lin + 0.7151522*g_lin + 0.0721750*b_lin
    Z = 0.0193339*r_lin + 0.1191920*g_lin + 0.9503041*b_lin
    return _np.stack([X, Y, Z], axis=-1)

def _xyz_to_lab(xyz):  # D65/2°, reference white
    # Ref white (D65)
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x = xyz[...,0] / Xn
    y = xyz[...,1] / Yn
    z = xyz[...,2] / Zn

    def f(t):
        eps = 216/24389  # ~0.008856
        kappa = 24389/27
        return _np.where(t > eps, _np.cbrt(t), (kappa*t + 16)/116)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return _np.stack([L, a, b], axis=-1)

def _rgb_to_lab(rgb01):
    return _xyz_to_lab(_rgb_to_xyz(rgb01))

def gen_distinct_palette(n_colors=40, grid_levels=(0, 51, 102, 153, 204, 255),
                         min_L=25, max_L=85, min_chroma=25, seed=7):
    """
    Generate n_colors distinct sRGB colors as RGBA tuples in [0,1].
    Filters candidates to avoid too dark/light and too gray; then greedy maximin in Lab.
    """
    rng = _np.random.RandomState(seed)

    # Build candidate RGBs on a coarse grid
    gl = _np.array(grid_levels, dtype=_np.uint8)
    R, G, B = _np.meshgrid(gl, gl, gl, indexing='ij')
    cand = _np.stack([R, G, B], axis=-1).reshape(-1, 3).astype(_np.float32) / 255.0

    # Exclude near-black/white by Lab L*, and exclude very low chroma
    lab = _rgb_to_lab(cand)
    L, a, b = lab[:,0], lab[:,1], lab[:,2]
    chroma = _np.sqrt(a*a + b*b)

    keep = (L >= min_L) & (L <= max_L) & (chroma >= min_chroma)
    cand = cand[keep]
    lab  = lab[keep]

    if len(cand) < n_colors:
        raise RuntimeError(f"Not enough candidates ({len(cand)}) for {n_colors} colors; "
                           f"loosen filters or add grid levels.")

    # Start seeds: pick a few high-chroma anchors (or random)
    # Here: pick the highest-chroma candidate as the first
    first_idx = _np.argmax(_np.sqrt(lab[:,1]**2 + lab[:,2]**2))
    selected = [first_idx]

    # To improve stability, pick second as farthest in Lab from first
    d2first = _np.sum((lab - lab[first_idx])**2, axis=1)
    second_idx = _np.argmax(d2first)
    selected.append(second_idx)

    # Precompute distance cache (we'll keep min distance to any selected)
    min_d2 = _np.minimum(
        _np.sum((lab - lab[first_idx])**2, axis=1),
        _np.sum((lab - lab[second_idx])**2, axis=1)
    )

    # Greedy maximin
    while len(selected) < n_colors:
        # exclude already chosen
        min_d2[selected] = -_np.inf
        next_idx = _np.argmax(min_d2)
        selected.append(next_idx)
        # update min distances
        d2 = _np.sum((lab - lab[next_idx])**2, axis=1)
        min_d2 = _np.maximum(min_d2, -_np.inf)  # keep shape
        min_d2 = _np.minimum(min_d2, d2)

    # Return as RGBA (matplotlib-friendly)
    out = []
    for idx in selected:
        r, g, b = cand[idx]
        out.append((float(r), float(g), float(b), 1.0))
    return out
