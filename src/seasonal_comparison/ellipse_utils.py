# ellipse_utils.py

import numpy as np
from matplotlib.patches import Ellipse
import os 
from seasonal_comparison.general_utils import robust_load_csv
from seasonal_comparison.image_stitching_utils import gps_lookup 

LAT_METERS_PER_DEGREE = 111320
LON_METERS_PER_DEGREE = 85390

def extract_timestamp_ns_from_path(p):
    filename = os.path.basename(p).replace("_downscaled", "")
    try:
        return int(os.path.splitext(filename)[0])
    except Exception:
        print(f"[!] Failed to extract timestamp from {p}")
        return None

def get_gps_center(image_paths):
    points = []
    for p in image_paths:
        print("p:",p)
        try:
            corners = gps_lookup(p)
            print("corners:",corners)
            if corners: points.extend(corners)
        except:
            continue
    if not points: return None
    lons, lats = zip(*points)
    return np.mean(lons), np.mean(lats)

def load_and_average_covariances(image_paths, cov_dir):
    cov_matrices = []
    for path in image_paths:
        ts_ns = extract_timestamp_ns_from_path(path)
        if ts_ns is None:
            print("ts_ns:",ts_ns)
            continue
        cov_path = os.path.join(cov_dir, f"{ts_ns}.csv")
       
        if not os.path.exists(cov_path):
            #try to find a covariance path thats close enough
            #165082581.3395741138
            partial_str = str(ts_ns)[:10]
            potential_paths = [x for x in os.listdir(cov_dir) if partial_str in x] 
            cov_path = min(
                potential_paths,
                key=lambda p: abs(ts_ns - extract_timestamp_ns_from_path(p)),
                default=None  
            )
            cov_path = os.path.join(cov_dir,cov_path)
        print("cov_path:",cov_path)
        cov = np.genfromtxt(cov_path,delimiter=" ")
        print("cov:",cov)
        if cov.shape[0] >= 2 and cov.shape[1] >= 2:
            cov_matrices.append(cov[:2, :2])

    print("cov_matrices: ",cov_matrices)
    print("np.mean(cov_matrices,axis=0):",np.mean(cov_matrices, axis=0))
    return np.mean(cov_matrices, axis=0) if cov_matrices else None


def plot_covariance_ellipse(ax, center, cov, n_std=2.0, edgecolor='blue', facecolor='none', alpha=1.0, linewidth=1.5, **kwargs):
    """
    Plots a 2D covariance ellipse on the given axes.
    
    Parameters:
        ax (matplotlib.axes): The matplotlib axis to plot on.
        center (tuple): (x, y) center coordinates (e.g., (lon, lat)).
        cov (2x2 np.ndarray): Covariance matrix.
        n_std (float): Number of standard deviations (default: 2 for ~95% confidence).
        edgecolor (str): Outline color of ellipse.
        facecolor (str): Fill color of ellipse.
        alpha (float): Transparency.
        linewidth (float): Line width.
        **kwargs: Additional arguments to Ellipse.
    """
    if cov.shape != (2, 2):
        raise ValueError("Covariance matrix must be 2x2.")

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    if np.any(np.isnan(vals)) or np.any(vals <= 0):
        print("[✗] Invalid eigenvalues, skipping ellipse.")
        return

    width, height = 2 * np.sqrt(vals)
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    print(f"[✓] Ellipse width={width}, height={height}, angle={angle}")

    ellipse = Ellipse(xy=center, width=width, height=height, angle=angle,
                      lw=2, facecolor='none', **kwargs)
    ax.add_patch(ellipse)

def scale_covariance_to_degrees(cov_matrix):
    """Scale covariance matrix from meters² to degrees²."""
    scale = np.diag([1 / LAT_METERS_PER_DEGREE, 1 / LON_METERS_PER_DEGREE])
    return scale @ cov_matrix @ scale.T

def create_uncertainty_ellipse(cov_matrix, center, ax, **kwargs):
    """Draws an uncertainty ellipse on the given matplotlib axis."""
    if cov_matrix is None:
        print("Covariance matrix is None; skipping ellipse.")
        return None

    try:
        scaled = scale_covariance_to_degrees(cov_matrix)
        eigenvalues, eigenvectors = np.linalg.eig(scaled)

        major_axis = 2 * np.sqrt(eigenvalues[0])
        minor_axis = 2 * np.sqrt(eigenvalues[1])
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        ellipse = Ellipse(xy=center, width=major_axis, height=minor_axis, angle=angle, **kwargs)
        return ellipse

    except Exception as e:
        print(f"[ERROR] Failed to create ellipse: {e}")
        return None


def is_inside_ellipse(cov_matrix, center, point, sigma=2.0):
    """Check if a point is within a Mahalanobis distance (default 2σ) from the center."""
    diff = np.array(point) - np.array(center)
    inv_cov = np.linalg.inv(cov_matrix)
    dist = np.sqrt(diff.T @ inv_cov @ diff)
    return dist <= sigma
