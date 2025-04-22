# ellipse_utils.py

import numpy as np
from matplotlib.patches import Ellipse

LAT_METERS_PER_DEGREE = 111320
LON_METERS_PER_DEGREE = 85390

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
