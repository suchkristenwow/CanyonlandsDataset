#!/usr/bin/env python3

"""
Utilities for working with GPS covariance ellipses, 
finding overlapping frames, and loading covariance data.
"""

import os
import glob
import numpy as np

from shapely.geometry import Polygon

from seasonal_comparison.general_utils import robust_load_csv, same_order_of_magnitude

# Constants for meter-to-degree scaling
LAT_METERS_PER_DEGREE = 111_320
LON_METERS_PER_DEGREE = 85_390


def scale_covariance_to_degrees(cov_matrix):
    """
    Scale a 2x2 covariance matrix from meters² to degrees².

    Args:
        cov_matrix (np.ndarray): Covariance matrix in meters.

    Returns:
        np.ndarray: Covariance matrix scaled to degrees.
    """
    scale = np.diag([1 / LAT_METERS_PER_DEGREE, 1 / LON_METERS_PER_DEGREE])
    return scale @ cov_matrix @ scale.T


def find_largest_overlap_subset(mpl_polygons):
    """
    Find the largest subset of polygons where all overlap.

    Args:
        mpl_polygons (list of MplPolygon): List of Matplotlib polygons.

    Returns:
        list of MplPolygon: Largest subset of overlapping polygons.
    """
    shapely_polygons = [Polygon(polygon.get_xy()) for polygon in mpl_polygons]

    G = nx.Graph()
    G.add_nodes_from(range(len(shapely_polygons)))

    for i in range(len(shapely_polygons)):
        for j in range(i + 1, len(shapely_polygons)):
            if shapely_polygons[i].intersects(shapely_polygons[j]):
                G.add_edge(i, j)

    largest_cc = max(nx.connected_components(G), key=len, default=[])
    return [mpl_polygons[i] for i in largest_cc]


def find_frames_inside_ellipse(cov_matrix, center, longitudes, latitudes, timestamps, threshold=1.0):
    """
    Find frames whose (longitude, latitude) falls inside the given covariance ellipse.

    Args:
        cov_matrix (np.ndarray): 2x2 covariance matrix.
        center (tuple): (lon, lat) center point.
        longitudes (np.ndarray): Array of longitude values.
        latitudes (np.ndarray): Array of latitude values.
        timestamps (np.ndarray): Array of timestamps.
        threshold (float): Threshold for Mahalanobis distance (default 1.0).

    Returns:
        list of (timestamp, lon, lat): Points inside the ellipse.
    """
    points = np.column_stack((longitudes, latitudes))
    deltas = points - np.array(center)
    inv_cov = np.linalg.inv(cov_matrix)

    # Compute Mahalanobis distance squared
    mahal_dists = np.einsum("ij,jk,ik->i", deltas, inv_cov, deltas)

    inside_mask = mahal_dists <= threshold
    return list(zip(timestamps[inside_mask], longitudes[inside_mask], latitudes[inside_mask]))


def precompute_timestamps(covariance_dir):
    """
    Precompute available timestamps from covariance matrix files.

    Args:
        covariance_dir (str): Path to covariance matrix directory.

    Returns:
        list of int: Sorted timestamps in nanoseconds.
    """
    print("[INFO] Precomputing timestamps...")
    filenames = glob.glob(os.path.join(covariance_dir, "*.csv"))
    timestamps = [int(os.path.basename(f).split(".")[0]) for f in filenames]
    return sorted(timestamps)


def load_covariance_matrix(covariance_dir, available_timestamps, timestamp, threshold=0.3):
    """
    Load the covariance matrix closest to a given timestamp.

    Args:
        covariance_dir (str): Directory containing covariance matrices.
        available_timestamps (list of int): Available timestamps (nanoseconds).
        timestamp (float): Target timestamp (nanoseconds).
        threshold (float): Maximum allowed delta in seconds (default 0.1).

    Returns:
        np.ndarray or None: Covariance matrix if within threshold, otherwise None.
    """
    
    if not same_order_of_magnitude(available_timestamps[0],timestamp):
        #print("WARN: These are not the same order of magnitude ... trying to correct")
        tmp_timestamp = timestamp * 10**9 
        if same_order_of_magnitude(available_timestamps[0],tmp_timestamp):
            timestamp = timestamp * 10**9 

    closest_timestamp = min(available_timestamps, key=lambda t: abs(t - timestamp))
    #print("closest_timestamp: ",closest_timestamp) 

    filename = os.path.join(covariance_dir, f"{closest_timestamp}.csv")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"[ERROR] Covariance matrix file not found: {filename}")

    #print("timestamp:",timestamp) 
    
    #this should be converted into secs 
    if len(str(timestamp)) == 19 and len(str(closest_timestamp)) == 19:
        #this is in nanoseconds 
        timestamp = timestamp * 1e-9
        closest_timestamp = closest_timestamp * 1e-9

    #print("delta: ",abs(timestamp - closest_timestamp)) 

    if abs(timestamp - closest_timestamp) <= threshold:
        return np.genfromtxt(filename)
    else:
        print(f"[INFO] No covariance file close enough (Δt={abs(timestamp - closest_timestamp):.2f}s).")
        return None
