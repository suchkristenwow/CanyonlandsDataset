from geopy.distance import geodesic
import os 
import numpy as np 
import glob  
from geopy.distance import geodesic
from collections import defaultdict, deque
from seasonal_comparison.image_stitching_utils import robust_load_csv

def is_inside_ellipse(cov_matrix, center, point, threshold=1.0):
    delta = np.array(point) - np.array(center)
    return delta.T @ np.linalg.inv(cov_matrix) @ delta <= threshold

def find_frames_inside_ellipse(cov_matrix, center, longitudes, latitudes, timestamps, threshold=1.0):
    """
    Vectorized check for which points fall inside the covariance ellipse.
    """
    # Stack coordinates: shape (N, 2)
    points = np.column_stack((longitudes, latitudes))
    deltas = points - np.array(center)

    # Inverse covariance matrix
    inv_cov = np.linalg.inv(cov_matrix)

    # Mahalanobis distance squared
    mahal_dists = np.einsum("ij,jk,ik->i", deltas, inv_cov, deltas)

    # Mask of points within the ellipse
    inside_mask = mahal_dists <= threshold

    # Select points inside
    return list(zip(timestamps[inside_mask], longitudes[inside_mask], latitudes[inside_mask]))

def are_coordinates_within_threshold(coord1, coord2, threshold_meters=0.1):
    """
    Checks if two GPS coordinates are within a certain distance threshold.

    Parameters:
        coord1 (tuple): First GPS coordinate as (latitude, longitude).
        coord2 (tuple): Second GPS coordinate as (latitude, longitude).
        threshold_meters (float): The distance threshold in meters.

    Returns:
        bool: True if the coordinates are within the threshold, False otherwise.
    """
    # Calculate the distance in meters using geodesic (Vincenty formula fallback)
    distance = geodesic(coord1, coord2).meters
    #print("distance: ",distance) 
    return distance <= threshold_meters

def find_closest_frame(frame_dir, timestamp):
    """Find the closest frame to the given timestamp."""
    frame_files = [f for f in os.listdir(frame_dir) if f.endswith(".png") or f.endswith(".jpg")]

    frame_timestamps = []
    file_to_ts = {}

    for file in frame_files:
        try:
            file_ts = 10 ** (-9) * int(file.split(".")[0])
        except:
            filename = file.split(".")[0]
            idx = filename.index("_")
            file_ts = 10 ** (-9) * int(filename[:idx])
        
        frame_timestamps.append(file_ts)
        file_to_ts[file] = file_ts

    # Now use the original input timestamp here!
    closest_timestamp = min(frame_timestamps, key=lambda t: abs(t - timestamp))
    delta_t = abs(closest_timestamp - timestamp)

    if delta_t > 0.3:
        print("Input timestamp:", timestamp)
        print("Closest frame timestamp:", closest_timestamp)
        raise OSError

    tstep_str = str(int(closest_timestamp * 1e9))
    matching_files = [
        os.path.join(frame_dir, f)
        for f in frame_files
        if tstep_str[:-5] in f
    ]

    if matching_files:
        return matching_files[0]

    matching_files = [
        os.path.join(frame_dir, f)
        for f in frame_files
        if tstep_str[:-6] in f
    ]

    if matching_files:
        return matching_files[0]

    raise FileNotFoundError(f"No matching file found for timestamp: {tstep_str}")

def precompute_timestamps(covariance_dir):
    """Precompute all available timestamps from the covariance matrix files."""
    print("precomputing timestamps ...")
    filenames = glob.glob(os.path.join(covariance_dir, "*.csv"))
    timestamps = [int(os.path.basename(f).split(".")[0]) for f in filenames]  # Convert nanoseconds to seconds
    return sorted(timestamps)

def load_covariance_matrix(covariance_dir,available_timestamps,timestamp,threshold=0.1):
    """Load the covariance matrix for the given timestamp."""

    # Find the closest available timestamp
    closest_timestamp = min(available_timestamps, key=lambda t: abs(t - timestamp)) 
    filename = os.path.join(covariance_dir, f"{closest_timestamp}.csv") 

    if not os.path.exists(filename):
        raise OSError 

    # Check if the closest timestamp is within 0.1 seconds
    if abs(timestamp - closest_timestamp) <= threshold:
        return np.genfromtxt(filename)
    else:
        # If no timestamp is close enough, return None
        return None