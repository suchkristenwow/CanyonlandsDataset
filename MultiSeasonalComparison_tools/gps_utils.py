from geopy.distance import geodesic
import os 
import numpy as np 
import glob  

def is_inside_ellipse(cov_matrix, center, point):
    """
    Check if a point is inside the covariance ellipse.
    :param cov_matrix: Covariance matrix (2x2)
    :param center: Center of the ellipse (longitude, latitude)
    :param point: Point to check (longitude, latitude)
    :return: True if inside, False otherwise
    """
    diff = np.array(point) - np.array(center)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    mahalanobis_dist = np.sqrt(diff.T @ inv_cov_matrix @ diff)
    return mahalanobis_dist <= 2  # Check for 2-sigma ellipse

def find_frames_inside_ellipse(cov_matrix, center, longitudes, latitudes, timestamps):
    """
    Find all frames whose points fall inside the covariance ellipse.
    :param cov_matrix: Covariance matrix (2x2)
    :param center: Center of the ellipse (longitude, latitude)
    :param longitudes: Array of longitude values
    :param latitudes: Array of latitude values
    :param timestamps: Array of timestamps
    :return: List of (timestamp, longitude, latitude) for frames inside the ellipse
    """
    frames_inside = []
    for i in range(len(longitudes)):
        point = (longitudes[i], latitudes[i])
        if is_inside_ellipse(cov_matrix, center, point):
            frames_inside.append((timestamps[i], longitudes[i], latitudes[i]))
    return frames_inside


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

def find_closest_frame(frame_dir,timestamp):
    """Find the closest frame to the given timestamp."""
    frame_files = [f for f in os.listdir(frame_dir) if f.endswith(".png") or f.endswith(".jpg")]
    frame_timestamps = [10 ** (-9) * int(f.split(".")[0]) for f in frame_files] 
    closest_timestamp = min(frame_timestamps, key=lambda t: abs(t - timestamp))
    tstep_str = str(int(closest_timestamp * 1e9))
    matching_files = [
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if tstep_str[:-5] in f
    ]

    if len(matching_files) > 0:
        return matching_files[0]
    else: 
        matching_files = [
            os.path.join(frame_dir, f)
            for f in os.listdir(frame_dir)
            if tstep_str[:-6] in f
        ] 
        return matching_files[0] 
    
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