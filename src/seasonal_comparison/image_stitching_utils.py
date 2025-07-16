#!/usr/bin/env python3

"""
Utilities for frame chunking, GPS corner extraction, and image cropping
for seasonal frame stitching.
"""

import os
import numpy as np
import cv2 as cv
from shapely.geometry import Polygon
from seasonal_comparison.general_utils import robust_load_csv

def chunk_filenames(filenames, chunk_size=5):
    """Split filenames into chunks of specified size."""
    return [filenames[i:i + chunk_size] for i in range(0, len(filenames), chunk_size)]


def check_corner_uniqueness(corners, tol=1e-7):
    """
    Check if four corners are unique and form a valid polygon.

    Args:
        corners (list of (lon, lat)): List of corner tuples.
        tol (float): Tolerance for coordinate equality.

    Returns:
        (bool, str): (is_valid, reason_for_failure or 'OK')
    """
    if len(corners) != 4:
        return False, "Expected 4 corners"

    unique = []
    for pt in corners:
        if not any(np.linalg.norm(np.array(pt) - np.array(other)) < tol for other in unique):
            unique.append(pt)

    if len(unique) < 4:
        return False, f"Only {len(unique)} unique corners"

    try:
        poly = Polygon(corners)
        if not poly.is_valid:
            return False, "Invalid polygon geometry"
        if len(poly.exterior.coords) != 5:  # 4 corners + closing point
            return False, f"Got {len(poly.exterior.coords)} exterior points"
    except Exception as e:
        return False, str(e)

    return True, "OK"


def crop_connected_region(image, area_thresh_ratio=0.05):
    """
    Crop the largest connected region from the input image.

    Args:
        image (np.ndarray): Input BGR image.
        area_thresh_ratio (float): Minimum ratio for keeping regions.

    Returns:
        np.ndarray: Cropped image.
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 15, 255, cv.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity=8)

    if num_labels <= 1:
        print("[!] No connected regions found — skipping crop.")
        return image

    largest_label = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
    mask = (labels == largest_label).astype(np.uint8) * 255

    coords = cv.findNonZero(mask)
    x, y, w, h = cv.boundingRect(coords)
    cropped = cv.bitwise_and(image, image, mask=mask)[y:y+h, x:x+w]

    return cropped


def gps_lookup(filepath):
    """
    Lookup GPS corner coordinates for a given image filepath.

    Args:
        filepath (str): Path to the image file.

    Returns:
        list of (lon, lat) tuples: Four corner points.
    """
    result_dir = os.path.dirname(os.path.dirname(filepath))

    if "Panos" in filepath:
        result_dir = os.path.dirname(result_dir)
        frustrum_corners = robust_load_csv(os.path.join(result_dir, "frustrum_corners.csv"))
    else:
        frustrum_corners = robust_load_csv(os.path.join(result_dir, "processed_results", "frustrum_corners.csv"))

    if "downscaled" in filepath:
        cleaned_filename = os.path.splitext(os.path.basename(filepath))[0]
        underscore_idx = cleaned_filename.index("_")
        timestamp = int(cleaned_filename[:underscore_idx]) * 1e-9
    else:
        timestamp = int(os.path.splitext(os.path.basename(filepath))[0]) * 1e-9

    if "Panos" not in filepath:
        idx = np.argmin(np.abs(frustrum_corners[:, 0] - timestamp))
        delta_t = np.abs(timestamp - frustrum_corners[idx, 0])
        if delta_t > 0.3:
            print(f"[!] Warning: Timestamp delta too large ({delta_t:.2f}s)")
            print(f"Target: {timestamp:.6f}, Closest: {frustrum_corners[idx, 0]:.6f}")
            input("WARNING ... CANNOT FIND CORRESPONDING FRUSTRUM CORNERS")
            return None

    if "May" in filepath:
        frustrum_corners = frustrum_corners[idx, 17:]
    elif "Nov" in filepath:
        if "Left" in filepath:
            frustrum_corners = frustrum_corners[idx, 17:25]
        elif "Right" in filepath:
            frustrum_corners = frustrum_corners[idx, 25:33]

    if frustrum_corners.shape[0] != 8:
        print(f"[!] Expected 8 GPS corner values, got {frustrum_corners.shape[0]} for {filepath}")
        raise OSError

    corners = []
    for i in range(4):
        lat = frustrum_corners[i*2]
        lon = frustrum_corners[i*2 + 1]
        corners.append((lon, lat))  # (x, y) order

    if len(corners) < 4:
        raise OSError

    return corners


def find_closest_file(directory, target_ts_sec, tolerance_sec=0.1):
    """
    Find the closest .png file to a target timestamp.

    Args:
        directory (str): Directory containing PNG files.
        target_ts_sec (float): Target timestamp (seconds).
        tolerance_sec (float): Maximum allowed time difference.

    Returns:
        str or None: Full path to matching file, or None if not found.
    """
    files = [f for f in os.listdir(directory) if f.endswith('.png')]
    if not files:
        return None

    ts_file_list = []
    for f in files:
        try:
            ts_ns = int(os.path.splitext(f)[0])
            ts_sec = ts_ns * 1e-9
            ts_file_list.append((ts_sec, f))
        except ValueError:
            continue

    if not ts_file_list:
        return None

    closest = min(ts_file_list, key=lambda x: abs(x[0] - target_ts_sec))
    closest_diff = abs(closest[0] - target_ts_sec)

    if closest_diff <= tolerance_sec:
        return os.path.join(directory, closest[1])
    else:
        print(f"[!] Could not find a filename within {tolerance_sec:.2f} sec threshold!")
        return None 

def check_image_sizes(image_paths):
    """
    Checks if all images in the list have the same shape.

    Args:
        image_paths (list of str): Paths to the image files.

    Returns:
        bool: True if all images are the same size, False otherwise.
    """
    if not image_paths:
        print("[WARN] No images to check.")
        return False

    reference_shape = None
    for path in image_paths:
        img = cv.imread(path)
        if img is None:
            print(f"[WARN] Could not load image: {path}")
            continue

        if reference_shape is None:
            reference_shape = img.shape
        elif img.shape != reference_shape:
            print(f"[MISMATCH] {path} has shape {img.shape}, expected {reference_shape}")
            return False
    return True 

def crop_black_border(image, tol=10):
    """
    Crops black borders from an image.
    `tol` is the tolerance for pixel brightness to be considered black.
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, tol, 255, cv.THRESH_BINARY)
    
    coords = cv.findNonZero(thresh)
    if coords is None:
        print("everything is black")
        raise OSError 

    x, y, w, h = cv.boundingRect(coords)
    cropped = image[y:y+h, x:x+w]
    return cropped