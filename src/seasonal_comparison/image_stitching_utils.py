#!/usr/bin/env python3

"""
Utilities for frame chunking, GPS corner extraction, and image cropping
for seasonal frame stitching.
"""
import os
import numpy as np
import cv2 as cv
from shapely.geometry import Polygon
from seasonal_comparison.general_utils import robust_load_csv, log_mem
import pickle 
import subprocess 
import uuid  

import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
plt.ioff() 

from geopy.distance import geodesic
from scipy.stats import circmean 
import tempfile
import math 
from pathlib import Path

def fix_fourth_corner_area_match(three_corners, desired_area):
    """
    Given three corners of a quadrilateral and the desired area,
    compute the fourth corner so that the resulting polygon matches the area.

    Parameters:
        three_corners: list of 3 (x, y) tuples (assumed to be in order)
        desired_area: float (in same units as coordinates, e.g., degrees or meters^2)

    Returns:
        list of 4 (x, y) tuples defining the full polygon.
    """
    if len(three_corners) != 3:
        raise ValueError("Expected exactly three corners")

    A, B, C = map(np.array, three_corners)

    # Step 1: compute vector AB and vector BC
    AB = B - A
    BC = C - B

    # Step 2: compute the unit normal vector to AB (90 degrees CCW)
    AB_norm = np.array([-AB[1], AB[0]])
    AB_norm /= np.linalg.norm(AB_norm)

    # Step 3: sweep a line from C along the direction of AB_norm
    # and find point D such that ABCD has the desired area
    def area_with_d(d):
        polygon = Polygon([A, B, C, d])
        return polygon.area

    # Step 4: brute force or binary search along the direction of AB_norm
    def find_best_d():
        best_d = None
        min_error = float('inf')
        for scale in np.linspace(-1.0, 1.0, 500):  # tune bounds as needed
            D = C + scale * AB_norm
            poly = Polygon([A, B, C, D])
            if not poly.is_valid or poly.is_empty:
                continue
            err = abs(poly.area - desired_area)
            if err < min_error:
                min_error = err
                best_d = D
        return best_d

    D = find_best_d()
    if D is None:
        raise RuntimeError("Unable to find a valid fourth corner")
    
    return [tuple(A), tuple(B), tuple(C), tuple(D)]

def fix_polygon_area(poly, desired_area, tol=0.1):
    """
    Given a 4-corner polygon (possibly invalid), fix it by removing one bad corner
    and computing a new one so that the resulting polygon has the desired area.
    """
    coords = list(poly.exterior.coords)[:-1]  # drop closing point
    if len(coords) != 4:
        raise ValueError("Polygon must have 4 corners")

    coords = [np.array(pt) for pt in coords]

    # Step 1: Find the two closest points (likely one is bad)
    min_dist = float('inf')
    close_pair = (0, 1)
    for i in range(4):
        for j in range(i + 1, 4):
            d = np.linalg.norm(coords[i] - coords[j])
            if d < min_dist:
                min_dist = d
                close_pair = (i, j)

    if min_dist > tol:
        print(f"[WARN] No close points found — unable to identify bad corner confidently.")
        raise RuntimeError("No clearly bad corner based on proximity.")

    # Step 2: Try fixing by removing each candidate bad corner and reconstructing
    best_poly = None
    min_err = float('inf')

    for bad_idx in close_pair:
        three = [tuple(coords[i]) for i in range(4) if i != bad_idx]
        try:
            candidate_pts = fix_fourth_corner_area_match(three, desired_area)
            candidate_poly = Polygon(candidate_pts)
            err = abs(candidate_poly.area - desired_area)
            if err < min_err:
                min_err = err
                best_poly = candidate_poly
        except Exception as e:
            print(f"[WARN] Skipping fix attempt due to error: {e}")
            continue

    if best_poly is None:
        raise RuntimeError("Unable to repair the polygon")

    return best_poly


def call_stitch_subprocess(frame_list, stitch_cfg, timeout_sec=60):
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        data_path = f.name
        pickle.dump({"frame_list": frame_list, "stitch_cfg": stitch_cfg}, f)

    output_path = f"/tmp/stitched_{uuid.uuid4().hex}.png"

    env = os.environ.copy()
    # Cap hidden thread fans that blow up RAM
    env.setdefault("OMP_NUM_THREADS","1")
    env.setdefault("OPENBLAS_NUM_THREADS","1")
    env.setdefault("MKL_NUM_THREADS","1")
    env.setdefault("NUMEXPR_NUM_THREADS","1")
    env.setdefault("OPENCV_OPENCL_RUNTIME","disabled")

    # IMPORTANT: check=False and capture outputs for logging
    try:
        res = subprocess.run(
            ["python3", "src/seasonal_comparison/img_stitch_worker.py", data_path, output_path],
            check=False, env=env, capture_output=True, text=True, timeout=timeout_sec
        )
    except subprocess.TimeoutExpired:
        print("[WARN] stitch subprocess timed out")
        try: os.remove(data_path)
        except OSError: pass
        return None
    finally:
        # clean the pickle, it's not needed after the subprocess starts
        try: os.remove(data_path)
        except OSError: pass

    if res.returncode != 0:
        # Non-zero means stitch failed (e.g., no matches). Just log and skip.
        if res.stdout: print("[stitch stdout]\n" + res.stdout.strip())
        if res.stderr: print("[stitch stderr]\n" + res.stderr.strip())
        try: 
            if os.path.exists(output_path): os.remove(output_path)
        except OSError:
            pass
        return None

    if os.path.exists(output_path):
        stitched_img = cv.imread(output_path)
        # optionally remove file after reading
        try: os.remove(output_path)
        except OSError: pass
        return stitched_img

    return None

def fix_fourth_corner(corners, tol=1e-8):
    """
    Identifies the bad corner (nearly colinear with two others), removes it,
    and reconstructs the rectangle using the other three.
    Returns new corner list and index of the removed corner.
    """
    import numpy as np

    pts = np.array(corners)

    for i in range(4):
        others = [pts[j] for j in range(4) if j != i]
        a, b, c = others

        ab = b - a
        ac = c - a
        bc = c - b

        if np.isclose(np.abs(np.cross(ab, ac)), 0, atol=tol) or \
           np.isclose(np.abs(np.cross(ab, bc)), 0, atol=tol) or \
           np.isclose(np.abs(np.cross(ac, bc)), 0, atol=tol):

            # Try to reconstruct a rectangle from the three non-colinear points
            # Use vector addition to compute the missing fourth point
            # Assume a right angle at 'a': d = b + c - a
            d = b + c - a
            new_pts = [tuple(a), tuple(b), tuple(d), tuple(c)]
            return reorder_corners(new_pts), i

    raise ValueError("No bad corner could be reliably identified")

def create_polygons(corners_list, debug_dir="./debug_polygon_plots"):
    """Convert list of corner coordinates into shapely Polygons. Save plots for invalid ones."""
    os.makedirs(debug_dir, exist_ok=True)
    polygons = []

    for idx, corners in enumerate(corners_list):
        if not check_corner_uniqueness(corners):
            print("[ERROR] Invalid frame corners:",corners)
            raise OSError 

        try:
            corners = reorder_corners(corners)
        except Exception as e:
            print(f"[WARN] Failed to reorder corners: {corners} ({e})")

        poly = Polygon(corners)
        if poly.is_valid and not poly.is_empty and poly.area > 0:
            polygons.append(poly)
        else:
            new_corners, bad_idx = fix_fourth_corner(corners)

            # Save debug plot
            fig, ax = plt.subplots()
            
            # Original polygon (red)
            xs, ys = zip(*corners)
            ax.plot(xs + (xs[0],), ys + (ys[0],), 'r-', marker='o', label='Original')
            
            # Reconstructed polygon (blue)
            xs_new, ys_new = zip(*new_corners)
            ax.plot(xs_new + (xs_new[0],), ys_new + (ys_new[0],), 'b-', marker='o', label='Fixed')

            # Mark the bad corner with a black star
            bad_x, bad_y = corners[bad_idx]
            ax.plot(bad_x, bad_y, 'k*', markersize=12, label='Bad Corner')

            ax.set_title(f"Invalid Polygon {bad_idx}")
            ax.set_aspect('equal')
            ax.legend()

            debug_path = os.path.join(debug_dir, f"invalid_polygon_{bad_idx}.png")
            plt.savefig(debug_path, dpi=150)
            plt.close(fig)
            print(f"[DEBUG] Saved invalid polygon plot to {debug_path}")
            raise OSError 
            '''
            poly = Polygon(new_corners)
            polygons.append(poly)
            ''' 

    return polygons
    
def create_polygon(corners, debug_dir="./debug_polygon_plots"): 
    corners = reorder_corners(corners)
    poly = Polygon(corners)
    if poly.is_valid and not poly.is_empty and poly.area > 0:
        return poly 
    else:
        print("corners: ",corners) 
        new_corners, bad_idx = fix_fourth_corner(corners)

        # Save debug plot
        fig, ax = plt.subplots()
        
        # Original polygon (red)
        xs, ys = zip(*corners)
        ax.plot(xs + (xs[0],), ys + (ys[0],), 'r-', marker='o', label='Original')
        
        # Reconstructed polygon (blue)
        xs_new, ys_new = zip(*new_corners)
        ax.plot(xs_new + (xs_new[0],), ys_new + (ys_new[0],), 'b-', marker='o', label='Fixed')

        # Mark the bad corner with a black star
        bad_x, bad_y = corners[bad_idx]
        ax.plot(bad_x, bad_y, 'k*', markersize=12, label='Bad Corner')

        ax.set_title(f"Invalid Polygon {bad_idx}")
        ax.set_aspect('equal')
        ax.legend()

        debug_path = os.path.join(debug_dir, f"invalid_polygon_{bad_idx}.png")
        plt.savefig(debug_path, dpi=150)
        plt.close(fig)
        print(f"[DEBUG] Saved invalid polygon plot to {debug_path}")
        raise OSError 

        #poly = Polygon(new_corners)
        
    return poly 

def reorder_corners(corner_tuples):
    """
    Reorder 4 (lon, lat) corner points into consistent order:
    [top-left, top-right, bottom-right, bottom-left]
    """
    pts = np.array(corner_tuples)
    pts = pts.reshape((4,2))
    # Sort by latitude (lat = y)
    sorted_by_lat = pts[np.argsort(pts[:, 1])]
    top_two = sorted_by_lat[-2:]
    bottom_two = sorted_by_lat[:2]

    # Sort each pair by longitude (lon = x)
    top_left, top_right = top_two[np.argsort(top_two[:, 0])]
    bottom_left, bottom_right = bottom_two[np.argsort(bottom_two[:, 0])]

    return [
        tuple(top_left),
        tuple(top_right),
        tuple(bottom_right),
        tuple(bottom_left)
    ]

def stitch_and_rotate_north(frames_to_stitch,stitch_cfg,output_path, processed_compass_headings):
    if not check_image_sizes(frames_to_stitch):
        raise OSError

    stitched_img = None
    try:
        stitched_img = call_stitch_subprocess(frames_to_stitch, stitch_cfg)
        if stitched_img is None:
            print("stitched_img is None ... moving on")
            return  # gracefully skip

        stitched_img = crop_connected_region(stitched_img)
    except Exception as e:
        print(f"Image stitching failed: {e}")
        if stitched_img is not None:
            del stitched_img
        return

    # Extract timestamps from filenames
    timestamps = [
        int(os.path.splitext(os.path.basename(path))[0]) for path in frames_to_stitch
    ]

    # Get headings for each frame
    headings = np.array([
        get_heading(ts, processed_compass_headings) for ts in timestamps
    ])

    headings_rad = np.deg2rad(headings) 
    avg_heading = np.rad2deg(circmean(headings_rad))
 
    # Compute mean using circular statistics
    mean_sin = np.mean(np.sin(headings_rad))
    mean_cos = np.mean(np.cos(headings_rad))
    circular_mean_rad = np.arctan2(mean_sin, mean_cos)
    circular_mean_deg = np.rad2deg(circular_mean_rad) % 360

    # Compute circular standard deviation
    R = np.sqrt(mean_sin**2 + mean_cos**2)
    circular_std_deg = np.rad2deg(np.sqrt(-2 * np.log(R)))  # circular std dev

    if circular_std_deg > 15:
        print("[WARN] High variance in heading ... skipping")
        print(f"[DEBUG] Headings for frames_to_stitch:")
        print(f"  circular mean: {circular_mean_deg:.2f}°")
        print(f"  circular std: {circular_std_deg:.2f}°")
        return 
    
    stitched_img = rotate_image_north(stitched_img, avg_heading) 
    stitched_img = crop_connected_region(stitched_img) 
    stitched_img = crop_black_border(stitched_img) 

    #print(f"Writing stitched image: {chunk_path}")
    cv.imwrite(output_path, stitched_img)

    del stitched_img

def get_pano_path(output_dir):
    if not [x for x in os.listdir(output_dir) if x[-3:] == "png"]:
        return os.path.join(output_dir,"0.png") 
    filename_nums = [
            int(os.path.splitext(f)[0])
            for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f)) and f.lower().endswith(".png")
        ]
    last_pano_num = max(filename_nums)
    return os.path.join(output_dir,str(last_pano_num + 1)+".png")

def _as_paths(items):
    """Accept list of strings or frameInstance objects."""
    out = []
    for it in items:
        out.append(it.frame_path if hasattr(it, "frame_path") else it)
    return out

def _safe_polygons(paths):
    polys = []
    for p in paths:
        corners = gps_lookup(p)
        if corners is None:
            continue
        try:
            poly = create_polygon(corners)
        except Exception as e:
            print(f"[WARN] create_polygon failed for {p}: {e}")
            continue
        if poly is None or getattr(poly, "is_empty", False):
            continue
        polys.append(poly)
    return polys

def _pano_path(output_dir, prefix, cluster_idx, part_idx):
    # Make the filename deterministic & unique per cluster/part
    return os.path.join(
        output_dir,
        f"{prefix}_cluster{int(cluster_idx):04d}_part{int(part_idx):02d}.png"
    )

def stitch_clusters(cluster_idx, frame_list, output_dir,
                    processed_compass_headings, stitch_cfg, prefix):
    """
    Create one or more stitched panos for a cluster and
    return { pano_path: [Polygon, ...], ... } and write a pickle.
    """
    #print("entered stitch clusters...")
    os.makedirs(output_dir, exist_ok=True)

    paths = _as_paths(frame_list)

    # Chunks: keep the original behavior (chunk only if many frames)
    chunks = chunk_filenames(paths) if len(paths) > 5 else [paths]

    panos = {}
    for part_idx, chunk in enumerate(chunks):
        #print("chunk:",chunk)
        pano_path = _pano_path(output_dir, prefix, cluster_idx, part_idx)

        # Choose your stitcher; if you don't have rotate-north, fall back.
        stitch_and_rotate_north(chunk, stitch_cfg, pano_path, processed_compass_headings) 
        #print("stitched and rotated successfuly!") 

        polygons = _safe_polygons(chunk)
        if not polygons:
            print(f"[WARN] No valid polygons for {pano_path}; skipping key.")
            continue
        panos[pano_path] = polygons

    # Sanity check: ensure elements are shapely Polygons
    for k, v in panos.items():
        if not v:
            print(f"[WARN] Empty polygon list for {k}.")
            continue
        if isinstance(v[0], list):
            raise TypeError(f"Expected shapely Polygons, got list for key {k}")

    # Persist pickle once per cluster
    pickle_path = os.path.join(output_dir, f"{prefix}_panos_{cluster_idx}.pickle")
    with open(pickle_path, "wb") as handle:
        #print("writing:", pickle_path)
        pickle.dump(panos, handle)

    return panos
        
def stitch_and_save(frame_list, output_dir, processed_compass_headings, stitch_cfg, prefix):
    """Helper function to stitch frames and save results."""
    panos = {}
    fused_img_count = 0
    chunks = chunk_filenames(frame_list)

    
    for chunk in chunks:
        log_mem()
        stitched_img = None 

        if not check_image_sizes(chunk):
            print("Images are different sizes!!")
            for path in chunk:
                img = cv.imread(path) 
                print(img.shape)
            print("chunk:",chunk)
            raise OSError

        try:
            #stitched_img = stitcher.stitch(chunk)
            stitched_img = call_stitch_subprocess(chunk,stitch_cfg)
            if stitched_img is None:
                print("stitched_img is None ... moving on")
                continue 
            stitched_img = crop_connected_region(stitched_img)
        except Exception as e:
            print(f"Image stitching failed: {e}")
            if stitched_img:
                del stitched_img 
            continue

        avg_heading = np.mean([
            get_heading(int(os.path.splitext(os.path.basename(path))[0]), processed_compass_headings)
            for path in chunk
        ])

        # Extract timestamps from filenames
        timestamps = [
            int(os.path.splitext(os.path.basename(path))[0]) for path in chunk
        ]

        # Get headings for each frame
        headings = np.array([
            get_heading(ts, processed_compass_headings) for ts in timestamps
        ])

        headings_rad = np.deg2rad(headings) 

        # Compute mean using circular statistics
        mean_sin = np.mean(np.sin(headings_rad))
        mean_cos = np.mean(np.cos(headings_rad))
        circular_mean_rad = np.arctan2(mean_sin, mean_cos)
        circular_mean_deg = np.rad2deg(circular_mean_rad) % 360

        # Compute circular standard deviation
        R = np.sqrt(mean_sin**2 + mean_cos**2)
        circular_std_deg = np.rad2deg(np.sqrt(-2 * np.log(R)))  # circular std dev

        if circular_std_deg > 15:
            print("[WARN] High variance in heading ... skipping")
            print(f"[DEBUG] Headings for chunk:")
            print(f"  circular mean: {circular_mean_deg:.2f}°")
            print(f"  circular std: {circular_std_deg:.2f}°")
            continue 
        
        stitched_img = rotate_image_north(stitched_img, avg_heading) 
        stitched_img = crop_connected_region(stitched_img) 
        stitched_img = crop_black_border(stitched_img) 

        chunk_path = os.path.join(output_dir, f"{fused_img_count}.png")
        #print(f"Writing stitched image: {chunk_path}")
        cv.imwrite(chunk_path, stitched_img)

        del stitched_img 

        fused_corners = [gps_lookup(path) for path in chunk]
        panos[chunk_path] = fused_corners

        fused_img_count += 1

    del chunk, chunks 

    pickle_path = os.path.join(output_dir, f"{prefix}_panos.pickle")

    with open(pickle_path, "wb") as handle:
        pickle.dump(panos, handle)

    return panos


def get_heading(timestamp, compass_data):    
    timestamps = compass_data[:, 0]
    headings_deg = compass_data[:, 1]
    if math.floor(math.log10(abs(timestamp))) > math.floor(math.log10(abs(timestamps[0]))):
        idx = np.abs(timestamps - timestamp*10**(-9)).argmin()
        if np.abs(timestamps[idx] - timestamp*10**(-9)) > 0.3:
            print("no valid heading found!")
            print(timestamps[0])
            print(timestamp*10**(-9))
            print("timestamp: ",timestamp)
            print(np.abs(timestamps - timestamp*10**(-9)))
            raise OSError 
    elif math.floor(math.log10(abs(timestamp))) == math.floor(math.log10(abs(timestamps[0]))):
        idx = np.abs(timestamps - timestamp).argmin() 
        if np.abs(timestamps[idx] - timestamp) > 0.3:
            print("no valid heading found!")
            print(timestamps[0])
            print(timestamp*10**(-9))
            print("timestamp: ",timestamp)
            print(np.abs(timestamps - timestamp*10**(-9)))
            raise OSError 
    
    return headings_deg[idx]

def rotate_image_north(image, heading_deg):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_matrix = cv.getRotationMatrix2D(center, -heading_deg, 1.0)
    return cv.warpAffine(image, rot_matrix, image.shape[1::-1], flags=cv.INTER_LINEAR) 


def index_image_dir(image_dir):
    index = {}
    for f in os.listdir(image_dir):
        if not f.endswith(".png"):
            continue
        basename = os.path.splitext(f)[0]
        try:
            ts = int(basename)
            index[ts] = os.path.join(image_dir, f)
        except ValueError:
            print(f"[WARN] Could not parse timestamp from {f}")
    return index

def find_from_index(index_dict, ts_seconds, threshold_sec=0.2):
    ts_ns = int(round(ts_seconds * 1e9))  # Convert to int nanoseconds
    all_keys = np.array(list(index_dict.keys()))

    if len(all_keys) == 0:
        print("[WARN] Empty index_dict!")
        return None

    # Compute absolute time difference
    diffs = np.abs(all_keys - ts_ns)
    min_idx = np.argmin(diffs)
    min_diff = diffs[min_idx]

    if min_diff > threshold_sec * 1e9:
        print(f"[WARN] Closest image too far: {min_diff/1e9:.3f}s for {ts_ns}")
        return None

    nearest_ts = all_keys[min_idx]
    return index_dict[nearest_ts]

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
            frustrum_corners = frustrum_corners[idx, 15:23]
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