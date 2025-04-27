import numpy as np 
import math 
import os 
from shapely.geometry import Polygon, MultiPoint
import cv2 as cv
import itertools
import tempfile
import matplotlib.pyplot as plt 
import shutil 
import gc 
from matplotlib.patches import Polygon as MplPolygon 
import psutil 
from seasonal_comparison.gps_utils import debug_corner_distances 
from seasonal_comparison.general_utils import robust_load_csv

MAX_FUSED_IMG_PX = 250*10**(3)

def plot_invalid_polygon(corners, title="Invalid Polygon"):
    fig, ax = plt.subplots()
    
    try:
        # Try creating it
        poly = Polygon(corners)

        if not poly.is_empty:
            x, y = poly.exterior.xy
            ax.plot(x, y, marker='o', linestyle='-', color='red')
        else:
            print("Polygon is empty.")
        
        # Also plot points individually
        for lon, lat in corners:
            ax.plot(lon, lat, 'bo')  # blue dots for raw points

        ax.set_aspect('equal')
        ax.set_title(title)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.savefig("./debug_polygon.png")
        plt.close() 
        
    except Exception as e:
        print(f"Error while plotting: {e}")

def chunk_filenames(filenames, chunk_size=5):
    return [filenames[i:i + chunk_size] for i in range(0, len(filenames), chunk_size)]

def check_all_same_size(image_paths):
    """
    Returns True if all images have the same (height, width), otherwise False.
    Also returns the shape of the first image and the path of any mismatched ones.
    """
    if not image_paths:
        print("[!] No image paths provided.")
        return True, None, []

    ref_shape = None
    mismatches = []

    for i, path in enumerate(image_paths):
        img = cv.imread(path)
        if img is None:
            print(f"[✗] Failed to load image: {path}")
            continue

        h, w = img.shape[:2]

        if ref_shape is None:
            ref_shape = (h, w)
        elif (h, w) != ref_shape:
            mismatches.append((path, (h, w)))

    all_same = len(mismatches) == 0
    return all_same, ref_shape, mismatches

def check_corner_uniqueness(corners, tol=1e-7):
    """
    Parameters:
        corners (list of (lon, lat)): List of tuples with GPS coordinates.
        tol (float): Tolerance for coordinate equality (in degrees).
    
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

def is_memory_critical(threshold=0.90):
    """Returns True if memory usage is above threshold (e.g., 90%)"""
    mem = psutil.virtual_memory()
    return mem.percent / 100.0 > threshold

def downscale_images(image_paths, scale=0.5):
    downscaled_paths = []
    for path in image_paths:
        img = cv.imread(path)
        if img is None:
            continue
        resized = cv.resize(img, (0, 0), fx=scale, fy=scale)
        temp_path = path.replace(".png", "_downscaled.png")
        cv.imwrite(temp_path, resized)
        downscaled_paths.append(temp_path)
    return downscaled_paths

def get_yaw_from_quaternion(qx, qy, qz, qw):
    # Assuming robot moves in 2D (flat ground)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

def polar_to_cartesian(range_m, bearing_rad):
    return np.array([
        range_m * np.cos(bearing_rad),
        range_m * np.sin(bearing_rad)
    ])

def cartesian_to_polar(vec):
    x, y = vec
    range_m = np.linalg.norm(vec)
    bearing_rad = np.arctan2(y, x)
    return range_m, bearing_rad

def get_ouster_to_corner(gps_to_cam_range, gps_to_cam_bearing,
                         cam_to_corner_range, cam_to_corner_bearing,
                         gps_to_ouster_range, gps_to_ouster_bearing):

    # Convert all vectors to Cartesian in GPS frame
    gps_to_cam = polar_to_cartesian(gps_to_cam_range, gps_to_cam_bearing)
    gps_to_ouster = polar_to_cartesian(gps_to_ouster_range, gps_to_ouster_bearing)

    # Rotate cam_to_corner into GPS frame using gps→cam bearing as camera yaw
    cam_to_corner_local = polar_to_cartesian(cam_to_corner_range, cam_to_corner_bearing)
    theta = gps_to_cam_bearing  # assuming this is also the camera's heading
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    cam_to_corner_global = rotation_matrix @ cam_to_corner_local

    # Compute ouster→corner
    ouster_to_corner = (gps_to_cam + cam_to_corner_global) - gps_to_ouster

    # Convert to range + bearing
    return cartesian_to_polar(ouster_to_corner)

def compute_corner_positions_from_ouster_pose(ouster_pose, corner_offsets):
    """
    Parameters:
    - ouster_pose: numpy array of shape (8,) -> [t, x, y, z, qx, qy, qz, qw]
    - corner_offsets: list of (range, bearing) tuples in the ouster frame

    Returns:
    - List of (x, y) global coordinates of each corner
    """
    _, x, y, _, qx, qy, qz, qw = ouster_pose
    yaw = get_yaw_from_quaternion(qx, qy, qz, qw)

    R = np.array([
        [math.cos(yaw), -math.sin(yaw)],
        [math.sin(yaw),  math.cos(yaw)]
    ])

    world_corners = []
    for r, theta in corner_offsets:
        local_vec = polar_to_cartesian(r, theta)
        global_vec = R @ local_vec
        corner_world = np.array([x, y]) + global_vec
        world_corners.append(tuple(corner_world))

    return world_corners

def make_image_path_dict(image_paths):
    """
    This function maps the image path to the LIOSAM odometry 
    """
    base_dir = os.path.dirname(os.path.dirname(image_paths[0]))
    processed_dir = os.path.join(base_dir,"processed_results")
    liosam_odometry = robust_load_csv(os.path.join(processed_dir,"liosam_odometry.csv"))
    image_path_coords = {}
    for image_path in image_paths:
        if "downscaled" in image_path:
            continue 
        timestamp_nsecs = float(os.path.splitext(os.path.basename(image_path))[0])

        idx = np.argmin(np.abs(liosam_odometry[:,0] - timestamp_nsecs * 10**(-9)))

        if np.abs(timestamp_nsecs* 10**(-9) - liosam_odometry[idx,0]) > 0.3:
            print(timestamp_nsecs* 10**(-9))
            print(liosam_odometry[idx,0])
            print(np.abs(timestamp_nsecs* 10**(-9) - liosam_odometry[idx,0]))
            print("WARNING:NO CORRESPONDING TIMESTAMP in the liosam data")
            continue 
    
        liosam_row_data = liosam_odometry[idx,:] 
        frame_corners = {
                'topRight': (0.0724, 0.78),
                'bottomRight': (0.0747, -1.07),
                'bottomLeft': (0.0745, -2.34),
                'topLeft': (0.0721, 2.09)
            }
        r3,b3 = 0.664, 0.192 #gps to ouster
        global_cam_offsets = [] 
        if "May" in image_path:
            r1,b1 = 0.703, 0.192 #gps -> camera
            for corner in frame_corners:
                r2,b2 = frame_corners[corner]
                range_oc, bearing_oc = get_ouster_to_corner(r1, b1, r2, b2, r3, b3)
                global_cam_offsets.append((range_oc,bearing_oc))
        if "Nov" in image_path: 
            if "Right" in image_path:
                r1,b1 = 0.679, 0.269 #gps -> camera 
                for corner in frame_corners:
                    r2,b2 = frame_corners[corner] 
                    range_oc, bearing_oc = get_ouster_to_corner(r1, b1, r2, b2, r3, b3)
                    global_cam_offsets.append((range_oc,bearing_oc))
            elif "Left" in image_path: 
                r1,b1 = 0.679, 0.132
                for corner in frame_corners:
                    r2,b2 = frame_corners[corner]
                    range_oc, bearing_oc = get_ouster_to_corner(r1, b1, r2, b2, r3, b3) 
                    global_cam_offsets.append((range_oc,bearing_oc)) 

        world_corners = compute_corner_positions_from_ouster_pose(liosam_row_data,global_cam_offsets)
        image_path_coords[image_path] = world_corners 

    return image_path_coords

def stitch_within_size_limit(chunk, stitcher, tmp_dir, chunk_id, max_px=MAX_FUSED_IMG_PX, max_recursion=2):
    """
    Returns:
        dict {output_path: [input_image_paths]}
    """
    result_dict = {}

    if not chunk or max_recursion < 0 or len(chunk) < 2:
        print(f"[!] Cannot stitch chunk {chunk_id} — falling back to saving individual images")
        for j, path in enumerate(chunk):
            img = cv.imread(path)
            if img is None:
                continue
            out_path = os.path.join(tmp_dir, f"chunk_{chunk_id}_{j}.png")
            cv.imwrite(out_path, img)
            result_dict[out_path] = [path]
        return result_dict

    est_fused_px, _ = estimate_fused_image_size(chunk)
    print(f"est_fused_px: {est_fused_px}")

    if est_fused_px < max_px:
        if not check_all_same_size(chunk)[0]:
            raise OSError(f"[✗] Images in chunk {chunk_id} not same size")
        stitched_img = stitcher.stitch(chunk)
        if stitched_img is not None:
            stitched = crop_connected_region(stitched_img)
            out_path = os.path.join(tmp_dir, f"chunk_{chunk_id}.png")
            cv.imwrite(out_path, stitched)
            result_dict[out_path] = chunk
            print(f"[✓] Wrote stitched chunk {chunk_id} to {out_path}")
            del stitched_img, stitched
        else:
            print(f"[✗] Stitching returned None for chunk {chunk_id}")
    else:
        print(f"[⚠️] Chunk {chunk_id} too large — trying to split further")
        mid = len(chunk) // 2
        left_results = stitch_within_size_limit(chunk[:mid], stitcher, tmp_dir, f"{chunk_id}_0", max_px, max_recursion - 1)
        right_results = stitch_within_size_limit(chunk[mid:], stitcher, tmp_dir, f"{chunk_id}_1", max_px, max_recursion - 1)
        result_dict.update(left_results)
        result_dict.update(right_results)

    return result_dict

def chunk_list(image_paths, max_delta_t=0.5, max_chunk_size=5):
    """
    Groups image paths into chunks where timestamps are within max_delta_t seconds.
    Each chunk is capped at max_chunk_size.

    Args:
        image_paths (list of str): Paths to images, with timestamps in filenames.
        max_delta_t (float): Maximum allowed time difference between consecutive images.
        max_chunk_size (int): Maximum number of images per chunk.

    Returns:
        List of chunks (list of lists of paths).
    """
    # Filter and sort
    paths_with_time = [(p, extract_timestamp_from_path(p)) for p in image_paths]
    paths_with_time = [(p, t) for p, t in paths_with_time if t is not None]
    paths_with_time.sort(key=lambda x: x[1])  # sort by time
    #print("paths_with_time: ",paths_with_time)

    chunks = []
    current_chunk = []

    for i, (path, ts) in enumerate(paths_with_time):
        if not current_chunk:
            current_chunk.append((path, ts))
            continue

        _, last_ts = current_chunk[-1]
        #print("ts-last_ts:",abs(ts - last_ts))
        if abs(ts - last_ts) <= max_delta_t and len(current_chunk) < max_chunk_size:
            current_chunk.append((path, ts))
        else:
            if len(current_chunk) >= 2:
                chunks.append([p for p, _ in current_chunk])
            current_chunk = [(path, ts)]

    # Add final chunk if valid
    if len(current_chunk) >= 2:
        chunks.append([p for p, _ in current_chunk])

    return chunks

def filter_imgs_by_overlap(image_path_coords, iou_threshold=0.95):
    polygons = {}
    for path, corners in image_path_coords.items():
        try:
            poly = Polygon(corners)
            if poly.is_valid and not poly.is_empty:
                polygons[path] = poly
        except Exception as e:
            print(f"[!] Failed to create polygon for {path}: {e}")

    kept_paths = set(polygons.keys())

    paths = list(polygons.keys())

    for i, path_i in enumerate(paths):
        poly_i = polygons[path_i]
        has_overlap = False
        for j, path_j in enumerate(paths):
            if i == j:
                continue
            poly_j = polygons[path_j]

            intersection_area = poly_i.intersection(poly_j).area
            union_area = poly_i.union(poly_j).area
            iou = intersection_area / union_area if union_area > 0 else 0

            if iou > 0:
                has_overlap = True
            
            if iou > iou_threshold:
                # Remove the one with the higher path (arbitrary tie-breaker)
                # print("this path has a lot of overlap")
                removed = max(path_i, path_j)
                kept_paths.discard(removed)
        
        if not has_overlap:
            # print("this path has no overlap ... removing")
            kept_paths.discard(path_i)
    #print("keeping {} images".format(len(kept_paths)))
    return sorted(kept_paths)

def crop_connected_region(image, area_thresh_ratio=0.05):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 15, 255, cv.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity=8)

    if num_labels <= 1:
        print("[!] No connected regions found — skipping crop.")
        return image

    # Find label of the largest non-background component
    largest_label = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])

    mask = (labels == largest_label).astype(np.uint8) * 255

    # Apply mask
    cleaned = cv.bitwise_and(image, image, mask=mask)

    # Crop to bounding box
    coords = cv.findNonZero(mask)
    x, y, w, h = cv.boundingRect(coords)
    cropped = cleaned[y:y+h, x:x+w]

    return cropped

def resize_and_save_images_uniform(image_paths, tmp_dir, target_size=None):
    """
    Resize all images to a uniform size (width, height), save to tmp_dir, and return new paths.
    """
    resized_paths = []
    for i, path in enumerate(image_paths):
        img = cv.imread(path)
        if img is None:
            raise IOError(f"Failed to load image: {path}")

        if target_size is None:
            target_size = (img.shape[1], img.shape[0])  # (width, height)

        resized = cv.resize(img, target_size)
        new_path = os.path.join(tmp_dir, f"uniform_{i}.png")
        cv.imwrite(new_path, resized)
        resized_paths.append(new_path)

    return resized_paths

def estimate_fused_image_size(image_paths):
    """
    Roughly estimate the total width and height of the stitched image,
    assuming simple horizontal or grid-like layout.

    Returns:
        total_pixels (int), (width, height)
    """
    total_width = 0
    max_height = 0

    for path in image_paths:
        img = cv.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        total_width += w  # side-by-side assumption
        max_height = max(max_height, h)
        del img

    total_pixels = total_width * max_height
    return total_pixels, (total_width, max_height)

def extract_timestamp_from_path(path):
    name = os.path.basename(path)
    if "downscaled" in name:
        name = name.split("_downscaled")[0]
    if "png" in name:
        name = name[:-4]
    try:
        return int(name) * 1e-9
    except:
        print("Error: Could not extract timestamp!")
        print("name: ",name)
        print(os.path.basename(path))
        print("path:",path)
        raise OSError

def fuse_chunks(chunk_paths, stitcher, out_dir=None, provenance_map=None):
    """
    Attempts to fuse image chunks and ensures that all chunks are represented in the result.

    Returns:
        dict: {fused_image_path: [list of input chunks used]}
    """
    print("fusing chunks ...")
    #Estimated fused image size:  (248668, (1162, 214))

    downscaled_chunk_paths = downscale_images(chunk_paths)
    resize_tmp_dir = tempfile.mkdtemp()

    tmp_dir_built = False
    if out_dir is None:
        out_dir = tempfile.mkdtemp()
        tmp_dir_built = True
    else:
        os.makedirs(out_dir, exist_ok=True)

    # Pre-resize all images once
    print("Preprocessing resized images...")
    resized_path_map = {}
    target_size = None
    for i, orig_path in enumerate(downscaled_chunk_paths):
        img = cv.imread(orig_path)
        if img is None:
            continue
        if target_size is None:
            target_size = (img.shape[1], img.shape[0])  # width, height
        resized = cv.resize(img, target_size)
        resized_path = os.path.join(resize_tmp_dir, f"{i}.png")
        cv.imwrite(resized_path, resized)
        resized_path_map[orig_path] = resized_path
        del img, resized
        gc.collect()

    unused = set(downscaled_chunk_paths)
    fused_results = {}
    fuse_id = 0

    mem = psutil.virtual_memory()
    print(f"[MEM] Used: {mem.used / 1e9:.2f} GB / {mem.total / 1e9:.2f} GB ({mem.percent}%)")

    print("Entering the while loop.")
    while unused:
        base = unused.pop()
        group = [base]
        used = set()

        for other in unused:
            mem = psutil.virtual_memory()
            print(f"[MEM] Used: {mem.used / 1e9:.2f} GB / {mem.total / 1e9:.2f} GB ({mem.percent}%)")

            test_paths = group + [other]
            
            if is_memory_critical(threshold=0.8):
                print("[⚠️] Memory usage high — skipping further fusing for current group.")
                break  # Stop trying to add more images to this group

            try:
                uniform_paths = [resized_path_map[p] for p in test_paths]
                #print("trying to fuse: {} paths".format(len(uniform_paths)))
                #print("Estimated fused image size: ",estimate_fused_image_size(uniform_paths))
                est_total_px, _ = estimate_fused_image_size(uniform_paths) 
                if MAX_FUSED_IMG_PX <= est_total_px:
                    print("[⚠️] WARNING Estimated image size too large.")
                    continue 
                
                if is_memory_critical(threshold=0.75):
                    print("Memory is critical - skipping stitching.")
                    continue 
                if not check_all_same_size(uniform_paths):
                    raise OSError 
                result = stitcher.stitch(uniform_paths)
                if result is not None:
                    #print("No result found :(")
                    group.append(other)
                    used.add(other)
                #print("Stitching was safely completed!")
                if len(group) >= 4:
                    print("[ℹ️] Group size cap reached — stopping additions.")
                    break
                    
                del result
                gc.collect()
            except Exception as e:
                print(f"[✗] Failed to stitch {group[-1]} with {other}: {e}")
                continue

        unused -= used

        try:
            uniform_paths = [resized_path_map[p] for p in group]
            est_total_px,_ = estimate_fused_image_size(uniform_paths) 
            if MAX_FUSED_IMG_PX <= est_total_px:
                print("[⚠️] WARNING Estimated image size too large.")
                continue 
            if not check_all_same_size(uniform_paths):
                raise OSError
            result = stitcher.stitch(uniform_paths)
            if result is not None:
                result = crop_connected_region(result)
                fused_path = os.path.join(out_dir, f"fused_{fuse_id}.png")
                cv.imwrite(fused_path, result)

                if provenance_map:
                    originals = []
                    for chunk_path in group:
                        originals.extend(provenance_map.get(chunk_path, [chunk_path]))
                    fused_results[fused_path] = originals
                else:
                    fused_results[fused_path] = group

                print(f"[✓] Saved fused image: {fused_path}")
                del result
                gc.collect()
            else:
                #print("result is None...")
                for chunk in group:
                    solo_img = cv.imread(chunk)
                    solo_out = os.path.join(out_dir, f"fused_{fuse_id}.png")
                    cv.imwrite(solo_out, solo_img)
                    del solo_img
                    if provenance_map:
                        originals = provenance_map.get(chunk, [chunk])
                        fused_results[solo_out] = originals
                    else:
                        fused_results[solo_out] = [chunk]
                    print(f"[•] Could not fuse — keeping {chunk} as its own")
                    fuse_id += 1
                    if len(group) >= 4:
                        print("Group size cap reached — stopping additions.")
                        break
                        
                continue

        except Exception as e:
            print(f"[✗] Final stitch failed for group: {e}")
            for chunk in group:
                solo_img = cv.imread(chunk)
                solo_out = os.path.join(out_dir, f"fused_{fuse_id}.png")
                cv.imwrite(solo_out, solo_img)
                del solo_img
                if provenance_map:
                    originals = provenance_map.get(chunk, [chunk])
                    fused_results[solo_out] = originals
                else:
                    fused_results[solo_out] = [chunk]
                print(f"[•] Exception fallback: saved {chunk} as its own")
                fuse_id += 1
            continue

        fuse_id += 1
        print()

    if tmp_dir_built:
        shutil.rmtree(out_dir)
    shutil.rmtree(resize_tmp_dir)

    return fused_results

def get_unique_sorted_corners(corners, debug_path=None):
    try:
        if len(corners) != 4:
            print(f"[⚠️] Expected 4 corners, got {len(corners)} — {debug_path}")
            return []

        centroid = MultiPoint(corners).centroid
        corners.sort(key=lambda pt: np.arctan2(pt[1] - centroid.y, pt[0] - centroid.x))

        unique = []
        for pt in corners:
            if not any(np.linalg.norm(np.array(pt) - np.array(other)) < 1e-9 for other in unique):
                unique.append(pt)

        if len(unique) < 4:
            print(f"[⚠️] Only {len(unique)} unique corners after sorting — {debug_path}")
            print(f"  Corners: {corners}")
            return []
        return unique
    except Exception as e:
        print(f"[✗] Failed to process corners for {debug_path}: {e}")
        return []

def plot_stitched_summary_grid(fused_dict_may, fused_dict_nov, save_path, figsize=(20, 15)):
    def get_image_size(path):
        img = cv.imread(path)
        if img is None:
            return 0
        h, w = img.shape[:2]
        return h * w

    def safe_gps_lookup(path):
        try:
            return gps_lookup(path)
        except Exception as e:
            print(f"[!] gps_lookup failed for {path}: {e}")
            return None

    # --- Filter to top 6 largest images per season ---
    may_sizes = [(p, get_image_size(p)) for p in fused_dict_may]
    top_may = {p for p, _ in sorted(may_sizes, key=lambda x: x[1], reverse=True)[:6]}
    fused_dict_may = {p: fused_dict_may[p] for p in top_may}

    nov_sizes = [(p, get_image_size(p)) for p in fused_dict_nov]
    top_nov = {p for p, _ in sorted(nov_sizes, key=lambda x: x[1], reverse=True)[:6]}
    fused_dict_nov = {p: fused_dict_nov[p] for p in top_nov}

    # --- Layout ---
    n_rows = max(len(fused_dict_may), len(fused_dict_nov), 1)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, 3, width_ratios=[1, 1.5, 1])  # wider center

    may_paths = list(fused_dict_may.keys())
    nov_paths = list(fused_dict_nov.keys())

    gps_ax = fig.add_subplot(gs[:, 1])
    gps_ax.set_title("Stitched Image GPS Footprints", fontsize=10)

    # --- Plot May stitched images ---
    for i, path in enumerate(may_paths):
        img = cv.imread(path)
        if img is None:
            continue
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(img_rgb)
        ax.set_title(f"May {path.split('/')[-1]}", fontsize=8)
        ax.axis("off")

        for sub_path in fused_dict_may[path]:
            corners = safe_gps_lookup(sub_path)
            if corners is None: continue
            corners = get_unique_sorted_corners(corners)
            if len(corners) < 4: 
                print("corners:",corners)
                raise OSError
            valid_corners = debug_corner_distances(corners)
            if valid_corners:
                poly = MplPolygon(corners, closed=True, facecolor='blue', alpha=0.15, edgecolor='none')
                gps_ax.add_patch(poly)

    # --- Plot Nov stitched images ---
    for i, path in enumerate(nov_paths):
        img = cv.imread(path)
        if img is None:
            continue
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        ax = fig.add_subplot(gs[i, 2])
        ax.imshow(img_rgb)
        ax.set_title(f"Nov {path.split('/')[-1]}", fontsize=8)
        ax.axis("off")

        for sub_path in fused_dict_nov[path]:
            corners = safe_gps_lookup(sub_path)
            if corners is None: continue
            corners = get_unique_sorted_corners(corners)
            if len(corners) < 4: 
                print(corners)
                raise OSError

            valid_corners = debug_corner_distances(corners)
            if valid_corners: 
                poly = MplPolygon(corners, closed=True, facecolor='red', alpha=0.15, edgecolor='none')
                gps_ax.add_patch(poly)

    # --- Finalize GPS plot ---
    gps_ax.set_xlabel("Longitude")
    gps_ax.set_ylabel("Latitude")
    gps_ax.set_aspect('equal', adjustable='box')
    gps_ax.grid(True)
    gps_ax.legend(handles=[
        plt.Line2D([0], [0], color='blue', label='May'),
        plt.Line2D([0], [0], color='red', label='Nov')
    ], fontsize=8, loc='upper right')

    # Set GPS plot bounds
    all_lons, all_lats = [], []
    for d in [fused_dict_may, fused_dict_nov]:
        for paths in d.values():
            for p in paths:
                corners = safe_gps_lookup(p)
                if corners and len(corners) == 4:
                    lons, lats = zip(*corners)
                    all_lons.extend(lons)
                    all_lats.extend(lats)
    if all_lons and all_lats:
        gps_ax.set_xlim(min(all_lons), max(all_lons))
        gps_ax.set_ylim(min(all_lats), max(all_lats))
    else:
        print("[⚠️] No valid GPS corners found — using default bounds.")

    # --- Save ---
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[✓] Saved stitched summary with GPS footprints to: {save_path}")


def gps_lookup(filepath):
    """
    Given the filepath of an image, return the gps coordinate of its corners
    """
    #print("entered gps lookup ...")
    result_dir = os.path.dirname(os.path.dirname(filepath))
    if "Panos" in filepath:
        #result_dir is in processed results already
        result_dir = os.path.dirname(result_dir)
        #print("Result_dir: ",result_dir)
        #print("trying to load in this path:",os.path.join(result_dir,"frustrum_corners.csv"))
        frustrum_corners = robust_load_csv(os.path.join(result_dir,"frustrum_corners.csv"))
    else: 
        #go up one level, into processed results
        frustrum_corners = robust_load_csv(os.path.join(result_dir,"processed_results/frustrum_corners.csv"))
    #print("filepath: ",filepath)
    if "downscaled" in filepath:
        cleaned_filepath = os.path.splitext(os.path.basename(filepath))[0]
        underscore_idx = cleaned_filepath.index("_")
        timestamp = int(cleaned_filepath[:underscore_idx]) * 10**(-9)
    else:
        #print("filepath:",filepath)
        timestamp = int(os.path.splitext(os.path.basename(filepath))[0]) * 10**(-9)

    if not "Panos" in filepath:
        idx = np.argmin(np.abs(frustrum_corners[:,0] - timestamp))
        #print(f"[gps_lookup] Closest match to {timestamp:.6f} is {frustrum_corners[idx,0]:.6f} (Δt={abs(timestamp - frustrum_corners[idx,0]):.3f}s)")
        if np.abs(timestamp - frustrum_corners[idx,0]) > 0.3:
            delta_t = np.abs(timestamp - frustrum_corners[idx,0]) 
            print("timestamp:",timestamp)
            print("closest timestamp frustrum_corners:",frustrum_corners[idx,0])
            print("delta_t:",delta_t)
            input("WARNING ... CANNOT FIND CORRESPONDING FRUSTRUM CORNERS")
            return None 

    if "May" in filepath:   
        frustrum_corners = frustrum_corners[idx,17:] 
    elif "Nov" in filepath: 
        if "Left" in filepath:    
            frustrum_corners = frustrum_corners[idx,17:25]
        elif "Right" in filepath: 
            frustrum_corners =  frustrum_corners[idx,25:33]

    if frustrum_corners.shape[0] != 8:
        print("filepath: ",filepath) 
        print("len(frustrum_corners[idx,:]): ",len(frustrum_corners))
        print(f"[!] Expected 8 GPS corner values, got {frustrum_corners.shape[0]} for {filepath}")
        raise OSError

    corners = []
    for i in range(4):
        lat_i = frustrum_corners[i*2]
        lon_i = frustrum_corners[i*2 + 1]
        corners.append((lon_i, lat_i))  # correct order: (x, y)
   
    if len(corners) < 4:
        raise OSError 

    return corners

def find_closest_file(directory, target_ts_sec, tolerance_sec=0.1):
    """
    Find the closest .png file in the directory based on the target timestamp (seconds),
    where filenames are nanosecond timestamps. Only return if within tolerance (seconds).
    
    Args:
        directory (str): Path to the directory containing .png files.
        target_ts_sec (float): Target timestamp in seconds.
        tolerance_sec (float): Acceptable time difference in seconds.

    Returns:
        str or None: Full path to the closest matching file, or None if no match.
    """
    files = [f for f in os.listdir(directory) if f.endswith('.png')]

    if not files:
        return None

    # Parse timestamps from filenames (remove .png, convert to seconds)
    ts_file_list = []
    for f in files:
        fname_no_ext = os.path.splitext(f)[0]
        try:
            ts_ns = int(fname_no_ext)
            ts_sec = ts_ns * 1e-9
            ts_file_list.append((ts_sec, f))
        except ValueError:
            continue  # skip files that don't have numeric names

    if not ts_file_list:
        return None

    # Find the file with minimum time difference
    closest = min(ts_file_list, key=lambda x: abs(x[0] - target_ts_sec))
    closest_diff = abs(closest[0] - target_ts_sec)

    if closest_diff <= tolerance_sec:
        return os.path.join(directory, closest[1])
    else:
        print("Could not find a filename within the desired threshold!")
        return None