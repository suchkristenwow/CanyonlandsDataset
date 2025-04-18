import numpy as np 
import math 
import os 
from shapely.geometry import Polygon
import cv2 as cv
import itertools
import tempfile
import matplotlib.pyplot as plt 

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

def robust_load_csv(path, min_cols=4, skip_header=1):
    import numpy as np

    data = np.genfromtxt(path,skip_header=skip_header) 

    try:
        if not np.isnan(data).all():
            print(f"✅ Loaded annotations using no delimiter")
            return data
    except:
        print("path:",path)
        print(data)

    delimiters = [',', '\t', ';']
    for delim in delimiters:
        try:
            data = np.genfromtxt(path, delimiter=delim, skip_header=skip_header)
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)

            if not np.isnan(data).all() and data.shape[1] >= min_cols:
                #print(f"✅ Loaded annotations using delimiter '{delim}'")
                return data
        except Exception as e:
            print(f"⚠️ Failed to load with delimiter '{delim}': {e}")

    raise ValueError(f"❌ Failed to load usable data from {path} with common delimiters.")

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

def chunk_list(lst, chunk_size=10):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

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
    print("keeping {} images".format(len(kept_paths)))
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

def fuse_chunks(chunk_paths, stitcher, out_dir=None):
    """
    Attempts to fuse image chunks and ensures that all chunks are represented in the result.

    Returns:
        dict: {fused_image_path: [list of input chunks used]}
    """
    tmp_dir_built = False 
    if out_dir is None:
        out_dir = tempfile.mkdtemp()
        tmp_dir_built = True 
    else:
        os.makedirs(out_dir, exist_ok=True)

    unused = set(chunk_paths)
    fused_results = {}
    fuse_id = 0

    while unused:
        base = unused.pop()
        group = [base]
        used = set()

        for other in unused:
            try:
                test_paths = group + [other]
                result = stitcher.stitch(test_paths)
                if result is not None:
                    group.append(other)
                    used.add(other)
            except Exception as e:
                print(f"[✗] Failed to stitch {group[-1]} with {other}: {e}")
                continue

        unused -= used

        try:
            result = stitcher.stitch(group)
            if result is not None:
                fused_path = os.path.join(out_dir, f"fused_{fuse_id}.png")
                cv.imwrite(fused_path, result)
                fused_results[fused_path] = group
                print(f"[✓] Saved fused image: {fused_path}")
            else:
                # Fall back to using each chunk individually
                for chunk in group:
                    solo_img = cv.imread(chunk)
                    solo_out = os.path.join(out_dir, f"fused_{fuse_id}.png")
                    cv.imwrite(solo_out, solo_img)
                    fused_results[solo_out] = [chunk]
                    print(f"[•] Could not fuse — keeping {chunk} as its own")
                    fuse_id += 1
                continue
        except Exception as e:
            print(f"[✗] Final stitch failed for group: {e}")
            # Same fallback
            for chunk in group:
                solo_img = cv.imread(chunk)
                solo_out = os.path.join(out_dir, f"fused_{fuse_id}.png")
                cv.imwrite(solo_out, solo_img)
                fused_results[solo_out] = [chunk]
                print(f"[•] Exception fallback: saved {chunk} as its own")
                fuse_id += 1
            continue

        fuse_id += 1

    if tmp_dir_built:
        shutil.rmtree(out_dir) 

    return fused_results

def plot_stitched_summary_grid(fused_dict_may, fused_dict_nov, save_path, figsize=(25, 15)):
    n_may = len(fused_dict_may)
    n_nov = len(fused_dict_nov)
    n_rows = max(n_may, n_nov)
    
    print("n_may: {}, n_nov: {}".format(n_may,n_nov))
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, 3, width_ratios=[1, 1, 1])

    may_paths = list(fused_dict_may.keys())
    nov_paths = list(fused_dict_nov.keys())

    gps_ax = fig.add_subplot(gs[:, 1])
    gps_ax.set_title("Stitched Image GPS Footprints", fontsize=10)

    # Left column — May images
    for i, path in enumerate(may_paths):
        img = cv.imread(path)
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(img_rgb)
        ax.set_title(f"May {os.path.basename(path)}", fontsize=8)
        ax.axis("off")

        # Plot GPS rectangles
        for input_img in fused_dict_may[path]:
            corners = gps_lookup(input_img)
            if corners and len(corners) == 4:
                poly = MplPolygon(corners, closed=True, edgecolor='none', facecolor='blue', alpha=0.5)
                gps_ax.add_patch(poly)

    # Right column — Nov images
    for i, path in enumerate(nov_paths):
        img = cv.imread(path)
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        ax = fig.add_subplot(gs[i, 2])
        ax.imshow(img_rgb)
        ax.set_title(f"Nov {os.path.basename(path)}", fontsize=8)
        ax.axis("off")

        for input_img in fused_dict_nov[path]:
            corners = gps_lookup(input_img)
            if corners and len(corners) == 4:
                poly = MplPolygon(corners, closed=True, edgecolor='none', facecolor='red', alpha=0.5)
                gps_ax.add_patch(poly)

    gps_ax.set_xlabel("Longitude")
    gps_ax.set_ylabel("Latitude")
    gps_ax.grid(True)
    gps_ax.set_aspect('equal', adjustable='box')

    blue_patch = plt.Line2D([0], [0], color='blue', label='May')
    red_patch = plt.Line2D([0], [0], color='red', label='Nov')
    gps_ax.legend(handles=[blue_patch, red_patch], loc="upper right", fontsize=8)
    plt.show() 
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[✓] Saved stitched summary with GPS footprints to: {save_path}")

def gps_lookup(filepath):
    """
    Given the filepath of an image, return the gps coordinate of its corners
    """
    result_dir = os.path.dirname(filepath)
    frustrum_corners = robust_load_csv(os.path.join(result_dir,"processed_results/frustrum_corners.csv"))
    timestamp = int(os.path.splitext(os.path.basename(filepath))[0]) * 10**(-9)
    idx = np.argmin(frustrum_corners[:,0] - timestamp)
    if np.abs(timestamp - frustrum_corners[idx,0]) > 0.2:
        raise OSError 

    if "May" in filepath:   
        frustrum_corners = frustrum_corners[idx,15:] 
    elif "Nov" in filepath: 
        if "Left" in filepath:    
            frustrum_corners = frustrum_corners[idx,17:23]
        elif "Right" in filepath: 
            frustrum_corners =  frustrum_corners[idx,25:33]

    corners = []
    for i in range(4):
        lat_i = frustrum_corners[i*2]
        lon_i = frustrum_corners[i*2 + 1]
        corners.append((lat_i,lon_i)) 
        
    return corners