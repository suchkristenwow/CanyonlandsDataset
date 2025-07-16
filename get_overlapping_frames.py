#!/usr/bin/env python3

"""
Fuse overlapping seasonal frames based on covariance ellipses
and generate comparison plots.
"""

import os
import csv
import gc
import pickle

import numpy as np
import cv2 as cv
cv.ocl.setUseOpenCL(False)

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.ioff() 
from shapely.geometry import Polygon
from shapely.ops import unary_union

# Local imports
from stitching import AffineStitcher, Stitcher 
from seasonal_comparison.gps_utils import (
    precompute_timestamps,
    load_covariance_matrix,
    find_frames_inside_ellipse,
    scale_covariance_to_degrees,
)
from seasonal_comparison.image_stitching_utils import (
    crop_connected_region,
    chunk_filenames,
    find_closest_file,
    gps_lookup,
    check_image_sizes,
    crop_black_border
)
from seasonal_comparison.general_utils import (
    robust_load_csv,
    log_mem,
    find_closest_index,
    load_config,
    parse_args,
)
from seasonal_comparison.plot_figs import (
    make_gps_plot,
    make_comparison_fig,
)

import shutil 
import tracemalloc
tracemalloc.start()
import threading 
import time 

import subprocess
import uuid

def call_stitch_subprocess(frame_list,stitch_cfg):
    result = subprocess.run(["python3", "src/seasonal_comparison/img_stitch_worker.py", frame_list, stitch_cfg], check=True)
    return result

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

def rotate_image_north(image, heading_deg):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_matrix = cv.getRotationMatrix2D(center, -heading_deg, 1.0)
    return cv.warpAffine(image, rot_matrix, image.shape[1::-1], flags=cv.INTER_LINEAR) 

def get_heading(timestamp, compass_data):    
    timestamps = compass_data[:, 0]
    headings_deg = compass_data[:, 1]
    idx = np.abs(timestamps - timestamp*10**(-9)).argmin()
    if np.abs(timestamps[idx] - timestamp*10**(-9)) > 0.3:
        print("no valid heading found!")
        print(timestamps[0])
        print(timestamp*10**(-9))
        print("timestamp: ",timestamp)
        print(np.abs(timestamps - timestamp*10**(-9)))
        raise OSError 
    return headings_deg[idx]

def fuse_front_facing_images(timestamp, may_front_frames, nov_front_frames, paths):
    print("[INFO] Starting front-facing image fusion thread...")
    output_dir_timestamp = os.path.join(paths["match_output_dir"], str(timestamp))
    front_facing_img_dir = os.path.join(output_dir_timestamp, "front_facing_imgs")
    os.makedirs(front_facing_img_dir, exist_ok=True)

    may_front_facing_img_dir = os.path.join(front_facing_img_dir, "May")
    os.makedirs(may_front_facing_img_dir, exist_ok=True)

    nov_front_facing_img_dir = os.path.join(front_facing_img_dir, "Nov")
    os.makedirs(nov_front_facing_img_dir, exist_ok=True)

    stitcher = Stitcher(detector="sift", confidence_threshold=0.2)

    for src_path in may_front_frames:
        filename = os.path.basename(src_path)
        shutil.copy(src_path, os.path.join(may_front_facing_img_dir, filename))

    try:
        stitched_front_img = stitcher.stitch(may_front_frames)
        out_path = os.path.join(may_front_facing_img_dir, "stitched_front.jpg")
        print(f"Writing: {out_path}")
        cv.imwrite(out_path, stitched_front_img)
        del stitched_front_img 
        cv.destroyAllWindows() 
    except Exception as e:
        print(f"[ERROR] May front-facing stitching failed: {e}")

    for src_path in nov_front_frames:
        filename = os.path.basename(src_path)
        shutil.copy(src_path, os.path.join(nov_front_facing_img_dir, filename))

    try:
        stitched_front_img = stitcher.stitch(nov_front_frames)
        out_path = os.path.join(nov_front_facing_img_dir, "stitched_front.jpg")
        print(f"Writing: {out_path}")
        cv.imwrite(out_path, stitched_front_img)
        del stitched_front_img
        cv.destroyAllWindows() 
    except Exception as e:
        print(f"[ERROR] Nov front-facing stitching failed: {e}")

    gc.collect()
    print("[INFO] Finished front-facing image fusion thread.")


def main():
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    stitch_cfg = config.get("stitching", {})
    
    os.makedirs(paths["match_output_dir"], exist_ok=True)

    may_image_index        = index_image_dir(paths["may_images"])
    may_front_image_index = index_image_dir(paths["may_front_images"])
    nov_left_image_index  = index_image_dir(paths["nov_left_images"])
    nov_right_image_index = index_image_dir(paths["nov_right_images"])
    nov_front_image_index = index_image_dir(paths["nov_front_images"])

    # Load GPS data
    may_data = np.genfromtxt(
        os.path.join(paths["may_results"], "frustrum_corners.csv"),
        delimiter=",",
        skip_header=1
    )
    nov_data = np.genfromtxt(
        os.path.join(paths["nov_results"], "frustrum_corners.csv"),
        delimiter=",",
        skip_header=1
    )

    may_timestamps = may_data[:, 0]
    cam_lat, cam_lon = may_data[:, 15], may_data[:, 16]
    nov_timestamps = nov_data[:, 0]
    left_cam_lat, left_cam_lon = nov_data[:, 23], nov_data[:, 24]
    right_cam_lat, right_cam_lon = nov_data[:, 33], nov_data[:, 34]
    '''
    writer.writerow(['timestamp', 'lat', 'lon', 'pointer_lat', 'pointer_lon',
        'ouster_lat', 'ouster_lon', 'frontLeftWheel_lat', 'frontLeftWheel_lon',
        'frontRightWheel_lat', 'frontRightWheel_lon', 'rearLeftWheel_lat', 'rearLeftWheel_lon',
        'rearRightWheel_lat', 'rearRightWheel_lon', 'cam_lat', 'cam_lon',
        'topRight_frame_lat', 'topRight_frame_lon', 'topLeft_frame_lat', 'topLeft_frame_lon',
        'bottomRight_frame_lat', 'bottomRight_frame_lon', 'bottomLeft_frame_lat', 'bottomLeft_frame_lon'])
    ''' 

    may_cov_dir = os.path.join(paths["may_results"], "covariance_matrices")
    may_cov_timestamps = precompute_timestamps(may_cov_dir)
    
    may_compass_headings = np.genfromtxt(
        os.path.join(paths["may_results"], "processed_compass_heading.csv"),
        delimiter=",",
        skip_header=1
    )

    nov_compass_headings = np.genfromtxt(
        os.path.join(paths["nov_results"], "processed_compass_heading.csv"),
        delimiter=",",
        skip_header=1
    )

    for i, timestamp in enumerate(may_cov_timestamps):
        print(f"\n[{i+1}/{len(may_cov_timestamps)}] Processing timestamp {timestamp}")

        cov_matrix = load_covariance_matrix(may_cov_dir, may_cov_timestamps, timestamp)
        if cov_matrix is None:
            print("[WARN] No matching covariance for this timestamp.")
            continue

        scaled_cov = scale_covariance_to_degrees(cov_matrix)
        i_may = find_closest_index(may_timestamps, timestamp)
        center = (cam_lon[i_may], cam_lat[i_may])

        # Find overlapping frames
        print("[INFO] Looking for overlapping frames...")
        may_frames_tuple = find_frames_inside_ellipse(scaled_cov, center, cam_lon, cam_lat, may_timestamps)
        left_frames_tuple = find_frames_inside_ellipse(scaled_cov, center, left_cam_lon, left_cam_lat, nov_timestamps)
        right_frames_tuple = find_frames_inside_ellipse(scaled_cov, center, right_cam_lon, right_cam_lat, nov_timestamps)

        if not (may_frames_tuple or left_frames_tuple or right_frames_tuple):
            print("[WARN] No overlapping frames found.")
            continue
        
        print("[INFO] finding frames ...")

        #print("[DEBUG] Trying to look up these timestamps:", [ts[0] for ts in may_frames_tuple]) 
        # Extract May frames
        may_frames = [
            find_from_index(may_image_index, ts[0])
            for ts in sorted(may_frames_tuple, key=lambda x: x[0])
            if find_from_index(may_image_index, ts[0])
        ]

        may_front_frames = [
            find_from_index(may_front_image_index, ts[0])
            for ts in sorted(may_frames_tuple, key=lambda x: x[0])
            if find_from_index(may_front_image_index, ts[0])
        ]

        # Extract November frames
        left_frames = [
            find_from_index(nov_left_image_index, ts[0])
            for ts in sorted(left_frames_tuple, key=lambda x: x[0])
            if find_from_index(nov_left_image_index, ts[0])
        ]

        right_frames = [
            find_from_index(nov_right_image_index, ts[0])
            for ts in sorted(right_frames_tuple, key=lambda x: x[0])
            if find_from_index(nov_right_image_index, ts[0])
        ]

        nov_front_frames = [
            find_from_index(nov_front_image_index, ts[0])
            for ts in sorted(left_frames_tuple, key=lambda x: x[0])
            if find_from_index(nov_front_image_index, ts[0])
        ]

        if not left_frames and not right_frames:
            raise OSError 

        if not (may_frames):
            raise OSError 

        output_dir_timestamp = os.path.join(paths["match_output_dir"], str(timestamp))
        os.makedirs(output_dir_timestamp, exist_ok=True)
        may_fused_img_dir = os.path.join(output_dir_timestamp, "May")
        nov_fused_img_dir = os.path.join(output_dir_timestamp, "Nov")
        os.makedirs(may_fused_img_dir, exist_ok=True)
        os.makedirs(nov_fused_img_dir, exist_ok=True) 

        # Copy over front-facing images
        front_facing_img_dir = os.path.join(output_dir_timestamp,"front_facing_imgs")
        os.makedirs(front_facing_img_dir, exist_ok=True)

        may_front_facing_img_dir = os.path.join(front_facing_img_dir,"May") 
        os.makedirs(may_front_facing_img_dir, exist_ok=True) 
        
        for src_path in may_front_frames:
            filename = os.path.basename(src_path) 
            dest_path = os.path.join(may_front_facing_img_dir,filename)
            shutil.copy(src_path,dest_path) 

        #start stitching thread 
        thread = threading.Thread(
            target=fuse_front_facing_images,
            args=(timestamp, may_front_frames, nov_front_frames, paths),
            daemon=True  # set to False if you need to join it later
        )
        thread.start()

        # May stitching
        may_panos = {}
        if os.path.exists(os.path.join(may_fused_img_dir, "may_panos.pickle")):
            with open(os.path.join(may_fused_img_dir, "may_panos.pickle"), "rb") as handle:
                may_panos = pickle.load(handle)
        else:
            may_panos = stitch_and_save(
                may_frames,
                may_fused_img_dir,
                may_compass_headings,
                stitch_cfg,
                prefix="may"
            )

        # November stitching
        nov_panos = {}
        if os.path.exists(os.path.join(nov_fused_img_dir, "nov_panos.pickle")):
            with open(os.path.join(nov_fused_img_dir, "nov_panos.pickle"), "rb") as handle:
                nov_panos = pickle.load(handle)
        else:
            nov_panos = stitch_and_save(
                left_frames + right_frames,
                nov_fused_img_dir,
                nov_compass_headings,
                stitch_cfg, 
                prefix="nov"
            )

        print(f"[INFO] Found {len(may_panos)} May panoramas and {len(nov_panos)} November panoramas.")
        if len(may_panos) == 0 and len(nov_panos) == 0:
            raise OSError 

        output_img_dir = os.path.join(output_dir_timestamp, "gps_plots")
        os.makedirs(output_img_dir, exist_ok=True)
        csv_path = os.path.join(output_dir_timestamp, "gps_plot_fused_images.csv")

        t0 = time.time()
        with open(csv_path, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["gps_plot_path", "may_fused_img_path", "nov_fused_img_path"])

            fused_img_counter = 0
            for fused_img_path in may_panos:
                print("[INFO:main] iterating over fused_img_path in may panos ...")
                log_mem()

                may_polygon_list = create_polygons(may_panos[fused_img_path])
                #print(f"[DEBUG] {len(may_polygon_list)} May polygons created for {fused_img_path}") 
                for nov_img_path in nov_panos:
                    #log_mem()

                    nov_polygon_list = create_polygons(nov_panos[nov_img_path])

                    if not (may_polygon_list and nov_polygon_list):
                        continue

                    may_union = unary_union(may_polygon_list)
                    nov_union = unary_union(nov_polygon_list)

                    max_centroid_dist = 1e-6 

                    may_centroid = may_union.centroid
                    nov_centroid = nov_union.centroid

                    if may_centroid.distance(nov_centroid) > max_centroid_dist:
                        #print(f"[INFO] Skipping comparison figs... Centroids too far: {may_centroid.distance(nov_centroid)} degrees")
                        continue
                    
                    overlap = unary_union(may_polygon_list).intersects(unary_union(nov_polygon_list))
                    if not overlap:
                        #print("[INFO] Skipping comparison figs... no overlap detected!") 
                        continue 

                    gps_plot_path = os.path.join(output_img_dir, f"gps_plot_{fused_img_counter}.png")
                    comparison_plot_path = os.path.join(output_img_dir, f"comparison_plot_{fused_img_counter}.png")
                
                    print("[INFO] Making gps plot ...")
                    make_gps_plot(may_data, nov_data, may_polygon_list, nov_polygon_list, nov_img_path, gps_plot_path)
                    log_mem() 

                    plt.close('all')
                    gc.collect() 

                    #Debugging memory creep
                    snapshot = tracemalloc.take_snapshot()
                    top_stats = snapshot.statistics('lineno')
                    for stat in top_stats[:10]:
                        print(stat)


                    print("[INFO] Making comparison fig ...")
                    make_comparison_fig(may_data, nov_data, may_polygon_list, nov_polygon_list, fused_img_path, nov_img_path, comparison_plot_path)
                    log_mem()  

                    plt.close('all')
                    gc.collect()

                    print("[INFO:main] memory usage after closing the debug figures ...")
                    log_mem()

                    #Debugging memory creep
                    snapshot = tracemalloc.take_snapshot()
                    top_stats = snapshot.statistics('lineno')
                    for stat in top_stats[:10]:
                        print(stat)

                    csv_writer.writerow([gps_plot_path, fused_img_path, nov_img_path])

                    fused_img_counter += 1

                del nov_polygon_list, may_polygon_list 
                nov_polygon_list = None; may_polygon_list = None;
                gc.collect()

        t1 = time.time() 
        print(f"[INFO] Figure drawing took {t1 - t0:.2f} seconds")  

        print("[INFO:main] iterating over the may covariance timestamps ...")
        log_mem()
        thread.join()  
        

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
        print(f"Writing stitched image: {chunk_path}")
        cv.imwrite(chunk_path, stitched_img)

        del stitched_img 

        fused_corners = [gps_lookup(path) for path in chunk]
        panos[chunk_path] = fused_corners

        fused_img_count += 1

    del chunk, chunks 

    pickle_path = os.path.join(output_dir, f"{prefix}_panos.pickle")
    #print(f"Saving to {pickle_path}")
    with open(pickle_path, "wb") as handle:
        pickle.dump(panos, handle)

    return panos

def reorder_corners(corner_tuples):
    """
    Reorder 4 (lon, lat) corner points counterclockwise around their centroid.
    """
    pts = np.array(corner_tuples)
    centroid = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:,1] - centroid[1], pts[:,0] - centroid[0])
    sort_order = np.argsort(angles)
    return [tuple(pt) for pt in pts[sort_order]]

def create_polygons(corners_list, debug_dir="./debug_polygon_plots"):
    """Convert list of corner coordinates into shapely Polygons. Save plots for invalid ones."""
    os.makedirs(debug_dir, exist_ok=True)
    polygons = []

    for idx, corners in enumerate(corners_list):
        try:
            corners = reorder_corners(corners)
        except Exception as e:
            print(f"[WARN] Failed to reorder corners: {corners} ({e})")

        poly = Polygon(corners)
        if poly.is_valid and not poly.is_empty and poly.area > 0:
            polygons.append(poly)
        else:
            print(f"[WARN] Invalid polygon from corners: {corners}")
            
            # Save debug plot
            fig, ax = plt.subplots()
            xs, ys = zip(*corners)
            ax.plot(xs + (xs[0],), ys + (ys[0],), 'r-', marker='o', label='Corners')
            ax.set_title(f"Invalid Polygon {idx}")
            ax.set_aspect('equal')
            ax.legend()

            debug_path = os.path.join(debug_dir, f"invalid_polygon_{idx}.png")
            plt.savefig(debug_path, dpi=150)
            plt.close(fig)
            print(f"[DEBUG] Saved invalid polygon plot to {debug_path}")

            poly = poly.buffer(0)
            polygons.append(poly)

    return polygons

if __name__ == "__main__":
    main()
