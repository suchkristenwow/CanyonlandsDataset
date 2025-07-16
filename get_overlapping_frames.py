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

def index_image_dir(image_dir):
    """
    Create a dict mapping timestamps (from filenames) to full file paths.
    Assumes filenames are like '1234567890123456789.png'
    """
    print("indexing this dir: ",image_dir)
    return {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir) if f.endswith(".png")
    }

def find_from_index(index_dict, ts):
    """
    Lookup timestamp in prebuilt index.
    Returns file path or None if not found.
    """
    return index_dict.get(ts, None)

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

        if not (left_frames_tuple or right_frames_tuple):
            print("[WARN] No overlapping frames found.")
            continue

        print("[INFO] finding frames ...")
        t0 = time.time() 
        """
        may_frames = [
            find_closest_file(paths["may_images"], ts[0])
            for ts in sorted(may_frames_tuple, key=lambda x: x[0])
            if find_closest_file(paths["may_images"], ts[0])
        ]

        may_front_frames = [
            find_closest_file(paths["may_front_images"], ts[0])
            for ts in sorted(may_frames_tuple, key=lambda x: x[0])
            if find_closest_file(paths["may_front_images"], ts[0])
        ]

        left_frames = [
            find_closest_file(paths["nov_left_images"], ts[0])
            for ts in sorted(left_frames_tuple, key=lambda x: x[0])
            if find_closest_file(paths["nov_left_images"], ts[0])
        ]
        right_frames = [
            find_closest_file(paths["nov_right_images"], ts[0])
            for ts in sorted(right_frames_tuple, key=lambda x: x[0])
            if find_closest_file(paths["nov_right_images"], ts[0])
        ]

        nov_front_frames = [
            find_closest_file(paths["nov_front_images"], ts[0])
            for ts in sorted(left_frames_tuple, key=lambda x: x[0])
            if find_closest_file(paths["nov_front_images"], ts[0])
        ]
        print("[INFO] Done finding overlapping frames!") 
        """

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
        t1 = time.time() 
        print(f"[INFO] finding frames took {t1 - t0:.2f} seconds")  

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
                    #print(f"[DEBUG] {len(nov_polygon_list)} Nov polygons created for {fused_img_path}")  

                    if not (may_polygon_list and nov_polygon_list):
                        continue

                    may_union = unary_union(may_polygon_list)
                    nov_union = unary_union(nov_polygon_list)

                    buffer_deg = 1e-7 #~1 meter ≈ 1e-5 degrees
                    max_centroid_dist = 1e-6 

                    may_centroid = may_union.centroid
                    nov_centroid = nov_union.centroid

                    if may_centroid.distance(nov_centroid) > max_centroid_dist:
                        #print(f"[SKIP] Centroids too far: {may_centroid.distance(nov_centroid)} degrees")
                        continue

                    # Minimal buffer just to handle numerical precision
                    overlap = may_union.buffer(buffer_deg).intersects(nov_union.buffer(buffer_deg))
                    if not overlap:
                        #print("[SKIP] No overlap even after minimal buffering.")
                        continue
                    else: 
                        print("[INFO] found overlapping frames!")  

                    gps_plot_path = os.path.join(output_img_dir, f"gps_plot_{fused_img_counter}.png")
                    comparison_plot_path = os.path.join(output_img_dir, f"comparison_plot_{fused_img_counter}.png")

                    print("[INFO] Making gps plot ...")
                    make_gps_plot(may_data, nov_data, may_polygon_list, nov_polygon_list, nov_img_path, gps_plot_path)
                    log_mem() 

                    plt.close('all')
                    gc.collect() 

                    #Debugging memory creep
                    '''
                    snapshot = tracemalloc.take_snapshot()
                    top_stats = snapshot.statistics('lineno')
                    for stat in top_stats[:10]:
                        print(stat)
                    '''

                    print("[INFO] Making comparison fig ...")
                    make_comparison_fig(may_data, nov_data, may_polygon_list, nov_polygon_list, fused_img_path, nov_img_path, comparison_plot_path)
                    log_mem()  

                    plt.close('all')
                    gc.collect()

                    input("Hol Up")

                    #Debugging memory creep
                    '''
                    snapshot = tracemalloc.take_snapshot()
                    top_stats = snapshot.statistics('lineno')
                    for stat in top_stats[:10]:
                        print(stat)
                    ''' 
                    #Memory Management 
                    plt.clf()
                    plt.close('all')
                    gc.collect()

                    print("[INFO:main] memory usage after closing the debug figures ...")
                    log_mem()

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
    """Stitch rotated images and save stitched panoramas + corner info."""
    #--medium_megapix -1 --low_megapix -1 --final_megapix -1
    stitcher = AffineStitcher(crop=False, confidence_threshold=stitch_cfg["confidence_threshold"], medium_megapix=-1, low_megapix=-1,final_megapix=-1) 
    if stitcher is None:
        print("stitcher is None right after initialization")
        raise OSError 

    panos = {}
    fused_img_count = 0
    chunks = chunk_filenames(frame_list)

    # Extract compass heading columns
    compass_timestamps = processed_compass_headings[:, 0]
    compass_headings_deg = processed_compass_headings[:, 1]

    # Directory to store rotated frames for inspection
    rotated_dir = os.path.join(output_dir, "..", "rotated_frames", prefix)
    os.makedirs(rotated_dir, exist_ok=True)

    for chunk in chunks:
        print("[INFO: stitch_and_save] iterating through chunks")
        log_mem()
        rotated_image_paths = []

        # Step 1: Rotate and save each image in the chunk
        for path in chunk:
            img = cv.imread(path)
            if img is None:
                print(f"[WARN] Failed to load image {path}")
                continue

            try:
                print("path:",path)
                ts = int(os.path.splitext(os.path.basename(path))[0])
                heading = get_heading(ts, processed_compass_headings)
                img_rotated = rotate_image_north(img, heading)
            except Exception as e:
                print(f"[ERROR] Error rotating image {path}: {e}")
                continue

            tmp_path = os.path.join(rotated_dir, f"rotated_{ts}.png")
            cv.imwrite(tmp_path, img_rotated)
            rotated_image_paths.append(tmp_path)

            del img, img_rotated
            img = None; img_rotated = None;

        # Step 2: Attempt stitching
        try:
            if not rotated_image_paths:
                print("[WARN] No valid images to stitch.")
                continue

            if not stitcher:
                print("[ERROR] stitcher is NONE")
                raise OSError 

            if not check_image_sizes(rotate_image_paths):
                print("[ERROR] these images are not all the same size")
                raise OSError  

            stitched_img = stitcher.stitch(rotated_image_paths)
            print("[INFO] cropping black padding!")
            stitched_img = crop_black_border(stitched_img)  # Crop black padding 

            # Optional: Delete rotated images to save disk
            for tmp_path in rotated_image_paths:
                try:
                    os.remove(tmp_path)
                except Exception as e:
                    print(f"[WARN] Failed to delete {tmp_path}: {e}")

        except Exception as e:
            print(f"[ERROR] Image stitching failed: {e}")
            stitched_img = None
            continue
            
        rotated_image_paths.clear() 
        rotate_image_paths = None 
        gc.collect() 

        # Step 3: Save stitched image if successful
        if stitched_img is None:
            continue

        chunk_path = os.path.join(output_dir, f"{fused_img_count}.png")
        print("[INFO] writing stitched image to: ",chunk_path)
        cv.imwrite(chunk_path, stitched_img)

        del stitched_img
        stitched_img = None 
        gc.collect() 

        # Step 4: Store corner metadata
        fused_corners = [gps_lookup(path) for path in chunk]
        panos[chunk_path] = fused_corners
        print(f"[DEBUG] {chunk_path} includes {len(fused_corners)} GPS frames") 

        fused_img_count += 1

    # Step 5: Save corner data as pickle
    pickle_path = os.path.join(output_dir, f"{prefix}_panos.pickle")
    print(f"Saving metadata to {pickle_path}")
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
