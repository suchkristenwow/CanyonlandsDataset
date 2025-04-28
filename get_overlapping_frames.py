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

import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union

# Local imports
from stitching import AffineStitcher
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


def main():
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    stitch_cfg = config.get("stitching", {})

    os.makedirs(paths["match_output_dir"], exist_ok=True)

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

    stitcher = AffineStitcher(crop=False, confidence_threshold=stitch_cfg["confidence_threshold"])

    for i, timestamp in enumerate(may_cov_timestamps):
        print(f"\n[{i+1}/{len(may_cov_timestamps)}] Processing timestamp {timestamp}")

        cov_matrix = load_covariance_matrix(may_cov_dir, may_cov_timestamps, timestamp)
        if cov_matrix is None:
            print("No matching covariance for this timestamp.")
            continue

        scaled_cov = scale_covariance_to_degrees(cov_matrix)
        i_may = find_closest_index(may_timestamps, timestamp)
        center = (cam_lon[i_may], cam_lat[i_may])

        # Find overlapping frames
        print("Looking for overlapping frames...")
        may_frames_tuple = find_frames_inside_ellipse(scaled_cov, center, cam_lon, cam_lat, may_timestamps)
        left_frames_tuple = find_frames_inside_ellipse(scaled_cov, center, left_cam_lon, left_cam_lat, nov_timestamps)
        right_frames_tuple = find_frames_inside_ellipse(scaled_cov, center, right_cam_lon, right_cam_lat, nov_timestamps)

        if not (left_frames_tuple or right_frames_tuple):
            print("No overlapping frames found.")
            continue

        may_frames = [
            find_closest_file(paths["may_images"], ts[0])
            for ts in sorted(may_frames_tuple, key=lambda x: x[0])
            if find_closest_file(paths["may_images"], ts[0])
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

        output_dir_timestamp = os.path.join(paths["match_output_dir"], str(timestamp))
        os.makedirs(output_dir_timestamp, exist_ok=True)
        may_fused_img_dir = os.path.join(output_dir_timestamp, "May")
        nov_fused_img_dir = os.path.join(output_dir_timestamp, "Nov")
        os.makedirs(may_fused_img_dir, exist_ok=True)
        os.makedirs(nov_fused_img_dir, exist_ok=True)

        # May stitching
        may_panos = {}
        if os.path.exists(os.path.join(may_fused_img_dir, "may_panos.pickle")):
            with open(os.path.join(may_fused_img_dir, "may_panos.pickle"), "rb") as handle:
                may_panos = pickle.load(handle)
        else:
            may_panos = stitch_and_save(
                stitcher,
                may_frames,
                may_fused_img_dir,
                prefix="may"
            )

        # November stitching
        nov_panos = {}
        if os.path.exists(os.path.join(nov_fused_img_dir, "nov_panos.pickle")):
            with open(os.path.join(nov_fused_img_dir, "nov_panos.pickle"), "rb") as handle:
                nov_panos = pickle.load(handle)
        else:
            nov_panos = stitch_and_save(
                stitcher,
                left_frames + right_frames,
                nov_fused_img_dir,
                prefix="nov"
            )

        print(f"Iterating over stitched images...")
        print(f"Found {len(may_panos)} May panoramas and {len(nov_panos)} November panoramas.")

        output_img_dir = os.path.join(output_dir_timestamp, "gps_plots")
        os.makedirs(output_img_dir, exist_ok=True)
        csv_path = os.path.join(output_dir_timestamp, "gps_plot_fused_images.csv")

        with open(csv_path, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["gps_plot_path", "may_fused_img_path", "nov_fused_img_path"])

            fused_img_counter = 0
            for fused_img_path in may_panos:
                print(f"Processing May panorama: {fused_img_path}")
                log_mem()

                may_polygon_list = create_polygons(may_panos[fused_img_path])
                for nov_img_path in nov_panos:
                    log_mem()

                    nov_polygon_list = create_polygons(nov_panos[nov_img_path])

                    if not (may_polygon_list and nov_polygon_list):
                        continue

                    overlap = unary_union(may_polygon_list).intersects(unary_union(nov_polygon_list))

                    if not overlap:
                        continue

                    gps_plot_path = os.path.join(output_img_dir, f"gps_plot_{fused_img_counter}.png")
                    comparison_plot_path = os.path.join(output_img_dir, f"comparison_plot_{fused_img_counter}.png")

                    make_gps_plot(may_data, nov_data, may_polygon_list, nov_polygon_list, nov_img_path, gps_plot_path)
                    make_comparison_fig(may_data, nov_data, may_polygon_list, nov_polygon_list, fused_img_path, nov_img_path, comparison_plot_path)

                    #Memory Management 
                    plt.clf()
                    plt.close()
                    gc.collect()

                    csv_writer.writerow([gps_plot_path, fused_img_path, nov_img_path])

                    fused_img_counter += 1

                del nov_polygon_list, may_polygon_list
                gc.collect() 

        log_mem()
        
def stitch_and_save(stitcher, frame_list, output_dir, prefix):
    """Helper function to stitch frames and save results."""
    panos = {}
    fused_img_count = 0
    chunks = chunk_filenames(frame_list)

    for chunk in chunks:
        log_mem()
        try:
            stitched_img = stitcher.stitch(chunk)
            stitched_img = crop_connected_region(stitched_img)
        except Exception as e:
            print(f"Image stitching failed: {e}")
            del stitched_img 
            continue

        if stitched_img is None:
            continue

        chunk_path = os.path.join(output_dir, f"{fused_img_count}.png")
        print(f"Writing stitched image: {chunk_path}")
        cv.imwrite(chunk_path, stitched_img)

        del stitched_img 

        fused_corners = [gps_lookup(path) for path in chunk]
        panos[chunk_path] = fused_corners

        fused_img_count += 1

    del chunk, chunks 

    pickle_path = os.path.join(output_dir, f"{prefix}_panos.pickle")
    print(f"Saving to {pickle_path}")
    with open(pickle_path, "wb") as handle:
        pickle.dump(panos, handle)

    return panos


def create_polygons(corners_list):
    """Convert list of corner coordinates into shapely Polygons."""
    polygons = []
    for corners in corners_list:
        poly = Polygon(corners)
        if poly.is_valid and not poly.is_empty and poly.area > 0:
            polygons.append(poly)
        else:
            poly = poly.buffer(0)
            polygons.append(poly)
    return polygons


if __name__ == "__main__":
    main()
