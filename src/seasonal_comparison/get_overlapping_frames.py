#!/usr/bin/env python3

import os
import argparse
import numpy as np
import cv2 as cv
cv.ocl.setUseOpenCL(False)
import matplotlib.pyplot as plt
import toml
from stitching import AffineStitcher
from seasonal_comparison.gps_utils import (
    precompute_timestamps,
    load_covariance_matrix,
    find_closest_frame,
    find_frames_inside_ellipse
)
from seasonal_comparison.ellipse_utils import scale_covariance_to_degrees
from seasonal_comparison.image_stitching_utils import (
    make_image_path_dict, 
    robust_load_csv,
    chunk_list,
    crop_connected_region, 
    fuse_chunks,
    downscale_images,
    plot_stitched_summary_grid
)    
import tempfile
import shutil 

import psutil
def log_mem(msg=""):
    usage = psutil.Process(os.getpid()).memory_info().rss / 1e9
    print(f"[{msg}] Memory: {usage:.2f} GB")

def find_closest_index(array, target, max_delta_t=None):
    """
    Returns the index of the element in the array closest to the target.
    If max_delta_t is specified, returns None if the closest value exceeds it.

    Parameters:
        array (np.ndarray): Array of floats.
        target (float): Target value to find the closest match to.
        max_delta_t (float, optional): Maximum allowed distance. Defaults to None.

    Returns:
        int or None: Index of the closest value, or None if no value is within max_delta_t.
    """
    array = np.asarray(array)
    idx = np.argmin(np.abs(array - target))
    closest_diff = abs(array[idx] - target)

    if max_delta_t is not None and closest_diff > max_delta_t:
        return None

    return idx

def parse_args():
    parser = argparse.ArgumentParser(description="Fuse overlapping seasonal frames from May and Nov.")
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config file")
    return parser.parse_args()

def load_all_images(paths, scale=1.0):
    images = []
    for path in paths:
        img = cv.imread(path)
        if img is None:
            print(f"[!] Failed to load: {path}")
            continue
        if scale < 1.0:
            h, w = img.shape[:2]
            img = cv.resize(img, (int(w * scale), int(h * scale)))
        images.append(img)
    return images

def load_config(config_path):
    return toml.load(config_path)

def stitch_images(image_paths, stitcher, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print("image_paths:",len(image_paths))
    print("set(image_paths)):",len(set(image_paths)))

    log_mem("Before stitching ...")
    if len(image_paths) < 10:
        print("There are fewer than 10 images — attempting direct stitch")
        try:
            stitched_img = stitcher.stitch(image_paths)
            if stitched_img is not None:
                stitched_img = crop_connected_region(stitched_img)
                out_path = os.path.join(out_dir, "fused_single.png")
                cv.imwrite(out_path, stitched_img)
                return {out_path: image_paths}
        except Exception as e:
            print(f"[✗] Stitching failed: {e}")
            return {}

    print(f"[i] Found {len(image_paths)} valid paths")
    path_dict = make_image_path_dict(image_paths)

    """
    filtered_img_paths = filter_imgs_by_overlap(path_dict)
    print(f"[i] {len(filtered_img_paths)} paths after overlap filtering")
    """ 
    print("len(image_paths):",len(image_paths))
    print("len(np.unique(image_paths)):",len(np.unique(image_paths)))

    img_chunks = chunk_list(image_paths, chunk_size=10)
    print(f"[i] Processing {len(img_chunks)} chunks")
    
    tmp_img_chunk_dir = tempfile.mkdtemp()
    chunk_output_paths = []
    chunk_to_original_paths = {}

    for i, chunk in enumerate(img_chunks):
        log_mem("Before processing chunk i:{} ...".format(i))
        print("there are {} images in this chunk".format(len(chunk)))
        chunk_dict = {}
        for path in chunk:
            print("path:",path)
            chunk_dict[path] = path_dict[path]
        print("chunk_dict:",chunk_dict)
        input("Wait")
        #filtered_chunk = filter_imgs_by_overlap(chunk_dict)
        filtered_chunk = downscale_images(chunk_dict,scale=0.5)
        print("filtered_chunk:",filtered_chunk)
        if len(filtered_chunk) < 2:
            print(f"[!] Skipping chunk {i} — too few images after filtering")
            continue

        try:
            stitched = stitcher.stitch(filtered_chunk)
            if stitched is None:
                print(f"[!] Stitching failed for chunk {i} — skipping")
                continue

            stitched = crop_connected_region(stitched)
            out_path = os.path.join(tmp_img_chunk_dir, f"chunk_{i}.png")
            cv.imwrite(out_path, stitched)
            chunk_output_paths.append(out_path)
            chunk_to_original_paths[out_path] = filtered_chunk
            print(f"[✓] Wrote stitched chunk {i} to {out_path}")

            del stitched 
            cv.waitKey(1)

        except Exception as e:
            print(f"[✗] Chunk {i} error: {e}")
            continue
        print("Safely processed chunk: {}".format(i))

    print("[i] Fusing stitched chunks...")
    fused_results = fuse_chunks(chunk_output_paths, stitcher, out_dir=out_dir)

    # Remap temp input chunks to original input images
    final_output_dict = {}
    for fused_path, used_chunk_paths in fused_results.items():
        all_originals = []
        for chunk_path in used_chunk_paths:
            originals = chunk_to_original_paths.get(chunk_path, [])
            all_originals.extend(originals)
        final_output_dict[fused_path] = all_originals

    shutil.rmtree(tmp_img_chunk_dir)
    return final_output_dict

def main():
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    stitch_cfg = config.get("stitching", {})

    # Load GPS data
    may_data = np.genfromtxt(f"{paths['may_results']}/frustrum_corners.csv", delimiter=",", skip_header=1)
    nov_data = np.genfromtxt(f"{paths['nov_results']}/frustrum_corners.csv", delimiter=",", skip_header=1)

    may_timestamps = may_data[:, 0]
    cam_lat, cam_lon = may_data[:, 15], may_data[:, 16]
    nov_timestamps = nov_data[:, 0]
    left_cam_lat, left_cam_lon = nov_data[:, 23], nov_data[:, 24]
    right_cam_lat, right_cam_lon = nov_data[:, 33], nov_data[:, 34]

    # Covariance
    may_cov_dir = os.path.join(paths["may_results"], "covariance_matrices")
    may_cov_timestamps = precompute_timestamps(may_cov_dir)

    # Stitcher
    settings = {"crop":False,"confidence_threshold":stitch_cfg['confidence_threshold']}
    stitcher = AffineStitcher(**settings)

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
        print("Looking for overlapping frames ...")
        may_frames = find_frames_inside_ellipse(scaled_cov, center, cam_lon, cam_lat, may_timestamps)
        print("len(may_frames): {}, len(set(may_frames)):{}".format(len(may_frames),len(set(may_frames))))
        left_frames = find_frames_inside_ellipse(scaled_cov, center, left_cam_lon, left_cam_lat, nov_timestamps)
        right_frames = find_frames_inside_ellipse(scaled_cov, center, right_cam_lon, right_cam_lat, nov_timestamps)

        if not (left_frames or right_frames):
            print("No overlapping frames found.")
            continue

        # Collect image paths
        may_ts = [x[0] for x in may_frames]
        #print("len(set(may_ts)):",len(set(may_ts)))
        may_paths = [find_closest_frame(paths["may_images"], x) for x in may_ts]
        #print("len(may_paths):{}, len(set(may_paths)):{}".format(len(may_paths),len(set(may_paths))))
        if len(set(may_paths)) <= 1:
            raise OSError 
        left_ts = [x[0] for x in left_frames]
        left_paths = [find_closest_frame(paths["nov_left_images"], x) for x in left_ts]
        if len(set(left_paths)) <= 1:
            raise OSError
        right_ts = [x[0] for x in right_frames]
        right_paths = [find_closest_frame(paths["nov_right_images"], x) for x in right_ts]
        if len(set(right_paths)) <= 1:
            raise OSError 

        # Stitch panoramas
        os.makedirs(f"{paths['may_results']}/Panos",exist_ok=True) 
        os.makedirs(f"{paths['may_results']}/Panos/"+str(timestamp),exist_ok=True)
        print("Calling stitch images ...")
        may_pano_dict = stitch_images(may_paths, stitcher,f"{paths['may_results']}/Panos/"+str(timestamp))
        print("may_pano_dict: ",may_pano_dict)
        input("Wait") 
        print()

        os.makedirs(f"{paths['nov_results']}/Panos/"+str(timestamp),exist_ok=True) 
        nov_pano_dict = stitch_images(left_paths + right_paths, stitcher,f"{paths['nov_results']}/Panos/"+str(timestamp))
        print("nov_pano_dict: ",nov_pano_dict)
        input("Wait") 
        print() 

        # Save side-by-side comparison
        os.makedirs(paths["match_output_dir"], exist_ok=True)
        output_path = os.path.join(paths["match_output_dir"], f"fused_{int(timestamp)}.png")
        
        plot_stitched_summary_grid(may_pano_dict,nov_pano_dict,output_path)

        input("Wait.")

if __name__ == "__main__":
    main()