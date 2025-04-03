#!/usr/bin/env python3

import os
import argparse
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import toml
from stitching import AffineStitcher
from gps_utils import (
    precompute_timestamps,
    load_covariance_matrix,
    find_closest_index,
    find_closest_frame,
    find_frames_inside_ellipse,
)
from ellipse_utils import scale_covariance_to_degrees

def parse_args():
    parser = argparse.ArgumentParser(description="Fuse overlapping seasonal frames from May and Nov.")
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config file")
    return parser.parse_args()


def load_config(config_path):
    return toml.load(config_path)


def stitch_images(image_paths, stitcher):
    try:
        return stitcher.stitch(image_paths) if image_paths else None
    except Exception as e:
        print(f"Stitching failed: {e}")
        return None


def save_side_by_side(img1, img2, save_path, figsize=(20, 10)):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    axs[0].set_title("May Fused")
    axs[0].axis("off")

    axs[1].imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    axs[1].set_title("Nov Fused")
    axs[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[✓] Saved: {save_path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    stitch_cfg = config.get("stitching", {})

    # Load GPS data
    may_data = np.genfromtxt(f"{paths['may_results']}/results.csv", delimiter=",", skip_header=1)
    nov_data = np.genfromtxt(f"{paths['nov_results']}/results.csv", delimiter=",", skip_header=1)

    may_timestamps = may_data[:, 0]
    cam_lat, cam_lon = may_data[:, 15], may_data[:, 16]
    nov_timestamps = nov_data[:, 0]
    left_cam_lat, left_cam_lon = nov_data[:, 23], nov_data[:, 24]
    right_cam_lat, right_cam_lon = nov_data[:, 33], nov_data[:, 34]

    # Covariance
    may_cov_dir = os.path.join(paths["may_results"], "covariance_matrices")
    may_cov_timestamps = precompute_timestamps(may_cov_dir)

    # Stitcher
    stitcher = AffineStitcher(
        crop=stitch_cfg.get("crop", False),
        confidence_threshold=stitch_cfg.get("confidence_threshold", 0.25)
    )

    for i, timestamp in enumerate(may_cov_timestamps):
        print(f"\n[{i+1}/{len(may_cov_timestamps)}] Processing timestamp {timestamp}")
        cov_matrix = load_covariance_matrix(may_cov_dir, may_cov_timestamps, timestamp)
        if cov_matrix is None:
            continue

        scaled_cov = scale_covariance_to_degrees(cov_matrix)
        i_may = find_closest_index(may_timestamps, timestamp)
        center = (cam_lon[i_may], cam_lat[i_may])

        # Find overlapping frames
        may_frames = find_frames_inside_ellipse(scaled_cov, center, cam_lon, cam_lat, may_timestamps)
        left_frames = find_frames_inside_ellipse(scaled_cov, center, left_cam_lon, left_cam_lat, nov_timestamps)
        right_frames = find_frames_inside_ellipse(scaled_cov, center, right_cam_lon, right_cam_lat, nov_timestamps)

        if not (left_frames or right_frames):
            print("  No overlapping frames found.")
            continue

        # Collect image paths
        may_paths = [find_closest_frame(paths["may_images"], ts) for ts, *_ in may_frames]
        left_paths = [find_closest_frame(paths["nov_left_images"], ts) for ts, *_ in left_frames]
        right_paths = [find_closest_frame(paths["nov_right_images"], ts) for ts, *_ in right_frames]

        # Stitch panoramas
        may_pano = stitch_images(may_paths, stitcher)
        nov_pano = stitch_images(left_paths + right_paths, stitcher)

        if may_pano is None or nov_pano is None:
            print("  Skipping due to stitching failure.")
            continue

        # Save side-by-side comparison
        os.makedirs(paths["match_output_dir"], exist_ok=True)
        output_path = os.path.join(paths["match_output_dir"], f"fused_{int(timestamp)}.png")
        save_side_by_side(may_pano, nov_pano, output_path)


if __name__ == "__main__":
    main()