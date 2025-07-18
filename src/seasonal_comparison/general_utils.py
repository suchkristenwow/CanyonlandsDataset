#!/usr/bin/env python3

"""
General file management utilites for seasonal comparison tools.
"""

import numpy as np
import argparse 
import toml
import psutil 
import os 
import math 


def get_ellipse_bounds(ellipse):
    center = np.array(ellipse.get_center())
    width = ellipse.width
    height = ellipse.height
    angle_deg = ellipse.angle
    angle_rad = np.deg2rad(angle_deg)

    # Sample N points on ellipse
    t = np.linspace(0, 2 * np.pi, 100)
    x = 0.5 * width * np.cos(t)
    y = 0.5 * height * np.sin(t)

    # Rotate points
    R = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    rotated_points = np.dot(R, np.vstack((x, y)))

    # Translate to center
    latlon_points = rotated_points + center.reshape(2, 1)

    # Extract bounds
    min_lon, max_lon = np.min(latlon_points[0]), np.max(latlon_points[0])
    min_lat, max_lat = np.min(latlon_points[1]), np.max(latlon_points[1])

    return min_lat, max_lat, min_lon, max_lon


def same_order_of_magnitude(a, b):
    if a == 0 or b == 0:
        return a == b  # Only 0 is same order as 0

    return math.floor(math.log10(abs(a))) == math.floor(math.log10(abs(b)))

def load_config(config_path):
    return toml.load(config_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Fuse overlapping seasonal frames from May and Nov.")
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config file")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive plot viewer")

    return parser.parse_args()

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
    if not same_order_of_magnitude(array[0],target):
        print("array[0]: ",array[0])
        print("target:",target)
        raise OSError 

    array = np.asarray(array)
    idx = np.argmin(np.abs(array - target))
    closest_diff = abs(array[idx] - target)

    if max_delta_t is not None and closest_diff > max_delta_t:
        return None

    return idx

def log_mem():
    usage = psutil.Process(os.getpid()).memory_info().rss / 1e9
    print(f"[INFO] Memory: {usage:.2f} GB")

def robust_load_csv(path, min_cols=4, skip_header=1):
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
