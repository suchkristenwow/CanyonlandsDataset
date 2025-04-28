#!/usr/bin/env python3

"""
General file management utilites for seasonal comparison tools.
"""

import numpy as np
import argparse 
import toml
import psutil 
import os 

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
