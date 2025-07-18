#!/usr/bin/env python3

"""
Fuse overlapping seasonal frames based on covariance ellipses
and generate comparison plots.
"""
import subprocess
import json
import tempfile
from seasonal_comparison.general_utils import log_mem, parse_args, load_config 
import os 
import numpy as np 
import subprocess

def main():
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"] 

    may_data = np.genfromtxt(
            os.path.join(paths["may_results"], "frustrum_corners.csv"),
            delimiter=",",
            skip_header=1
        )
    timestamps = may_data[:,0] 

    chunk_size = 3

    for i in range(0, len(timestamps), chunk_size):
        batch = timestamps[i:i+chunk_size]
        print(f"[INFO] Processing batch {i//chunk_size + 1}")
        log_mem() 

        # Save config and batch to temp file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            data_path = f.name
            json.dump({
                "config_path": args.config,
                "timestamps": batch.tolist()  # ✅ convert np.ndarray to list
            }, f)
        # Launch subprocess
        subprocess.run(["python3", "src/seasonal_comparison/timestamp_batch_worker.py", data_path], check=True) 
        input("End of batch process")

if __name__ == "__main__":
    main()
