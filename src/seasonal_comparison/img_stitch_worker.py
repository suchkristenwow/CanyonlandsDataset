# src/seasonal_comparison/img_stitch_worker.py
import sys
import pickle
import cv2 as cv
from stitching import AffineStitcher

def main(pickle_path, output_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    frame_list = data["frame_list"]
    stitch_cfg = data["stitch_cfg"]

    stitcher = AffineStitcher(crop=False, confidence_threshold=stitch_cfg["confidence_threshold"])
    try:
        stitched_img = stitcher.stitch(frame_list)
    except Exception as e:
        print(f"[ERROR] Stitching failure: {e}")
        sys.exit(2)  # <-- non-zero exit to signal failure to parent

    print(f"[INFO] Writing stitched image to {output_path}")
    cv.imwrite(output_path, stitched_img)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("[USAGE] python img_stitch_worker.py input_pickle_path output_image_path")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
