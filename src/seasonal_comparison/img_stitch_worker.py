# stitch_worker.py
import sys
import pickle
import cv2 as cv
from stitching import AffineStitcher

def main(frame_list_path,stitch_cfg):
    print("frame_list: ",frame_list) 
    print("stitch_cfg: ",stitch_cfg) 
    stitcher = AffineStitcher(crop=False, confidence_threshold=stitch_cfg["confidence_threshold"])
    stitched_img = stitcher.stitch(frame_list)
    return stitched_img 

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2])
