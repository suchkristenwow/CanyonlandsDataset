# stitch_worker.py
import sys
import pickle
import cv2 as cv
from stitching import Stitcher

def main(frame_list_path):
    with open(frame_list_path, 'rb') as f:
        frame_list = pickle.load(f)

    stitcher = Stitcher()
    stitched_img = stitcher.stitch(frame_list)
    return stitched_img 

if __name__ == '__main__':
    main(sys.argv[1])
