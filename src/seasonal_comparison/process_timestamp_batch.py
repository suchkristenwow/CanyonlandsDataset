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
    crop_black_border,
    stitch_and_save,
    stitch_clusters, 
    index_image_dir,
    find_from_index,
    create_polygons,
    create_polygon,
    fix_polygon_area 
)
from seasonal_comparison.general_utils import (
    robust_load_csv,
    log_mem,
    find_closest_index,
    load_config,
    parse_args,
    get_ellipse_bounds
)
from seasonal_comparison.plot_figs import (
    make_gps_plot,
    make_comparison_fig,
)

from seasonal_comparison.geometry_utils import (
    cluster_overlapping_polygons, 
    cluster_overlapping_frame_instances, 
    find_intersecting_clusters
) 


import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.pyplot as plt
plt.ioff() 
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes 
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from shapely.geometry import Polygon
from shapely.ops import unary_union

import os
import csv
import gc
import pickle
import shutil 
import threading 
import tracemalloc
from pathlib import Path 

import numpy as np
import cv2 as cv
cv.ocl.setUseOpenCL(False)

LAT_METERS_PER_DEGREE = 111_320
LON_METERS_PER_DEGREE = 85_390

class frameInstance:
    def __init__(self,frame_path):
        self.frame_path = frame_path 
        self.polygon_obj = None 

class seasonalComparer:
    def __init__(self,config):
        paths = config["paths"]
        self.paths = paths 
        self.stitch_cfg = config.get("stitching", {}) 

        os.makedirs(paths["match_output_dir"], exist_ok=True)

        self.may_image_index        = index_image_dir(paths["may_images"])
        self.may_front_image_index = index_image_dir(paths["may_front_images"])
        self.nov_left_image_index  = index_image_dir(paths["nov_left_images"])
        self.nov_right_image_index = index_image_dir(paths["nov_right_images"])
        self.nov_front_image_index = index_image_dir(paths["nov_front_images"])

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

        self.may_data = may_data 
        self.nov_data = nov_data 

        self.may_timestamps = may_data[:, 0]
        self.cam_lat, self.cam_lon = may_data[:, 15], may_data[:, 16]
        self.nov_timestamps = nov_data[:, 0]
        self.left_cam_lat, self.left_cam_lon = nov_data[:, 23], nov_data[:, 24]
        self.right_cam_lat, self.right_cam_lon = nov_data[:, 33], nov_data[:, 34]

        '''
        may_data: 
        writer.writerow(['timestamp', 'lat', 'lon', 'pointer_lat', 'pointer_lon',
            'ouster_lat', 'ouster_lon', 'frontLeftWheel_lat', 'frontLeftWheel_lon',
            'frontRightWheel_lat', 'frontRightWheel_lon', 'rearLeftWheel_lat', 'rearLeftWheel_lon',
            'rearRightWheel_lat', 'rearRightWheel_lon', 'cam_lat', 'cam_lon',
            'topRight_frame_lat', 'topRight_frame_lon', 'topLeft_frame_lat', 'topLeft_frame_lon',
            'bottomRight_frame_lat', 'bottomRight_frame_lon', 'bottomLeft_frame_lat', 'bottomLeft_frame_lon'])

        nov_data: 
        row = [
                t, lat, lon, pointer_lat, pointer_lon,
                ouster_lat, ouster_lon,
                wheel_coords['frontLeft'][0], wheel_coords['frontLeft'][1],
                wheel_coords['frontRight'][0], wheel_coords['frontRight'][1],
                wheel_coords['rearLeft'][0], wheel_coords['rearLeft'][1],
                wheel_coords['rearRight'][0], wheel_coords['rearRight'][1],
                left_frame_coords['topRight'][0], left_frame_coords['topRight'][1],
                left_frame_coords['topLeft'][0], left_frame_coords['topLeft'][1],
                left_frame_coords['bottomRight'][0], left_frame_coords['bottomRight'][1],
                left_frame_coords['bottomLeft'][0], left_frame_coords['bottomLeft'][1],
                cam_lat, cam_lon,
                right_frame_coords['topRight'][0], right_frame_coords['topRight'][1],
                right_frame_coords['topLeft'][0], right_frame_coords['topLeft'][1],
                right_frame_coords['bottomRight'][0], right_frame_coords['bottomRight'][1],
                right_frame_coords['bottomLeft'][0], right_frame_coords['bottomLeft'][1],
                cam1_lat, cam1_lon
            ]
        ''' 

        self.may_cov_dir = os.path.join(paths["may_results"], "covariance_matrices")
        self.may_cov_timestamps = precompute_timestamps(self.may_cov_dir)
        
        self.may_compass_headings = np.genfromtxt(
            os.path.join(paths["may_results"], "processed_compass_heading.csv"),
            delimiter=",",
            skip_header=1
        )

        self.nov_compass_headings = np.genfromtxt(
            os.path.join(paths["nov_results"], "processed_compass_heading.csv"),
            delimiter=",",
            skip_header=1
        ) 

        self.front_stitch_thread = None 

        self.may_frame_area = None 
        self.nov_frame_area = None 

    def stitch_front_imgs(self, timestamp, output_dir_timestamp, may_front_frames, nov_front_frames):
        front_facing_img_dir = os.path.join(output_dir_timestamp, "front_facing_imgs")
        os.makedirs(front_facing_img_dir, exist_ok=True)

        may_front_facing_img_dir = os.path.join(front_facing_img_dir, "May")
        os.makedirs(may_front_facing_img_dir, exist_ok=True)

        for src_path in may_front_frames:
            filename = os.path.basename(src_path)
            dest_path = os.path.join(may_front_facing_img_dir, filename)
            shutil.copy(src_path, dest_path)

        # Launch stitching in background
        self.front_stitch_thread = threading.Thread(
            target=self.fuse_front_facing_images,
            args=(timestamp, may_front_frames, nov_front_frames),
            daemon=False  
        )
        self.front_stitch_thread.start()

    def get_frames(self,scaled_cov,center):
        frames_dict = {}

        # Find overlapping frames
        print("[INFO] Looking for overlapping frames...")
        may_frames_tuple = find_frames_inside_ellipse(scaled_cov, center, self.cam_lon, self.cam_lat, self.may_timestamps)
        left_frames_tuple = find_frames_inside_ellipse(scaled_cov, center, self.left_cam_lon, self.left_cam_lat, self.nov_timestamps)
        right_frames_tuple = find_frames_inside_ellipse(scaled_cov, center, self.right_cam_lon, self.right_cam_lat, self.nov_timestamps)
        
        '''
        print(f"[DEBUG] Found {len(may_frames_tuple)} overlapping May frames")
        print(f"[DEBUG] May timestamps in region: {[ts[0] for ts in may_frames_tuple]}") 

        print(f"[DEBUG] Found {len(left_frames_tuple)} overlapping left frames")
        print(f"[DEBUG] Nov:Left timestamps in region: {[ts[0] for ts in left_frames_tuple]}")

        print(f"[DEBUG] Found {len(right_frames_tuple)} overlapping right frames")
        print(f"[DEBUG] Nov:Right timestamps in region: {[ts[0] for ts in right_frames_tuple]}")
        ''' 

        if not (may_frames_tuple or left_frames_tuple or right_frames_tuple):
            print("[WARN] No overlapping frames found.")
            return 

        # Extract May frames
        may_frames = [
            find_from_index(self.may_image_index, ts[0])
            for ts in sorted(may_frames_tuple, key=lambda x: x[0])
            if find_from_index(self.may_image_index, ts[0])
        ]
        frames_dict['may_frames'] = may_frames 

        may_front_frames = [
            find_from_index(self.may_front_image_index, ts[0])
            for ts in sorted(may_frames_tuple, key=lambda x: x[0])
            if find_from_index(self.may_front_image_index, ts[0])
        ]
        frames_dict['may_front_frames'] = may_front_frames 

        # Extract November frames
        left_frames = [
            find_from_index(self.nov_left_image_index, ts[0])
            for ts in sorted(left_frames_tuple, key=lambda x: x[0])
            if find_from_index(self.nov_left_image_index, ts[0])
        ]
        frames_dict['left_frames'] = left_frames

        right_frames = [
            find_from_index(self.nov_right_image_index, ts[0])
            for ts in sorted(right_frames_tuple, key=lambda x: x[0])
            if find_from_index(self.nov_right_image_index, ts[0])
        ]
        frames_dict['right_frames'] = right_frames

        nov_front_frames = [
            find_from_index(self.nov_front_image_index, ts[0])
            for ts in sorted(left_frames_tuple, key=lambda x: x[0])
            if find_from_index(self.nov_front_image_index, ts[0])
        ]
        frames_dict['nov_front_frames'] = nov_front_frames

        if not left_frames and not right_frames:
            raise OSError 

        if not (may_frames):
            raise OSError 

        self.debug_ellipse_plot(scaled_cov, center, frames_dict)
        self.cluster_polygons_plot(scaled_cov,center,frames_dict) 
        #input("Check debug plot")

        return frames_dict 

    def cluster_polygons_plot(self, cov, center, frames): 
        fig, ax = plt.subplots(figsize=(6, 6))  # square aspect ratio
        #first, plot the May ellipse 
        # Compute eigenvalues and eigenvectors
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        # Calculate angle of ellipse rotation (in degrees)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Width and height of the ellipse (2*stddevs)
        width, height = 2 * np.sqrt(vals)

        ellipse = Ellipse(xy=center, width=width, height=height, angle=theta, edgecolor='blue', facecolor='none')

        min_lat, max_lat, min_lon, max_lon = get_ellipse_bounds(ellipse) 

        ax.add_patch(ellipse) 
        ax.plot(center[0], center[1], 'bo')  # center point 

        may_frames = frames['may_frames']
        left_frames = frames['left_frames']
        right_frames = frames['right_frames'] 

        #plot the may frames 
        may_corners = []
        for frame in may_frames: 
            may_corners.append(gps_lookup(frame))
        may_polys = create_polygons(may_corners)
        print("Plotting may frames ...")
        '''
        for poly in may_polys:
            if not poly.is_valid or poly.is_empty:
                print(f"[WARN] Skipping invalid polygon {i}")
                continue
            if not self.may_frame_area:
                self.may_frame_area = poly.area 
            else: 
                if np.abs(self.may_frame_area - poly.area) / self.may_frame_area > 0.015:
                    poly = fix_polygon_area(poly,self.may_frame_area) 
            x, y = poly.exterior.xy
            ax.plot(x, y, color="blue", linewidth=1)
        '''
        
        # Cluster and assign colors
        clusters = cluster_overlapping_polygons(may_polys)
        cmap = cm.get_cmap('tab10', len(clusters))  # or 'Set3', 'tab20', etc.

        for cluster_idx, cluster in enumerate(clusters):
            color = cmap(cluster_idx)
            for poly in cluster:
                if not poly.is_valid or poly.is_empty:
                    continue
                x, y = poly.exterior.xy
                ax.fill(x, y, color=color, alpha=0.4, edgecolor='blue', linewidth=1) 
   
        #plot the nov frames 
        nov_corners = []
        for frame in left_frames + right_frames: 
            nov_corners.append(gps_lookup(frame))
        nov_polys = create_polygons(nov_corners)
        print("Plotting nov frames ...")

        '''
        for poly in nov_polys:
            if not poly.is_valid or poly.is_empty:
                print(f"[WARN] Skipping invalid polygon {i}")
                continue
            if not self.nov_frame_area:
                self.nov_frame_area = poly.area 
            else: 
                if np.abs(self.nov_frame_area - poly.area) / self.nov_frame_area > 0.015:
                    print(type(poly))
                    poly = fix_polygon_area(poly,self.nov_frame_area) 
            x, y = poly.exterior.xy
            ax.plot(x, y, color="red", linewidth=1)
        '''

        clusters = cluster_overlapping_polygons(nov_polys)
        cmap = cm.get_cmap('tab10', len(clusters))  # or 'Set3', 'tab20', etc.

        for cluster_idx, cluster in enumerate(clusters):
            color = cmap(cluster_idx)
            for poly in cluster:
                if not poly.is_valid or poly.is_empty:
                    continue
                x, y = poly.exterior.xy
                ax.fill(x, y, color=color, alpha=0.4, edgecolor='red', linewidth=1) 

        #add scale bar 
        # --- Add scale bar ---
        scalebar_meters = 1  # Desired scale bar length in meters
        lat = center[1]       # center is (lon, lat)

        # Approximate conversion: meters to degrees longitude
        delta_deg_lon = scalebar_meters / LON_METERS_PER_DEGREE

        # Place the scale bar slightly below the ellipse
        scalebar_x = center[0] - delta_deg_lon / 2
        scalebar_y = center[1] - 0.0003  # adjust vertically below ellipse

        # Draw scale bar line
        x0_data, x1_data = ax.get_xlim()
        x_width_data = x1_data - x0_data
        rel_bar_width = delta_deg_lon / x_width_data  # scale bar width in relative axes units

        # Draw scale bar line in axes-relative coordinates
        ax.plot(
            [0.05, 0.05 + rel_bar_width],
            [0.05, 0.05],
            transform=ax.transAxes,
            color='black',
            linewidth=3
        )
        aspect_correction = 1 / np.cos(np.radians((min_lat + max_lat) / 2))
        ax.set_aspect(aspect_correction)

        ax.grid(True) 

        deg_lat_buffer = 1.5 / LAT_METERS_PER_DEGREE
        deg_lon_buffer = 1.5 /LON_METERS_PER_DEGREE 

        # Add label for scale bar 
        ax.text(
            0.05 + rel_bar_width / 2,  # center of the scale bar
            0.055,                     # slightly above the bar
            f"{scalebar_meters} m",
            transform=ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=10
        )

        ax.set_xlim(center[0] - deg_lon_buffer, center[0] + deg_lon_buffer)  # center[0] = lon
        ax.set_ylim(center[1] - deg_lat_buffer, center[1] + deg_lat_buffer)  # center[1] = lat

        #save   
        output_path = os.path.join(self.paths["match_output_dir"], f"frame_clusters_{int(center[0]*1e6)}_{int(center[1]*1e6)}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight",pad_inches=0.2)
        plt.close(fig)
        print(f"[INFO] wrote: {output_path}")

    def debug_ellipse_plot(self, cov, center, frames): 
        fig, ax = plt.subplots(figsize=(6, 6))  # square aspect ratio
        #first, plot the May ellipse 
        # Compute eigenvalues and eigenvectors
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        # Calculate angle of ellipse rotation (in degrees)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Width and height of the ellipse (2*stddevs)
        width, height = 2 * np.sqrt(vals)

        ellipse = Ellipse(xy=center, width=width, height=height, angle=theta, edgecolor='blue', facecolor='none')

        min_lat, max_lat, min_lon, max_lon = get_ellipse_bounds(ellipse) 

        ax.add_patch(ellipse) 
        ax.plot(center[0], center[1], 'bo')  # center point 

        may_frames = frames['may_frames']
        left_frames = frames['left_frames']
        right_frames = frames['right_frames'] 

        #plot the may frames 
        may_corners = []
        for frame in may_frames: 
            may_corners.append(gps_lookup(frame))
        may_polys = create_polygons(may_corners)
        print("Plotting may frames ...")
        for poly in may_polys:
            if not poly.is_valid or poly.is_empty:
                print(f"[WARN] Skipping invalid polygon {i}")
                continue
            if not self.may_frame_area:
                self.may_frame_area = poly.area 
            else: 
                if np.abs(self.may_frame_area - poly.area) / self.may_frame_area > 0.015:
                    poly = fix_polygon_area(poly,self.may_frame_area) 
            x, y = poly.exterior.xy
            ax.plot(x, y, color="blue", linewidth=1)

        #plot the nov frames 
        nov_corners = []
        for frame in left_frames + right_frames: 
            nov_corners.append(gps_lookup(frame))
        nov_polys = create_polygons(nov_corners)
        print("Plotting nov frames ...")
        for poly in nov_polys:
            if not poly.is_valid or poly.is_empty:
                print(f"[WARN] Skipping invalid polygon {i}")
                continue
            if not self.nov_frame_area:
                self.nov_frame_area = poly.area 
            else: 
                if np.abs(self.nov_frame_area - poly.area) / self.nov_frame_area > 0.015:
                    print(type(poly))
                    poly = fix_polygon_area(poly,self.nov_frame_area) 
            x, y = poly.exterior.xy
            ax.plot(x, y, color="red", linewidth=1)

        #add scale bar 
        # --- Add scale bar ---
        scalebar_meters = 1  # Desired scale bar length in meters
        lat = center[1]       # center is (lon, lat)

        # Approximate conversion: meters to degrees longitude
        delta_deg_lon = scalebar_meters / LON_METERS_PER_DEGREE

        # Place the scale bar slightly below the ellipse
        scalebar_x = center[0] - delta_deg_lon / 2
        scalebar_y = center[1] - 0.0003  # adjust vertically below ellipse

        # Draw scale bar line
        x0_data, x1_data = ax.get_xlim()
        x_width_data = x1_data - x0_data
        rel_bar_width = delta_deg_lon / x_width_data  # scale bar width in relative axes units

        # Draw scale bar line in axes-relative coordinates
        ax.plot(
            [0.05, 0.05 + rel_bar_width],
            [0.05, 0.05],
            transform=ax.transAxes,
            color='black',
            linewidth=3
        )
        aspect_correction = 1 / np.cos(np.radians((min_lat + max_lat) / 2))
        ax.set_aspect(aspect_correction)

        ax.grid(True) 

        deg_lat_buffer = 1.5 / LAT_METERS_PER_DEGREE
        deg_lon_buffer = 1.5 /LON_METERS_PER_DEGREE 

        # Add label for scale bar 
        ax.text(
            0.05 + rel_bar_width / 2,  # center of the scale bar
            0.055,                     # slightly above the bar
            f"{scalebar_meters} m",
            transform=ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=10
        )

        ax.set_xlim(center[0] - deg_lon_buffer, center[0] + deg_lon_buffer)  # center[0] = lon
        ax.set_ylim(center[1] - deg_lat_buffer, center[1] + deg_lat_buffer)  # center[1] = lat

        #save   
        output_path = os.path.join(self.paths["match_output_dir"], f"debug_ellipse_{int(center[0]*1e6)}_{int(center[1]*1e6)}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight",pad_inches=0.2)
        plt.close(fig)
        print(f"[INFO] Saved debug plot to {output_path}")

    def process_timestamp(self,i,timestamp):
        print(f"\n[{i+1}/{len(self.may_cov_timestamps)}] Processing timestamp {timestamp}")

        cov_matrix = load_covariance_matrix(self.may_cov_dir, self.may_cov_timestamps, timestamp)
        if cov_matrix is None:
            print("[WARN] No matching covariance for this timestamp.")
            return

        scaled_cov = scale_covariance_to_degrees(cov_matrix)
        i_may = find_closest_index(self.may_timestamps, timestamp, max_delta_t=0.3)
        center = (self.cam_lon[i_may], self.cam_lat[i_may])

        #print("this is center: ",center)
        frames = self.get_frames(scaled_cov,center) 
        if not frames:
            return 

        may_frames = frames['may_frames']
        may_front_frames = frames['may_front_frames']
        left_frames = frames['left_frames']
        right_frames = frames['right_frames'] 
        nov_front_frames = frames['nov_front_frames']

        output_dir_timestamp = os.path.join(self.paths["match_output_dir"],  str(int(timestamp * 10**9)))
        #print("output_dir_timestamp: ",output_dir_timestamp)

        os.makedirs(output_dir_timestamp, exist_ok=True)
        may_fused_img_dir = os.path.join(output_dir_timestamp, "May")
        nov_fused_img_dir = os.path.join(output_dir_timestamp, "Nov")
        os.makedirs(may_fused_img_dir, exist_ok=True)
        os.makedirs(nov_fused_img_dir, exist_ok=True) 

        self.stitch_front_imgs(timestamp, output_dir_timestamp, may_front_frames, nov_front_frames) 

        may_polys = []
        for frame in may_frames: 
            #may_corners.append(gps_lookup(frame))
            frame_i  = frameInstance(frame)
            may_corners = gps_lookup(frame) 
            frame_i.polygon_obj = create_polygon(may_corners)
            may_polys.append(frame_i) 
        
        clusters = cluster_overlapping_frame_instances(may_polys) 
        for i,cluster in enumerate(clusters):
            stitch_clusters(
                    i,
                    cluster, 
                    may_fused_img_dir, 
                    self.may_compass_headings,
                    self.stitch_cfg, 
                    prefix="may"
                )

        nov_polys = []
        for frame in left_frames + right_frames: 
            #may_corners.append(gps_lookup(frame))
            frame_i  = frameInstance(frame)
            nov_corners = gps_lookup(frame) 
            frame_i.polygon_obj = create_polygon(nov_corners)
            nov_polys.append(frame_i) 
        
        clusters = cluster_overlapping_frame_instances(nov_polys) 
        for i,cluster in enumerate(clusters):
            stitch_clusters(
                    i,
                    cluster, 
                    nov_fused_img_dir, 
                    self.nov_compass_headings,
                    self.stitch_cfg, 
                    prefix="nov"
                )

        if hasattr(self, "front_stitch_thread"):
            try:
                self.front_stitch_thread.join()
            except Exception as e:
                print(f"[ERROR] Failed to join front stitching thread: {e}")

        self.visualize_overlapping_frames(timestamp, output_dir_timestamp,may_fused_img_dir,nov_fused_img_dir)
        input("Hol Up") 

    def visualize_overlapping_frames(self,timestamp,output_dir_timestamp,may_dir,nov_dir):
        cov_matrix = load_covariance_matrix(self.may_cov_dir, self.may_cov_timestamps, timestamp)
        scaled_cov = scale_covariance_to_degrees(cov_matrix)
        i_may = find_closest_index(self.may_timestamps, timestamp, max_delta_t=0.3)
        center = (self.cam_lon[i_may], self.cam_lat[i_may]) 

        print("visualize overlapping frames") 
        output_fig_dir = os.path.join(output_dir_timestamp, "plots")
        #each pickle contains a dict where the keys are the fused image path and the entries are a list of corresponding polygon object  
        may_pano_pickles = [str(f) for f in Path(may_dir).glob("*.pickle") if f.is_file()]
        nov_pano_pickles = [str(f) for f in Path(nov_dir).glob("*.pickle") if f.is_file()]

        for may_pano_pickle in may_pano_pickles: 
            #print("loading this pickle: ",may_pano_pickle)
            with open(may_pano_pickle,"rb") as handle:
                may_pano_dict = pickle.load(handle)
                path = list(may_pano_dict.keys())[0]
                if isinstance(may_pano_dict[path][0],list):
                    #need to find where this bad dict is being written bc it ruins everything, for now just skip it 
                    continue 
            for nov_pano_pickle in nov_pano_pickles:
                #print("loading this pickle: ",nov_pano_pickle)
                with open(nov_pano_pickle,"rb") as handle: 
                    nov_pano_dict = pickle.load(handle)  
                path = list(nov_pano_dict.keys())[0]
                if isinstance(nov_pano_dict[path][0],list):
                    #need to find where this bad dict is being written bc it ruins everything, for now just skip it 
                    continue 
                for fused_may_path in may_pano_dict: 
                    for fused_nov_path in nov_pano_dict: 
                        #may/nov pano dict has path keys and list of corresponding Polygon objects (the camera frustrums fused together saved to path)
                        intersecting_frames = find_intersecting_clusters(may_pano_dict[fused_may_path],nov_pano_dict[fused_nov_path]) 
                        #this is a list of tuples such that if (0,1) then A0 intersects with B1  
                        if not intersecting_frames:
                            return
                        
                        fig = plt.figure(figsize=(15, 5))  # Wider figure
                        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])  # 3 columns
                        # ----- Left: May fused image -----
                        ax0 = fig.add_subplot(gs[0])
                        may_img = cv.imread(fused_may_path)
                        may_img = cv.cvtColor(may_img, cv.COLOR_BGR2RGB)  # OpenCV loads in BGR, need to convert to RGB
                        ax0.imshow(may_img)
                        ax0.axis('off')
                        ax0.set_title(f"May Fused Image\n{os.path.basename(fused_may_path)}", fontsize=10)

                        # ----- Right: November fused image -----
                        ax2 = fig.add_subplot(gs[2])
                        nov_img = cv.imread(fused_nov_path)
                        nov_img = cv.cvtColor(nov_img, cv.COLOR_BGR2RGB)
                        ax2.imshow(nov_img)
                        ax2.axis('off')
                        ax2.set_title(f"November Fused Image\n{os.path.basename(fused_nov_path)}", fontsize=10)

                        # ----- Center: GPS plot -----
                        ax1 = fig.add_subplot(gs[1])

                        plotted_may_clusters = []
                        plotted_nov_clusters = []
                        for intersecting_tupe in intersecting_frames:
                            if not intersecting_tupe[0] in plotted_may_clusters:
                                plotted_may_clusters.append(intersecting_tupe[0]) 
                                may_polygon = may_pano_dict[fused_may_path][intersecting_tupe[0]]
                                x, y = may_polygon.exterior.xy  # get boundary coordinates
                                ax1.fill(x, y, color='blue', alpha=0.1) 
                            if not intersecting_tupe[1] in plotted_nov_clusters: 
                                plotted_nov_clusters.append(intersecting_tupe[1]) 
                                nov_polygon = nov_pano_dict[fused_nov_path][intersecting_tupe[1]]
                                x, y = nov_polygon.exterior.xy  # get boundary coordinates
                                ax1.fill(x, y, color='red', alpha=0.1)  
                        
                        # Add covariance ellipse 
                        vals, vecs = np.linalg.eigh(scaled_cov)
                        order = vals.argsort()[::-1]
                        vals = vals[order]
                        vecs = vecs[:, order]
                        # Calculate angle of ellipse rotation (in degrees)
                        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

                        # Width and height of the ellipse (2*stddevs)
                        width, height = 2 * np.sqrt(vals)

                        ellipse = Ellipse(
                            xy=center,
                            width=width,
                            height=height,
                            angle=theta,
                            edgecolor='blue',
                            facecolor='none',
                            linestyle='dotted'  # or use '--' for dashed
                        )

                        ax1.add_patch(ellipse) 
                        ax1.plot(center[0], center[1], 'bo')  # center point 

                        # Setup GPS plot
                        ax1.set_xlabel("Longitude")
                        ax1.set_ylabel("Latitude")

                        may_fused = mpatches.Patch(facecolor='blue', alpha=0.25, label='May Fused Area')
                        nov_fused = mpatches.Patch(facecolor='red', alpha=0.25, label='Nov Fused Area')
                        ellipse_handle = mlines.Line2D([], [], color='blue', linestyle='dotted', label='May Frame Covariance Ellipse')

                        """
                        may_merged_polygon = unary_union(may_pano_dict[fused_may_path])
                        nov_merged_polygon = unary_union(nov_pano_dict[fused_nov_path]) 

                        # Find the bounding box of both fused polygons
                        may_bounds = may_merged_polygon.bounds  # (minx, miny, maxx, maxy)
                        nov_bounds = nov_merged_polygon.bounds  # (minx, miny, maxx, maxy)

                        # Merge the bounds
                        min_lon = min(may_bounds[0], nov_bounds[0])
                        min_lat = min(may_bounds[1], nov_bounds[1])
                        max_lon = max(may_bounds[2], nov_bounds[2])
                        max_lat = max(may_bounds[3], nov_bounds[3])

                        # Set axis limits
                        ax1.set_xlim(min_lon, max_lon)
                        ax1.set_ylim(min_lat, max_lat)
                        """ 

                        ax1.legend(
                            #may_outline, nov_left_outline, nov_right_outline, 
                            handles=[may_fused, nov_fused, ellipse_handle],
                            loc='lower center',
                            bbox_to_anchor=(0.5, 1.22),
                            ncol=3,
                            fontsize=8,
                            frameon=False
                        )

                        plt.tight_layout(rect=[0, 0, 1, 0.95])  
                        output_path = os.path.join(self.paths["match_output_dir"], f"overlapping_frames" + str(int(timestamp*10**9)) + ".png")
                        print(f"Writing {output_path}")
                        plt.savefig(output_path)
                        input("Check plot") 

                        fig.clf()
                        plt.close(fig)
                        gc.collect()

    def make_comparison_plots(self,output_dir_timestamp,may_panos,nov_panos):
        output_img_dir = os.path.join(output_dir_timestamp, "gps_plots")
        os.makedirs(output_img_dir, exist_ok=True)
        csv_path = os.path.join(output_dir_timestamp, "gps_plot_fused_images.csv")

        with open(csv_path, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["gps_plot_path", "may_fused_img_path", "nov_fused_img_path"])

            fused_img_counter = 0
            for fused_img_path in may_panos:
                print("[INFO:main] iterating over fused_img_path in may panos ...")
                log_mem()

                may_polygon_list = create_polygons(may_panos[fused_img_path])
                
                for nov_img_path in nov_panos:
                    nov_polygon_list = create_polygons(nov_panos[nov_img_path])

                    if not (may_polygon_list and nov_polygon_list):
                        return 

                    may_union = unary_union(may_polygon_list)
                    nov_union = unary_union(nov_polygon_list)

                    max_centroid_dist = 1e-6 

                    may_centroid = may_union.centroid
                    nov_centroid = nov_union.centroid

                    if may_centroid.distance(nov_centroid) > max_centroid_dist:
                        #print(f"[INFO] Skipping comparison figs... Centroids too far: {may_centroid.distance(nov_centroid)} degrees")
                        return 
                    
                    overlap = unary_union(may_polygon_list).intersects(unary_union(nov_polygon_list))
                    if not overlap:
                        #print("[INFO] Skipping comparison figs... no overlap detected!") 
                        return  

                    gps_plot_path = os.path.join(output_img_dir, f"gps_plot_{fused_img_counter}.png")
                    comparison_plot_path = os.path.join(output_img_dir, f"comparison_plot_{fused_img_counter}.png")
                
                    print("[INFO] Making gps plot ...")
                    make_gps_plot(self.may_data, self.nov_data, may_polygon_list, nov_polygon_list, nov_img_path, gps_plot_path)
                    log_mem() 

                    plt.close('all')
                    gc.collect() 

                    #Debugging memory creep
                    snapshot = tracemalloc.take_snapshot()
                    top_stats = snapshot.statistics('lineno')
                    for stat in top_stats[:10]:
                        print(stat)


                    print("[INFO] Making comparison fig ...")
                    make_comparison_fig(self.may_data, self.nov_data, may_polygon_list, nov_polygon_list, fused_img_path, nov_img_path, comparison_plot_path)
                    log_mem()  

                    plt.close('all')
                    gc.collect()

                    print("[INFO:main] memory usage after closing the debug figures ...")
                    log_mem()

                    #Debugging memory creep
                    snapshot = tracemalloc.take_snapshot()
                    top_stats = snapshot.statistics('lineno')
                    for stat in top_stats[:10]:
                        print(stat)

                    csv_writer.writerow([gps_plot_path, fused_img_path, nov_img_path])

                    fused_img_counter += 1

                del nov_polygon_list, may_polygon_list, may_union, nov_union
                nov_polygon_list = None; may_polygon_list = None;
                gc.collect()

        print("[INFO:main] iterating over the may covariance timestamps ...")
        log_mem()
    
    def fuse_front_facing_images(self, timestamp, may_front_frames, nov_front_frames):
        print("[INFO] Starting front-facing image fusion thread...")
        output_dir_timestamp = os.path.join(self.paths["match_output_dir"], str(int(timestamp * 10**9)))
        print("output_dir_timestamp: ",output_dir_timestamp)

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
            cv.destroyAllWindows() 
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
            cv.destroyAllWindows() 
        except Exception as e:
            print(f"[ERROR] Nov front-facing stitching failed: {e}")

        gc.collect()
        print("[INFO] Finished front-facing image fusion thread.")