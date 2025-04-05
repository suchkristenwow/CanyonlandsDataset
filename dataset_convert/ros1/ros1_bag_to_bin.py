#!/usr/bin/env python3
import yaml
from tqdm import tqdm
import os
import numpy as np
# np.set_printoptions(suppress=True)
from typing import Tuple
import shutil
from array import array
import json 

# ROS imports
import rosbag
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import NavSatFix, CameraInfo
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry

# Scipy
from scipy.spatial.transform import Rotation as R, Slerp


def interpolate_pose(timestamps, positions, quaternions, target_timestamp):
    """
    Interpolate the pose (position and orientation) for a given target timestamp.
    """
    # Find the interval for interpolation
    idx = np.searchsorted(timestamps, target_timestamp) - 1

    # Check for out-of-bounds and assign default values if necessary
    if idx < 0 or idx >= len(timestamps) - 1:
        return None, None  # Or use -1, -1 for both position and orientation if preferred

    t0, t1 = timestamps[idx], timestamps[idx + 1]
    p0, p1 = positions[idx], positions[idx + 1]
    q0, q1 = quaternions[idx], quaternions[idx + 1]

    # Perform linear interpolation for position
    ratio = float((target_timestamp - t0) / (t1 - t0))  # Convert Decimal to float for ratio
    print(f"t0: {t0}, t1: {t1}, ratio: {ratio}")
    interp_position = (1 - ratio) * p0 + ratio * p1

    # Perform SLERP for orientation
    rotations = R.from_quat([q0, q1])
    slerp = Slerp([float(t0), float(t1)], rotations)  # Convert Decimals to floats for Slerp
    interp_orientation = slerp(float(target_timestamp)).as_quat()

    return interp_position, interp_orientation

def lerp(q0, q1, t):
    """
    Linearly interpolate between two quaternions (q0, q1) by factor t.
    """
    q0 = np.array(q0)
    q1 = np.array(q1)
    
    return (1 - t) * q0 + t * q1

def pointcloud_msg_to_numpy(msg: pc2, datatype=np.float32): # -> NDArray[np.float32]:
    """
    """
    # Decode the point cloud-- ours has five float elts:
    field_names = ['x', 'y', 'z', 'intensity', 'reflectivity']
    points = pc2.read_points(msg, field_names=field_names, skip_nans=True)
    pointcloud_numpy_all = np.array(list(points), dtype=datatype)
    
    pointcloud_numpy = pointcloud_numpy_all[:, :4]  # Only keep the x, y, z, intensity values

    return pointcloud_numpy


def parse_gnss_msg(msg: NavSatFix) -> Tuple[float, float, float]:
    """
    Parses a ROS NavSatFix message to extract latitude, longitude, and altitude.

    Args:
        msg (NavSatFix): A ROS NavSatFix message.

    Returns:
        Tuple[float, float, float]: A tuple containing latitude, longitude, and altitude.
    """
    latitude = msg.latitude
    longitude = msg.longitude
    altitude = msg.altitude
    return latitude, longitude, altitude

 
def transform_msg_to_numpy(msg, offset=None):
    """
    """
    translation = np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z])
    quaternion = np.array([msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w])

    if offset is not None:
        off_t, off_q = offset
        t += off_t
        w = quaternion[3] * off_q[3] - np.dot(quaternion[:3], off_q[:3])
        v = quaternion[3] * off_q[:3] + off_q[3] * quaternion[:3] + np.cross(quaternion[:3], off_q[:3])
        quaternion[3] = w
        quaternion[:3] = v

    return translation, quaternion


def path_to_numpy(msg): 
    """
    """
    # Preallocate dicts for the poses in the path
    path_quat_ts_data_dict = {}

    # Loop over each pose in the path message
    for pose_msg in msg.poses:
        msg_header_time = f"{pose_msg.header.stamp.to_sec():.20f}"
        odom_quat_flat_numpy = pose_msg_to_numpy(pose_msg)
        path_quat_ts_data_dict[msg_header_time] = odom_quat_flat_numpy

    return path_quat_ts_data_dict


def odom_msg_to_numpy(msg: Odometry): # -> NDArray[np.float32]:
    """
    """
    odom_quat_np = pose_msg_to_numpy(msg.pose)
    return odom_quat_np


def pose_msg_to_numpy(pose_msg: Pose): # -> NDArray[np.float32]: 
    """
    """
    odom_quat_np = np.asarray([pose_msg.pose.position.x, 
                               pose_msg.pose.position.y, 
                               pose_msg.pose.position.z,
                               pose_msg.pose.orientation.x,
                               pose_msg.pose.orientation.y,
                               pose_msg.pose.orientation.z,
                               pose_msg.pose.orientation.w])
    return odom_quat_np


def decode_realsense_image(msg, display_image=False):
    """
    Decode a ROS sensor_msgs/Image message into a numpy array.
    """
    if msg.encoding == "16UC1":  # Monochrome depth image
        dtype = np.uint16
        channels = 1
    elif msg.encoding == "rgb8":  # RGB image
        dtype = np.uint8
        channels = 3
    else:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")

    # Convert the image data to a numpy array and reshape it
    image = np.frombuffer(msg.data, dtype=dtype).reshape(
        msg.height, msg.width, channels
    )

    return image

class BagParser:
    def __init__(self, log_paths_dict, robot_name):
        # Set paths for run
        for key, value in log_paths_dict.items():
            setattr(self, key, value)

        # Set topics
        self.ouster_points_topic = f"{robot_name}/ouster/points"
        self.odom_topic = f"{robot_name}/lio_sam/mapping/odometry"
        self.path_topic = f"{robot_name}/lio_sam/mapping/path" 
        self.gnss_1_topic = f"{robot_name}/gnss_1/llh_position"
        self.gnss_2_topic = f"{robot_name}/gnss_2/llh_position"

        # self.gnss_ekf_topic = f"{robot_name}/ekf/llh_position"
        # self.imu_data_topic = f"{robot_name}/imu/data"
        # self.gnss_ekf_heading_topic = f"{robot_name}/ekf/dual_antenna_heading"
        # self.camera_depth_image_topic = f"{robot_name}/camera/depth/image_rect_raw"
        # self.camera_depth_info_topic = f"{robot_name}/camera/depth/camera_info"
        # self.camera_rgb_image_topic = f"{robot_name}/camera/color/image_raw"
        # self.camera_rgb_info_topic = f"{robot_name}/camera/color/camera_info"
        # self.transforms_topic = "/tf"
        # self.static_transforms_topic = "/tf_static"
        
        # Dictionary that maps topics to their respective handling functions. Keys can be used as wanted topics.
        self.topics_handlers_dict = {
            self.ouster_points_topic: self.handle_ouster_pointcloud,
            self.odom_topic: self.handle_odometry,
            self.path_topic: self.handle_path, 
            self.gnss_1_topic: self.handle_gnss1_gps, 
            self.gnss_2_topic: self.handle_gnss2_gps, 
            # self.imu_data_topic: self.handle_imu_data,
            # self.camera_rgb_image_topic: self.handle_rgb_image,
            # self.camera_rgb_info_topic:self.handle_rgb_cam_info, 
            # self.camera_depth_image_topic: self.handle_depth_image,
            # self.camera_depth_info_topic: self.handle_depth_cam_info, 
        }

        # Initialize number of data to 0 so that it corresponds with line in timestamp text file
        self.lidar_pc_num = 0
        # self.camera_rgb_num = 0
        # self.camera_depth_num = 0

        # Initialize dictionaries for timestamps and data, where dict[timestamp] = data
        self.ouster_ts_index_dict = {}    
        self.gnss1_ts_data_dict = {}  
        self.gnss2_ts_data_dict = {}   
        self.odom_ts_data_dict = {}
        self.fin_path_ts_data_dict = {}
        # self.imu_ts_data_dict = {} 
        # self.camera_depth_ts_index_dict = {}   
        # self.camera_rgb_ts_index_dict = {}

        self.path_msg = None
        # self.rgb_info_initted = False 
        # self.rgb_info = None 
        # self.depth_info_initted = False  
        # self.depth_info =  None 

    # def handle_rgb_cam_info(self,msg,msg_header_time): 
    #     if not self.rgb_info_initted: 
    #         self.rgb_info_initted = True 
    #         self.rgb_info = cam_info_to_json(msg)  

    # def handle_rgb_image(self, msg, msg_header_time):
    #     image = utils.image_msg_to_numpy(msg=msg)

    #     # Save rgb image
    #     filename = f"{self.camera_rgb_path}/unsorted_camera_rgb_image_{self.camera_rgb_num}.bin"
    #     image.tofile(filename)

    #     # Store timestamp and index
    #     self.camera_rgb_ts_index_dict[msg_header_time] = self.camera_rgb_num
    #     self.camera_rgb_num += 1

    # def handle_depth_cam_info(self,msg,msg_header_time): 
    #     if not self.depth_info_initted: 
    #         self.depth_info_initted = True 
    #         self.depth_info = utils.cam_info_to_json(msg) 

    # def handle_depth_image(self, msg, msg_header_time):
    #     image = utils.image_msg_to_numpy(msg, datatype=np.uint16, num_channels=1)

    #     # Save depth image
    #     filename = f"{self.camera_depth_path}/unsorted_camera_depth_image_{self.camera_depth_num}.bin"
    #     image.tofile(filename)

    #     # Store timestamp and index
    #     self.camera_depth_ts_index_dict[msg_header_time] = self.camera_depth_num
    #     self.camera_depth_num += 1

    # def handle_imu_data(self, msg, msg_header_time):
    #     # Decode IMU
    #     imu_numpy = imu_msg_to_numpy(msg)
    #     self.imu_ts_data_dict[msg_header_time] = imu_numpy[:6]  # Do not save orientation data

    def handle_path(self, msg, msg_header_time=None): 
        self.path_msg = msg

    def handle_odometry(self, msg, msg_header_time):
        odom_quat_flat = odom_msg_to_numpy(msg)
        self.odom_ts_data_dict[msg_header_time] = odom_quat_flat
        # gt_data_file.write(str(p_x) + ' ' + str(p_y) + ' ' + str(p_z) + ' ' + str(q_x) + ' ' + str(q_y) + ' ' + str(q_z) + ' ' + str(q_w) + '\n')

    def handle_gnss1_gps(self, msg, msg_header_time):
        latlonalt = parse_gnss_msg(msg)
        self.gnss1_ts_data_dict[msg_header_time] = latlonalt

    def handle_gnss2_gps(self, msg, msg_header_time):
        latlonalt = parse_gnss_msg(msg)
        self.gnss2_ts_data_dict[msg_header_time] = latlonalt

    def handle_ouster_pointcloud(self, msg, msg_header_time):
        pointcloud = pointcloud_msg_to_numpy(msg)

        # Save point cloud
        pointcloud_filename = f"{self.lidar_pc_bin_path}/unsorted_lidar_pointcloud_{self.lidar_pc_num}.bin"
        pointcloud.tofile(pointcloud_filename)

        # Store timestamp and index
        self.ouster_ts_index_dict[msg_header_time] = self.lidar_pc_num
        self.lidar_pc_num += 1

    def read_bag(self, rosbag_path):
        # Open specified rosbag file
        bag = rosbag.Bag(rosbag_path)

        for topic, msg, bag_time in bag.read_messages(self.topics_handlers_dict.keys()):
            # print(f"Processing message from topic: {topic}")
            # msg_time = f"{msg.header.stamp.secs}.{msg.header.stamp.nsecs}"
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                msg_header_time = f"{msg.header.stamp.to_sec():.20f}"
            elif hasattr(msg, 'transforms'):
                msg_header_time = None #f"{msg.transforms.header.stamp.to_sec()}"

            # Dispatch message to the corresponding handler function
            self.topics_handlers_dict[topic](msg, msg_header_time)

        # Assign final path attributes after bag has been completely read
        self.fin_path_ts_data_dict = path_to_numpy(self.path_msg)

        # Close the bag file
        bag.close()

    def write_data_to_files(self):
        # Helper function to handle timestamps, renaming, and saving data
        def save_and_rename(timestamp_file_prefix, ts_index_dict, timestamp_path, data_path, filename_prefix):
            timestamps_np = np.array(sorted(ts_index_dict.keys()))
            np.savetxt(f'{timestamp_path}/{timestamp_file_prefix}timestamps.txt', timestamps_np, fmt='%s')

            for idx, timestamp in enumerate(timestamps_np):
                original_index = ts_index_dict[timestamp]
                new_index = idx + 1
                original_filename = f"{data_path}/unsorted_{filename_prefix}_{original_index}.bin"
                new_filename = f"{data_path}/{filename_prefix}_{new_index}.bin"
                os.rename(original_filename, new_filename)
        
        # # Write and rename CAMERA RGB data
        # save_and_rename("rgb_", self.camera_rgb_ts_index_dict, self.camera_path, self.camera_rgb_path, "camera_rgb_image")
        
        # # Write and rename CAMERA DEPTH data
        # save_and_rename("depth_", self.camera_depth_ts_index_dict, self.camera_path, self.camera_depth_path, "camera_depth_image")
        
        # Write and rename LIDAR data
        save_and_rename("", self.ouster_ts_index_dict, self.lidar_path, self.lidar_pc_bin_path, "lidar_pointcloud")

        # # Write IMU timestamps and data
        # imu_timestamps_np = np.array(sorted(self.imu_ts_data_dict.keys()))
        # np.savetxt(f'{self.imu_path}/timestamps.txt', imu_timestamps_np, fmt='%s')

        # imu_data_np = np.array([self.imu_ts_data_dict[timestamp] for timestamp in imu_timestamps_np])
        # np.savetxt(f'{self.imu_path}/imu_data.txt', imu_data_np)

        # Save RTK GPS to txt file
        gnss_1_timestamps_np = np.array(sorted(self.gnss1_ts_data_dict.keys()))
        np.savetxt(f"{self.gps_path}/gnss_1_timestamps.txt", gnss_1_timestamps_np, fmt="%s")
        gnss_1_np = np.array([self.gnss1_ts_data_dict[timestamp] for timestamp in gnss_1_timestamps_np])
        np.savetxt(f"{self.gps_path}/gnss_1_data.txt", gnss_1_np)

        gnss_2_timestamps_np = np.array(sorted(self.gnss2_ts_data_dict.keys()))
        np.savetxt(f"{self.gps_path}/gnss_2_timestamps.txt", gnss_2_timestamps_np, fmt="%s")
        gnss_2_np = np.array([self.gnss2_ts_data_dict[timestamp] for timestamp in gnss_2_timestamps_np])
        np.savetxt(f"{self.gps_path}/gnss_2_data.txt", gnss_2_np)

        # Write odometry timestamps and data
        odom_timestamps_np = np.array(sorted(self.odom_ts_data_dict.keys()))
        np.savetxt(f'{self.poses_path}/odom_timestamps.txt', odom_timestamps_np, fmt='%s')
        odom_quat_np = np.array([self.odom_ts_data_dict[timestamp] for timestamp in odom_timestamps_np])
        np.savetxt(f'{self.poses_path}/groundtruth_odom.txt', odom_quat_np) 

        # Write final path timestamps and data to txt files
        fin_path_timestamps_np = np.array(sorted(self.fin_path_ts_data_dict.keys()))
        np.savetxt(f'{self.poses_path}/path_timestamps.txt', fin_path_timestamps_np, fmt='%s')
        fin_path_quat_np = np.array([self.fin_path_ts_data_dict[timestamp] for timestamp in fin_path_timestamps_np])
        np.savetxt(f'{self.poses_path}/groundtruth_path.txt', fin_path_quat_np) 

        # # Save Camera calib info 
        # with open(self.camera_path + '/rgb_cam_info.json','w') as f: 
        #     json.dump(self.rgb_info,f,indent=4) 

        # with open(self.camera_path + '/depth_cam_info.json','w') as f:
        #     json.dump(self.depth_info,f,indent=4)   

class DatasetToBin:
    def __init__(self, seq_name, input_bag_path, output_bin_dir):
        self.seq_name = seq_name
        self.input_bag_path = input_bag_path
        self.root_output_bin_dir = output_bin_dir

    def bag_to_kitti(self):
        print(f"Binarizing Sequence: {self.seq_name}")

        # Create KITTI-style directory for seq
        kitti_paths_dict = self.create_rootdir_and_subdirs()
        kitti_directory_path = kitti_paths_dict['directory_path']
        print(f"        - Created KITTI-style directory: {kitti_directory_path}")

        # Init bagparser object
        bag_parser = BagParser(log_paths_dict=kitti_paths_dict)

        # Parse rosbag data
        print(f"        - Parsing rosbag: {self.input_bag_path}")
        bag_parser.read_bag(rosbag_path=self.input_bag_path)

        # Save data
        bag_parser.write_data_to_files()

    def create_rootdir_and_subdirs(self):
        # Dictionary to hold all necessary subdirectory paths
        run_dir_paths_dict = {
            'directory_path': self.root_output_bin_dir,
            'poses_path': os.path.join(self.root_output_bin_dir, "poses"),
            'lidar_path': os.path.join(self.root_output_bin_dir, "lidar"),
            'lidar_pc_bin_path': os.path.join(self.root_output_bin_dir, "lidar", "pointclouds"),
            'gps_path': os.path.join(self.root_output_bin_dir, "gps"),
            # 'imu_path': os.path.join(directory_path, "imu"),
            # 'camera_path': os.path.join(directory_path, "camera"),
            # 'camera_rgb_path': os.path.join(directory_path, "camera", "images", "rgb"),
            # 'camera_depth_path': os.path.join(directory_path, "camera", "images", "depth")
        }

        # Create all directories based on the paths in the dictionary
        for path in run_dir_paths_dict.values():
            os.makedirs(path, exist_ok=True)

        return run_dir_paths_dict


def main():
    # Paths
    seq_name = ""
    input_bag_path = ""
    output_bin_dir = ""

    dataset2bin = DatasetToBin(seq_name, input_bag_path, output_bin_dir)
    dataset2bin.bag_to_kitti()  # Convert


if __name__ == "__main__":
    main()