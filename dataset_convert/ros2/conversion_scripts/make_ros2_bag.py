import os
import rclpy

from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions
from rclpy.serialization import serialize_message
from rosbag2_py._storage import TopicMetadata
from sensor_msgs.msg import PointCloud2, PointField, Image
from builtin_interfaces.msg import Time

# For TFS
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped

# For images
import cv2
from cv_bridge import CvBridge

# General
import numpy as np
from decimal import Decimal
import struct
from tqdm import tqdm
from collections import namedtuple
from scipy.spatial.transform import Rotation as R, Slerp
# import open3d as o3d

def convert_unix_nsec_to_rostime_msg(timestamp_ns):
    timestamp_sec = timestamp_ns // 1_000_000_000
    timestamp_nsec = timestamp_ns % 1_000_000_000

    time_msg = Time()
    time_msg.sec = int(timestamp_sec)
    time_msg.nanosec = int(timestamp_nsec)
    return time_msg

def read_timestamps(timestamp_path):
    """
    """
    with open(timestamp_path, "r") as ts_file:
        timestamps_sec = [Decimal(line.strip()) for line in ts_file]

    return timestamps_sec
    

def quaternion_pose_to_4x4(trans, quat):
    rotation_matrix = R.from_quat(quat).as_matrix()

    # Create the 4x4 transformation matrix
    transformation_matrix = np.eye(4)  # Start with an identity matrix
    transformation_matrix[:3, :3] = rotation_matrix  # Set the rotation part
    transformation_matrix[:3, 3] = trans  # Set the translation part

    return transformation_matrix


def read_quat_poses(poses_path, pose_ts_path):
    """
    """
    timestamps = read_timestamps(pose_ts_path)
    quat_poses_dict = {}
    with open(poses_path, "r") as file:
        for idx, line in enumerate(file):
            quat_pose = line.strip().split()
            timestamp = timestamps[idx]
            quat_poses_dict[timestamp] = [float(element) for element in quat_pose]
    return quat_poses_dict


def create_pointcloud2_msg(points, header_time_ns, frame_id):
    """
    Creates a PointCloud2 message from a numpy array of points.
    """
    msg = PointCloud2()
    msg.header.stamp = header_time_ns
    msg.header.frame_id = frame_id

    msg.height = 1  # Unordered point cloud
    msg.width = points.shape[0]

    # Point fields (x, y, z, intensity)
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        # PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12  # 4 fields * 4 bytes/field
    msg.row_step = msg.point_step * points.shape[0]
    msg.is_dense = True

    # Convert numpy array to byte data
    msg.data = np.asarray(points, dtype=np.float32).tobytes()

    return msg


def create_tf_msg(quaternion_tf, header_timestamp, parent_frame_id, child_frame_id):
    """
    """
    transform_msg = TransformStamped()
    transform_msg.header.stamp = header_timestamp
    transform_msg.header.frame_id = parent_frame_id
    transform_msg.child_frame_id = child_frame_id
    transform_msg.transform.translation.x = quaternion_tf[0]
    transform_msg.transform.translation.y = quaternion_tf[1]
    transform_msg.transform.translation.z = quaternion_tf[2]
    transform_msg.transform.rotation.x = quaternion_tf[3]
    transform_msg.transform.rotation.y = quaternion_tf[4]
    transform_msg.transform.rotation.z = quaternion_tf[5]
    transform_msg.transform.rotation.w = quaternion_tf[6]

    return transform_msg


class ROS2BagCreator():
    camera_names = ["Front_Facing_Images", "Left_Down_Facing_Images", "Right_Down_Facing_Images"]
    def __init__(self, dataset_root_dir, environment):
        # Set input/output dirs
        self.sequence_dir = os.path.join(dataset_root_dir, environment)
        output_dir = os.path.join(dataset_root_dir, "bags", "ros2", environment)

        # Init ROS2 bag writer
        self.writer = SequentialWriter()
        storage_options = StorageOptions(
            uri=output_dir,
            storage_id='sqlite3'
        )
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        self.writer.open(storage_options, converter_options)

        # Write data to bag
        print(f"self.sequence_dir: {self.sequence_dir}")

        # Write camera data
        self.cv_bridge = CvBridge()
        self.write_all_camera_data()

        # Write lidar data
        self.write_all_lidar_data()

        # self.write_all_tfs()

    def write_all_camera_data(self):
        ''' Reads all png files from each camera dir and converts to Image in ROS2. '''
        for cam in self.camera_names:
            img_topic_name = f'{cam}/image'
            img_topic_metadata = TopicMetadata(
                name=img_topic_name,
                type='sensor_msgs/msg/Image',
                serialization_format='cdr'
            )
            self.writer.create_topic(img_topic_metadata)
            
            camera_frame_id = "map" #f"{cam}_frame"
            camera_path = os.path.join(self.sequence_dir, cam)
            image_names = [f for f in os.listdir(camera_path) if f.endswith(".png") and os.path.isfile(os.path.join(camera_path, f))]
            image_names.sort(key=lambda x: int(os.path.splitext(x)[0]))
            image_filepaths = [os.path.join(camera_path, image_name) for image_name in image_names]

            for idx, img_png_filepath in tqdm(enumerate(image_filepaths), total=len(image_filepaths), desc=f"Processing images for {cam}"):
                filename = os.path.basename(img_png_filepath)
                timestamp_ns = int(os.path.splitext(filename)[0])     # Strip the .png extension and convert to int
                header_time_ns = convert_unix_nsec_to_rostime_msg(timestamp_ns)
                msg_time_ns = timestamp_ns
                self.write_single_image(img_topic_name, img_png_filepath, header_time_ns, msg_time_ns, camera_frame_id)

    def write_single_image(self, topic_name, img_filepath, header_time_msg, msg_time_ns, frame_id):
        image = cv2.imread(img_filepath, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"[Warning] Could not read image at {img_filepath}")
            return

        ros_img = self.cv_bridge.cv2_to_imgmsg(image, encoding="passthrough")
        ros_img.header.stamp = header_time_msg
        ros_img.header.frame_id = frame_id

        self.writer.write(topic_name, serialize_message(ros_img), msg_time_ns)

    def write_all_lidar_data(self):
        """ """
        lidar_topic_name = "ouster/points"
        pc_topic_metadata = TopicMetadata(
            name=lidar_topic_name,
            type='sensor_msgs/msg/PointCloud2',
            serialization_format='cdr'
        )
        self.writer.create_topic(pc_topic_metadata)

        lidar_frame_id = "map" #f"lidar_frame"
        lidar_path = os.path.join(self.sequence_dir, "PointClouds")
        scan_names = [f for f in os.listdir(lidar_path) if f.endswith(".bin") and os.path.isfile(os.path.join(lidar_path, f))]
        scan_names.sort(key=lambda x: int(os.path.splitext(x)[0]))
        scan_bin_filepaths = [os.path.join(lidar_path, scan_name) for scan_name in scan_names]
        
        for idx, scan_bin_filepath in tqdm(enumerate(scan_bin_filepaths), total=len(scan_bin_filepaths), desc="Processing point clouds"):
            filename = os.path.basename(scan_bin_filepath)
            timestamp_ns = int(os.path.splitext(filename)[0])
            header_time_ns = convert_unix_nsec_to_rostime_msg(timestamp_ns)
            msg_time_ns = timestamp_ns

            self.write_single_pointcloud(lidar_topic_name, scan_bin_filepath, header_time_ns, msg_time_ns, lidar_frame_id)

    def write_single_pointcloud(self, lidar_topic_name, scan_bin_filepath, header_time_ns, msg_time, pc_frame_id):
        """ """
        point_cloud_data = np.fromfile(scan_bin_filepath, dtype=np.float32).reshape(-1, 3) # Assuming x, y, z, intensity
        pointcloud_msg = create_pointcloud2_msg(point_cloud_data, header_time_ns, pc_frame_id)
        self.writer.write(lidar_topic_name, serialize_message(pointcloud_msg), msg_time)

def main(args=None):
    rclpy.init(args=args)

    # Path to root of CU-MULTI Dataset directory
    dataset_root_dir = '/root/Datasets/canyonlands_dataset'
    environment = "1conmod"
    datapath = os.path.join(dataset_root_dir, environment)

    ROS2BagCreator(dataset_root_dir, environment)


if __name__ == '__main__':
    main()