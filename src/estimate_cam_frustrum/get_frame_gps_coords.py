#!/usr/bin/env python

import rospy
import tf
import numpy as np
import argparse
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from nmea_msgs.msg import Sentence
from collections import deque
import os
import csv
from std_msgs.msg import Bool

def get_cardinal_direction(heading_degrees):
    if (heading_degrees >= 345 or heading_degrees < 15):
        return "N"
    elif (heading_degrees >= 15 and heading_degrees < 75):
        return "NE"
    elif (heading_degrees >= 75 and heading_degrees < 105):
        return "E"
    elif (heading_degrees >= 105 and heading_degrees < 165):
        return "SE"
    elif (heading_degrees >= 165 and heading_degrees < 195):
        return "S"
    elif (heading_degrees >= 195 and heading_degrees < 255):
        return "SW"
    elif (heading_degrees >= 255 and heading_degrees < 285):
        return "W"
    elif (heading_degrees >= 285 and heading_degrees < 345):
        return "NW"

class GPSProjector:
    def __init__(self, season, plot_name):
        rospy.init_node("gps_projector_node")
        self.season = season
        self.plot_name = plot_name
        self.result_dir = os.path.join(f"{season}_results", plot_name)
        os.makedirs(self.result_dir, exist_ok=True)

        # File paths
        self.yaw_heading_path = os.path.join(self.result_dir, "yawHeading_data.csv")
        self.transforms_path = os.path.join(self.result_dir, "robotTransforms.csv")

        # Subscribers
        rospy.Subscriber("/lio_sam/mapping/odometry", Odometry, self.odom_callback)
        rospy.Subscriber("/nmea_sentence", Sentence, self.nmea_callback)
        rospy.Subscriber("/gps/fix", NavSatFix, self.gps_callback)

        self.yaw = None
        self.angularVel = None
        self.yaw_window = deque()
        self.tf_listener = tf.TransformListener(cache_time=rospy.Duration(10))
        self.filtered_lat = None
        self.filtered_lon = None
        self.position_covar = []
        self.saved_static_transforms = False

        if not os.path.exists(self.yaw_heading_path):
            with open(self.yaw_heading_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["timestamp", "yaw", "angular_vel (yaw)", "true_course",
                                 "lat", "lon", "filtered_lat", "filtered_lon"] +
                                [f"position_covar{i}" for i in range(9)])

    def euler_from_quaternion(self, q):
        quaternion = (q.x, q.y, q.z, q.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        return euler[2]

    def gps_callback(self, msg):
        self.filtered_lat = msg.latitude
        self.filtered_lon = msg.longitude
        self.position_covar = msg.position_covariance

    def get_navsat_transforms(self):
        if self.saved_static_transforms:
            return
        try:
            listener = self.tf_listener

            def get_radius_yaw(from_frame, to_frame):
                trans, _ = listener.lookupTransform(from_frame, to_frame, rospy.Time(0))
                return np.linalg.norm(trans[:2]), np.arctan2(trans[1], trans[0])

            transforms = []
            links = [
                "H03/horiz_ouster_sensor",
                "H03/front_left_wheel_link",
                "H03/front_right_wheel_link",
                "H03/rear_left_wheel_link",
                "H03/rear_right_wheel_link"
            ]

            for link in links:
                radius, yaw = get_radius_yaw("navsat_link", link)
                transforms += [radius, yaw]

            with open(self.transforms_path, mode='a', newline='') as file:
                csv.writer(file).writerow(transforms)

            self.saved_static_transforms = True

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"TF lookup failed: {e}")

    def odom_callback(self, msg):
        msg_tstamp = msg.header.stamp.to_sec()
        orientation = msg.pose.pose.orientation
        yaw = self.euler_from_quaternion(orientation)
        self.yaw = yaw
        self.yaw_window.append((msg_tstamp, yaw))

        while self.yaw_window and (msg_tstamp - self.yaw_window[0][0] > 1.0):
            self.yaw_window.popleft()

        if len(self.yaw_window) >= 2:
            t_start, yaw_start = self.yaw_window[0]
            t_end, yaw_end = self.yaw_window[-1]
            time_diff = t_end - t_start
            yaw_diff = (yaw_end - yaw_start + np.pi) % (2 * np.pi) - np.pi
            self.angularVel = yaw_diff / time_diff

        self.get_navsat_transforms()

    def nmea_callback(self, msg):
        msg_tstamp = msg.header.stamp.to_sec()
        if msg.sentence.startswith("$GNRMC"):
            try:
                parts = msg.sentence.split(',')
                if len(parts) >= 9 and parts[8]:
                    heading = float(parts[8])
                    if len(parts) >= 11 and parts[10]:
                        heading += float(parts[10])
                else:
                    return
                lat = 10 ** (-2) * float(parts[3])
                lon = -10 ** (-2) * float(parts[5])
                if heading and heading != 0:
                    cardinal = get_cardinal_direction(heading)
                    rospy.loginfo(f"[{msg_tstamp}] Heading: {heading}° ({cardinal})")

                    with open(self.yaw_heading_path, mode='a', newline='') as file:
                        position_covar_list = list(self.position_covar[:9]) if self.position_covar else [None]*9

                        csv.writer(file).writerow([
                            msg_tstamp, self.yaw, self.angularVel, heading,
                            lat, lon, self.filtered_lat, self.filtered_lon
                        ] + position_covar_list)

            except Exception as e:
                print("parts:",parts)
                rospy.logwarn(f"Error parsing NMEA: {e}")
                raise OSError 
            
    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True, help="e.g., May or Nov")
    parser.add_argument("--plot_name", required=True, help="e.g., 1conmod")
    args = parser.parse_args()

    GPSProjector(args.season, args.plot_name).run()
