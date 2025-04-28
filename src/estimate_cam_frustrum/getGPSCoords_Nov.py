import numpy as np
import argparse
from geopy.distance import geodesic
from geopy import Point
import math
import os
import csv

def inverseVincenty(lat1, lon1, lat2, lon2):
    point1 = (lat1, lon1)
    point2 = (lat2, lon2)
    range_m = geodesic(point1, point2).meters

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lon1_rad = math.radians(lon1)
    lon2_rad = math.radians(lon2)

    delta_lon = lon2_rad - lon1_rad
    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
    bearing_rad = math.atan2(x, y)
    bearing_deg = (math.degrees(bearing_rad) + 360) % 360

    return range_m, bearing_deg

def calculate_new_gps(lat, lon, range_m, bearing_deg):
    start_point = Point(lat, lon)
    destination = geodesic(meters=range_m).destination(start_point, bearing_deg)
    return destination.latitude, destination.longitude

def main(plot_name):
    input_dir = f"Nov_results/{plot_name}"
    measurements = np.genfromtxt(os.path.join(input_dir, "yawHeading_data.csv"), delimiter=",", skip_header=1)
    compass_data = np.genfromtxt(os.path.join(input_dir, "processed_compass_heading.csv"), delimiter=",", skip_header=1)

    os.makedirs(input_dir, exist_ok=True)
    output_path = os.path.join(input_dir, "results.csv")

    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        for measurement in measurements:
            t = measurement[0]
            lat, lon = measurement[4], measurement[5]

            if not isinstance(lat, float) or not isinstance(lon, float):
                continue
            if np.isnan(lat) or np.isnan(lon):
                continue
            if int(lat) != 38 or int(lon) != -109:
                continue

            idx = np.argmin(np.abs(compass_data[:, 0] - t))
            if np.abs(compass_data[idx, 0] - t) > 0.1:
                if 0 < idx < len(compass_data) - 1:
                    compass_heading_t = np.mean([compass_data[idx - 1, 1], compass_data[idx + 1, 1]])
                else:
                    raise OSError("No valid compass heading interpolation.")
            else:
                compass_heading_t = compass_data[idx, 1]

            if np.isnan(compass_heading_t):
                print("Warning: compass heading is nan")
                continue

            compass_heading_rad = np.deg2rad(compass_heading_t)
            pointer_lat, pointer_lon = calculate_new_gps(lat, lon, 1.5, compass_heading_t)

            ouster_range = np.linalg.norm([0.664, 0.192])
            ouster_bearing_rad = compass_heading_rad + (np.pi / 2 - np.arctan2(0.664, 0.192))
            ouster_lat, ouster_lon = calculate_new_gps(lat, lon, ouster_range, np.rad2deg(ouster_bearing_rad % (2 * np.pi)))

            wheels = {
                'frontLeft': (0.557, 0.097),
                'frontRight': (0.557, -0.474),
                'rearLeft': (0.045, 0.097),
                'rearRight': (0.045, -0.474)
            }

            wheel_coords = {}
            for wheel, (x, y) in wheels.items():
                range_m = np.linalg.norm([x, y])
                bearing_rad = compass_heading_rad - (np.pi / 2 - np.arctan2(x, y))
                lat_lon = calculate_new_gps(lat, lon, range_m, np.rad2deg(bearing_rad % (2 * np.pi)))
                wheel_coords[wheel] = lat_lon

            cam_range = np.linalg.norm([0.679, 0.132])
            cam_bearing_rad = compass_heading_rad + (np.pi / 2 - np.arctan2(0.679, 0.132))
            cam_lat, cam_lon = calculate_new_gps(lat, lon, cam_range, np.rad2deg(cam_bearing_rad % (2 * np.pi)))

            frame_corners = {
                'topRight': (0.0724, 0.78),
                'bottomRight': (0.0747, -1.07),
                'bottomLeft': (0.0745, -2.34),
                'topLeft': (0.0721, 2.09)
            }

            left_frame_coords = {}
            for corner, (range_m, bearing_rad) in frame_corners.items():
                corner_bearing_rad = compass_heading_rad + bearing_rad
                left_frame_coords[corner] = calculate_new_gps(cam_lat, cam_lon, range_m, np.rad2deg(corner_bearing_rad % (2 * np.pi)))

            cam1_range = np.linalg.norm([0.679, 0.269])
            cam1_bearing_rad = compass_heading_rad + (np.pi / 2 - np.arctan2(0.679, 0.269))
            cam1_lat, cam1_lon = calculate_new_gps(lat, lon, cam1_range, np.rad2deg(cam1_bearing_rad % (2 * np.pi)))

            right_frame_coords = {}
            for corner, (range_m, bearing_rad) in frame_corners.items():
                corner_bearing_rad = compass_heading_rad + bearing_rad
                right_frame_coords[corner] = calculate_new_gps(cam1_lat, cam1_lon, range_m, np.rad2deg(corner_bearing_rad % (2 * np.pi)))

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
            writer.writerow(row)

    print(f"✅ Results saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project robot components onto GPS map.")
    parser.add_argument("--plot_name", required=True, help="Plot name (e.g. '1conmod')")
    args = parser.parse_args()
    main(args.plot_name)
