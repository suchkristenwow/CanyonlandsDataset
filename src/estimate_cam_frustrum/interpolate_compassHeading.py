import csv
import os
import argparse
import numpy as np
from collections import deque

# === Utility Functions ===
def normalize_angle(angle):
    return (angle + 180) % 360 - 180

def circular_mean(headings):
    sum_sin = sum(np.sin(np.radians(h)) for h in headings)
    sum_cos = sum(np.cos(np.radians(h)) for h in headings)
    return np.degrees(np.arctan2(sum_sin, sum_cos)) % 360

def is_heading_in_range(x, min_theta, max_theta):
    x, min_theta, max_theta = x % 360, min_theta % 360, max_theta % 360
    return min_theta <= x <= max_theta if min_theta <= max_theta else x >= min_theta or x <= max_theta

def generate_evenly_spaced_headings(theta0, theta1, num_points, clockwise=None):
    theta0 %= 360
    theta1 %= 360
    if theta0 == theta1:
        return [theta0] * num_points
    cw_dist = (theta1 - theta0) % 360
    ccw_dist = (theta0 - theta1) % 360
    if clockwise is None:
        clockwise = cw_dist <= ccw_dist
    headings = np.linspace(theta0, theta0 + cw_dist, num_points) if clockwise else np.linspace(theta0, theta0 - ccw_dist, num_points)
    return list(headings % 360)

# === Core Classes ===
class AngularVelocityProcessor:
    def __init__(self, auto_annots, manual_annots, max_ang_vel=50.0):
        self.window = deque()
        self.angular_velocity = 0.0
        self.max_ang_vel = max_ang_vel
        self.error_flag = False
        self.last_time = None
        self.last_yaw = None
        self.last_heading = None
        self.auto_annots = auto_annots
        self.manual_annots = manual_annots

    def get_annotated_heading(self, t):
        idx = np.abs(self.auto_annots[:, 0] - t).argmin()
        t_annot = self.auto_annots[idx, 0]
        heading = self.auto_annots[idx, 1] if abs(t_annot - t) <= 0.3 else None
        return heading, t_annot

    def update_deque(self, t, heading):
        self.window.append((t, heading))
        while self.window and self.window[0][0] < t - 1:
            self.window.popleft()
        if len(self.window) > 1:
            dt = self.window[-1][0] - self.window[0][0]
            if dt > 0:
                diff = normalize_angle(self.window[-1][1] - self.window[0][1])
                self.angular_velocity = diff / dt
                if abs(self.angular_velocity) > self.max_ang_vel:
                    return True
        return False

    def update(self, row):
        t, yaw_rad = row[0], row[1]
        yaw_deg = normalize_angle(np.degrees(yaw_rad))
        annotated_heading, annot_t = self.get_annotated_heading(t)
        if annotated_heading is not None:
            self.last_time = annot_t
            self.last_yaw = yaw_deg
            self.last_heading = annotated_heading
            self.error_flag = self.update_deque(t, annotated_heading)
            updated_heading = annotated_heading
            uncertainty = 1.571
        elif self.last_time is not None and self.last_heading is not None:
            yaw_diff = normalize_angle(self.last_yaw - yaw_deg)
            updated_heading = normalize_angle(self.last_heading + yaw_diff)
            self.error_flag = self.update_deque(t, updated_heading)
            uncertainty = 2.356
        else:
            updated_heading = normalize_angle(yaw_deg)
            self.error_flag = True
            uncertainty = np.pi

        if not 0 < updated_heading < 360:
            updated_heading = updated_heading % 360

        if len(self.manual_annots.shape) == 2:
            for annot in self.manual_annots:
                if annot[0] <= t <= annot[1] and not is_heading_in_range(updated_heading, annot[2], annot[3]):
                    updated_heading = circular_mean([annot[2], annot[3]])
                    uncertainty = np.deg2rad(100)
        else:
            annot = self.manual_annots
            if annot[0] <= t <= annot[1] and not is_heading_in_range(updated_heading, annot[2], annot[3]):
                updated_heading = circular_mean([annot[2], annot[3]])
                uncertainty = np.deg2rad(100)

        return updated_heading, uncertainty

    def get_angular_velocity(self):
        return self.angular_velocity

    def has_error(self):
        return self.error_flag

def smooth_error_ranges(data, error_indices):
    smoothed = data.copy()
    if not error_indices:
        return smoothed

    ranges = []
    current = [error_indices[0]]
    for idx in error_indices[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            ranges.append(current)
            current = [idx]
    ranges.append(current)

    for r in ranges:
        start, end = r[0] - 1, r[-1] + 1
        if start < 0 or end >= len(data):
            continue
        h0, h1 = smoothed[start]["updated_heading"], smoothed[end]["updated_heading"]
        interpolated = generate_evenly_spaced_headings(h0, h1, len(r))
        for i, idx in enumerate(r):
            smoothed[idx]["updated_heading"] = interpolated[i]
    return smoothed

def robust_load_csv(path, min_cols=4, skip_header=1):
    import numpy as np

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
                print(f"✅ Loaded annotations using delimiter '{delim}'")
                return data
        except Exception as e:
            print(f"⚠️ Failed to load with delimiter '{delim}': {e}")

    raise ValueError(f"❌ Failed to load usable data from {path} with common delimiters.")

# === Main Script ===
def main(season, plot_name, base_path, manual_annots_path):
    os.makedirs(base_path, exist_ok=True)

    auto_annots = np.genfromtxt(os.path.join(base_path, "data.csv"), delimiter=",")

    manual_annots = robust_load_csv(manual_annots_path) 

    raw_data = np.genfromtxt(os.path.join(base_path, "yawHeading_data.csv"), delimiter=",", skip_header=1)

    processor = AngularVelocityProcessor(auto_annots, manual_annots)
    processed_data = []
    error_indices = []

    for i, row in enumerate(raw_data):
        timestamp = row[0]
        yaw_deg = normalize_angle(np.degrees(row[1]))
        heading, uncertainty = processor.update(row)
        angular_velocity = processor.get_angular_velocity()
        error_flag = processor.has_error()
        if error_flag:
            error_indices.append(i)
        processed_data.append({
            "timestamp": timestamp,
            "yaw_deg": yaw_deg,
            "updated_heading": heading % 360,
            "angular_velocity": angular_velocity,
            "uncertainty_radians": uncertainty,
            "error_flag": error_flag
        })

    if error_indices:
        processed_data = smooth_error_ranges(processed_data, error_indices)

    output_file = os.path.join(base_path, "processed_compass_heading.csv")
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["timestamp", "updated_heading", "uncertainty"])
        writer.writeheader()
        for row in processed_data:
            writer.writerow({
                "timestamp": row["timestamp"],
                "updated_heading": row["updated_heading"],
                "uncertainty": row["uncertainty_radians"]
            })

    print(f"✅ Saved processed heading data to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process compass heading data with error smoothing.")
    parser.add_argument("--season", required=True, help="Season (e.g. Nov, May)")
    parser.add_argument("--plot_name", required=True, help="Plot name (e.g. 1conmod)")
    parser.add_argument("--annotation_path", required=True)
    parser.add_argument("--results_path")

    args = parser.parse_args()
    main(args.season, args.plot_name, args.results_path, args.annotation_path)
