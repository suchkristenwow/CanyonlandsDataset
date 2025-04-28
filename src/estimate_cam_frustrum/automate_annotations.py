import numpy as np
import argparse
import os
from interpolate_compassHeading import is_heading_in_range, circular_mean, robust_load_csv

def get_range_index(ranges, value):
    for i, (start, end) in enumerate(ranges):
        if start <= value <= end:
            return i
    return None

def main(season, annotation_path, results_path):
    raw_data_path = os.path.join(results_path,f"{season}_results/{plot_name}/yawHeading_data.csv") 
    save_path = os.path.join(results_path,f"{season}_results/{plot_name}/data.csv") 

    hand_annotations = robust_load_csv(annotation_path) 
    print("hand_annotations:", hand_annotations)

    try:
        hand_annotations = hand_annotations[1:, :]
        init_tf = hand_annotations[0, 1]
        annotated_time_ranges = [(row[0], row[1]) for row in hand_annotations]
    except:
        print("Only one annotation")
        init_tf = hand_annotations[1]
        annotated_time_ranges = [(hand_annotations[0], hand_annotations[1])]

    raw_data = np.genfromtxt(raw_data_path, delimiter=",", skip_header=True)

    annotations = []
    for row in raw_data:
        row_tstep = row[0]
        measured_yaw = row[1]
        angular_vel = row[2]
        true_course = row[3]

        idx = get_range_index(annotated_time_ranges, row_tstep)
        if idx is None:
            continue

        if hand_annotations.ndim == 1:  # Single annotation row
            theta_0 = hand_annotations[2]
            theta_1 = hand_annotations[3]
        else:
            annotation = hand_annotations[idx]
            theta_0 = annotation[2]
            theta_1 = annotation[3]

        if is_heading_in_range(true_course, theta_0, theta_1):
            annotations.append([row_tstep, true_course, measured_yaw, angular_vel])
        else:
            # Check if it's in the initial transform zone (start time == 0)
            if hand_annotations.ndim == 1 or hand_annotations[0][0] == 0:
                avg_heading = circular_mean([theta_0, theta_1])
                annotations.append([row_tstep, avg_heading, measured_yaw, angular_vel])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savetxt(save_path, np.array(annotations), delimiter=",")
    print(f"✅ Saved compass heading data to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and save valid compass heading data based on annotations.")
    parser.add_argument("--season", required=True, help="Season name (e.g., 'Nov', 'May')")
    parser.add_argument("--annotation_path", required=True)
    parser.add_argument("--results_path")

    args = parser.parse_args()
    main(args.season, args.annotation_path, args.results_path)