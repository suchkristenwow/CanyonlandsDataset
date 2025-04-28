import subprocess
import time
import os
import argparse
import signal
import toml

def source_catkin_workspace(workspace_path):
    return f"source {workspace_path}/devel/setup.bash && "

def launch_process(cmd, shell=True, env=None):
    print(f"🚀 Launching: {cmd}")
    return subprocess.Popen(cmd, shell=shell, executable="/bin/bash", env=env)

def run_bag(bag_dir, plot_name, season, topics): 
    bag_path = os.path.join(bag_dir, f"{plot_name}_{season}2022.bag")
    
    if not os.path.exists(bag_path):
        raise FileNotFoundError(f"❌ Bag file not found: {bag_path}")

    topic_args = " ".join(topics)
    cmd = f"rosbag play {bag_path} --topics {topic_args}"
    
    print(cmd)
    return launch_process(cmd)

def file_has_data(filepath):
    if not os.path.exists(filepath):
        return False
    with open(filepath, 'r') as f:
        lines = f.readlines()
        return len(lines) > 1

def main(plot_name, season, config_path):
    config = toml.load(config_path)

    workspace_path = config["paths"]["workspace_path"]
    bag_dir = config["paths"]["bag_dir"]
    season_dir = config["paths"]["season_dir_Nov"]  # Assume you add this path for November

    results_dir = f"{season}_results/{plot_name}"
    os.makedirs(results_dir, exist_ok=True)

    ros_env = os.environ.copy()
    ros_env["ROS_PACKAGE_PATH"] = f"{workspace_path}/src"

    source_cmd = source_catkin_workspace(workspace_path)

    # Step 2: Run supporting scripts with arguments
    gps_fix_cmd = source_cmd + f"python3 publish_gps_fix.py --season Nov --plot_name {plot_name}"
    imu_cmd = source_cmd + f"python3 publish_IMU_correct.py --season Nov --plot_name {plot_name}"
    frame_cmd = source_cmd + f"python3 get_frame_gps_coords.py --season Nov --plot_name {plot_name}"
    
    if not os.path.exists(os.path.join(results_dir,"yawHeading_data.csv")) or not file_has_data(os.path.join(results_dir,"yawHeading_data.csv")): 
        # Step 1: Launch LIO-SAM
        liosam_cmd = source_cmd + f"roslaunch {workspace_path}/src/LIO-SAM/launch/run_restorebot_Nov_noViz.launch"
        liosam_proc = launch_process(liosam_cmd, env=ros_env)

        gps_proc = launch_process(gps_fix_cmd, env=ros_env)
        imu_proc = launch_process(imu_cmd, env=ros_env)
        frame_proc = launch_process(frame_cmd, env=ros_env)

        time.sleep(5)  # Let nodes spin up

        # Step 3: Run the bag file (filtered)
        print(f"▶️ Playing bag for Nov/{plot_name}")
        bag_proc = run_bag(bag_dir, plot_name, "Nov", topics=[
            "/nmea_sentence",
            "/H03/imu/data",
            "/H03/horiz/os_cloud_node/points",
            "/cam_front/image_color"
        ])
        bag_proc.wait()
        print("📦 Bag playback finished.")

        # Step 4: Terminate launched processes
        for proc, name in [(liosam_proc, "LIO-SAM"), (gps_proc, "GPS Fix"), (imu_proc, "IMU Correction"), (frame_proc, "Frame Coords")]:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
                print(f"✅ {name} shut down cleanly.")
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"⚠️ {name} killed after timeout.")

        print("🎉 All processes completed.")

    # Step 5: Validate yawHeading_data.csv
    yaw_file = os.path.join(results_dir, "yawHeading_data.csv")
    if not file_has_data(yaw_file):
        raise RuntimeError("❌ yawHeading_data.csv has no data.")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Step 6: Run automate_annotations.py
    auto_annotate_cmd = f"python3 {os.path.join(script_dir, 'automate_annotations.py')} --season Nov --plot_name {plot_name}"
    subprocess.check_call(auto_annotate_cmd, shell=True, executable="/bin/bash")

    data_csv = os.path.join(results_dir, "data.csv")
    if not file_has_data(data_csv):
        raise RuntimeError("❌ data.csv has no data after automate_annotations.py")

    # Step 7: Run interpolate_compassHeading.py
    interp_cmd = f"python3 {os.path.join(script_dir, 'interpolate_compassHeading.py')} --season Nov --plot_name {plot_name}"
    subprocess.check_call(source_cmd + interp_cmd, shell=True, executable="/bin/bash")

    proc_heading = os.path.join(results_dir, "processed_compass_heading.csv")
    if not file_has_data(proc_heading):
        raise RuntimeError("❌ processed_compass_heading.csv has no data after interpolation")

    # Step 8: Run getGPSCoords_Nov.py
    gps_coords_cmd = f"python3 {os.path.join(script_dir, 'getGPSCoords_Nov.py')} --plot_name {plot_name}"
    subprocess.check_call(source_cmd + gps_coords_cmd, shell=True, executable="/bin/bash")

    print("✅ All steps completed successfully.")

    # --- Step 9: Second rosbag run (nmea only) + GPS/IMU Fix ---
    print("▶️ Replaying bag (nmea only) + GPS/IMU + Uncertainty Estimation")

    gps_proc2 = launch_process(gps_fix_cmd, env=ros_env)
    imu_proc2 = launch_process(imu_cmd, env=ros_env)
    uncertainty_cmd = source_cmd + f"python3 gps_uncertainty_estimation.py --season Nov --plot_name {plot_name}"
    uncertainty_proc = launch_process(uncertainty_cmd, env=ros_env)

    bag_proc2 = run_bag(bag_dir, plot_name, season, topics=["/nmea_sentence"]) 
    bag_proc2.wait()
    print("📦 Second bag playback finished.")

    for proc, name in [(gps_proc2, "GPS Fix (2nd)"), (imu_proc2, "IMU Correction (2nd)"), (uncertainty_proc, "Uncertainty Estimation")]:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
            print(f"✅ {name} shut down cleanly.")
        except subprocess.TimeoutExpired:
            proc.kill()
            print(f"⚠️ {name} killed after timeout.")

    print("✅✅✅ Full pipeline completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot_name", required=True)
    parser.add_argument("--config", required=True, help="Path to TOML config file")
    args = parser.parse_args()

    main(args.plot_name, args.config)