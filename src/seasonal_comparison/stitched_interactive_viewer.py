import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import cv2 as cv
import numpy as np
import os
from seasonal_comparison.ellipse_utils import plot_covariance_ellipse

def launch_interactive_viewer(fused_dict, timestamp, may_cov_dir, nov_cov_dir):
    """
    Interactive viewer to inspect fused images and optionally show combined covariance ellipses.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.3)

    fused_paths = list(fused_dict.keys())
    fused_names = [os.path.basename(p) for p in fused_paths]

    # Dropdown selector
    ax_dropdown = plt.axes([0.1, 0.15, 0.8, 0.1])
    dropdown = widgets.RadioButtons(ax_dropdown, fused_names, active=0)

    # Checkbox for toggling covariance
    ax_check = plt.axes([0.1, 0.05, 0.3, 0.05])
    checkbox = widgets.CheckButtons(ax_check, ['Show Covariance'], [False])

    def extract_timestamp_ns_from_path(p):
        try:
            return int(os.path.splitext(os.path.basename(p))[0])
        except Exception:
            print(f"[!] Failed to extract timestamp from {p}")
            return None

    def load_and_average_covariances(image_paths, cov_dir):
        cov_matrices = []
        for path in image_paths:
            ts_ns = extract_timestamp_ns_from_path(path)
            if ts_ns is None:
                continue
            cov_path = os.path.join(cov_dir, f"{ts_ns}.csv")
            if not os.path.exists(cov_path):
                print(f"[!] Covariance file not found: {cov_path}")
                continue
            try:
                cov = np.genfromtxt(cov_path, delimiter=',')
                if cov.shape[0] >= 2 and cov.shape[1] >= 2:
                    cov2d = cov[:2, :2]
                    cov_matrices.append(cov2d)
            except Exception as e:
                print(f"[✗] Failed to load {cov_path}: {e}")
                continue

        if not cov_matrices:
            return None
        return np.mean(cov_matrices, axis=0)

    def get_gps_center(image_paths):
        # crude centroid based on all 4 corners if available
        from seasonal_comparison.image_stitching_utils import gps_lookup
        all_points = []
        for p in image_paths:
            try:
                corners = gps_lookup(p)
                if corners:
                    all_points.extend(corners)
            except:
                continue
        if not all_points:
            return None
        lons, lats = zip(*all_points)
        return np.mean(lons), np.mean(lats)

    def update(val):
        ax.clear()
        selected_name = dropdown.value_selected
        fused_path = [p for p in fused_paths if os.path.basename(p) == selected_name][0]
        constituent_paths = fused_dict[fused_path]

        # Load and show fused image
        fused_img = cv.imread(fused_path)
        if fused_img is not None:
            fused_img = cv.cvtColor(fused_img, cv.COLOR_BGR2RGB)
            ax.imshow(fused_img)
        ax.set_title(f"Fused Image: {selected_name}")
        ax.axis("off")

        if checkbox.get_status()[0]:  # Show Covariance checkbox is on
            cov_dir = may_cov_dir if "May" in fused_path else nov_cov_dir
            mean_cov = load_and_average_covariances(constituent_paths, cov_dir)
            gps_center = get_gps_center(constituent_paths)

            if mean_cov is not None and gps_center is not None:
                print(f"[✓] Plotting covariance ellipse at {gps_center}")
                plot_covariance_ellipse(ax, gps_center, mean_cov, edgecolor='purple')
            else:
                print("[!] Could not compute covariance or center.")

        plt.draw()

    dropdown.on_clicked(update)
    checkbox.on_clicked(update)

    update(None)
    plt.show()