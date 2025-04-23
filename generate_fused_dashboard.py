from nbformat import v4 as nbf
import nbformat

nb = nbf.new_notebook()
cells = []

# Cell 1: Markdown header
cells.append(nbf.new_markdown_cell("""\
# 🌱 Fused Image Dashboard: May vs Nov

Interactive dashboard for exploring stitched images from two seasons.
- ✅ 6 May and 6 Nov stitched images always visible
- ✅ Center GPS plot updated by toggles
- ✅ Each image has toggles to show GPS frame polygons and covariance ellipses
- 🚫 Frames with malformed corner distances (triangles) are automatically skipped
"""))

# Cell 2: Imports
cells.append(nbf.new_code_cell("""\
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint
from matplotlib.patches import Polygon as MplPolygon
from ipywidgets import VBox, HBox, Checkbox, Output, Layout
from IPython.display import display, clear_output

from seasonal_comparison.ellipse_utils import plot_covariance_ellipse
from seasonal_comparison.image_stitching_utils import gps_lookup, get_unique_sorted_corners
from seasonal_comparison.gps_utils import load_covariance_matrix, debug_corner_distances
"""))

# Cell 3: Load pano dicts
cells.append(nbf.new_code_cell("""\
import pickle

with open("may_pano_dict.pickle", "rb") as f:
    may_pano_dict = pickle.load(f)
with open("nov_pano_dict.pickle", "rb") as f:
    nov_pano_dict = pickle.load(f)

may_cov_dir = "/media/kristen/easystore2/RestorebotData/May2022_final/1conmod/processed_results/covariance_matrices"
nov_cov_dir = "/media/kristen/easystore2/RestorebotData/Nov2022/1conmod/processed_results/covariance_matrices"
"""))

# Cell 4: Utilities
cells.append(nbf.new_code_cell("""\
def extract_timestamp_ns_from_path(p):
    filename = os.path.basename(p).replace("_downscaled", "")
    try:
        return int(os.path.splitext(filename)[0])
    except Exception:
        print(f"[!] Failed to extract timestamp from {p}")
        return None

def load_and_average_covariances(image_paths, cov_dir):
    cov_matrices = []
    for path in image_paths:
        ts_ns = extract_timestamp_ns_from_path(path)
        if ts_ns is None:
            #print("ts_ns:",ts_ns)
            continue
        cov_path = os.path.join(cov_dir, f"{ts_ns}.csv")
       
        if not os.path.exists(cov_path):
            #try to find a covariance path thats close enough
            #165082581.3395741138
            partial_str = str(ts_ns)[:10]
            potential_paths = [x for x in os.listdir(cov_dir) if partial_str in x] 
            cov_path = min(
                potential_paths,
                key=lambda p: abs(ts_ns - extract_timestamp_ns_from_path(p)),
                default=None  
            )
            cov_path = os.path.join(cov_dir,cov_path)
        #print("cov_path:",cov_path)
        cov = np.genfromtxt(cov_path,delimiter=" ")
        #print("cov:",cov)
        if cov.shape[0] >= 2 and cov.shape[1] >= 2:
            cov_matrices.append(cov[:2, :2])

    return np.mean(cov_matrices, axis=0) if cov_matrices else None

def get_gps_center(image_paths):
    points = []
    for p in image_paths:
        #print("p:",p)
        try:
            corners = gps_lookup(p)
            #print("corners:",corners)
            if corners: points.extend(corners)
        except:
            continue
    if not points: return None
    lons, lats = zip(*points)
    return np.mean(lons), np.mean(lats)

"""))

# Cell 5: GPS plotting
cells.append(nbf.new_code_cell("""\
gps_out = Output(layout=Layout(border='1px solid black'))

def update_gps_plot(states):
    with gps_out:
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        all_lons, all_lats = [], []

        for label, enabled, show_cov, fused_path, constituent_paths, color, cov_dir in states:
            if not enabled: continue
            for frame_path in constituent_paths:
                corners = gps_lookup(frame_path)
                if not corners: continue
                corners = get_unique_sorted_corners(corners)
                if len(corners) < 4: continue
                if not debug_corner_distances(corners, tol=0.2): continue
                poly = MplPolygon(corners, closed=True, facecolor=color, alpha=0.15, edgecolor='none')
                ax.add_patch(poly)
                lons, lats = zip(*corners)
                all_lons.extend(lons)
                all_lats.extend(lats)

            if show_cov:
                cov = load_and_average_covariances(constituent_paths, cov_dir)
                center = get_gps_center(constituent_paths)
                if cov is not None and center is not None:
                    print(f"[✓] Plotting ellipse at center={center} with cov=\n{cov}")

                    # Check for valid ellipse shape
                    vals, vecs = np.linalg.eigh(cov)
                    order = vals.argsort()[::-1]
                    vals = vals[order]
                    vecs = vecs[:, order]

                    if np.any(vals <= 0):
                        print("[✗] Skipping ellipse — non-positive eigenvalues.")
                        continue

                    width, height = 2 * np.sqrt(vals)
                    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                    print(f"[→] Ellipse size: width={width}, height={height}, angle={angle}")

                    from matplotlib.patches import Ellipse
                    ellipse = Ellipse(xy=center, width=width, height=height, angle=angle,
                                    edgecolor=color, facecolor='none', lw=2, label=f"Covariance ({label[:3]})")
                    ax.add_patch(ellipse)

                    # Add a centroid point in the same color
                    ax.plot(center[0], center[1], 'o', color=color, label=f"Center ({label[:3]})")

                    # Force zoom for debug (optional — override global bounds)
                    buffer_lon = 0.001
                    buffer_lat = 0.001
                    ax.set_xlim(center[0] - buffer_lon, center[0] + buffer_lon)
                    ax.set_ylim(center[1] - buffer_lat, center[1] + buffer_lat)

        if all_lons and all_lats:
            ax.set_xlim(min(all_lons), max(all_lons))
            ax.set_ylim(min(all_lats), max(all_lats))

        ax.set_title("GPS Footprints (Toggleable)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right", fontsize="small") 
        ax.grid(True)
        plt.show()
"""))

# Cell 6: Widget control builder
cells.append(nbf.new_code_cell("""\
from IPython.display import display, Image as IPyImage
import PIL.Image

def build_controls(fused_dict, color, cov_dir):
    controls, widgets = [], []
    for fused_path, frame_paths in list(fused_dict.items())[:6]:
        label = os.path.basename(fused_path)
        show_frames = Checkbox(value=True, description=label[:8])
        show_cov = Checkbox(value=False, description="Cov")
        hbox = HBox([show_frames, show_cov])
        widgets.append(hbox)
        controls.append((label, show_frames, show_cov, fused_path, frame_paths, color, cov_dir))
    return controls, widgets

may_controls, may_widgets = build_controls(may_pano_dict, "blue", may_cov_dir)
nov_controls, nov_widgets = build_controls(nov_pano_dict, "red", nov_cov_dir)

def collect_states():
    return [(label, sf.value, sc.value, fpath, frames, col, covd)
            for (label, sf, sc, fpath, frames, col, covd) in may_controls + nov_controls]
"""))

# Cell 7: Display logic
cells.append(nbf.new_code_cell("""\
def load_and_resize_image(path, max_height=150):
    try:
        img = PIL.Image.open(path)
        if img.height > max_height:
            scale = max_height / img.height
            img = img.resize((int(img.width * scale), max_height))
        return img
    except Exception as e:
        print(f"[!] Could not load image {path}: {e}")
        return None

def make_image_column(fused_dict, widgets):
    col = []
    for (fused_path, _), w in zip(list(fused_dict.items())[:6], widgets):
        img = load_and_resize_image(fused_path)
        out = Output()
        with out:
            if img:
                display(img)
        col.append(VBox([out, w]))
    return VBox(col)

for _, cb1, cb2, *_ in may_controls + nov_controls:
    cb1.observe(lambda change: update_gps_plot(collect_states()), names="value")
    cb2.observe(lambda change: update_gps_plot(collect_states()), names="value")

left = make_image_column(may_pano_dict, may_widgets)
right = make_image_column(nov_pano_dict, nov_widgets)

display(HBox([left, gps_out, right]))
update_gps_plot(collect_states())
"""))

# Save the notebook
nb['cells'] = cells
with open("fused_dashboard.ipynb", "w") as f:
    nbformat.write(nb, f)

print("✅ Notebook created: fused_dashboard.ipynb")
print("📂 Open it in JupyterLab and run: Kernel > Restart Kernel and Run All")
