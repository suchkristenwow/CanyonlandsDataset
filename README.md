# Canyonlands Dataset Toolkit

Welcome to the **Canyonlands Dataset Toolkit** — a collection of Python tools designed to interact with the Canyonlands Dataset, available here: <LINK TO DATASET>

This repository provides a set of scripts that:

- Search for corresponding image frames between seasons
- Use covariance ellipsoids to identify well-localized frame clusters
- Stitch image frames into coherent sets for cross-seasonal analysis
- Compare sets of frames for visual changes over time

<p align="center"> <img src="imgs/20221108_100625.jpg" width="600"/> </p>

---

## 🧰 Key Features

- **GPS-Based Frame Matching**: Efficiently search and retrieve frames within spatial proximity across seasonal datasets.
- **Covariance Ellipsoid Filtering**: Select only those sets of frames that lie within a statistically meaningful localization boundary.
- **Temporal Stitching & Comparison**: Compare matched frame sets to observe environmental or scene-level changes between seasons.

---

## Getting Started

### 1. Download the Data
The data are divided between May and November, and subsequently by plot. 

<p align="left"> <img src="imgs/canyonlands_filestructure.png" width="250"/> </p>

The original ROS1 rosbags are separately included under bags. To decompress, simply run:
```
rosbag decompress plotName.bag 
```

### 2. Clone the Repository & Install Dependencies 

```
git clone https://github.com/suchkristenwow/CanyonlandsDataset
cd ./CanyonlandsDataset
pip install .
```

### 3. Example Usage: ROS 2 Rebagification
### 4. Example Usage: Multi-Seasonal Comparison
Set the desired paths and stitching parameters inside configs/your_config.toml, then run:

    python get_overlapping_frames.py --config configs/your_config.toml


This script iterates over each frame in May and searches for overlapping frames from the same plot in November whose centroid falls within the covariance associated with that timestamp. These frames are then temporarily chunked (to manage image size and memory usage) and stitched.In the center the frustrum of these frames is plotted in GPS coordinates. The paths to the constituent images are mapped to each stitched image and pickled. 

#### Example Output
<p align="center"> <img src="imgs/comparison_plot21.png" width="800"/> </p> 


## Dataset Configuration Files
The dataset includes all necessary calibration and configuration files to support usage and reproducibility:

LIOSAM Configuration Files: Located under configs/LIOSAM/.

- These YAML configuration files allow users to replicate odometry and mapping results in the map frame using the provided GPS and IMU data. 

## Down-Facing Camera Frustrum Estimation Pipeline 
We provide all the scripts used to estimate the GPS coordinates of the down-facing camera frames. For your convenience, you need only to run either ``processing_pipeline_May.py`` or ``processing_pipeline_Nov.py`` along with the desired plot name and configuration file containing the relevant paths.  
Additionally, to visualize the down-facing camera frustrum while playing the ROS bag, you can use either ``publishDownCamFrames_May.py`` or ``publishDownCamFrames_Nov.py`` 