# Canyonlands Dataset Toolkit

Welcome to the **Canyonlands Dataset Toolkit** — a collection of Python tools designed to interact with the Canyonlands Dataset, available here: link

This repository provides a set of scripts that:

- Search for corresponding image frames between seasons
- Use covariance ellipsoids to identify well-localized frame clusters
- Stitch image frames into coherent sets for cross-seasonal analysis
- Compare sets of frames for visual changes over time

---

## 🧰 Key Features

- **GPS-Based Frame Matching**: Efficiently search and retrieve frames within spatial proximity across seasonal datasets.
- **Covariance Ellipsoid Filtering**: Select only those sets of frames that lie within a statistically meaningful localization boundary.
- **Temporal Stitching & Comparison**: Align and analyze matched frame sets to observe environmental or scene-level changes between seasons.

---

## 📁 Repository Structure

seasonal_frame_tools/ ├── gps_frame_search.py # Find matching frames across seasons by GPS location ├── ellipsoid_filter.py # Group frames within a localization ellipsoid ├── frame_stitcher.py # Stitch matched frames into sets for comparison ├── compare_sets.py # Visual/structural comparison tools for frame sets ├── utils/ │ └── gps_utils.py # Helper functions for GPS & ENU conversions ├── README.md # You’re here! └── requirements.txt # Python dependencies

---

## Getting Started

### 1. Clone the Repository
### 2. Install Dependencies 
### 3. Example Usage


## Example Output


## Reference 
