#!/usr/bin/env python3

import os

import numpy as np
import cv2 as cv

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon

from shapely.geometry import MultiPoint
from shapely.ops import unary_union

from geographiclib.geodesic import Geodesic

from seasonal_comparison.image_stitching_utils import check_corner_uniqueness

def find_largest_overlap_subset(polygons):
    """
    Given a list of polygons, find the largest subset where all polygons overlap at least partially.
    
    Input:
        polygons (list of shapely.geometry.Polygon)
    
    Output:
        overlapping_polygons (list of shapely.geometry.Polygon)
    """
    if not polygons:
        return []

    # Start with the largest polygon
    sorted_polys = sorted(polygons, key=lambda p: p.area, reverse=True)

    best_subset = []
    best_union = None

    for i, poly in enumerate(sorted_polys):
        current_subset = [poly]
        current_union = poly

        for other in sorted_polys[i+1:]:
            if current_union.intersects(other):
                current_subset.append(other)
                current_union = unary_union([current_union, other])

        if len(current_subset) > len(best_subset):
            best_subset = current_subset
            best_union = current_union

    return best_subset

def compute_fourth_corner(A, B, C):
    """ Given three points (lat, lon), compute the fourth assuming rectangle """
    geod = Geodesic.WGS84

    inv = geod.Inverse(A[0], A[1], B[0], B[1])
    azimuth_AB = inv['azi1']
    distance_AB = inv['s12']

    proj = geod.Direct(C[0], C[1], azimuth_AB, distance_AB)
    D_lat = proj['lat2']
    D_lon = proj['lon2']

    return (D_lat, D_lon)

def fix_invalid_corners(corners):
    """
    Given up to 4 GPS (lon, lat) corners (even if bad),
    return 4 corners forming a rectangle.

    Input:
        corners (list of (lon, lat)) - Can have duplicate or unordered points

    Output:
        fixed_corners (list of (lon, lat)) - Properly ordered rectangle corners
    """
    if len(corners) < 3:
        raise ValueError("Need at least 3 points to fix corners.")

    # Step 1: Remove duplicates
    unique_pts = []
    tol = 1e-7
    for pt in corners:
        if not any(np.linalg.norm(np.array(pt) - np.array(other)) < tol for other in unique_pts):
            unique_pts.append(pt)

    if len(unique_pts) < 3:
        raise ValueError("Not enough unique points to define a rectangle.")

    if len(unique_pts) == 3:
        # Step 2: Compute missing fourth point
        p1, p2, p3 = unique_pts

        # In GPS coordinates (lon, lat):
        # We'll assume relatively small area -> locally flat
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)

        # Find vectors
        v1 = p2 - p1
        v2 = p3 - p1
        v3 = p3 - p2

        # Find which two points are most orthogonal
        # Cosine of angle between vectors
        def cos_angle(v, w):
            v_norm = np.linalg.norm(v)
            w_norm = np.linalg.norm(w)
            if v_norm == 0 or w_norm == 0:
                return 1.0
            return np.dot(v, w) / (v_norm * w_norm)

        c12 = abs(cos_angle(v1, v2))
        c13 = abs(cos_angle(v1, v3))
        c23 = abs(cos_angle(v2, v3))

        # Pick the pair with smallest cosine (most orthogonal)
        min_c = min(c12, c13, c23)

        if min_c == c12:
            # v1 and v2 are most orthogonal
            origin = p1
            pt2 = p2
            pt3 = p3
        elif min_c == c13:
            origin = p2
            pt2 = p1
            pt3 = p3
        else:
            origin = p3
            pt2 = p1
            pt3 = p2

        # Compute 4th point: origin + (pt2 - origin) + (pt3 - origin)
        fourth = pt2 + (pt3 - origin)

        fixed = [tuple(origin), tuple(pt2), tuple(fourth), tuple(pt3)]

    else:
        # Already 4 points — just reorder safely
        multipoint = MultiPoint(unique_pts)
        hull = multipoint.convex_hull
        if hull.geom_type != 'Polygon' or len(hull.exterior.coords) < 5:
            raise ValueError("Invalid convex hull from 4 points.")
        fixed = list(hull.exterior.coords)[:-1]  # drop duplicate last point

    return fixed

def reorder_corners(corner_tuples):
    """
    Given 4 corner points (lon, lat), returns them reordered counterclockwise around centroid.
    """
    # Convert to numpy for easy math
    pts = np.array(corner_tuples)
    
    # Compute centroid
    centroid = np.mean(pts, axis=0)
    
    # Compute angles from centroid
    angles = np.arctan2(pts[:,1] - centroid[1], pts[:,0] - centroid[0])
    
    # Sort points by angle (counterclockwise)
    sort_order = np.argsort(angles)
    ordered_pts = pts[sort_order]
    
    return [tuple(pt) for pt in ordered_pts]

def make_comparison_fig(may_data,nov_data,may_polygon_list,nov_polygon_list,fused_img_path,nov_img_path,output_path):
    fig = plt.figure(figsize=(15, 5))  # Wider figure
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])  # 3 columns

    # ----- Left: May fused image -----
    ax0 = fig.add_subplot(gs[0])
    may_img = cv.imread(fused_img_path)
    may_img = cv.cvtColor(may_img, cv.COLOR_BGR2RGB)  # OpenCV loads in BGR, need to convert to RGB
    ax0.imshow(may_img)
    ax0.axis('off')
    ax0.set_title(f"May Fused Image\n{os.path.basename(fused_img_path)}", fontsize=10)

    # ----- Center: GPS plot -----
    ax1 = fig.add_subplot(gs[1])

    # Plot filled fused polygons
    may_overlap_subset = find_largest_overlap_subset(may_polygon_list)
    for polygon in may_overlap_subset:
        coords = list(polygon.exterior.coords)[:-1]  # drop closing duplicate
        ordered_corners = reorder_corners(coords)
        ordered_corners = fix_invalid_corners(ordered_corners)
        # First for ax
        patch = MplPolygon(ordered_corners, fill=True, facecolor='blue', alpha=0.25)
        ax0.add_patch(patch)
        # Then create a *new* patch for ax1
        patch_copy = MplPolygon(ordered_corners, fill=True, facecolor='blue', alpha=0.25)
        ax1.add_patch(patch_copy)
 

    nov_overlap_subset = find_largest_overlap_subset(nov_polygon_list)
    poly_color = "orange" if "Right" in nov_img_path else "red"
    for polygon in nov_overlap_subset:
        coords = list(polygon.exterior.coords)[:-1]  # drop closing duplicate
        ordered_corners = reorder_corners(coords)
        ordered_corners = fix_invalid_corners(ordered_corners)
        # First for ax
        patch = MplPolygon(ordered_corners, fill=True, facecolor=poly_color, alpha=0.25)
        ax0.add_patch(patch)
        # Then create a *new* patch for ax1
        patch_copy = MplPolygon(ordered_corners, fill=True, facecolor=poly_color, alpha=0.25)
        ax1.add_patch(patch_copy)

    # Setup GPS plot
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    if may_overlap_subset:
        may_merged_polygon = unary_union(may_polygon_list)
    else:
        raise OSError 

    if nov_overlap_subset:
        nov_merged_polygon = unary_union(nov_polygon_list)
    else:
        raise OSError 

    # Find the bounding box of both fused polygons
    may_bounds = may_merged_polygon.bounds  # (minx, miny, maxx, maxy)
    nov_bounds = nov_merged_polygon.bounds  # (minx, miny, maxx, maxy)

    # Merge the bounds
    min_lon = min(may_bounds[0], nov_bounds[0])
    min_lat = min(may_bounds[1], nov_bounds[1])
    max_lon = max(may_bounds[2], nov_bounds[2])
    max_lat = max(may_bounds[3], nov_bounds[3])


    # Set axis limits
    ax1.set_xlim(min_lon, max_lon)
    ax1.set_ylim(min_lat, max_lat)

    # Correct aspect ratio for GPS distortions
    aspect_correction = np.cos(np.deg2rad((min_lat + max_lat) / 2))
    ax1.set_aspect(aspect_correction)

    ax1.set_title("Down-Facing Camera Frames", fontsize=10)

    # Add legend
    '''
    may_outline = mpatches.Patch(facecolor='none', edgecolor='blue', label='May Frames')
    nov_left_outline = mpatches.Patch(facecolor='none', edgecolor='red', label='Nov Frames (Left)')
    nov_right_outline = mpatches.Patch(facecolor='none', edgecolor='orange', label='Nov Frames (Right)')
    '''
    may_fused = mpatches.Patch(facecolor='blue', alpha=0.25, label='May Fused Area')
    nov_fused = mpatches.Patch(facecolor=poly_color, alpha=0.25, label='Nov Fused Area')

    ax1.legend(
        #may_outline, nov_left_outline, nov_right_outline, 
        handles=[may_fused, nov_fused],
        loc='lower center',
        bbox_to_anchor=(0.5, 1.22),
        ncol=2,
        fontsize=8,
        frameon=False
    )

    # ----- Right: November fused image -----
    ax2 = fig.add_subplot(gs[2])
    nov_img = cv.imread(nov_img_path)
    nov_img = cv.cvtColor(nov_img, cv.COLOR_BGR2RGB)
    ax2.imshow(nov_img)
    ax2.axis('off')
    ax2.set_title(f"November Fused Image\n{os.path.basename(nov_img_path)}", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    print(f"Writing {output_path}")
    plt.savefig(output_path)
    
    fig.clf()
    plt.close(fig)

    return fig 

def make_gps_plot(may_data,nov_data,may_polygon_list,nov_polygon_list,nov_img_path,output_path):
    fig, ax = plt.subplots()
    
    # Plot fused May polygons (filled)
    may_overlap_subset = find_largest_overlap_subset(may_polygon_list)
    for polygon in may_overlap_subset:
        coords = list(polygon.exterior.coords)
        ordered_corners = reorder_corners(coords)   
        ordered_corners = fix_invalid_corners(ordered_corners)

        # For ax
        patch = MplPolygon(ordered_corners, fill=True, facecolor='blue', alpha=0.25)
        ax.add_patch(patch)

    # Plot fused Nov polygons (filled)
    if "Right" in nov_img_path:
        poly_color = "orange"
    else:
        poly_color = "red"

    nov_overlap_subset = find_largest_overlap_subset(nov_polygon_list)
    for polygon in nov_overlap_subset:
        coords = list(polygon.exterior.coords)
        ordered_corners = reorder_corners(coords)
        ordered_corners = fix_invalid_corners(ordered_corners)
        # For ax
        patch = MplPolygon(ordered_corners, fill=True, facecolor=poly_color, alpha=0.25)
        ax.add_patch(patch)

    # Create proxy artists
    may_fused = mpatches.Patch(facecolor='blue', alpha=0.25, label='May Fused Area')
    nov_fused = mpatches.Patch(facecolor=poly_color, alpha=0.25, label='Nov Fused Area')

    # Add legend
    ax.legend(
        handles=[may_fused, nov_fused],
        loc='center left',
        bbox_to_anchor=(-0.35, 1.2),
        fontsize=8
        )

    if may_polygon_list:
        may_merged_polygon = unary_union(may_polygon_list)
    else:
        raise OSError 

    if nov_polygon_list:
        nov_merged_polygon = unary_union(nov_polygon_list)
    else:
        raise OSError 

    # Find the bounding box of both fused polygons
    may_bounds = may_merged_polygon.bounds  # (minx, miny, maxx, maxy)
    nov_bounds = nov_merged_polygon.bounds  # (minx, miny, maxx, maxy)

    # Merge the bounds
    min_lon = min(may_bounds[0], nov_bounds[0])
    min_lat = min(may_bounds[1], nov_bounds[1])
    max_lon = max(may_bounds[2], nov_bounds[2])
    max_lat = max(may_bounds[3], nov_bounds[3])

    # Set axis limits
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)

    aspect_correction = np.cos(np.deg2rad((min_lat+max_lat)/2))
    ax.set_aspect(aspect_correction)

    # Set aspect ratio to equal (important for GPS maps!)
    ax.set_aspect('equal', adjustable='box')

    print(f"Writing {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    fig.clf()
    plt.close(fig)

    return fig 