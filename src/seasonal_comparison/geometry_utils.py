from shapely.strtree import STRtree
import networkx as nx
from image_stitching_utils import create_polygon 

def cluster_overlapping_polygons(polygons, min_overlap_area=0.0):
    valid_polygons = [p for p in polygons if p.is_valid and not p.is_empty]
    if not valid_polygons:
        return []

    tree = STRtree(valid_polygons)
    polygon_to_index = {p: i for i, p in enumerate(valid_polygons)}  # use polygon object itself

    #print("Known keys:", list(polygon_to_index.keys()))

    G = nx.Graph()
    for i, poly in enumerate(valid_polygons):
        G.add_node(i)
        for j in tree.query(poly):
            candidate = valid_polygons[j]
            #print("candidate:", candidate)
            if candidate == poly:
                continue  # skip self
            if poly.intersects(candidate):
                if min_overlap_area > 0 and poly.intersection(candidate).area < min_overlap_area:
                    continue
                j = polygon_to_index.get(candidate)
                if j is not None:
                    G.add_edge(i, j)

    return [[valid_polygons[i] for i in component] for component in nx.connected_components(G)]

def cluster_overlapping_frame_instances(frame_instance, min_overlap_area=1e-16):
    valid_polygons = [p.polygon_obj for p in frame_instance if p.polygon_obj.is_valid and not p.polygon_obj.is_empty]
    if not valid_polygons:
        return []

    tree = STRtree(valid_polygons)
    polygon_to_index = {p: i for i, p in enumerate(valid_polygons)}  # use polygon object itself

    G = nx.Graph()
    for i, poly in enumerate(valid_polygons):
        G.add_node(i)
        for j in tree.query(poly):
            candidate = valid_polygons[j]
            #print("candidate:", candidate)
            if candidate == poly:
                continue  # skip self
            if poly.intersects(candidate):
                if min_overlap_area > 0 and poly.intersection(candidate).area < min_overlap_area:
                    continue
                j = polygon_to_index.get(candidate)
                if j is not None:
                    G.add_edge(i, j)
    return [[frame_instance[i].frame_path for i in component] for component in nx.connected_components(G)] 

from shapely.ops import unary_union

def find_intersecting_clusters(clustersA, clustersB):
    """
    Finds all pairs of clusters (from A and B) that intersect.
    
    Returns:
        A list of tuples: (i, j) where clustersA[i] intersects with clustersB[j]
    """

    if isinstance(clustersA[0],list):
        print("uh oh ... trying to recover in find_intersecting_clusters")
        clustersA = [create_polygon(x) for x in clustersA]
    if isinstance(clustersB[0],list):
        print("uh oh ... trying to recover in find_intersecting_clusters")
        clustersB = [create_polygon(x) for x in clustersB] 

    unionA = [unary_union(cluster) for cluster in clustersA]
    unionB = [unary_union(cluster) for cluster in clustersB]
    
    intersecting_pairs = []
    for i, ua in enumerate(unionA):
        for j, ub in enumerate(unionB):
            if ua.intersects(ub):
                intersecting_pairs.append((i, j))
    return intersecting_pairs
 