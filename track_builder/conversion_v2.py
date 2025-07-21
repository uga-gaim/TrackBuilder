import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors

def points_to_smooth_lines_fast(df, return_geo=False, method='vectorized'):
    """
    Optimized version with multiple algorithm choices
    
    Methods:
    - 'vectorized': Fast vectorized nearest neighbor (recommended)
    - 'sklearn': Uses sklearn's optimized nearest neighbors
    - 'precomputed': Pre-computes distance matrix (good for small groups)
    """
    all_columns = df.columns

    # Choose the reordering function based on method
    if method == 'vectorized':
        reorder_func = nearest_neighbor_vectorized
    elif method == 'sklearn':
        reorder_func = nearest_neighbor_sklearn
    elif method == 'precomputed':
        reorder_func = nearest_neighbor_precomputed
    else:
        raise ValueError("Method must be 'vectorized', 'sklearn', or 'precomputed'")

    reordered = (
        df
            .groupby('mmsi')
            .filter(lambda x: len(x) > 1)
            .reset_index(drop=True)
            .groupby('mmsi')[all_columns]
            .apply(reorder_func)
            .reset_index(drop=True)
    )

    if not return_geo:
        return reordered

    # Rest of your original code...
    reordered = gpd.GeoDataFrame(
        data=reordered,
        geometry=gpd.points_from_xy(
            x=reordered.cell_ll_lon,
            y=reordered.cell_ll_lat),
        crs='EPSG:4326'
    )

    tracks = reordered.groupby('mmsi').agg({
        'date': ['min', 'max'],
        'hours': 'sum',
        'fishing_hours': ['sum', 'count'],
        'geometry': lambda x: LineString(x),
    })

    tracks.columns = ['start_date', 'end_date', 'hours', 'fishing_hours', 'n_pixels', 'geometry']
    return gpd.GeoDataFrame(tracks, geometry=tracks.geometry)

def nearest_neighbor_vectorized(df, col_x='cell_ll_lon', col_y='cell_ll_lat'):
    """
    Fastest method: Vectorized nearest neighbor using numpy broadcasting
    """
    coords = df[[col_x, col_y]].values
    n = len(coords)
    
    if n <= 1:
        return df
    
    # Pre-allocate arrays
    path = np.zeros(n, dtype=int)
    unvisited = np.ones(n, dtype=bool)
    
    # Start from first point
    current = 0
    path[0] = current
    unvisited[current] = False
    
    for i in range(1, n):
        # Vectorized distance calculation to all unvisited points
        current_coord = coords[current]
        unvisited_coords = coords[unvisited]
        
        # Calculate squared distances (avoid sqrt for speed)
        distances_sq = np.sum((unvisited_coords - current_coord)**2, axis=1)
        
        # Find nearest unvisited point
        nearest_local_idx = np.argmin(distances_sq)
        nearest_global_idx = np.where(unvisited)[0][nearest_local_idx]
        
        path[i] = nearest_global_idx
        current = nearest_global_idx
        unvisited[current] = False
    
    return df.iloc[path]

def nearest_neighbor_sklearn(df, col_x='cell_ll_lon', col_y='cell_ll_lat'):
    """
    Use sklearn's optimized nearest neighbor search
    """
    coords = df[[col_x, col_y]].values
    n = len(coords)
    
    if n <= 1:
        return df
    
    # Build KDTree for fast nearest neighbor queries
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='kd_tree').fit(coords)
    
    path = [0]
    unvisited = set(range(1, n))
    
    while unvisited:
        current = path[-1]
        # Get all distances from current point
        distances, indices = nbrs.kneighbors([coords[current]], n_neighbors=n)
        
        # Find nearest unvisited point
        for idx in indices[0]:
            if idx in unvisited:
                path.append(idx)
                unvisited.remove(idx)
                break
    
    return df.iloc[path]

def nearest_neighbor_precomputed(df, col_x='cell_ll_lon', col_y='cell_ll_lat'):
    """
    Pre-compute distance matrix (best for small groups < 1000 points)
    """
    coords = df[[col_x, col_y]].values
    n = len(coords)
    
    if n <= 1:
        return df
    
    # Pre-compute distance matrix
    dist_matrix = distance_matrix(coords, coords)
    
    path = [0]
    unvisited = set(range(1, n))
    
    while unvisited:
        current = path[-1]
        # Find nearest unvisited point using pre-computed distances
        min_dist = float('inf')
        nearest_idx = None
        
        for idx in unvisited:
            if dist_matrix[current, idx] < min_dist:
                min_dist = dist_matrix[current, idx]
                nearest_idx = idx
        
        path.append(nearest_idx)
        unvisited.remove(nearest_idx)
    
    return df.iloc[path]

# Alternative: Simple chronological ordering (often works well for GPS tracks)
def chronological_reorder(df, time_col='date'):
    """
    Simple alternative: order by timestamp (often better than nearest neighbor for GPS tracks)
    """
    return df.sort_values(time_col).reset_index(drop=True)

def points_to_smooth_lines_chrono(df, return_geo=False, time_col='date'):
    """
    Alternative function using chronological ordering instead of spatial
    """
    all_columns = df.columns

    reordered = (
        df
            .groupby('mmsi')
            .filter(lambda x: len(x) > 1)
            .reset_index(drop=True)
            .groupby('mmsi')[all_columns]
            .apply(lambda x: chronological_reorder(x, time_col))
            .reset_index(drop=True)
    )

    if not return_geo:
        return reordered

    reordered = gpd.GeoDataFrame(
        data=reordered,
        geometry=gpd.points_from_xy(
            x=reordered.cell_ll_lon,
            y=reordered.cell_ll_lat),
        crs='EPSG:4326'
    )

    tracks = reordered.groupby('mmsi').agg({
        'date': ['min', 'max'],
        'hours': 'sum',
        'fishing_hours': ['sum', 'count'],
        'geometry': lambda x: LineString(x),
    })

    tracks.columns = ['start_date', 'end_date', 'hours', 'fishing_hours', 'n_pixels', 'geometry']
    return gpd.GeoDataFrame(tracks, geometry=tracks.geometry)

# Usage examples:
# result = points_to_smooth_lines_fast(df, method='vectorized')  # Fastest
# result = points_to_smooth_lines_fast(df, method='sklearn')     # Good for large groups
# result = points_to_smooth_lines_fast(df, method='precomputed') # Good for small groups
# result = points_to_smooth_lines_chrono(df)                     # Often better alternative