"""
GPS Track Optimization Module

This module provides optimized algorithms for converting scattered GPS point data into
smooth, continuous vessel tracks. It offers multiple approaches for reordering GPS points
to create realistic vessel trajectories, with performance optimizations for different
dataset sizes and characteristics.

The main workflow is:
1. Group GPS points by vessel identifier (MMSI)
2. Reorder points within each group to create logical trajectories
3. Optionally convert to geographic LineString objects for spatial analysis

Key concepts:
- Point reordering: Arranging GPS points to minimize jumps and create smooth tracks
- Nearest neighbor algorithms: Different approaches with speed/accuracy trade-offs
- Chronological ordering: Time-based alternative that often works well for GPS data
- GeoPandas integration: Converting point data to spatial line geometries

Available algorithms:
- Vectorized nearest neighbor (fastest, recommended for most cases)
- Sklearn nearest neighbor (good balance of speed and flexibility)
- Precomputed distance matrix (best for small groups < 1000 points)
- Chronological ordering (simple time-based alternative)
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors


def points_to_smooth_lines_fast(df: pd.DataFrame, return_geo: bool = False, method: str = 'vectorized') -> pd.DataFrame:
    """
    Optimized version with multiple algorithm choices for creating smooth vessel tracks.
    
    This is the main function for converting scattered GPS points into ordered tracks.
    It groups points by vessel (MMSI) and applies the specified reordering algorithm
    to create logical trajectories that minimize spatial jumps between consecutive points.
    
    Parameters:
    - df (pd.DataFrame): GPS tracking data with required columns:
                        ['mmsi', 'cell_ll_lon', 'cell_ll_lat', 'date', 'hours', 'fishing_hours']
    - return_geo (bool): If True, returns GeoPandas DataFrame with LineString geometries
                        If False, returns regular DataFrame with reordered points (default: False)
    - method (str): Algorithm to use for point reordering:
                   'vectorized' - Fast vectorized nearest neighbor (recommended)
                   'sklearn' - Uses sklearn's optimized nearest neighbors  
                   'precomputed' - Pre-computes distance matrix (good for small groups)
    
    Returns:
    - pd.DataFrame or gpd.GeoDataFrame: 
        If return_geo=False: Reordered point data with same columns as input
        If return_geo=True: GeoDataFrame with track summaries and LineString geometries
                           Columns: ['start_date', 'end_date', 'hours', 'fishing_hours', 'n_pixels', 'geometry']
    
    Raises:
    - ValueError: If method is not one of the supported options
    
    Notes:
    - Filters out vessels with only 1 point (can't form tracks)
    - Performance varies by method and dataset size
    - Vectorized method recommended for most use cases
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

    # Apply reordering algorithm to each vessel group
    reordered = (
        df
            .groupby('mmsi')
            .filter(lambda x: len(x) > 1)  # Remove single-point vessels
            .reset_index(drop=True)
            .groupby('mmsi')[all_columns]
            .apply(reorder_func)  # Apply chosen reordering algorithm
            .reset_index(drop=True)
    )

    if not return_geo:
        return reordered

    # Convert to GeoPandas and create track summaries
    reordered = gpd.GeoDataFrame(
        data=reordered,
        geometry=gpd.points_from_xy(
            x=reordered.cell_ll_lon,
            y=reordered.cell_ll_lat),
        crs='EPSG:4326'
    )

    # Aggregate points into track summaries with LineString geometries
    tracks = reordered.groupby('mmsi').agg({
        'date': ['min', 'max'],              # Track time span
        'hours': 'sum',                      # Total hours
        'fishing_hours': ['sum', 'count'],   # Fishing activity
        'geometry': lambda x: LineString(x), # Create line from points
    })

    tracks.columns = ['start_date', 'end_date', 'hours', 'fishing_hours', 'n_pixels', 'geometry']
    return gpd.GeoDataFrame(tracks, geometry=tracks.geometry)


def nearest_neighbor_vectorized(df: pd.DataFrame, col_x: str = 'cell_ll_lon', col_y: str = 'cell_ll_lat') -> pd.DataFrame:
    """
    Fastest method: Vectorized nearest neighbor using numpy broadcasting.
    
    Creates a track by starting from the first point and repeatedly finding the
    nearest unvisited point. Uses vectorized numpy operations for maximum speed.
    Recommended for most use cases due to excellent performance.
    
    Parameters:
    - df (pd.DataFrame): GPS points for a single vessel
    - col_x (str): Column name for longitude coordinates (default: 'cell_ll_lon')
    - col_y (str): Column name for latitude coordinates (default: 'cell_ll_lat')
    
    Returns:
    - pd.DataFrame: Input dataframe with rows reordered to form smooth track
    
    Notes:
    - Uses squared distances to avoid expensive sqrt calculations
    - Implements greedy nearest neighbor approach
    - Time complexity: O(n²) but with optimized vectorized operations
    - Memory efficient - no distance matrix storage
    """
    coords = df[[col_x, col_y]].values
    n = len(coords)
    
    if n <= 1:
        return df
    
    # Pre-allocate arrays for performance
    path = np.zeros(n, dtype=int)
    unvisited = np.ones(n, dtype=bool)
    
    # Start from first point (could be optimized to start from centroid)
    current = 0
    path[0] = current
    unvisited[current] = False
    
    # Greedy nearest neighbor algorithm
    for i in range(1, n):
        # Vectorized distance calculation to all unvisited points
        current_coord = coords[current]
        unvisited_coords = coords[unvisited]
        
        # Calculate squared distances (avoid sqrt for speed)
        distances_sq = np.sum((unvisited_coords - current_coord)**2, axis=1)
        
        # Find nearest unvisited point
        nearest_local_idx = np.argmin(distances_sq)
        nearest_global_idx = np.where(unvisited)[0][nearest_local_idx]
        
        # Add to path and mark as visited
        path[i] = nearest_global_idx
        current = nearest_global_idx
        unvisited[current] = False
    
    return df.iloc[path]


def nearest_neighbor_sklearn(df: pd.DataFrame, col_x: str = 'cell_ll_lon', col_y: str = 'cell_ll_lat') -> pd.DataFrame:
    """
    Use sklearn's optimized nearest neighbor search with KDTree.
    
    Leverages sklearn's highly optimized KDTree implementation for nearest neighbor
    queries. Good balance between performance and flexibility, especially for larger
    datasets where tree-based search becomes more efficient.
    
    Parameters:
    - df (pd.DataFrame): GPS points for a single vessel
    - col_x (str): Column name for longitude coordinates (default: 'cell_ll_lon')
    - col_y (str): Column name for latitude coordinates (default: 'cell_ll_lat')
    
    Returns:
    - pd.DataFrame: Input dataframe with rows reordered to form smooth track
    
    Notes:
    - Uses KDTree for O(log n) nearest neighbor queries
    - Overall complexity still O(n²) due to greedy approach
    - More memory efficient than precomputed matrix
    - Good for datasets with complex spatial distributions
    """
    coords = df[[col_x, col_y]].values
    n = len(coords)
    
    if n <= 1:
        return df
    
    # Build KDTree for fast nearest neighbor queries
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='kd_tree').fit(coords)
    
    path = [0]
    unvisited = set(range(1, n))
    
    # Greedy nearest neighbor using KDTree
    while unvisited:
        current = path[-1]
        # Get all distances from current point (sorted by distance)
        distances, indices = nbrs.kneighbors([coords[current]], n_neighbors=n)
        
        # Find nearest unvisited point
        for idx in indices[0]:
            if idx in unvisited:
                path.append(idx)
                unvisited.remove(idx)
                break
    
    return df.iloc[path]


def nearest_neighbor_precomputed(df: pd.DataFrame, col_x: str = 'cell_ll_lon', col_y: str = 'cell_ll_lat') -> pd.DataFrame:
    """
    Pre-compute distance matrix (best for small groups < 1000 points).
    
    Calculates all pairwise distances upfront, then uses this matrix for
    fast nearest neighbor lookups. Most efficient for small datasets where
    the O(n²) memory cost is acceptable.
    
    Parameters:
    - df (pd.DataFrame): GPS points for a single vessel
    - col_x (str): Column name for longitude coordinates (default: 'cell_ll_lon')
    - col_y (str): Column name for latitude coordinates (default: 'cell_ll_lat')
    
    Returns:
    - pd.DataFrame: Input dataframe with rows reordered to form smooth track
    
    Notes:
    - Pre-computes full n×n distance matrix
    - Memory usage: O(n²) - can be prohibitive for large datasets
    - Fastest lookups once matrix is computed
    - Recommended only for small groups (< 1000 points)
    - Uses Euclidean distance in coordinate space
    """
    coords = df[[col_x, col_y]].values
    n = len(coords)
    
    if n <= 1:
        return df
    
    # Pre-compute distance matrix (expensive for large n)
    dist_matrix = distance_matrix(coords, coords)
    
    path = [0]
    unvisited = set(range(1, n))
    
    # Greedy nearest neighbor using precomputed distances
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


def chronological_reorder(df: pd.DataFrame, time_col: str = 'date') -> pd.DataFrame:
    """
    Simple alternative: order by timestamp (often better than nearest neighbor for GPS tracks).
    
    For many GPS tracking scenarios, chronological ordering produces more realistic
    tracks than spatial nearest neighbor approaches. This is because GPS points are
    typically collected in temporal sequence as vessels move.
    
    Parameters:
    - df (pd.DataFrame): GPS points for a single vessel
    - time_col (str): Column name containing timestamp data (default: 'date')
    
    Returns:
    - pd.DataFrame: Input dataframe sorted by timestamp
    
    Notes:
    - Fastest possible approach - O(n log n) due to sorting
    - Often produces more realistic tracks than spatial methods
    - Recommended first approach for most GPS tracking data
    - Assumes temporal sampling roughly follows vessel movement
    - No spatial optimization - purely time-based
    """
    return df.sort_values(time_col).reset_index(drop=True)


def points_to_smooth_lines_chrono(df: pd.DataFrame, return_geo: bool = False, time_col: str = 'date') -> pd.DataFrame:
    """
    Alternative function using chronological ordering instead of spatial optimization.
    
    This version uses simple time-based ordering rather than spatial nearest neighbor
    algorithms. Often produces better results for GPS tracking data where points are
    collected in temporal sequence following actual vessel movement.
    
    Parameters:
    - df (pd.DataFrame): GPS tracking data with required columns:
                        ['mmsi', 'cell_ll_lon', 'cell_ll_lat', 'date', 'hours', 'fishing_hours']
    - return_geo (bool): If True, returns GeoPandas DataFrame with LineString geometries
                        If False, returns regular DataFrame with reordered points (default: False)
    - time_col (str): Column name containing timestamp data (default: 'date')
    
    Returns:
    - pd.DataFrame or gpd.GeoDataFrame:
        If return_geo=False: Chronologically ordered point data
        If return_geo=True: GeoDataFrame with track summaries and LineString geometries
    
    Notes:
    - Much faster than spatial optimization methods
    - Often produces more realistic tracks for GPS data
    - Recommended as first approach before trying spatial methods
    - Assumes GPS points collected in temporal sequence
    """
    all_columns = df.columns

    # Apply chronological reordering to each vessel group
    reordered = (
        df
            .groupby('mmsi')
            .filter(lambda x: len(x) > 1)  # Remove single-point vessels
            .reset_index(drop=True)
            .groupby('mmsi')[all_columns]
            .apply(lambda x: chronological_reorder(x, time_col))
            .reset_index(drop=True)
    )

    if not return_geo:
        return reordered

    # Convert to GeoPandas and create track summaries
    reordered = gpd.GeoDataFrame(
        data=reordered,
        geometry=gpd.points_from_xy(
            x=reordered.cell_ll_lon,
            y=reordered.cell_ll_lat),
        crs='EPSG:4326'
    )

    # Aggregate points into track summaries with LineString geometries
    tracks = reordered.groupby('mmsi').agg({
        'date': ['min', 'max'],              # Track time span
        'hours': 'sum',                      # Total hours
        'fishing_hours': ['sum', 'count'],   # Fishing activity
        'geometry': lambda x: LineString(x), # Create line from points
    })

    tracks.columns = ['start_date', 'end_date', 'hours', 'fishing_hours', 'n_pixels', 'geometry']
    return gpd.GeoDataFrame(tracks, geometry=tracks.geometry)