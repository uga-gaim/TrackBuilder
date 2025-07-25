"""
Ship Tracking Data Matching Module

This module provides functionality for matching ship tracking data between different datasets,
specifically ASTD (ship tracking) data and GFW (Global Fishing Watch) data. It uses spatial
and temporal buffers to identify vessels that appear in both datasets.

The main workflow is:
1. Sample points from ASTD track data at regular intervals
2. Create spatial buffers around each sampled point
3. Find GFW vessels (MMSIs) that pass through ALL buffers on matching dates
4. Return candidate MMSIs that likely represent the same vessel

Key concepts:
- ASTD: Ship tracking dataset with precise position data
- GFW: Global Fishing Watch dataset with vessel MMSI identifiers
- Buffer: Spatial area around a point used for matching tolerance
- MMSI: Maritime Mobile Service Identity - unique vessel identifier
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def identify_matching(df_astd_one_track: pd.DataFrame, df_gfw: pd.DataFrame, 
                     skip_every_n_astd: int = 45, astd_buffer_radius: float = 0.2) -> list:
    """
    Identify GFW vessels (MMSIs) that match an ASTD track using spatial and temporal filtering.
    
    This function samples points from an ASTD track at regular intervals, creates spatial
    buffers around each point, and finds GFW vessels that pass through ALL buffers on
    matching dates. This intersection approach helps identify vessels that follow the
    same route with high confidence.
    
    Parameters:
    - df_astd_one_track (pd.DataFrame): ASTD tracking data for a single vessel track
                                       Required columns: ['date_time_utc', 'latitude', 'longitude']
    - df_gfw (pd.DataFrame): GFW dataset with vessel positions
                            Required columns: ['mmsi', 'date', 'cell_ll_lat', 'cell_ll_lon']
    - skip_every_n_astd (int): Sample every Nth ASTD point for efficiency (default: 45)
                              Higher values = fewer sample points = faster but less precise
    - astd_buffer_radius (float): Buffer radius around each ASTD point in decimal degrees (default: 0.2)
                                 Larger values = more tolerant matching but more false positives
    
    Returns:
    - list: List of MMSI numbers (strings/ints) that pass through all buffers
           Empty list if no matches found
    
    Notes:
    - Uses intersection logic: vessels must pass through ALL sampled buffers
    - Matches require both spatial overlap AND temporal overlap (same date)
    - Buffer radius is in decimal degrees (approximate, not precise distance)
    """
    
    # Convert date columns to datetime objects
    df_astd_copy = df_astd_one_track.copy()
    df_gfw_copy = df_gfw.copy()
    
    print(f"ASTD rows: {len(df_astd_copy)}, GFW rows: {len(df_gfw_copy)}")
    
    # Convert ASTD date_time_utc to datetime
    df_astd_copy['date_time_utc'] = pd.to_datetime(df_astd_copy['date_time_utc'])
    
    # Convert GFW date to datetime
    df_gfw_copy['date'] = pd.to_datetime(df_gfw_copy['date'])
    
    # Initialize variables for sampling
    row_count = 0
    sampled_rows = []
    
    # Sort the ASTD data by time to ensure chronological sampling
    df_astd_sorted = df_astd_copy.sort_values('date_time_utc')
    
    # Sample every nth row to reduce computational load while maintaining track coverage
    for idx, row in df_astd_sorted.iterrows():
        row_count += 1
        if row_count == skip_every_n_astd:
            sampled_rows.append(row)
            row_count = 0  # Reset counter
    
    # Convert to DataFrame for easier handling
    df_sampled = pd.DataFrame(sampled_rows)
    
    if df_sampled.empty:
        print("Warning: No ASTD points were sampled")
        return []
    
    print(f"Sampled {len(df_sampled)} ASTD points to check")
    
    # Create simple degree-based buffers for sampled rows
    # Note: This is approximate - for precise distance buffers, use create_buffer_around_point()
    buffer_results = []
    for _, row in df_sampled.iterrows():
        buffer_info = {
            'min_lat': row['latitude'] - astd_buffer_radius,
            'max_lat': row['latitude'] + astd_buffer_radius,
            'min_lon': row['longitude'] - astd_buffer_radius,
            'max_lon': row['longitude'] + astd_buffer_radius,
            'date': row['date_time_utc'].date()  # Extract date for temporal matching
        }
        buffer_results.append(buffer_info)
    
    # Add buffer info to sampled DataFrame for reference
    df_sampled = df_sampled.copy()
    df_sampled['buffer_min_lat'] = [b['min_lat'] for b in buffer_results]
    df_sampled['buffer_max_lat'] = [b['max_lat'] for b in buffer_results]
    df_sampled['buffer_min_lon'] = [b['min_lon'] for b in buffer_results]
    df_sampled['buffer_max_lon'] = [b['max_lon'] for b in buffer_results]
    df_sampled['buffer_date'] = [b['date'] for b in buffer_results]
    
    # Find MMSIs that pass through ALL buffers with matching dates
    # Start with all unique MMSIs as potential candidates
    mmsi_candidates = set(df_gfw_copy['mmsi'].unique())
    print(f"Starting with {len(mmsi_candidates)} unique MMSIs")
    
    # Progressive filtering: for each buffer, keep only MMSIs that pass through it
    # This intersection approach ensures vessels must follow the entire sampled route
    for i, (_, astd_row) in enumerate(df_sampled.iterrows()):
        print(f"Processing buffer {i+1}/{len(df_sampled)}")
        
        # Convert ASTD datetime to date for comparison with GFW data
        astd_date = astd_row['date_time_utc'].date()
        
        # Find GFW points in this buffer with matching date
        # Spatial filter: within buffer bounds
        # Temporal filter: same date
        mmsi_in_this_buffer = df_gfw_copy[
            (df_gfw_copy['cell_ll_lat'] >= astd_row['buffer_min_lat']) &
            (df_gfw_copy['cell_ll_lat'] <= astd_row['buffer_max_lat']) &
            (df_gfw_copy['cell_ll_lon'] >= astd_row['buffer_min_lon']) &
            (df_gfw_copy['cell_ll_lon'] <= astd_row['buffer_max_lon']) &
            (df_gfw_copy['date'].dt.date == astd_date)
        ]['mmsi'].unique()
        
        print(f"  Found {len(mmsi_in_this_buffer)} MMSIs in this buffer")
        
        # Store previous candidates to show elimination progress
        previous_candidates = mmsi_candidates.copy()
        
        # Intersection: only keep MMSIs that are in this buffer AND were in previous buffers
        mmsi_candidates = mmsi_candidates.intersection(set(mmsi_in_this_buffer))
        print(f"  {len(mmsi_candidates)} MMSIs remain after intersection")
        
        # Show which MMSIs were eliminated this round (for debugging)
        eliminated = previous_candidates - mmsi_candidates
        if eliminated:
            print(f"  Eliminated MMSIs: {list(eliminated)}")
        
        # Early termination if no candidates remain
        if not mmsi_candidates:
            print("  No candidates remain, stopping early")
            print(f"  Last MMSIs before elimination were: {list(previous_candidates)}")
            break
        else:
            print(f"  Current remaining MMSIs: {list(mmsi_candidates)}")
    
    # Print final result
    print(f"Final MMSIs found: {list(mmsi_candidates)}")
    return list(mmsi_candidates)


def create_buffer_around_point(lat: float, lon: float, buffer_distance_meters: int) -> dict:
    """
    Create a proper circular buffer around a point using accurate distance calculations.
    
    This function creates a precise circular buffer around a geographic point by:
    1. Converting coordinates to a projected coordinate system
    2. Creating a buffer in meters
    3. Converting back to lat/lon for compatibility
    
    More accurate than simple degree-based buffers, especially at different latitudes
    where degree distances vary significantly.
    
    Parameters:
    - lat (float): Latitude of center point in decimal degrees
    - lon (float): Longitude of center point in decimal degrees  
    - buffer_distance_meters (int): Buffer radius in meters
    
    Returns:
    - dict: Dictionary containing buffer bounds and geometry:
           {
               'min_lat': float,    # Southern boundary
               'max_lat': float,    # Northern boundary  
               'min_lon': float,    # Western boundary
               'max_lon': float,    # Eastern boundary
               'geometry': Polygon  # Shapely geometry object
           }
    
    Notes:
    - Uses Web Mercator projection (EPSG:3857) for distance calculations
    - Input coordinates assumed to be in WGS84 (EPSG:4326)
    - More computationally expensive than degree-based buffers but more accurate
    """
    # Create point geometry - input data is in WGS84 (EPSG:4326)
    point = Point(lon, lat)
    gdf = gpd.GeoDataFrame([1], geometry=[point], crs='EPSG:4326')
    
    # Convert to a projected CRS for accurate distance calculations
    # Using Web Mercator (EPSG:3857) - good for global datasets
    # Note: For regional analysis, consider using a local UTM zone for higher accuracy
    gdf_proj = gdf.to_crs('EPSG:3857')
    
    # Create buffer (in meters) around the projected point
    buffered = gdf_proj.buffer(buffer_distance_meters)
    
    # Convert back to lat/lon (WGS84) for compatibility with input datasets
    buffered_latlon = gpd.GeoDataFrame(geometry=buffered).to_crs('EPSG:4326')
    
    # Get the bounding box of the buffer for easy spatial filtering
    bounds = buffered_latlon.bounds.iloc[0]
    
    return {
        'min_lat': bounds['miny'],
        'max_lat': bounds['maxy'],
        'min_lon': bounds['minx'],
        'max_lon': bounds['maxx'],
        'geometry': buffered_latlon.geometry.iloc[0]
    }

