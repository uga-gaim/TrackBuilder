import pandas as pd
import main
import geopandas as gpd
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
import numpy as np
from conversion import points_to_smooth_lines
from conversion_v2 import points_to_smooth_lines_fast
import plotly.graph_objects as go
from shapely.geometry import Point


def identify_matching(df_astd_one_track, df_gfw, skip_every_n_astd=45, astd_buffer_radius=0.2):
    
    # Convert date columns to datetime objects
    df_astd_copy = df_astd_one_track.copy()
    df_gfw_copy = df_gfw.copy()
    
    print(f"ASTD rows: {len(df_astd_copy)}, GFW rows: {len(df_gfw_copy)}")
    
    # Convert ASTD date_time_utc to datetime
    df_astd_copy['date_time_utc'] = pd.to_datetime(df_astd_copy['date_time_utc'])
    
    # Convert GFW date to datetime
    df_gfw_copy['date'] = pd.to_datetime(df_gfw_copy['date'])
    
    # Initialize variables
    row_count = 0
    sampled_rows = []
    
    # Sort the astd df by time
    df_astd_sorted = df_astd_copy.sort_values('date_time_utc')
    
    # Get every nth row
    for idx, row in df_astd_sorted.iterrows():
        row_count += 1
        if row_count == skip_every_n_astd:
            sampled_rows.append(row)
            row_count = 0
    
    # Convert to DataFrame for easier handling
    df_sampled = pd.DataFrame(sampled_rows)
    
    if df_sampled.empty:
        return []
    
    print(f"Sampled {len(df_sampled)} ASTD points to check")
    
    # Create simple degree-based buffers for sampled rows
    buffer_results = []
    for _, row in df_sampled.iterrows():
        buffer_info = {
            'min_lat': row['latitude'] - astd_buffer_radius,
            'max_lat': row['latitude'] + astd_buffer_radius,
            'min_lon': row['longitude'] - astd_buffer_radius,
            'max_lon': row['longitude'] + astd_buffer_radius,
            'date': row['date_time_utc'].date()
        }
        buffer_results.append(buffer_info)
    
    # Add buffer info to sampled DataFrame
    df_sampled = df_sampled.copy()
    df_sampled['buffer_min_lat'] = [b['min_lat'] for b in buffer_results]
    df_sampled['buffer_max_lat'] = [b['max_lat'] for b in buffer_results]
    df_sampled['buffer_min_lon'] = [b['min_lon'] for b in buffer_results]
    df_sampled['buffer_max_lon'] = [b['max_lon'] for b in buffer_results]
    df_sampled['buffer_date'] = [b['date'] for b in buffer_results]
    
    # Find MMSIs that pass through ALL buffers with matching dates
    mmsi_candidates = set(df_gfw_copy['mmsi'].unique())
    print(f"Starting with {len(mmsi_candidates)} unique MMSIs")
    
    # For each buffer, filter MMSIs that pass through it AND have matching date
    for i, (_, astd_row) in enumerate(df_sampled.iterrows()):
        print(f"Processing buffer {i+1}/{len(df_sampled)}")
        
        # Convert astd datetime to date for comparison
        astd_date = astd_row['date_time_utc'].date()
        
        # Find GFW points in this buffer with matching date
        mmsi_in_this_buffer = df_gfw_copy[
            (df_gfw_copy['cell_ll_lat'] >= astd_row['buffer_min_lat']) &
            (df_gfw_copy['cell_ll_lat'] <= astd_row['buffer_max_lat']) &
            (df_gfw_copy['cell_ll_lon'] >= astd_row['buffer_min_lon']) &
            (df_gfw_copy['cell_ll_lon'] <= astd_row['buffer_max_lon']) &
            (df_gfw_copy['date'].dt.date == astd_date)
        ]['mmsi'].unique()
        
        print(f"  Found {len(mmsi_in_this_buffer)} MMSIs in this buffer")
        
        # Store previous candidates to show what's being eliminated
        previous_candidates = mmsi_candidates.copy()
        
        # Only keep MMSIs that are in this buffer AND were in previous buffers
        mmsi_candidates = mmsi_candidates.intersection(set(mmsi_in_this_buffer))
        print(f"  {len(mmsi_candidates)} MMSIs remain after intersection")
        
        # Show which MMSIs were eliminated this round
        eliminated = previous_candidates - mmsi_candidates
        if eliminated:
            print(f"  Eliminated MMSIs: {list(eliminated)}")
        
        # If no candidates remain, show the last ones before breaking
        if not mmsi_candidates:
            print("  No candidates remain, stopping early")
            print(f"  Last MMSIs before elimination were: {list(previous_candidates)}")
            break
        else:
            print(f"  Current remaining MMSIs: {list(mmsi_candidates)}")
    
    # Print final result
    print(f"Final MMSIs found: {list(mmsi_candidates)}")
    return list(mmsi_candidates)


def create_buffer_around_point(lat, lon, buffer_distance_meters):
    """
    Create a proper circular buffer around a point
    lat, lon: coordinates from your ASTD/GFW data
    buffer_distance_meters: buffer radius in meters
    """
    # Create point geometry - your data is in EPSG:4326
    point = Point(lon, lat)
    gdf = gpd.GeoDataFrame([1], geometry=[point], crs='EPSG:4326')
    
    # Convert to a projected CRS for accurate distance calculations
    # Using Web Mercator (EPSG:3857) - good for global datasets
    gdf_proj = gdf.to_crs('EPSG:3857')
    
    # Create buffer (in meters)
    buffered = gdf_proj.buffer(buffer_distance_meters)
    
    # Convert back to lat/lon for compatibility with your datasets
    buffered_latlon = gpd.GeoDataFrame(geometry=buffered).to_crs('EPSG:4326')
    
    # Get the bounds of the buffer
    bounds = buffered_latlon.bounds.iloc[0]
    
    return {
        'min_lat': bounds['miny'],
        'max_lat': bounds['maxy'],
        'min_lon': bounds['minx'],
        'max_lon': bounds['maxx'],
        'geometry': buffered_latlon.geometry.iloc[0]
    }


