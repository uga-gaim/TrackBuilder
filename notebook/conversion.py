import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd

from pyproj import CRS
from shapely.geometry import LineString
from scipy.spatial import distance_matrix


def points_to_smooth_lines(df, return_geo=False):
    all_columns = df.columns

    # Use the TS algorithm to reorder rows for each MMSI
    reordered = (
        df
            .groupby('mmsi')
            .filter(lambda x: len(x) > 1)
            .reset_index(drop=True)
            .groupby('mmsi')[all_columns]
            .apply(traveling_salesman_reorder)
            .reset_index(drop=True)
    )

    if not return_geo:
        return df

    # Convert to points GDF
    reordered = gpd.GeoDataFrame(
        data=reordered,
        geometry=gpd.points_from_xy(
            x=reordered.cell_ll_lon,
            y=reordered.cell_ll_lat),
        crs='EPSG:4326'
    )

    # Merge points to polylines
    tracks = reordered.groupby('mmsi').agg({
        'date': ['min', 'max'],
        'hours': 'sum',
        'fishing_hours': ['sum', 'count'],
        'geometry': lambda x: LineString(x),
    })

    # Housekeeping
    tracks.columns = ['start_date', 'end_date', 'hours', 'fishing_hours', 'n_pixels', 'geometry']
    return gpd.GeoDataFrame(tracks, geometry=tracks.geometry)


def traveling_salesman_reorder(df, col_id='mmsi', col_x='cell_ll_lon', col_y='cell_ll_lat', project_wgs84_coords=True):

    assert col_id in df, f'Column {col_id} does not exist'
    assert col_x in df, f'Column {col_x} does not exist'
    assert col_y in df, f'Column {col_y} does not exist'
    assert df[col_id].nunique() == 1, 'Can only solve for one track ID at a time!'

    if project_wgs84_coords:

        # Create a local CRS for better coordinate representation
        local_crs = CRS(
            proj='aeqd',
            lat_0=df[col_y].mean(),
            lon_0=df[col_x].mean(),
            datum='WGS84',
            units='m'
        )
        
        coords = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(df[col_x], df[col_y]),
            crs='EPSG:4326',
        ).to_crs(local_crs).get_coordinates().values

    else:
        coords = df[[col_x, col_y]].values
    
    # Compute distance matrix
    dist_matrix = distance_matrix(coords, coords)

    # Create a fully connected graph
    G = nx.complete_graph(len(df))
    for i in range(len(df)):
        for j in range(len(df)):
            if i != j:
                G[i][j]['weight'] = dist_matrix[i, j]

    # Solve TSP with approximation
    tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False)
    
    # Reorder the original dataframe by TSP path
    return df.iloc[tsp_path]
