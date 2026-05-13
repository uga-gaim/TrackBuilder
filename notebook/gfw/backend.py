import os
import warnings


import pandas as pd
import numpy as np

import track_builder as tb
import geopandas as gpd
from shapely import box

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

import matplotlib.animation as animation
import imageio.v2 as imageio

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature

from tqdm.notebook import tqdm, trange
from IPython.display import Image, display


## Constant
parquet_path = "../../examples/data/"


## General
def select_subset(df, sample_per_day=50):
    # get gfw samples
    # work = (
    #     merged
    #     .groupby(['mmsi', 'date'], group_keys=False)
    #     .sample(10, random_state=42)
    # )

    work = (
        df
        .sort_values(['mmsi', 'date'])  # or include timestamp if you have one
        .groupby(['mmsi', 'date'], group_keys=False)
        .apply(lambda g: g if len(g) <= sample_per_day else g.iloc[np.linspace(0, len(g)-1, sample_per_day, dtype=int)])
    )

    work.sort_index(inplace=True)

    return work


## Functions ASTD
def load_data(parquet_file, source, year, **kwargs):
    parquet_file = parquet_path + parquet_file
    if os.path.exists(parquet_file):
        data =  pd.read_parquet(parquet_file)
        print(f"Loaded data {parquet_file}, parameters ignored")
    else:
        data = tb.load_astd_monthly(base_path=source, year=year, **kwargs)
        data.to_parquet(parquet_file)
        print(f"Loaded data {parquet_file} from {source}")

    return data

def load_periods(parquet_file, source, periods, **kwargs):
    parquet_file = parquet_path + parquet_file
    if os.path.exists(parquet_file):
        period =  pd.read_parquet(parquet_file)
        print(f"Loaded period {parquet_file}, parameters ignored")
    else:
        period = tb.load_astd_periods(base_path=source, periods=periods, **kwargs)
        period.to_parquet(parquet_file)
        print(f"Loaded period {parquet_file} from {source}")

    return period

def load_tracks(parquet_file, data, **kwargs):
    parquetLOG_file = parquet_path + "LOG" + parquet_file
    parquet_file = parquet_path + parquet_file
    if os.path.exists(parquet_file):
        tracks =  pd.read_parquet(parquet_file)
        print(f"Loaded tracks {parquet_file}, parameters ignored")
        if kwargs.get("return_logs", False) and os.path.exists(parquetLOG_file):
            print(f"Loaded tracks logs")
            track_logs =  pd.read_parquet(parquetLOG_file)
            return tracks, track_logs
        else:
            print(f"No logs found {parquetLOG_file}")
    else:
        data['month'] = data['month'].astype(str)
        tracks = tb.build_ship_tracks(data, **kwargs)
        print(f"Loaded tracks {parquet_file} from {data}")
        if kwargs.get("return_logs", False):
            print(f"Loaded tracks logs")
            tracks[0].to_parquet(parquet_file)
            tracks[1].to_parquet(parquetLOG_file)
    return tracks


## Function GFW
def load_data_gfw(parquet_file, source, year, months=None):
    parquet_file = parquet_path + parquet_file
    if os.path.exists(parquet_file):
        data =  pd.read_parquet(parquet_file)
        print(f"Loaded data {parquet_file}, parameters ignored")
    else:
        dfs = []
        for file in os.listdir(source):
            if file.endswith(".csv") and str(year) in file:

                month = int(file.split("-")[6])

                if months is None or month in months:
                    print("Working with :", file)
                    path = os.path.join(source, file)
                    df = pd.read_csv(path)
                    dfs.append(df)

        if not dfs:
            raise ValueError("No matching CSV files found")

        data = pd.concat(dfs, ignore_index=True)
        data.to_parquet(parquet_file)
        print(f"Loaded data {parquet_file} from {source}")

    return data

def load_periods_gfw(parquet_file, source, periods):
    parquet_file = parquet_path + parquet_file

    if os.path.exists(parquet_file):
        data = pd.read_parquet(parquet_file)
        print(f"Loaded data {parquet_file}, parameters ignored")
    else:
        dfs = []
        for group_year in os.listdir(source):
            parts = group_year.split("-")
            if parts[0] != "mmsi":
                continue

            file_year = int(parts[-1])

            if file_year in periods:
                print("Working with :", group_year)
                for file in os.listdir(source + group_year):
                    if file.endswith(".csv"):
                        month = int(file.split("-")[6])

                        if periods[file_year] is None or month in periods[file_year]:
                            path = os.path.join(source + group_year, file)
                            df = pd.read_csv(path)
                            dfs.append(df)

        if not dfs:
            raise ValueError("No matching CSV files found")

        data = pd.concat(dfs, ignore_index=True)
        data.to_parquet(parquet_file)
        print(f"Saved data to {parquet_file}")

    return data
