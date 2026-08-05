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


############
## Constant
############

parquet_path = "../../examples/data/"

############
## General
############

def select_subset(df, sample_per_day=50):
    """
    Function to select a specific subset from the dataframe
    > For each mmsi, we get x sample per day for everyday
    """
    ## Get random samples
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

def filter_latlon(df, lat=None, lon=None, max_point_allowed=0, gfw=True, lonlat=['grid_lon', 'grid_lat']):
    """
    Filter remove all ships with points in trajectory going under - lat or lon
    """
    if gfw:
        if lat is not None:
            mask = df[lonlat[1]] < lat * 10
            counts = mask.groupby(df['mmsi']).sum()

            valid_mmsi = counts[counts <= max_point_allowed].index
            df = df[df['mmsi'].isin(valid_mmsi)].copy()

        if lon is not None:
            mask = df[lonlat[0]] < lon * 10
            counts = mask.groupby(df['mmsi']).sum()

            valid_mmsi = counts[counts <= max_point_allowed].index
            df = df[df['mmsi'].isin(valid_mmsi)].copy()
    else:
        if lat is not None:
            mask = df[lonlat[1]] < lat * 10

            counts = mask.groupby(
                [df['month'], df['shipid']]
            ).sum()

            valid_pairs = counts[counts <= max_point_allowed].index

            df = (
                df.set_index(['month', 'shipid'])
                       .loc[valid_pairs]
                       .reset_index()
            ).copy()

        if lon is not None:
            mask = df[lonlat[0]] < lon * 10

            counts = mask.groupby(
                [df['month'], df['shipid']]
            ).sum()

            valid_pairs = counts[counts <= max_point_allowed].index

            df = (
                df.set_index(['month', 'shipid'])
                       .loc[valid_pairs]
                       .reset_index()
            ).copy()

    return df

############
## ASTD Functions
############

def load_data(parquet_file, source, year, **kwargs):
    """
    Load data from a parquet file if exists, otherwise loads the csv (for selected year) and save as parquet
    """
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
    """"
    Load data from a parquet file if exists, otherwise loads the csv (for selected periods) and save as parquet
    """
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
    """"
    Load tracks and logs as parquet files if exists, otherwise we generate TrackBuilder tracks with corresponding logs and save them as parquet
    """
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


############
## GFW Functions
############

def load_data_gfw(parquet_file, source, year, months=None):
    """"
    Load data from a parquet file if exists, otherwise loads the csv (for selected year) and save as parquet
    """
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
    """"
    Load data from a parquet file if exists, otherwise loads the csv (for selected periods) and save as parquet
    """
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

############
## Charts Functions
############

def clean_bar(data, labels, label='Multiple Candidates', is_legend=False, color='black', bins=None):
    bins = np.arange(data.min() - 0.5, data.max() + 1.5, 1) if bins is None else bins
    plt.hist(data, bins=bins, color=color, edgecolor='white')
    
    plt.xticks(range(1, data.max()+1)) if bins is None else ""
    plt.grid(axis='y', alpha=0.4)

    if is_legend:
        plt.legend()
    
    plt.xlabel(labels[0] if len(labels)>=1 else "Data Index", fontdict={"size":11})
    plt.ylabel(labels[1] if len(labels)>1 else "Count", fontdict={"size":11})

    filename = f"{label}{str(labels[0])}_{str(labels[1])}"
    path = os.path.join("300DPI", filename + ".png")

    if os.path.exists(path):
        i = 1
        while os.path.exists(os.path.join("300DPI", f"{filename}_{i}.png")):
            i += 1
        plt.savefig(os.path.join("300DPI", f"{filename}_{i}.png"), dpi=300)
    else:
        plt.savefig(path, dpi=300)

    plt.show()

    return plt


def clean_permonth(data, labels, label='Multiple Candidates', color="black"):
    data.plot(kind='bar', figsize=(9, 6), width=0.8, color=color)
    plt.grid(axis='y', alpha=0.3)
    
    plt.xlabel(labels[0] if len(labels)>=1 else "Data Index", fontdict={"size":11})
    plt.ylabel(labels[1] if len(labels)>1 else "Count", fontdict={"size":11})


    filename = f"{label}{str(labels[0])}_{str(labels[1])}"
    path = os.path.join("300DPI", filename + ".png")

    if os.path.exists(path):
        i = 1
        while os.path.exists(os.path.join("300DPI", f"{filename}_{i}.png")):
            i += 1
        plt.savefig(os.path.join("300DPI", f"{filename}_{i}.png"), dpi=300)
    else:
        plt.savefig(path, dpi=300)

    plt.show()


def special_bar(x, y, labels, label, color="black"):
    import matplotlib.dates as mdates
    import os
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(x, y, color=color, width=25)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('  %b'))

    sec = ax.secondary_xaxis(location=-0.08)
    sec.xaxis.set_major_locator(mdates.YearLocator(month=6))
    sec.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    sec.tick_params('x', length=0)
    sec.spines['bottom'].set_linewidth(0)
    # plt.xticks(rotation=0)

    ax.set_xlabel("")
    sec.set_xlabel(labels[0] if len(labels)>=1 else "Data Index", fontdict={"size" : 11})
    plt.ylabel(labels[1] if len(labels)>1 else "Count", fontdict={"size" : 11})

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()


    filename = f"{label}{str(labels[0])}_{str(labels[1])}"
    path = os.path.join("300DPI", filename + ".png")

    if os.path.exists(path):
        i = 1
        while os.path.exists(os.path.join("300DPI", f"{filename}_{i}.png")):
            i += 1
        plt.savefig(os.path.join("300DPI", f"{filename}_{i}.png"), dpi=300)
    else:
        plt.savefig(path, dpi=300)

    plt.show()


def evolution(to_plot, xlabel="Date", ylabel="Count", color="black", label="default", legend=False):
    fig, ax = plt.subplots(figsize=(9, 6))

    to_plot.plot(ax=ax, color=color)
    plt.grid(alpha=0.3)

    plt.xlabel(xlabel, fontdict={"size" : 11})
    plt.ylabel(ylabel, fontdict={"size" : 11})
    if legend:
        ax.legend()
    else:
        ax.get_legend().remove()

    filename = f"{label}{xlabel}_{ylabel}"
    path = os.path.join("300DPI", filename + ".png")

    if os.path.exists(path):
        i = 1
        while os.path.exists(os.path.join("300DPI", f"{filename}_{i}.png")):
            i += 1
        plt.savefig(os.path.join("300DPI", f"{filename}_{i}.png"), dpi=300)
    else:
        plt.savefig(path, dpi=300)

    plt.show()


def evolution_permonth(to_plot, xlabel="Date", ylabel="Count", color="black", label="default"):
    import matplotlib.dates as mdates
    import os
    
    to_plot.index = to_plot.index.to_timestamp(how='start')
    avg = to_plot.mean()
    
    fig, ax = plt.subplots(figsize=(9, 6))
    if avg != 0.0:
        ax.axhline(avg, color='red', linestyle='--', label=f'Average = {avg:.3f}')
    
    to_plot.plot(ax=ax, color=color)
    plt.grid(alpha=0.3)
    # ax.set_xlim(to_plot.index.min(), to_plot.index.max())
    
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('  %b'))
    
    sec = ax.secondary_xaxis(location=-0.08)
    sec.xaxis.set_major_locator(mdates.YearLocator(month=6))
    sec.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    sec.tick_params('x', length=0)
    sec.spines['bottom'].set_linewidth(0)
    # plt.xticks(rotation=0)
    
    ax.set_xlabel("")
    sec.set_xlabel(xlabel, fontdict={"size" : 11})
    plt.ylabel(ylabel, fontdict={"size" : 11})
    
    ax.legend()

    filename = f"{label}{xlabel}_{ylabel}"
    path = os.path.join("300DPI", filename + ".png")

    if os.path.exists(path):
        i = 1
        while os.path.exists(os.path.join("300DPI", f"{filename}_{i}.png")):
            i += 1
        plt.savefig(os.path.join("300DPI", f"{filename}_{i}.png"), dpi=300)
    else:
        plt.savefig(path, dpi=300)

    plt.show()


def evolution_permonth_both(to_plot, to_plot2, xlabels=["Date", ""], ylabels=["Count", "Quantity"], color="black", label="default"):
    import matplotlib.dates as mdates
    import os
    try:
        to_plot.index = to_plot.index.to_timestamp(how='start')
        to_plot2.index = to_plot2.index.to_timestamp(how='start')
    except:
        pass
    fig, ax = plt.subplots(figsize=(9, 6))

    col1 = "#000000"
    ax.plot(to_plot.index, to_plot.values, color=col1, label=ylabels[0].split(" ")[-1].title())

    ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis

    color = '#D72638'
    ax2.plot(to_plot2.index, to_plot2.values, color=color, label=ylabels[1].split(" ")[-1].title())
    ax2.tick_params(axis='y', labelcolor=color)

    plt.grid(alpha=0.3)
    # ax.set_xlim(to_plot.index.min(), to_plot.index.max())

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('  %b'))
    ax.tick_params(axis='y', labelcolor=col1)

    sec = ax.secondary_xaxis(location=-0.08)
    sec.xaxis.set_major_locator(mdates.YearLocator(month=6))
    sec.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    sec.tick_params('x', length=0)
    sec.spines['bottom'].set_linewidth(0)
    # plt.xticks(rotation=0)

    ax.set_xlabel(xlabels[1], fontdict={"size" : 11})
    sec.set_xlabel(xlabels[0], fontdict={"size" : 11})
    ax.set_ylabel(ylabels[0], fontdict={"size" : 11, "color": col1})
    ax2.set_ylabel(ylabels[1], fontdict={"size" : 11, "color": color})

    # Get handles and labels from both axes
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Create a single legend
    ax.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

    filename = f"{label}{str(xlabels)}_{str(ylabels)}"
    path = os.path.join("300DPI", filename + ".png")

    if os.path.exists(path):
        i = 1
        while os.path.exists(os.path.join("300DPI", f"{filename}_{i}.png")):
            i += 1
        plt.savefig(os.path.join("300DPI", f"{filename}_{i}.png"), dpi=300)
    else:
        plt.savefig(path, dpi=300)

    plt.show()


def boxplot_(data, x, y, xlabel="Year", ylabel="Ratio", label='default', size=(7,4)):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Publication style
    sns.set_theme(style="whitegrid", context="paper")

    fig, ax = plt.subplots(figsize=size)

    sns.boxplot(
        data=data,
        x=x,
        y=y,
        color="#4C72B0",
        width=0.6,
        ax=ax,
        linewidth=2,
        boxprops=dict(edgecolor='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        medianprops=dict(color='black', linewidth=2)
    )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    sns.despine()
    plt.tight_layout()

    filename = f"{label}{xlabel}_{ylabel}"
    path = os.path.join("300DPI", filename + ".png")

    if os.path.exists(path):
        i = 1
        while os.path.exists(os.path.join("300DPI", f"{filename}_{i}.png")):
            i += 1
        plt.savefig(os.path.join("300DPI", f"{filename}_{i}.png"), dpi=300)
    else:
        plt.savefig(path, dpi=300)
    plt.show()



def evolution_permonth_both_sameXY(to_plot, to_plot2, xlabels=["Date", ""], ylabels=["Count", "Quantity"], color="black", label="default"):
    import matplotlib.dates as mdates
    import os
    try:
        to_plot.index = to_plot.index.to_timestamp(how='start')
        to_plot2.index = to_plot2.index.to_timestamp(how='start')
    except:
        pass
    fig, ax = plt.subplots(figsize=(9, 6))

    col1 = "#000000"
    ax.plot(to_plot.index, to_plot.values, color=col1, label=ylabels[0].split(" ")[-1].title())

    color = '#D72638'
    ax.plot(to_plot2.index, to_plot2.values, color=color, label=ylabels[1].split(" ")[-1].title())

    ax.tick_params(axis='y', labelcolor=color)

    plt.grid(alpha=0.3)
    # ax.set_xlim(to_plot.index.min(), to_plot.index.max())

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('  %b'))
    ax.tick_params(axis='y', labelcolor=col1)

    sec = ax.secondary_xaxis(location=-0.08)
    sec.xaxis.set_major_locator(mdates.YearLocator(month=6))
    sec.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    sec.tick_params('x', length=0)
    sec.spines['bottom'].set_linewidth(0)
    # plt.xticks(rotation=0)

    ax.set_xlabel(xlabels[1], fontdict={"size" : 11})
    sec.set_xlabel(xlabels[0], fontdict={"size" : 11})
    ax.set_ylabel(ylabels[0], fontdict={"size" : 11})

    # Get handles and labels from both axes
    handles1, labels1 = ax.get_legend_handles_labels()

    # Create a single legend
    ax.legend(handles1, labels1, loc='upper left')

    filename = f"{label}{str(xlabels)}_{str(ylabels)}"
    path = os.path.join("300DPI", filename + ".png")

    if os.path.exists(path):
        i = 1
        while os.path.exists(os.path.join("300DPI", f"{filename}_{i}.png")):
            i += 1
        plt.savefig(os.path.join("300DPI", f"{filename}_{i}.png"), dpi=300)
    else:
        plt.savefig(path, dpi=300)

    plt.show()
