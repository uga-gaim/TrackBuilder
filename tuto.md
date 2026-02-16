# TrackBuilder — Getting Started Tutorial

> **Welcome!** This tutorial walks you through every step of the `one_track_build` demo notebook.
> Whether you are a student, a researcher, or just curious about Arctic shipping data, you will learn how to **load** raw ship positions, **build** continuous vessel trajectories, and **visualize** the results — all with a few lines of Python.

---

## Table of Contents

1. [What is TrackBuilder?](#1-what-is-trackbuilder)
2. [Prerequisites & Installation](#2-prerequisites--installation)
3. [Project Layout](#3-project-layout)
4. [Step 0 — Import and Configuration](#step-0--import-and-configuration)
   - [What is `sys.path`?](#what-is-syspath)
   - [Configuration variables explained](#configuration-variables-explained)
5. [Step 1 — Loading ASTD Data (Sampled)](#step-1--loading-astd-data-sampled)
   - [`load_astd_monthly()` in depth](#load_astd_monthly-in-depth)
   - [What does `sampling` do?](#what-does-sampling-do)
   - [What does `remove_nan_rows` do?](#what-does-remove_nan_rows-do)
6. [Step 2 — Loading Full ASTD Data](#step-2--loading-full-astd-data)
7. [Step 3 — Building Ship Tracks](#step-3--building-ship-tracks)
   - [`build_ship_tracks()` in depth](#build_ship_tracks-in-depth)
   - [Matching strategies](#matching-strategies)
   - [How the algorithm works (simplified)](#how-the-algorithm-works-simplified)
   - [Understanding the output table](#understanding-the-output-table)
8. [Step 4 — Inspecting a Single Track](#step-4--inspecting-a-single-track)
   - [`load_track_data()` in depth](#load_track_data-in-depth)
   - [`plot_individual_track()` in depth](#plot_individual_track-in-depth)
9. [Step 5 — Multi-Track Visualization](#step-5--multi-track-visualization)
   - [`build_light_multi_track_data()` in depth](#build_light_multi_track_data-in-depth)
   - [`plot_ship_tracks()` in depth](#plot_ship_tracks-in-depth)
10. [Other Useful Functions](#10-other-useful-functions)
    - [`load_astd_data()`](#load_astd_data)
    - [`load_astd_periods()`](#load_astd_periods)
    - [`find_track_candidates()`](#find_track_candidates)
    - [`get_track_statistics()`](#get_track_statistics)
    - [`export_figure()`](#export_figure)
    - [`remove_unrealistic_points()`](#remove_unrealistic_points)
    - [`compute_typical_speeds_by_astd_cat()`](#compute_typical_speeds_by_astd_cat)
11. [Configuration Reference (`config.py`)](#11-configuration-reference-configpy)
12. [FAQ / Troubleshooting](#12-faq--troubleshooting)

---

## 1. What is TrackBuilder?

When you observe ships in the Arctic using AIS (Automatic Identification System) data, each ship emits position reports. In the **ASTD** (Arctic Ship Traffic Data) datasets, these reports are grouped by a `shipid` — but this identifier only lasts for **one calendar month**. The same physical vessel gets a *different* `shipid` in January and in February.

**TrackBuilder** solves this problem: it reconnects monthly segments into continuous trajectories by analyzing the **spatial** and **temporal** proximity of segment start/end points, combined with vessel attributes (type, flag, ice class, size).

```
Month 1         Month 2         Month 3
 shipid=A  →→→  shipid=B  →→→  shipid=C    ← same physical ship
       ↑                                    
       └── TrackBuilder links them into track_id = 1
```

---

## 2. Prerequisites & Installation

### System Requirements

- **Python ≥ 3.8**
- A package manager: `pip` (or `conda`)

### Install TrackBuilder

```bash
# 1. Clone the repository
git clone https://github.com/uga-gaim/TrackBuilder.git
cd TrackBuilder

# 2. Install (regular user)
pip install .

# 3. OR install in development mode (recommended if you plan to modify the code)
pip install -e ".[dev]"
```

### Required Python Libraries

These are installed automatically:

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation (DataFrames) |
| `numpy` | Numerical computations |

### Optional (but recommended)

| Library | Purpose |
|---------|---------|
| `plotly` | Interactive map visualizations |
| `tqdm` | Progress bars when loading data |
| `geopandas` | Geospatial operations (shapefiles) |
| `shapely` | Geometry objects |

Install them with:

```bash
pip install plotly tqdm geopandas shapely
```

---

## 3. Project Layout

```
TrackBuilder/
├── track_builder/          ← The Python package
│   ├── __init__.py         ← Exposes all public functions
│   ├── api.py              ← Public API (re-exports everything)
│   ├── config.py           ← Constants, speed caps, zones, strategies
│   ├── track.py            ← Track-building algorithm
│   ├── core/               ← Internal helpers
│   │   ├── io_helpers.py   ← File loading, CSV parsing, sampling
│   │   ├── track_helpers.py← Data cleaning, segments, Haversine, speeds
│   │   └── vis_helpers.py  ← Color palettes, map styles, hover data
│   ├── io/
│   │   └── astd_loader.py  ← load_astd_data, load_astd_monthly, etc.
│   └── visualization/
│       └── visualizer.py   ← plot_ship_tracks, plot_individual_track
├── demo/
│   └── one_track_build.ipynb  ← THE notebook this tutorial explains
├── pyproject.toml          ← Package metadata
└── README.md
```

> **Tip**: You never need to import internal modules directly. Everything is available via `import track_builder as tb`.

---

## Step 0 — Import and Configuration

This is the first code cell in the notebook:

```python
import sys
import os

project_path = r"C:\Users\lamin\Documents\maitrise\ASTD\TrackBuilder"
if project_path not in sys.path:
    sys.path.append(project_path)

import track_builder as tb

BASE_PATH = r"C:\Users\lamin\Documents\maitrise\ASTD\data"
YEAR      = 2019
MONTHS_TO_LOAD = [1, 2, 3, 4, 5]
USECOLS   = "default"
SAMPLING  = [0, -1]

COLS_REQUIRED = [
    "shipid",
    "date_time_utc",
    "latitude",
    "longitude",
    "astd_cat",
    "flagname",
    "iceclass",
    "sizegroup_gt"
]
```

### What is `sys.path`?

`sys.path` is the list of directories where Python looks for packages. When you run a notebook from `demo/`, Python doesn't automatically know where `track_builder` lives. Adding `project_path` to `sys.path` tells Python: "also look here for packages."

> **Note**: If you installed TrackBuilder with `pip install .` or `pip install -e .`, you **don't** need the `sys.path` trick — Python already knows where the package is.

### Configuration variables explained

| Variable | Meaning |
|----------|---------|
| `BASE_PATH` | The folder on your computer where the raw ASTD CSV files are stored. Every CSV file in that folder (recursively) will be searched. |
| `YEAR` | The year of data you want to load (e.g. `2019`). |
| `MONTHS_TO_LOAD` | A list of month numbers. `[1, 2, 3, 4, 5]` means January through May. |
| `USECOLS` | Which columns to keep when reading CSVs. `"default"` loads a predefined list of essential columns (see `ASTD_USEFUL_COLS` in `config.py`): `shipid`, `date_time_utc`, `astd_cat`, `dist_nextpoint`, `sec_nextpoint`, `longitude`, `latitude`, `flagname`, `iceclass`, `sizegroup_gt`. |
| `SAMPLING` | Controls **which rows** to load from each CSV. `[0, -1]` means "keep only the first and last day of the month." This makes loading fast for testing. |
| `COLS_REQUIRED` | After loading, any row missing a value in one of these columns is dropped. This ensures your data is clean before building tracks. |

---

## Step 1 — Loading ASTD Data (Sampled)

```python
df_for_track = tb.load_astd_monthly(
    BASE_PATH, YEAR, months=MONTHS_TO_LOAD, progress=True,
    usecols=USECOLS, sampling=SAMPLING, remove_nan_rows=COLS_REQUIRED
)
```

This loads **only the first and last day** of each month (January–May 2019). Why? Because the track-building algorithm only needs the start and end positions of each monthly segment. Loading the full dataset would be slow and unnecessary for this step.

### `load_astd_monthly()` in depth

```python
tb.load_astd_monthly(
    base_path,           # Root folder with your ASTD CSVs
    year,                # Integer year (e.g. 2019)
    months=None,         # List of month numbers, or None = all 12 months
    progress=True,       # Show a tqdm progress bar
    **kwargs             # All extra arguments are forwarded to load_astd_data()
)
```

**What happens under the hood:**

1. **File discovery**: Recursively scans `base_path` for all `.csv` files.
2. **Year/month filtering**: Keeps only files whose name contains a pattern like `201901`, `2019-01`, or `2019_01` matching the requested year and months.
3. **Delegation**: Calls `load_astd_data()` with the matching files.

### What does `sampling` do?

The `sampling` parameter controls how many rows to load from each CSV file:

| Value | Behavior |
|-------|----------|
| `None` | Load **all rows** (full dataset). |
| `[0, -1]` | Load only the **first and last day** of the month in each file. Ideal for the track-building step where you only need segment boundaries. |
| `[0, 14, -1]` | Load the 1st, 15th, and last day of the month. |
| `0.5` (float) | Random sampling: keep ~50% of rows. |

> **How does `[0, -1]` work?**  
> Internally, the function `sample_by_day_of_month()` finds all unique days in the file, interprets `0` as the first day and `-1` as the last day (like Python list indexing), and keeps only rows on those dates.

### What does `remove_nan_rows` do?

| Value | Behavior |
|-------|----------|
| `None` | Don't drop any rows for missing values. |
| `"default"` or `"essential"` | Drop rows where any of the essential columns (`ASTD_USEFUL_COLS`) is NaN. |
| A list of column names | Drop rows where any of the listed columns is NaN. |

In our notebook, we pass `COLS_REQUIRED` — so any row missing `shipid`, `date_time_utc`, `latitude`, `longitude`, `astd_cat`, `flagname`, `iceclass`, or `sizegroup_gt` is removed.

---

## Step 2 — Loading Full ASTD Data

```python
df = tb.load_astd_monthly(
    BASE_PATH, YEAR, months=MONTHS_TO_LOAD, progress=True,
    usecols=USECOLS, sampling=None, remove_nan_rows=COLS_REQUIRED
)
```

Same function, but with `sampling=None`. This loads **every single row** from the 5 months of data. This full dataset is needed later for visualization — you want to see every position point of a ship, not just the first and last day.

> **Memory tip**: If your dataset is very large (millions of rows), consider loading only the months you need, or using the `quality_threshold_minutes` parameter to discard ships with very sparse data.

---

## Step 3 — Building Ship Tracks

```python
tracks = tb.build_ship_tracks(
    df_for_track,
    matching_strategy="balanced",
)
```

This is the **core** of TrackBuilder. It takes the sampled data (first + last day only) and connects monthly segments into multi-month trajectories.

### `build_ship_tracks()` in depth

```python
tb.build_ship_tracks(
    astd_data,                          # DataFrame with ship positions
    *,
    max_time_gap_hours=96,              # Max hours between end of segment A and start of segment B
    max_distance_km=1200,               # Max km between end of A and start of B
    min_track_length=1,                 # Minimum number of segments for a track to be kept
    matching_strategy="conservative",   # "conservative", "balanced", or "aggressive"
    w_time=0.4,                         # Weight for time gap in the scoring formula
    w_dist=0.4,                         # Weight for distance in the scoring formula
    w_speed=0.2,                        # Weight for implied speed in the scoring formula
    gap_days_no_penalty=3.0,            # Days of gap before penalty kicks in
    gap_penalty_per_day=0.05,           # Extra penalty per day of gap beyond threshold
    return_logs=False,                  # If True, also returns a DataFrame of all decisions made
    typical_speeds=None,                # Optional dict {category: speed_kmh}, auto-computed if None
)
```

**Returns:** A DataFrame with columns `['month', 'segment_id', 'track_id']`.

If `return_logs=True`, returns a tuple `(tracks_df, logs_df)` where `logs_df` details every candidate evaluated and why it was accepted or rejected.

### Matching Strategies

The `matching_strategy` parameter controls how strict the algorithm is:

| Strategy | Behavior | Multipliers (time, distance, speed) |
|----------|----------|--------------------------------------|
| `"conservative"` | Strict — fewer but more confident tracks | `(0.9, 0.9, 0.85)` |
| `"balanced"` | Middle ground (recommended to start) | `(1.0, 1.0, 1.0)` |
| `"aggressive"` | Relaxed — more tracks but higher risk of errors | `(1.2, 1.2, 1.15)` |

These multipliers are applied to the `max_time_gap_hours`, `max_distance_km`, and speed limits respectively. For example, with `"conservative"`, the effective time gap becomes `96 × 0.9 = 86.4 hours`.

### How the Algorithm Works (Simplified)

The algorithm follows four steps:

#### Step (i) — Segment Preparation

Each `(shipid, month)` pair in your data becomes a **segment**. For each segment, the algorithm records:
- **Start position** (latitude, longitude) and **start time** (first observation)
- **End position** and **end time** (last observation)
- **Attributes**: ship type (`astd_cat`), flag (`flagname`), ice class (`iceclass`), size group (`sizegroup_gt`)

This is done by the internal function `_prepare_segments()`, which calls:
- `clean_data()` — lowercases text fields, parses dates, validates coordinates, removes invalid rows
- `get_segment_summaries()` — groups by `(shipid, month)` and extracts the first/last observation

#### Step (ii) — Candidate Generation

For each segment `A` ending in month `M`, the algorithm looks at all segments starting in month `M+1` (and beyond) and asks: "Could this be the same ship?"

A candidate `B` is **rejected** if:
- The time gap (end of A → start of B) is negative or too large
- The distance (end of A → start of B) is too large
- The implied speed to travel from A to B is unrealistically fast
- The ship attributes (type, flag, etc.) are inconsistent

#### Step (iii) — Scoring

Surviving candidates receive a **score** (lower = better):

```
score = w_time × dt_norm + w_dist × dd_norm + w_speed × speed_ratio + gap_penalty
```

Where:
- `dt_norm` = time gap normalized by max allowed
- `dd_norm` = distance normalized by max allowed
- `speed_ratio` = implied speed / typical speed for this vessel category
- `gap_penalty` = extra cost if the gap exceeds `gap_days_no_penalty` days

#### Step (iv) — Greedy Selection

Segments are processed in chronological order. For each unassigned segment:
1. Create a new `track_id`
2. Try to extend the track by finding the best-scoring candidate in the next month
3. If found and not yet assigned, add it to the track and repeat from its position
4. Stop extending when no valid candidate remains

The result is a mapping `(month, segment_id) → track_id`.

### Understanding the Output Table

```
   month   segment_id  track_id
0  2019-01     12345        1
1  2019-02     67890        1
2  2019-03     11111        1
3  2019-01     22222        2
4  2019-02     33333        2
```

- **`month`**: The calendar month of the segment (format `YYYY-MM`)
- **`segment_id`**: The original `shipid` in the ASTD data for that month
- **`track_id`**: The identifier assigned by TrackBuilder — same `track_id` = same physical ship across months

---

## Step 4 — Inspecting a Single Track

```python
# Filter for tracks containing more than one segment (multi-month tracks)
track_counts = tracks['track_id'].value_counts()
linked_tracks = track_counts[track_counts > 1].index.tolist()

if linked_tracks:
    target_track_id = linked_tracks[0]
    print(f"Visualizing Track ID: {target_track_id}")

    # Load the full position data for this specific track
    track_data_df = tb.load_track_data(
        track_ids=target_track_id,
        track_table=tracks,
        base_path=BASE_PATH,
        chunksize=500_000
    )

    fig = tb.plot_individual_track(
        track_id=target_track_id,
        track_table=tracks,
        astd_data=track_data_df,
        title=f"Reconstructed Trajectory | Track ID: {target_track_id}"
    )
    fig.show()
else:
    print("No multi-segment tracks found in the current sample.")
```

Let's break this down:

### Finding Multi-Month Tracks

```python
track_counts = tracks['track_id'].value_counts()
linked_tracks = track_counts[track_counts > 1].index.tolist()
```

`value_counts()` counts how many segments belong to each `track_id`. If a track has more than 1 segment, it means TrackBuilder successfully linked segments across multiple months — that's a confirmed trajectory.

### `load_track_data()` in depth

```python
tb.load_track_data(
    track_ids,           # A single ID or a list of IDs
    track_table,         # The tracks DataFrame from build_ship_tracks()
    *,
    base_path=None,      # Path to raw CSV files (defaults to ASTD_DATA_PATH env variable)
    progress=True,       # Show tqdm progress bar
    chunksize=50_000,    # Read CSVs in chunks of this size (memory efficiency)
    use_preprocessing=True  # Apply remove_unrealistic_points() after loading
)
```

**What it does:**

1. Looks up the `track_table` to find which `(segment_id, month)` pairs belong to the requested `track_ids`
2. Determines which CSV files contain those months
3. Reads only the relevant files (batch I/O — each file is opened only once even for multiple tracks)
4. Filters rows to keep only positions matching the segment IDs for the correct months
5. Optionally cleans the data with `remove_unrealistic_points()`
6. Returns a DataFrame with all position points plus a `track_id` column

**Why `chunksize`?**  
ASTD CSV files can be large (millions of rows). Reading in chunks (`chunksize=500_000`) means the function processes 500,000 rows at a time instead of loading the entire file into memory at once. This prevents memory issues on machines with limited RAM.

### `plot_individual_track()` in depth

```python
tb.plot_individual_track(
    track_id,            # The ID of the track to visualize
    track_table,         # The tracks DataFrame
    astd_data,           # Position data (from load_track_data)
    *,
    show_segments=True,  # Show individual segment lines with markers
    map_style="open-street-map",  # Map background
    height=700,          # Plot height in pixels
    title=None,          # Custom title (auto-generated if None)
    extra_cols_priority=None,  # Extra columns to show in hover tooltip
    color_by=None,       # Column to use for coloring
)
```

**What it does:**

1. Resolves latitude, longitude, and time columns from the data
2. Joins the position data with the track table using `(segment_id, month)` keys
3. Colors each month's segment differently (using the Plotly qualitative palette)
4. Draws lines connecting consecutive positions within each segment
5. Adds hover tooltips showing the date, segment ID, coordinates, and any extra columns
6. Returns a Plotly `Figure` object — call `.show()` to display it

**Color coding:** Each month gets a unique color. The legend at the bottom of the map shows which color corresponds to which `YYYYMM` month.

---

## Step 5 — Multi-Track Visualization

```python
work = tb.build_light_multi_track_data(
    track_table=tracks,
    specific_track_ids=[490],
    positions_df=df
)

fig = tb.plot_ship_tracks(
    work,
    color_by="track_id",
    color_mode="categorical",
    show_points=False,
    map_style="open-street-map",
    title="Tracks (light multi-track sample)",
)
fig.update_layout(showlegend=False)
fig.show()
```

### `build_light_multi_track_data()` in depth

This function prepares a "light" (potentially subsampled) DataFrame with positions for one or many tracks, ready for visualization.

```python
tb.build_light_multi_track_data(
    track_table,                  # Result of build_ship_tracks()
    track_sampling=None,          # How many/which tracks to include
    *,
    specific_track_ids=None,      # Explicit list of track IDs to include
    positions_df=None,            # Pre-loaded position DataFrame (faster)
    n_tracks_length=None,         # Filter by track length
    base_path=None,               # Path to CSVs (if positions_df is None)
    chunksize=50_000,             # Chunk size for reading CSVs
    progress=True,                # Show progress bar
    point_stride=10,              # Keep 1 out of every N points (subsampling)
    random_state=42,              # Seed for reproducible random sampling
    preprocess_positions=True,    # Apply remove_unrealistic_points()
    region=None,                  # Geographic filter: "canada", "russia", "norway", etc.
    bounding_box=None,            # Custom geographic bounding box
)
```

#### Two Modes of Operation

| Mode | When? | Description |
|------|-------|-------------|
| **From memory** | `positions_df` is provided | Filters the pre-loaded DataFrame. Fast — no disk I/O. |
| **From disk** | `positions_df` is `None` | Reads CSVs via `load_track_data()`. Slower but doesn't require pre-loading. |

In the notebook, we pass `positions_df=df` (the full 5-month dataset we loaded in Step 2), so no additional file reading is needed.

#### Track Selection Parameters

| Parameter | Example | Effect |
|-----------|---------|--------|
| `specific_track_ids=[490]` | Explicit | Show only track 490 |
| `specific_track_ids=[1, 5, 10]` | Explicit | Show tracks 1, 5, and 10 |
| `track_sampling=20` | Random | Randomly pick 20 tracks |
| `track_sampling=[0, 100]` | Range | Tracks from index 0 to 100 |
| `track_sampling=[0, -1]` | Range | All tracks (0 to last) |
| `n_tracks_length=3` | Filter | Only tracks with ≥ 3 monthly segments |
| `n_tracks_length=[2, 5]` | Filter | Only tracks with 2 to 5 segments |

> If neither `specific_track_ids` nor `track_sampling` is provided, it defaults to randomly sampling 20 tracks.

#### Regional Filtering

You can restrict tracks to specific Arctic regions:

```python
work = tb.build_light_multi_track_data(
    track_table=tracks,
    track_sampling=50,
    positions_df=df,
    region="canada"  # Only keep positions in Canadian Arctic waters
)
```

Available regions (defined in `config.py`):

| Region | Approximate Area |
|--------|-----------------|
| `"canada"` | Northwest Passage (lon: -141 to -50, lat: 60-85) |
| `"norway"` | Barents Sea / Svalbard (lon: 5-35, lat: 60-82) |
| `"russia"` | Northern Sea Route (lon: 50 to -168, lat: 65-85) |
| `"usa"` | Alaska Arctic (lon: -170 to -140, lat: 60-75) |
| `"iceland"` | Iceland EEZ (lon: -25 to -12, lat: 63-67) |

### `plot_ship_tracks()` in depth

```python
tb.plot_ship_tracks(
    df,                          # DataFrame with positions
    track_ids=None,              # Filter to specific track IDs
    *,
    color_by=None,               # Column to color-code tracks
    color_mode="categorical",    # "auto", "categorical", or "continuous"
    color_lines=False,           # Color the lines (not just points)
    max_categories=20,           # Max distinct colors for categorical mode
    show_points=False,           # Show position markers
    show_start_end=False,        # Show START (ring) and END (dot) markers
    map_style=None,              # "open-street-map", "satellite", "dark", etc.
    title=None,                  # Map title
    date_from=None,              # Filter: only show data after this date
    date_to=None,                # Filter: only show data before this date
    ship_types=None,             # Filter: only show these ship types
    flags=None,                  # Filter: only show these flag states
    height=650,                  # Plot height in pixels
    zoom=None,                   # Map zoom level
    center=None,                 # Map center {"lat": ..., "lon": ...}
    extra_cols_priority=None,    # Extra columns in hover tooltip
    point_opacity=1.0,           # Opacity of point markers
    fixed_color=None,            # Force all tracks to one color
)
```

**Key features:**
- Automatically detects latitude/longitude/time columns
- Groups data by track and draws one line per track
- Hover tooltip shows date, track ID, and additional attributes
- START markers appear as rings (hollow circles), END markers as solid dots
- Returns a Plotly `Figure` — interactive zoom, pan, hover

#### Color Modes

| Mode | Use Case |
|------|----------|
| `"categorical"` | Color by ship type, flag, or track_id (discrete colors) |
| `"continuous"` | Color by a numeric value like speed (gradient) |
| `"auto"` | Automatically chooses based on the number of unique values |

#### Map Styles

| Style name | Description |
|------------|-------------|
| `"open-street-map"` or `"osm"` | Free, no token needed (default) |
| `"satellite"` | Satellite imagery (requires Mapbox token) |
| `"dark"` | Dark theme (requires Mapbox token) |
| `"light"` | Light theme (requires Mapbox token) |
| `"streets"` | Street map (requires Mapbox token) |

---

## 10. Other Useful Functions

These functions are part of the TrackBuilder API but are not used directly in the demo notebook. They can be very useful in your own analyses.

### `load_astd_data()`

The general-purpose data loader. `load_astd_monthly()` actually calls this function under the hood.

```python
df = tb.load_astd_data(
    file_paths,                     # A file, a list of files, or a directory
    pattern=None,                   # Glob pattern (e.g. "ASTD_*.csv")
    usecols=None,                   # Columns to keep ("default" or a list)
    remove_nan_rows=None,           # Drop rows with NaN in these columns
    sampling=None,                  # Row sampling (float, list of day indices, or None)
    infer_datetime_cols=True,       # Auto-parse date columns
    standardize_cols=True,          # Rename columns to canonical names
    quality_threshold_minutes=0,    # Drop ship-months with sparse data
    progress=True,                  # Show progress bar
)
```

Examples:

```python
# Load a single file
df = tb.load_astd_data('ASTD_area_level3_201907.csv')

# Load all CSVs in a directory
df = tb.load_astd_data('/path/to/data/', pattern='ASTD_*.csv')

# Load multiple files
df = tb.load_astd_data(['file1.csv', 'file2.csv'])
```

### `load_astd_periods()`

Load data spanning multiple years and specific months:

```python
df = tb.load_astd_periods(
    base_path='/data/astd/',
    periods={
        2019: [10, 11, 12],   # Oct, Nov, Dec 2019
        2020: [1, 2, 3]       # Jan, Feb, Mar 2020
    },
    usecols="default",
    progress=True
)
```

This is handy when you need winter data that crosses the year boundary.

### `find_track_candidates()`

Explore which segments could follow a given segment:

```python
candidates = tb.find_track_candidates(
    segment_id='12345',         # The shipid to investigate
    month='2019-01',            # The month of this segment
    astd_data=df,               # Full position data
    top_n=5,                    # Return at most 5 candidates
    matching_strategy="balanced"
)
print(candidates)
```

Returns a DataFrame with columns like `segment_id`, `match_score_simple`, `distance_km_fd`, `implied_v_kmh`, `dt_hours`.

### `get_track_statistics()`

Get summary statistics about the tracks you built:

```python
stats = tb.get_track_statistics(tracks, df)
print(f"Total tracks: {stats['n_tracks']}")
print(f"Total segments: {stats['n_segments']}")
print(f"Average track length: {stats['avg_length']:.1f} months")
print(f"Longest track: {stats['max_length']} months")
print(f"Tracks by ship type:\n{stats['by_ship_type']}")
```

### `export_figure()`

Save any Plotly figure to a file:

```python
tb.export_figure(fig, "my_map.html")   # Interactive HTML
tb.export_figure(fig, "my_map.png")    # Static image (needs kaleido)
tb.export_figure(fig, "my_map.pdf")    # PDF (needs kaleido)
```

### `remove_unrealistic_points()`

Clean up position data by removing "ghost" points — positions that imply physically impossible speeds:

```python
df_clean = tb.remove_unrealistic_points(df, multiplier=3.0)
```

**How it works:**
1. Computes typical speeds per vessel category from the data itself
2. For each consecutive pair of positions for a ship, checks if the implied speed exceeds `typical_speed × multiplier` (capped by literature-based maximum speeds)
3. Removes points that have no valid link (forward or backward) to a neighbor

### `compute_typical_speeds_by_astd_cat()`

Compute data-driven typical speeds per ship category:

```python
speeds = tb.compute_typical_speeds_by_astd_cat(df)
print(speeds)
#     astd_cat                 typical_speed_kmh  n_ships_used
# 0   bulk carriers            14.5               120
# 1   container ships          21.3               85
# ...
```

This uses the 90th percentile of per-ship average speeds, computed from daily samples. The result is used internally by `build_ship_tracks()` to validate candidate matches.

---

## 11. Configuration Reference (`config.py`)

### `ASTD_USEFUL_COLS`

The "default" columns loaded when `usecols="default"`:

```python
['shipid', 'date_time_utc', 'astd_cat', 'dist_nextpoint',
 'sec_nextpoint', 'longitude', 'latitude', 'flagname', 'iceclass', 'sizegroup_gt']
```

### `_LIT_CAPS_KMH`

Literature-based maximum speeds per vessel category (km/h). Used as a safety cap when filtering unrealistic transitions:

| Category | Max Speed (km/h) | Source |
|----------|-------------------|--------|
| Container ships | 68.0 | ~36.5 knots |
| Refrigerated cargo ships | 50.0 | ~27 knots |
| Passenger/Cruise ships | 56.0 | ~30 knots |
| Ro-Ro cargo ships | 47.0 | ~25 knots |
| Gas tankers | 41.0 | ~22 knots |
| General cargo ships | 34.0 | ~18 knots |
| Bulk carriers | 30.0 | ~16 knots |
| Crude oil / Oil product / Chemical tankers | 32.0 | ~17 knots |
| Fishing vessels | 28.0 | ~15 knots |
| Offshore service / supply | 47.0 | ~25 knots |

### `_LIMIT_MULTIPLIERS`

Strategy-dependent multipliers applied to max time gap, max distance, and speed limit:

```python
"conservative": (0.9, 0.9, 0.85)
"balanced":     (1.0, 1.0, 1.0)
"aggressive":   (1.2, 1.2, 1.15)
```

### `ARCTIC_ZONES`

Predefined bounding boxes `(min_lon, max_lon, min_lat, max_lat)` for Arctic regions (used by `build_light_multi_track_data(region=...)`).

---

## 12. FAQ / Troubleshooting

### Q: I get `FileNotFoundError: No CSV files found`

Make sure `BASE_PATH` points to the directory containing your ASTD CSV files. The files should have names containing patterns like `201901` or `2019-01` or `2019_01`.

### Q: The track-building step is very slow

- Use `sampling=[0, -1]` for the data you feed to `build_ship_tracks()` — you only need the first and last day of each month.
- Reduce the number of months.
- Use `matching_strategy="conservative"` to reduce the candidate search space.

### Q: I get very few (or no) multi-month tracks

- Try `matching_strategy="aggressive"` to relax the matching constraints.
- Increase `max_time_gap_hours` and `max_distance_km`.
- Check that your data covers consecutive months — gaps of 2+ months make linking harder.

### Q: How do I debug the matching decisions?

Use `return_logs=True`:

```python
tracks, logs = tb.build_ship_tracks(df_for_track, matching_strategy="balanced", return_logs=True)
print(logs.head(20))
# Shows why each candidate was accepted or rejected
```

### Q: The map doesn't display / I see a blank plot

- Make sure `plotly` is installed: `pip install plotly`
- If using JupyterLab, you may need: `pip install jupyterlab plotly`
- For `"satellite"` or `"dark"` styles, you need a Mapbox token. Stick with `"open-street-map"` (free, no token needed).

### Q: How do I save a figure?

```python
tb.export_figure(fig, "output.html")  # Interactive HTML file
```

### Q: Can I use my own data (not ASTD)?

Yes! As long as your DataFrame has these columns:
- `shipid` — a monthly vessel identifier
- `date_time_utc` — datetime of each position report
- `latitude`, `longitude` — coordinates
- `astd_cat` — vessel type category

You can rename your columns to match, or use the automatic column standardization (`standardize_cols=True`).

---

**Happy tracking!** If you have questions or suggestions, feel free to open an issue on [GitHub](https://github.com/uga-gaim/TrackBuilder/issues).
