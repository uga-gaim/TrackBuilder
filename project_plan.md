# TrackBuilder Package - High-Level API Design

The following high-level functions are planned to be exposed to the API.

## 1. ASTD Data Loading Functions

### `load_astd_data(file_paths, **kwargs)`
**Purpose**: Simplified loading and preprocessing of ASTD datasets
```python
# Single file
df = tb.load_astd_data('ASTD_area_level3_201907.csv')

# Multiple files with auto-merging
files = ['ASTD_area_level3_201907.csv', 'ASTD_area_level3_201908.csv'] 
df = tb.load_astd_data(files)

# Directory loading with pattern matching
df = tb.load_astd_data('/path/to/astd/2019/', pattern='ASTD_*.csv')
```

**Key Features**:
- Auto-detects file format (CSV separator, encoding)
- Standardizes column names and data types
- __Validates coordinates and removes invalid records__ (Need to discuss the criteria)
- Handles datetime parsing for different formats
- __Optional data cleaning__ (remove rows with low reporting frequency)
- Progress bars for multiple files (`tqdm`)

### `load_astd_monthly(base_path, year, months=None)`
**Purpose**: Convenient loading of monthly ASTD files
```python
# Load specific months
df = tb.load_astd_monthly('/data/astd/', 2019, months=[7, 8, 9])

# Load entire year
df = tb.load_astd_monthly('/data/astd/', 2019)
```

## 2. Ship Location Visualization Functions

### `plot_ship_tracks(data, track_ids=None, **options)`
**Purpose**: Map visualization of ship movements
```python
# Plot all tracks
fig = tb.plot_ship_tracks(df)

# Plot specific tracks
fig = tb.plot_ship_tracks(df, track_ids=['track_001', 'track_002'])

# Customization options
fig = tb.plot_ship_tracks(df, 
                          color_by='ship_type',
                          show_points=True,
                          map_style='satellite',
                          title='Arctic Shipping Routes 2019')
fig.show()
```

**Key Features**:
- __Interactive Plotly maps with hover information__ (Need to decide whether static or interactive)
- Multiple color schemes (by ship type, flag, time, track)
- Optional point markers with timestamps
- Configurable map styles and zoom levels
- Export options (HTML, PNG, PDF)
- Built-in filtering by date ranges, ship types, etc.

### `plot_individual_track(track_id, track_table, astd_data, **options)`
**Purpose**: Detailed view of a single track across months
```python
# Detailed single track view
fig = tb.plot_individual_track('track_001', tracks, df,
                               show_segments=True)
```

## 3. Track Matching and Building Functions

### `build_ship_tracks(astd_data, **options)`
**Purpose**: Main function to connect ship segments across time periods
```python
# Basic track building
tracks = tb.build_ship_tracks(df)

# Advanced options
tracks = tb.build_ship_tracks(df,
                             max_time_gap_hours=72,
                             max_distance_km=1000,
                             min_track_length=2,
                             matching_strategy='conservative')
```

**Returns**: DataFrame with columns `['month', 'segment_id', 'track_id']`

### `find_track_candidates(segment_id, month, astd_data, top_n=5)`
**Purpose**: Find potential matches for a specific ship segment
```python
# Find candidates for a specific segment
candidates = tb.find_track_candidates('ship_12345', '2019-07', df, top_n=3)
print(candidates)
# Returns: DataFrame with ranking, segment_id, match_score, distance, etc.
```

### `get_track_statistics(track_table, astd_data)`
**Purpose**: Analyze track building results and quality metrics
```python
# Get comprehensive track analysis
stats = tb.get_track_statistics(tracks, df)
print(f"Total tracks: {stats['n_tracks']}")
print(f"Average track length: {stats['avg_length']:.1f} months")
print(f"Longest track: {stats['max_length']} months")
```

## 4. Additional Utility Functions

### `export_tracks(track_table, astd_data, format='geojson', filename=None)`
**Purpose**: Export tracks in various formats for GIS analysis
```python
# Export as GeoJSON for QGIS/ArcGIS
tb.export_tracks(tracks, df, format='geojson', 'arctic_tracks_2019.geojson')

# Export as Shapefile
tb.export_tracks(tracks, df, format='shapefile', 'tracks.shp')

# Export summary table
tb.export_tracks(tracks, df, format='csv', 'track_summary.csv')
```

The design can be and will be subject to changes. `:D` Other than creating this function, the structure of the module also needs to be improved.