import numpy as np
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean up the ship data.

    Standardizes text fields, parses datetime columns, validates coordinates,
    and removes invalid records. Essential preprocessing step for all analysis.

    Parameters:
    - df (pd.DataFrame): Raw ship tracking dataset

    Returns:
    - pd.DataFrame: Cleaned dataset ready for analysis
                   May be empty if no valid data remains after cleaning
    """
    df = df.copy()

    # Make text columns lowercase and clean
    text_cols = ['astd_cat', 'flagname', 'iceclass', 'sizegroup_gt']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    # Fix datetime - try multiple formats
    try:
        df['date_time_utc'] = pd.to_datetime(df['date_time_utc'])
    except:
        print("Warning: Could not parse date_time_utc column")
        print(f"Sample date values: {df['date_time_utc'].head()}")
        return df

    # Fix coordinates
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    # Check for valid coordinate ranges
    df.loc[df['latitude'] > 90, 'latitude'] = np.nan
    df.loc[df['latitude'] < -90, 'latitude'] = np.nan
    df.loc[df['longitude'] > 180, 'longitude'] = np.nan
    df.loc[df['longitude'] < -180, 'longitude'] = np.nan

    # Remove bad coordinates
    before = len(df)
    df = df.dropna(subset=['latitude', 'longitude'])
    after = len(df)
    if before != after:
        print(f"Removed {before - after} rows with bad coordinates")

    # Check if we have any data left
    if len(df) == 0:
        print("Warning: No valid data remaining after cleaning")
        return df

    print(f"Data sample after cleaning:")
    print(f"  Date range: {df['date_time_utc'].min()} to {df['date_time_utc'].max()}")
    print(f"  Ship types: {df['astd_cat'].unique()}")
    print(f"  Unique ships: {df['shipid'].nunique()}")

    return df


def create_ship_signature(ship_row: pd.Series) -> str:
    """
    Create a unique signature for ship characteristics.

    Combines multiple ship attributes into a single string that can be used
    for matching segments that likely represent the same physical ship.

    Parameters:
    - ship_row (pd.Series): Single row containing ship characteristics

    Returns:
    - str: Ship signature string combining type, flag, ice class, and size
           e.g., "container ships|panama|none|10000-19999"
    """
    return f"{ship_row['astd_cat']}|{ship_row['flagname']}|{ship_row['iceclass']}|{ship_row['sizegroup_gt']}"


def get_segment_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary information for each ship segment (shipid).

    Creates segment summaries containing temporal and spatial information for each
    unique shipid in the dataset. Each shipid represents a ship's appearance during
    a specific time period.

    Parameters:
    - df (pd.DataFrame): Clean ship tracking data with datetime and position columns

    Returns:
    - pd.DataFrame: Segment summaries with columns: shipid, month, start_time, end_time,
                   start_lat, start_lon, end_lat, end_lon, astd_cat, flagname,
                   iceclass, sizegroup_gt, ship_signature
    """
    segments = []

    print(f"Creating segments for {df['shipid'].nunique()} unique shipids")

    for ship_id in df['shipid'].unique():
        ship_data = df[df['shipid'] == ship_id].copy()
        ship_data = ship_data.sort_values('date_time_utc')

        if len(ship_data) == 0:
            continue

        # Get time period and position info
        start_time = ship_data['date_time_utc'].iloc[0]
        end_time = ship_data['date_time_utc'].iloc[-1]
        month = start_time.strftime('%Y-%m')  # Use start month for grouping

        segment = {
            'shipid': ship_id,
            'month': month,
            'start_time': start_time,
            'end_time': end_time,
            'start_lat': ship_data['latitude'].iloc[0],
            'start_lon': ship_data['longitude'].iloc[0],
            'end_lat': ship_data['latitude'].iloc[-1],
            'end_lon': ship_data['longitude'].iloc[-1],
            'astd_cat': ship_data['astd_cat'].iloc[0],
            'flagname': ship_data['flagname'].iloc[0],
            'iceclass': ship_data['iceclass'].iloc[0],
            'sizegroup_gt': ship_data['sizegroup_gt'].iloc[0],
            # Add ship characteristics signature for matching
            'ship_signature': create_ship_signature(ship_data.iloc[0])
        }
        segments.append(segment)

    result_df = pd.DataFrame(segments)
    print(f"Created {len(result_df)} segments")
    if len(result_df) > 0:
        print(f"Sample segment: {result_df.iloc[0]['ship_signature']}")

    return result_df


def to_ts(x):
    return pd.to_datetime(x, utc=True, errors="coerce")


def haversine_km(lat1, lon1, lat2, lon2):
    """Fallback Haversine si _core.haversine_km n'existe pas."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def _compute_speed_kmh_between_rows(grp: pd.DataFrame) -> pd.Series:
    """
    grp: points of a single ship_id for a single day (sorted by date_time_utc).
    Returns a Series of speeds (km/h) between consecutive points, indexed starting at the 2nd point.
    """
    # Case 1: use dist_nextpoint/sec_nextpoint if available
    if {'dist_nextpoint', 'sec_nextpoint'}.issubset(grp.columns):
        dist_m = grp['dist_nextpoint'].to_numpy()[1:]
        secs = grp['sec_nextpoint'].to_numpy()[1:]
        km = dist_m / 1000.0
        hours = np.where(secs > 0, secs / 3600.0, np.nan)
        speeds = np.where(hours > 0, km / hours, np.nan)
        return pd.Series(speeds, index=grp.index[1:])

    # Cas 2: fallback lat/lon/time
    lat = grp['latitude'].to_numpy()
    lon = grp['longitude'].to_numpy()
    t = pd.to_datetime(grp['date_time_utc']).to_numpy()
    if len(grp) < 2:
        return pd.Series([], dtype=float)
    km = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dt_h = (t[1:] - t[:-1]) / np.timedelta64(1, 'h')
    speeds = np.where(dt_h > 0, km / dt_h, np.nan)
    return pd.Series(speeds, index=grp.index[1:])


def compute_typical_speeds_by_astd_cat(
        df: pd.DataFrame,
        *,
        n_per_day: int = 4,
        clip_kmh: float = 80.0,
        min_points_per_ship: int = 5
) -> pd.DataFrame:
    """
    Compute 'typical' speeds by astd_cat in three steps:
      (1) per ship_id & day: speeds between consecutive points (random sample n_per_day)
      (2) per ship_id: mean of speeds > 0
      (3) per astd_cat: 90th percentile of ship-level means

    Required columns: ship_id/shipid, date_time_utc, latitude, longitude, astd_cat.
    Uses dist_nextpoint/sec_nextpoint when available.
    """

    needed = {'shipid', 'date_time_utc', 'latitude', 'longitude', 'astd_cat'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(list(missing))}")

    work = df.copy()
    work['date_time_utc'] = pd.to_datetime(work['date_time_utc'])
    work['date'] = work['date_time_utc'].dt.date
    work = work.sort_values(['shipid', 'date', 'date_time_utc'])

    rows = []
    for (sid, d), grp in work.groupby(['shipid', 'date'], sort=False):
        if len(grp) < 2:
            continue
        v = _compute_speed_kmh_between_rows(grp)
        if v.empty:
            continue
        v = v.replace([np.inf, -np.inf], np.nan).dropna()
        v = v[(v > 0) & (v <= clip_kmh)]
        if v.empty:
            continue
        if len(v) > n_per_day:
            v = v.sample(n_per_day, random_state=42)
        rows.append(pd.DataFrame({'shipid': sid, 'date': d, 'speed_kmh': v.values}))

    if not rows:
        return pd.DataFrame(columns=['astd_cat', 'typical_speed_kmh', 'n_ships_used'])

    speeds = pd.concat(rows, ignore_index=True)

    ship_cat = work.drop_duplicates('shipid')[['shipid', 'astd_cat']]
    speeds = speeds.merge(ship_cat, on='shipid', how='left')

    # agg per ship_id
    ship_means = (speeds.groupby('shipid', as_index=False)
                  .agg(mean_speed_kmh=('speed_kmh', 'mean'),
                       n_samples=('speed_kmh', 'size')))
    ship_means = ship_means[ship_means['n_samples'] >= min_points_per_ship]
    ship_means = ship_means.merge(ship_cat, on='shipid', how='left')

    # final agg per astd_cat
    out = (ship_means.groupby('astd_cat')
           .agg(typical_speed_kmh=('mean_speed_kmh', lambda x: x.quantile(0.9)),
                n_ships_used=('shipid', 'nunique'))
           .reset_index()
           .sort_values('typical_speed_kmh'))
    return out


# ---------------------------
def get_all_point_to_point_speeds(
    astd_data: pd.DataFrame, 
    n_per_day: int = 10, 
    clip_kmh: int = 110
) -> pd.DataFrame:
    """
    Extracts all filtered point-to-point speeds and their categories
    for outlier analysis and visualization.

    This is an adaptation of compute_typical_speeds_q90, but returns
    the raw speed samples instead of the aggregated Q90.

    Parameters
    ----------
    astd_data : pd.DataFrame
        The raw ASTD DataFrame. Must contain:
        'shipid', 'date_time_utc', 'astd_cat', 
        and speed columns ('dist_nextpoint', 'sec_nextpoint') 
        or coordinates ('latitude', 'longitude').
    n_per_day : int, optional
        To avoid memory issues with millions of points, sample
        this many speed readings per ship, per day. (default is 10)
    clip_kmh : int, optional
        Upper speed limit to filter out absurd GPS errors. (default is 110)

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ['astd_cat', 'speed_kmh']
    """
    work = astd_data.copy()
    if 'date' not in work.columns:
        # Ensure 'date_time_utc' is datetime
        work['date_time_utc'] = to_ts(work['date_time_utc'])
        work['date'] = work['date_time_utc'].dt.date

    rows = []
    # Group by ship AND day
    grouped = work.groupby(['shipid', 'date'])

    # Use tqdm for a progress bar if available
    try:
        from tqdm import tqdm
        iterable = tqdm(grouped, desc="[track_helpers] Computing p-t-p speeds")
    except ImportError:
        iterable = grouped

    for (sid, d), grp in iterable:
        if len(grp) < 2:
            continue
        
        # Use the existing helper to get speeds for this group
        v = _compute_speed_kmh_between_rows(grp) 
        
        if v.empty:
            continue
        
        # Clean inf/-inf (from 0-sec gaps) and NaNs
        v = v.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Filter speeds: > 0 (stopped) and < absurd limit
        # (Using > 0 as per compute_typical_speeds_q90)
        v = v[(v > 0) & (v <= clip_kmh)]
        
        if v.empty:
            continue
        
        # Sample to keep the dataset manageable
        if len(v) > n_per_day: 
            v = v.sample(n_per_day, random_state=42)
        
        rows.append(pd.DataFrame({'shipid': sid, 'speed_kmh': v.values}))

    if not rows:
        return pd.DataFrame(columns=['astd_cat', 'speed_kmh'])

    speeds_df = pd.concat(rows, ignore_index=True)
    
    # Get the category for each shipid
    # (Using the first-seen category for each ship)
    ship_cat_map = work.drop_duplicates('shipid', keep='first').set_index('shipid')['astd_cat']
    speeds_df['astd_cat'] = speeds_df['shipid'].map(ship_cat_map)

    return speeds_df[['astd_cat', 'speed_kmh']].dropna()


def calculate_improved_match_score(candidates: pd.DataFrame, ship_type: str, current_segment: pd.Series) -> np.ndarray:
    """
    Improved scoring system that heavily penalizes unrealistic scenarios.

    Calculates a composite match score based on multiple factors including distance,
    time, speed reasonableness, temporal continuity, and position logic.
    Lower scores indicate better matches.

    Parameters:
    - candidates (pd.DataFrame): Candidates with calculated metrics
    - ship_type (str): Type of ship for speed expectations
    - current_segment (pd.Series): Current segment for position continuity

    Returns:
    - np.ndarray: Array of match scores (0-1+ scale, lower is better)
    """
    # Expected typical speeds for ship types (km/h) - more conservative
    typical_speeds = {
        'unknown': 12,
        'fishing vessels': 8,
        'passenger ships': 25,
        'oil product tankers': 15,
        'other activities': 12,
        'general cargo ships': 15,
        'ro-ro cargo ships': 20,
        'cruise ships': 25,
        'refrigerated cargo ships': 15,
        'chemical tankers': 15,
        'bulk carriers': 12,
        'other service offshore vessels': 10,
        'offshore supply ships': 8,
        'crude oil tankers': 14,
        'container ships': 22,
        'gas tankers': 16,
    }

    typical_speed = typical_speeds.get(ship_type, 15)

    # Normalize components (0-1 scale, lower is better)
    distance_score = np.minimum(candidates['distance_km'] / 500, 1.0)  # Normalize by 500km (stricter)
    time_score = np.minimum(candidates['time_gap_hours'] / 72, 1.0)  # Normalize by 3 days (stricter)

    # Speed deviation score (heavily penalize unrealistic speeds)
    speed_deviation = np.abs(candidates['implied_speed'] - typical_speed) / typical_speed
    speed_score = np.minimum(speed_deviation, 2.0)  # Allow up to 200% deviation

    # Month continuity score (heavily favor consecutive months)
    month_score = candidates['month_gap'].apply(lambda x: 0.0 if x == 1 else 0.5 if x == 2 else 1.0)

    # Position continuity - check if the route makes geographical sense
    # (This could be enhanced with actual shipping route data)
    position_score = calculate_position_continuity_score(candidates, current_segment)

    # Weighted combination - prioritize continuity and realistic movement
    total_score = (
            0.3 * distance_score +  # Distance traveled
            0.2 * speed_score +  # Speed reasonableness
            0.2 * time_score +  # Time gap
            0.2 * month_score +  # Month continuity
            0.1 * position_score  # Position logic
    )

    return total_score


def calculate_position_continuity_score(candidates: pd.DataFrame, current_segment: pd.Series) -> np.ndarray:
    """
    Calculate a score based on position continuity and geographical logic.

    Simple heuristic to penalize extreme coordinate jumps that might indicate
    unrealistic ship movements.

    Parameters:
    - candidates (pd.DataFrame): Candidates with start positions
    - current_segment (pd.Series): Current segment with end position

    Returns:
    - np.ndarray: Array of position scores (0-1 scale, lower is better)
    """
    # Simple heuristic: penalize extreme direction changes
    current_lat = current_segment['end_lat']
    current_lon = current_segment['end_lon']

    scores = []
    for _, candidate in candidates.iterrows():
        # Calculate bearing change (simplified)
        lat_change = candidate['start_lat'] - current_lat
        lon_change = candidate['start_lon'] - current_lon

        # Penalize extreme coordinate jumps
        coord_jump = abs(lat_change) + abs(lon_change)
        score = min(coord_jump / 10.0, 1.0)  # Normalize by 10 degrees
        scores.append(score)

    return np.array(scores)