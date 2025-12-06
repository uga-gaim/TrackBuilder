import numpy as np
import pandas as pd
from track_builder.config import _LIT_CAPS_KMH


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

    print(f"Data after cleaning:")
    print(f"  Date range: {df['date_time_utc'].min()} to {df['date_time_utc'].max()}")
    print(f"  Ship types: {df['astd_cat'].unique()}")
    print(f"  Unique ships: {df['shipid'].nunique()}")

    return df

def remove_unrealistic_points(
    astd_data: pd.DataFrame,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Remove unrealistic AIS points by checking whether consecutive position fixes are physically reachable
    given ship-category typical speeds and literature caps.
    This function performs a vectorized cleanup of an input AIS-like DataFrame by:
    - ensuring date_time_utc is a timezone-aware datetime and sorting by shipid and time;
    - deriving a per-row speed limit (km/h) from category typical speeds multiplied by speed_margin,
        capped by literature maxima (or defaulting to 80 km/h when unknown);
    - computing forward neighbor distances (Haversine, km) and elapsed time (hours) per ship;
    - marking a forward link valid if the observed distance <= speed_limit * elapsed_time;
    - propagating that validity backward to the previous point on the same ship;
    - keeping only points that participate in at least one valid forward or backward link;
    - returning a cleaned copy with temporary columns removed.
    Args:
            astd_data (pandas.DataFrame): Input table of AIS/spatial points. Required columns:
                    - 'shipid' (identifier grouping successive observations by vessel)
                    - 'date_time_utc' (datetime-like, will be coerced to timezone-aware if needed)
                    - 'latitude', 'longitude' (decimal degrees)
                    - 'astd_cat' (category used to look up typical speeds)
            speed_margin (float, optional): Multiplicative margin applied to the typical speed
                    (typical_speed_kmh * speed_margin) to produce the working speed limit.
                    Defaults to 1.5.
    Returns:
            pandas.DataFrame: A filtered copy of the input containing only points that belong to
            at least one physically plausible movement link (forward or backward). Temporary
            helper columns are removed.
    Notes:
            - Units: distances are computed in kilometers, speeds in km/h, elapsed times in hours.
            - The function relies on external helpers/values:
                    - compute_typical_speeds_by_astd_cat(df) -> DataFrame with columns ['astd_cat', 'typical_speed_kmh']
                    - _LIT_CAPS_KMH: mapping of normalized category -> literature speed cap (km/h)
                    - haversine_km(lat1, lon1, lat2, lon2) -> distance in km
            - If typical speed and literature cap are both missing for a category, a default cap of 80 km/h is used.
            - Points with zero elapsed time between identical timestamps are treated as invalid if a non-zero distance is observed.
            - The function prints progress/summary messages and returns an empty or original-like DataFrame
                when input is None or empty.
    Raises:
            KeyError/TypeError: May be raised if required columns are missing or have incompatible types.
    astd_data: pd.DataFrame,
    """
    if astd_data is None or len(astd_data) == 0:
        return astd_data

    df = astd_data.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df['date_time_utc']):
        df['date_time_utc'] = pd.to_datetime(df['date_time_utc'], utc=True)
        
    df = df.sort_values(['shipid', 'date_time_utc'])

    # define the limit speed
    print("computing typical speeds...")
    typ = compute_typical_speeds_by_astd_cat(df)
    
    # Mapping typical speeds
    df['temp_cat'] = df['astd_cat'].astype(str).str.lower().str.strip()
    typ['astd_cat_norm'] = typ['astd_cat'].astype(str).str.lower().str.strip()
    typical_lookup = dict(zip(typ['astd_cat_norm'], typ['typical_speed_kmh']))
    
    typical_val = df['temp_cat'].map(typical_lookup)
    lit_cap_val = df['temp_cat'].map(_LIT_CAPS_KMH)
    
    # Default to 80 km/h if unknown
    limit_series = typical_val * multiplier
    limit_series = np.where(
        typical_val.notna() & lit_cap_val.notna(),
        np.minimum(limit_series, lit_cap_val),
        np.where(lit_cap_val.notna(), lit_cap_val, 80.0)
    )
    df['speed_limit'] = limit_series

    # compute neighbors
    
    g = df.groupby('shipid')
    next_lat = g['latitude'].shift(-1)
    next_lon = g['longitude'].shift(-1)
    next_time = g['date_time_utc'].shift(-1)
    
    # Real distance (Haversine)
    dist_fwd = haversine_km(df['latitude'], df['longitude'], next_lat, next_lon)
    
    # Elapsed time (in hours)
    time_fwd_h = (next_time - df['date_time_utc']).dt.total_seconds() / 3600.0
    
   
    # What distance could this ship physically have covered during this time?
    # Ex: if Limit=30km/h and Time=2h -> Max=60km. If Time=0h -> Max=0km.
    max_possible_dist = df['speed_limit'] * time_fwd_h
    
    # Validation: Did the ship travel less than its theoretical maximum?
    # Note: If time_fwd_h = 0 and dist > 0 (instant teleportation), this returns False. Correct.
    valid_fwd = (dist_fwd <= max_possible_dist)
    valid_fwd = valid_fwd.fillna(False)


    # Same logic: if the link (i -> i+1) is valid, then the link (i+1 -> i) is also valid.
    # We use vectorized shift(1) and check that we remain on the same ship
    is_same_ship_prev = (df['shipid'] == df['shipid'].shift(1))
    valid_bwd = valid_fwd.shift(1).fillna(False) & is_same_ship_prev

    
    # A point is kept if it has at least one logical link (before or after)
    mask_keep = valid_fwd | valid_bwd

    cleaned = df[mask_keep].copy()
    
    # Final cleaning
    cols_to_drop = ['speed_limit', 'temp_cat']
    cleaned = cleaned.drop(columns=[c for c in cols_to_drop if c in cleaned.columns])
    
    n_dropped = len(df) - len(cleaned)
    if n_dropped > 0:
        print(f"Cleaning completed: {n_dropped} 'ghost' or aberrant points removed.")

    return cleaned

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
    # Work on a copy to avoid SettingWithCopy warnings
    df = df.copy()

    # 1. Ensure datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['date_time_utc']):
        df['date_time_utc'] = pd.to_datetime(df['date_time_utc'], utc=True)

    # 2. Create a period column for monthly grouping
    # This ensures Jan data is separated from Feb data for the same ship
    df['period_month'] = df['date_time_utc'].dt.to_period('M')

    segments = []

    # 3. Group by ShipID AND Month
    grouped = df.groupby(['shipid', 'period_month'])

    print(f"Creating segments for {len(grouped)} unique ship-months (segments)...")

    # 4. Iterate through groups
    for (ship_id, period), group in grouped:
        # Sort is essential to identify the true start and end of the segment
        group = group.sort_values('date_time_utc')
        
        if len(group) == 0:
            continue

        # Extract boundaries (first and last row of the month)
        start_row = group.iloc[0]
        end_row = group.iloc[-1]
        
        # Format month as string "YYYY-MM" for compatibility
        month_str = str(period)

        segment = {
            'shipid': ship_id,
            'month': month_str,
            'start_time': start_row['date_time_utc'],
            'end_time': end_row['date_time_utc'],
            'start_lat': start_row['latitude'],
            'start_lon': start_row['longitude'],
            'end_lat': end_row['latitude'],
            'end_lon': end_row['longitude'],
            # Safe metadata retrieval
            'astd_cat': start_row.get('astd_cat', 'unknown'),
            'flagname': start_row.get('flagname', 'unknown'),
            'iceclass': start_row.get('iceclass', 'unknown'),
            'sizegroup_gt': start_row.get('sizegroup_gt', 'unknown'),
            # Signature for matching logic
            'ship_signature': create_ship_signature(start_row)
        }
        segments.append(segment)

    # 5. Create final DataFrame
    result_df = pd.DataFrame(segments)
    
    print(f"Successfully created {len(result_df)} monthly segments.")
    if len(result_df) > 0:
        print(f"Sample segment: {result_df.iloc[0]['ship_signature']} in {result_df.iloc[0]['month']}")

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
    typical_speeds = _LIT_CAPS_KMH

    typical_speed = typical_speeds.get(ship_type, 15)

    # Normalize components (0-1 scale, lower is better)
    distance_score = np.minimum(candidates['distance_km'] / 500, 1.0)  # Normalize by 500km (stricter)
    time_score = np.minimum(candidates['time_gap_hours'] / 72, 1.0)  # Normalize by 3 days (stricter)

    # Speed deviation score (heavily penalize unrealistic speeds)
    speed_deviation = np.abs(candidates['implied_speed'] - typical_speed) / typical_speed
    speed_score = np.minimum(speed_deviation, 2.0)  # Allow up to 200% deviation

    # Month continuity score (heavily favor consecutive months)
    month_score = candidates['month_gap'].apply(lambda x: 0.0 if x == 1 else 0.5 if x == 2 else 1.0)

    # Weighted combination - prioritize continuity and realistic movement
    total_score = (
            0.3 * distance_score +  # Distance traveled
            0.2 * speed_score +  # Speed reasonableness
            0.2 * time_score +  # Time gap
            0.3 * month_score  # Month continuity
    )

    return total_score


def _norm_str(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    return s.lower()


def filter_attr_consistency_tolerant(
        cand_df,
        cur_row,
        *,
        attrs=("flagname", "iceclass", "astd_cat", "sizegroup_gt"),
        log=None
):
    """
    Tolerant of missing fields:
      - If the current OR candidate value is missing for a given attribute -> do NOT filter on that attribute.
      - Otherwise, require strict equality (case- and whitespace-insensitive).
    """
    df = cand_df
    for col in attrs:
        if col not in df.columns:
            continue

        cur_v = _norm_str(cur_row.get(col, None))
        # if cur_v is missing, do not filter on this attribute
        if cur_v is None:
            continue

        # normalize candidate values
        cand_v = df[col].map(_norm_str)

        # only filter where candidate value is known
        known_mask = cand_v.notna()
        if known_mask.any():
            match_mask = (cand_v == cur_v) | (~known_mask)  # tolerate unknowns on candidate side
            if log is not None:
                for _, r in df.loc[known_mask & ~match_mask].iterrows():
                    log(r, "filter", f"{col}_mismatch")
            df = df.loc[match_mask]
            if df.empty:
                return df
        # if all candidate values are missing -> nothing to do for this attribute

    return df
