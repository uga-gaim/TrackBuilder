import numpy as np
import pandas as pd


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
    grp: points d'UN ship_id sur UN jour (triés par date_time_utc).
    Retourne une série de vitesses (km/h) entre points consécutifs, indexée à partir du 2e point.
    """
    # Cas 1: colonnes FTP (dist depuis point précédent et secondes depuis point précédent)
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
    Calcule des vitesses 'typiques' par astd_cat, en 3 niveaux:
      (1) par ship_id & jour: vitesses entre points consécutifs (échantillon aléatoire n_per_day)
      (2) par ship_id: moyenne des vitesses > 0
      (3) par astd_cat: médiane des moyennes ship_id

    Colonnes requises: ship_id/shipid, date_time_utc, latitude, longitude, astd_cat.
    Utilise dist_nextpoint/sec_nextpoint si disponibles (exports FTP).
    """
    # Harmonise ship_id
    if 'shipid' in df.columns and 'ship_id' not in df.columns:
        df = df.rename(columns={'shipid': 'ship_id'}).copy()

    needed = {'ship_id', 'date_time_utc', 'latitude', 'longitude', 'astd_cat'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(list(missing))}")

    work = df.copy()
    work['date_time_utc'] = pd.to_datetime(work['date_time_utc'])
    work['date'] = work['date_time_utc'].dt.date
    work = work.sort_values(['ship_id', 'date', 'date_time_utc'])

    rows = []
    for (sid, d), grp in work.groupby(['ship_id', 'date'], sort=False):
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
        rows.append(pd.DataFrame({'ship_id': sid, 'date': d, 'speed_kmh': v.values}))

    if not rows:
        return pd.DataFrame(columns=['astd_cat', 'typical_speed_kmh', 'n_ships_used'])

    speeds = pd.concat(rows, ignore_index=True)
    # Récupère la catégorie (on prend la première valeur rencontrée pour ce ship)
    ship_cat = work.drop_duplicates('ship_id')[['ship_id', 'astd_cat']]
    speeds = speeds.merge(ship_cat, on='ship_id', how='left')

    # Moyenne par ship_id  filtre sur le nb d'échantillons gardés
    ship_means = (speeds.groupby('ship_id', as_index=False)
                  .agg(mean_speed_kmh=('speed_kmh', 'mean'),
                       n_samples=('speed_kmh', 'size')))
    ship_means = ship_means[ship_means['n_samples'] >= min_points_per_ship]
    ship_means = ship_means.merge(ship_cat, on='ship_id', how='left')

    # Agrégation finale: médiane robuste par type
    out = (ship_means.groupby('astd_cat')
           .agg(typical_speed_kmh=('mean_speed_kmh', 'median'),
                n_ships_used=('ship_id', 'nunique'))
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
        'ship_id', 'date_time_utc', 'astd_cat', 
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
    grouped = work.groupby(['ship_id', 'date'])

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
        
        rows.append(pd.DataFrame({'ship_id': sid, 'speed_kmh': v.values}))

    if not rows:
        return pd.DataFrame(columns=['astd_cat', 'speed_kmh'])

    speeds_df = pd.concat(rows, ignore_index=True)
    
    # Get the category for each ship_id
    # (Using the first-seen category for each ship)
    ship_cat_map = work.drop_duplicates('ship_id', keep='first').set_index('ship_id')['astd_cat']
    speeds_df['astd_cat'] = speeds_df['ship_id'].map(ship_cat_map)

    return speeds_df[['astd_cat', 'speed_kmh']].dropna()