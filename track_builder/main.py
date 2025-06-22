import math
import pandas as pd
import numpy as np

def build_track_table(dataset):
    """
    Build ship tracking table by connecting ship segments
    """
    # Check if we have the columns we need
    needed_cols = ['shipid', 'date_time_utc', 'lat', 'lon', 'astd_cat', 'flagname', 'iceclass', 'sizegroup_gt']
    for col in needed_cols:
        if col not in dataset.columns:
            print(f"Missing column: {col}")
            return pd.DataFrame(columns=['month', 'segment_id', 'track_id'])
    
    if len(dataset) == 0:
        print("Dataset is empty")
        return pd.DataFrame(columns=['month', 'segment_id', 'track_id'])
    
    print(f"Processing {len(dataset)} records")
    
    # Clean up the data
    data = clean_data(dataset)
    data = add_ship_bucket(data)
    
    # Get start and end points for each ship
    starts, ends = get_start_end_points(data)
    
    print(f"Found {len(starts)} starts and {len(ends)} ends")
    
    # Keep track of which ships belong to which tracks
    ship_tracks = {}
    track_num = 1
    
    # Go through each ship that ends and try to find what starts next
    for i in range(len(ends)):
        end_ship = ends.iloc[i]
        end_id = end_ship['shipid']
        
        # Skip if we already assigned this ship to a track
        if end_id in ship_tracks:
            continue
        
        # Find ships that could be the same ship continuing
        possible_matches = find_matches(end_ship, starts)
        
        if len(possible_matches) > 0:
            # Score each match to find the best one
            scored_matches = score_matches(end_ship, possible_matches)
            best_match = pick_best_match(scored_matches)
            
            if best_match is not None:
                start_id = best_match['shipid']
                
                # If the starting ship already has a track, use that track
                if start_id in ship_tracks:
                    track_id = ship_tracks[start_id]
                else:
                    # Make a new track
                    track_id = f"track_{track_num:03d}"
                    ship_tracks[start_id] = track_id
                    track_num += 1
                
                # Put the ending ship in the same track
                ship_tracks[end_id] = track_id
            else:
                # No good match, make a new track for this ship
                track_id = f"track_{track_num:03d}"
                ship_tracks[end_id] = track_id
                track_num += 1
        else:
            # No matches found, make a new track
            track_id = f"track_{track_num:03d}"
            ship_tracks[end_id] = track_id
            track_num += 1
    
    # Build the final table
    results = []
    for ship_id, track_id in ship_tracks.items():
        ship_data = data[data['shipid'] == ship_id]
        if len(ship_data) > 0:
            month = pd.to_datetime(ship_data['date_time_utc'].iloc[0]).strftime('%Y-%m')
            results.append({
                'month': month,
                'segment_id': ship_id,
                'track_id': track_id
            })
    
    # Handle ships that didn't get matched
    all_ships = set(data['shipid'].unique())
    unmatched_ships = all_ships - set(ship_tracks.keys())
    
    for ship_id in unmatched_ships:
        track_id = f"track_{track_num:03d}"
        ship_data = data[data['shipid'] == ship_id]
        if len(ship_data) > 0:
            month = pd.to_datetime(ship_data['date_time_utc'].iloc[0]).strftime('%Y-%m')
            results.append({
                'month': month,
                'segment_id': ship_id,
                'track_id': track_id
            })
            track_num += 1
    
    print(f"Created {track_num - 1} tracks")
    return pd.DataFrame(results)


def clean_data(df):
    """Clean up the ship data"""
    df = df.copy()
    
    # Make text columns lowercase and clean
    text_cols = ['astd_cat', 'flagname', 'iceclass', 'sizegroup_gt']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
    
    # Fix datetime
    df['date_time_utc'] = pd.to_datetime(df['date_time_utc'])
    
    # Fix coordinates
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    
    # Remove bad coordinates
    before = len(df)
    df = df.dropna(subset=['lat', 'lon'])
    after = len(df)
    if before != after:
        print(f"Removed {before - after} rows with bad coordinates")
    
    return df


def add_ship_bucket(df):
    """Add a bucket column to group similar ships"""
    df = df.copy()
    
    # Combine ship characteristics into one string
    df['bucket'] = (df['astd_cat'].astype(str) + '_' + 
                   df['flagname'].astype(str) + '_' + 
                   df['iceclass'].astype(str) + '_' + 
                   df['sizegroup_gt'].astype(str))
    
    return df


def distance_between_points(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS points in km"""
    R = 6371  # Earth radius in km
    
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = (math.sin(dlat/2)**2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def time_diff_hours(time1, time2):
    """Get time difference in hours"""
    diff = abs(time2 - time1)
    return diff.total_seconds() / 3600


def get_start_end_points(df):
    """Find first and last points for each ship"""
    df = df.copy()
    df['date'] = df['date_time_utc'].dt.date
    
    # Find first and last dates for each ship
    date_ranges = df.groupby('shipid')['date'].agg(['min', 'max']).reset_index()
    date_ranges.columns = ['shipid', 'first_date', 'last_date']
    
    # Merge back
    df_with_dates = df.merge(date_ranges, on='shipid')
    
    # Get start points (first day)
    starts = df_with_dates[df_with_dates['date'] == df_with_dates['first_date']].copy()
    starts = starts.sort_values(['shipid', 'date_time_utc']).groupby('shipid').first().reset_index()
    
    # Get end points (last day)
    ends = df_with_dates[df_with_dates['date'] == df_with_dates['last_date']].copy()
    ends = ends.sort_values(['shipid', 'date_time_utc']).groupby('shipid').last().reset_index()
    
    # Clean up extra columns
    starts = starts.drop(columns=['date', 'first_date', 'last_date'])
    ends = ends.drop(columns=['date', 'first_date', 'last_date'])
    
    return starts, ends


def find_matches(end_ship, starts_df):
    """Find possible matches for an ending ship"""
    max_time = 12  # hours
    max_distance = 100  # km
    
    # Only look at ships with same characteristics
    bucket = end_ship['bucket']
    candidates = starts_df[starts_df['bucket'] == bucket].copy()
    
    if len(candidates) == 0:
        return pd.DataFrame()
    
    # Calculate time and distance for each candidate
    end_time = end_ship['date_time_utc']
    end_lat = end_ship['lat']
    end_lon = end_ship['lon']
    
    # Only consider ships that start AFTER this one ends
    candidates = candidates[candidates['date_time_utc'] > end_time]
    
    if len(candidates) == 0:
        return pd.DataFrame()
    
    candidates['time_diff'] = candidates['date_time_utc'].apply(
        lambda x: time_diff_hours(end_time, x)
    )
    
    candidates['distance'] = candidates.apply(
        lambda row: distance_between_points(end_lat, end_lon, row['lat'], row['lon']), 
        axis=1
    )
    
    # Filter by thresholds
    good_candidates = candidates[
        (candidates['time_diff'] <= max_time) & 
        (candidates['distance'] <= max_distance)
    ]
    
    return good_candidates


def score_matches(end_ship, candidates):
    """Score each candidate match"""
    if len(candidates) == 0:
        return candidates
    
    candidates = candidates.copy()
    
    # Calculate implied speed
    candidates['speed'] = candidates.apply(
        lambda row: row['distance'] / row['time_diff'] if row['time_diff'] > 0 else 999,
        axis=1
    )
    
    # Score each candidate (lower is better)
    scores = []
    for _, candidate in candidates.iterrows():
        score = calculate_score(candidate, end_ship['astd_cat'])
        scores.append(score)
    
    candidates['score'] = scores
    return candidates


def calculate_score(candidate, ship_type):
    """Calculate match score (lower is better)"""
    distance = candidate['distance']
    time_diff = candidate['time_diff']
    speed = candidate['speed']
    
    # Expected speeds for different ship types (km/h)
    speed_ranges = {
        "fishing vessels": (5, 15),
        "passenger ships": (20, 40),
        "oil product tankers": (15, 25),
        "other activities": (10, 20),
        "general cargo ships": (15, 25),
        "ro-ro cargo ships": (20, 30),
        "cruise ships": (25, 40),
        "refrigerated cargo ships": (15, 25),
        "chemical tankers": (15, 25),
        "bulk carriers": (12, 20),
        "other service offshore vessels": (5, 15),
        "offshore supply ships": (5, 12),
        "crude oil tankers": (14, 24),
        "container ships": (20, 35),
        "gas tankers": (16, 26),
    }
    
    # Get expected speed range
    if ship_type in speed_ranges:
        min_speed, max_speed = speed_ranges[ship_type]
    else:
        min_speed, max_speed = (10, 25)  # default
    
    avg_speed = (min_speed + max_speed) / 2
    
    # Normalize scores (0-1, where 1 is worst)
    distance_score = min(distance / 100, 1.0)  # normalize by max distance
    time_score = min(time_diff / 12, 1.0)      # normalize by max time
    speed_penalty = abs(speed - avg_speed) / (max_speed - min_speed)
    speed_score = min(speed_penalty, 1.0)
    
    # Combine scores (weighted average)
    total_score = (0.4 * distance_score + 0.4 * speed_score + 0.2 * time_score)
    
    return total_score


def pick_best_match(candidates):
    """Pick the best match from scored candidates"""
    if len(candidates) == 0:
        return None
    
    # Remove invalid matches (infinite scores)
    good_candidates = candidates[candidates['score'] != float('inf')]
    
    if len(good_candidates) == 0:
        return None
    
    # Sort by score and pick the best one
    best_match = good_candidates.sort_values('score').iloc[0]
    return best_match






    




