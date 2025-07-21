import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def build_track_table(dataset):
    """
    Build ship tracking table by connecting ship segments across time periods.
    Works with any data density - daily, weekly, monthly, or custom periods.
    """
    # Check if we have the columns we need
    needed_cols = ['shipid', 'date_time_utc', 'latitude', 'longitude', 'astd_cat', 'flagname', 'iceclass', 'sizegroup_gt']
    for col in needed_cols:
        if col not in dataset.columns:
            print(f"Missing column: {col}")
            print(f"Available columns: {list(dataset.columns)}")
            return pd.DataFrame(columns=['month', 'segment_id', 'track_id'])
    
    if len(dataset) == 0:
        print("Dataset is empty")
        return pd.DataFrame(columns=['month', 'segment_id', 'track_id'])
    
    print(f"Processing {len(dataset)} records")
    
    # Clean up the data
    data = clean_data(dataset)
    
    # Get segments (each shipid represents a time period for a ship)
    segments = get_segment_summaries(data)
    
    print(f"Found {len(segments)} segments across {segments['month'].nunique()} time periods")
    
    # Build tracks by connecting segments chronologically with improved logic
    tracks = build_tracks_improved(segments)
    
    # Create the final output table
    results = []
    for track_id, segment_ids in tracks.items():
        for segment_id in segment_ids:
            segment_info = segments[segments['shipid'] == segment_id].iloc[0]
            results.append({
                'month': segment_info['month'],
                'segment_id': segment_id,
                'track_id': track_id
            })
    
    print(f"Created {len(tracks)} tracks")
    return pd.DataFrame(results)

def build_tracks_improved(segments):
    """
    Improved track building with stricter continuity requirements and better scoring
    """
    tracks = {}
    track_num = 1
    unassigned = set(segments['shipid'].tolist())
    
    # Sort segments by start time to process chronologically
    segments_sorted = segments.sort_values('start_time').reset_index(drop=True)
    
    for _, segment in segments_sorted.iterrows():
        if segment['shipid'] not in unassigned:
            continue  # Already assigned to a track
        
        # Start a new track with this segment
        track_id = f"track_{track_num:03d}"
        current_track = [segment['shipid']]
        unassigned.remove(segment['shipid'])
        
        # Keep looking for subsequent segments that could be the same ship
        current_segment = segment
        max_consecutive_gaps = 1  # Allow at most 1 month gap
        consecutive_gaps = 0
        
        while True:
            next_segment = find_next_segment_improved(current_segment, segments_sorted, unassigned)
            if next_segment is not None:
                # Check for temporal continuity
                months_gap = calculate_month_gap(current_segment['month'], next_segment['month'])
                
                if months_gap <= 2:  # Max 2 month gap (current + 1 gap + next)
                    current_track.append(next_segment['shipid'])
                    unassigned.remove(next_segment['shipid'])
                    current_segment = next_segment
                    consecutive_gaps = 0 if months_gap == 1 else consecutive_gaps + 1
                else:
                    # Gap too large, stop this track
                    break
                    
                # Stop if we've had too many gaps
                if consecutive_gaps > max_consecutive_gaps:
                    break
            else:
                break  # No more segments to connect
        
        tracks[track_id] = current_track
        track_num += 1
    
    return tracks

def calculate_month_gap(month1, month2):
    """Calculate the number of months between two month strings (YYYY-MM format)"""
    try:
        date1 = datetime.strptime(month1, '%Y-%m')
        date2 = datetime.strptime(month2, '%Y-%m')
        
        # Calculate difference in months
        month_diff = (date2.year - date1.year) * 12 + (date2.month - date1.month)
        return month_diff
    except:
        return float('inf')  # If parsing fails, treat as infinite gap

def find_next_segment_improved(current_segment, all_segments, unassigned):
    """
    Improved segment matching with stricter criteria and better scoring
    """
    ship_type = current_segment['astd_cat'].lower()
    
    # Find candidates: same ship characteristics, starts after current ends, unassigned
    candidates = all_segments[
        (all_segments['shipid'].isin(unassigned)) &
        (all_segments['ship_signature'] == current_segment['ship_signature']) &
        (all_segments['start_time'] > current_segment['end_time'])
    ].copy()
    
    if len(candidates) == 0:
        return None
    
    # Calculate metrics for each candidate
    candidates = calculate_candidate_metrics(current_segment, candidates)
    
    # Apply stricter filtering
    valid_candidates = apply_improved_filters(candidates, ship_type, current_segment)
    
    if len(valid_candidates) == 0:
        return None
    
    # Score candidates with improved scoring system
    valid_candidates = valid_candidates.copy()
    valid_candidates['match_score'] = calculate_improved_match_score(valid_candidates, ship_type, current_segment)
    
    # Only accept candidates with good scores (lower threshold)
    score_threshold = 0.4  # Reject candidates with scores above 0.4
    acceptable_candidates = valid_candidates[valid_candidates['match_score'] <= score_threshold]
    
    if len(acceptable_candidates) == 0:
        return None
    
    # Return the best match
    best_match = acceptable_candidates.sort_values('match_score').iloc[0]
    return best_match

def calculate_candidate_metrics(current_segment, candidates):
    """Calculate all metrics needed for candidate evaluation"""
    current_end_time = current_segment['end_time']
    current_end_lat = current_segment['end_lat']
    current_end_lon = current_segment['end_lon']
    
    candidates['time_gap_hours'] = candidates['start_time'].apply(
        lambda x: (x - current_end_time).total_seconds() / 3600
    )
    
    candidates['distance_km'] = candidates.apply(
        lambda row: distance_between_points(
            current_end_lat, current_end_lon,
            row['start_lat'], row['start_lon']
        ), axis=1
    )
    
    # Calculate implied speed (km/h)
    candidates['implied_speed'] = candidates.apply(
        lambda row: row['distance_km'] / row['time_gap_hours'] if row['time_gap_hours'] > 0 else float('inf'),
        axis=1
    )
    
    # Calculate month gap
    candidates['month_gap'] = candidates.apply(
        lambda row: calculate_month_gap(current_segment['month'], row['month']),
        axis=1
    )
    
    return candidates

def apply_improved_filters(candidates, ship_type, current_segment):
    """Apply improved filtering with stricter constraints"""
    
    # Stricter ship type constraints: (max_speed_kmh, max_time_gap_hours, max_distance_km)
    ship_constraints = {
        'unknown': (30, 72, 1000),  # Reduced from (50, 168, 2000)
        'fishing vessels': (25, 48, 600),  # Reduced from (30, 72, 800)
        'passenger ships': (45, 36, 800),  # Reduced from (60, 48, 1200)
        'oil product tankers': (30, 96, 1500),  # Reduced from (40, 168, 2500)
        'other activities': (25, 72, 800),  # Reduced from (40, 168, 1500)
        'general cargo ships': (25, 96, 1200),  # Reduced from (35, 168, 2000)
        'ro-ro cargo ships': (35, 48, 1000),  # Reduced from (45, 72, 1500)
        'cruise ships': (40, 36, 700),  # Reduced from (50, 48, 1000)
        'refrigerated cargo ships': (25, 96, 1200),  # Reduced from (35, 168, 2000)
        'chemical tankers': (30, 96, 1400),  # Reduced from (40, 168, 2200)
        'bulk carriers': (22, 96, 1200),  # Reduced from (30, 168, 2000)
        'other service offshore vessels': (20, 48, 400),  # Reduced from (25, 72, 600)
        'offshore supply ships': (20, 48, 400),  # Reduced from (25, 72, 600)
        'crude oil tankers': (25, 96, 1500),  # Reduced from (35, 168, 2500)
        'container ships': (35, 96, 2000),  # Reduced from (50, 168, 3000)
        'gas tankers': (30, 96, 1400),  # Reduced from (40, 168, 2200)
    }
    
    # Get constraints for this ship type
    max_speed, max_time_gap, max_distance = ship_constraints.get(
        ship_type, ship_constraints['general cargo ships']
    )
    
    # Apply basic constraints
    valid = candidates[
        (candidates['time_gap_hours'] <= max_time_gap) &
        (candidates['distance_km'] <= max_distance) &
        (candidates['implied_speed'] <= max_speed) &
        (candidates['month_gap'] <= 2)  # Max 2 month gap
    ]
    
    # Additional logic-based filters
    
    # 1. Penalize very long time gaps even if within limits
    if len(valid) > 1:
        # Prefer candidates with shorter time gaps
        median_gap = valid['time_gap_hours'].median()
        valid = valid[valid['time_gap_hours'] <= median_gap * 1.5]
    
    # 2. Reject unrealistic speeds (too slow or too fast for extended periods)
    if len(valid) > 1:
        valid = valid[
            (valid['implied_speed'] >= 2) &  # Minimum 2 km/h (not stationary)
            (valid['implied_speed'] <= max_speed * 0.8)  # 80% of max speed
        ]
    
    # 3. Continuity check - prefer next immediate month when possible
    if len(valid) > 1:
        immediate_next = valid[valid['month_gap'] == 1]
        if len(immediate_next) > 0:
            valid = immediate_next
    
    return valid

def calculate_improved_match_score(candidates, ship_type, current_segment):
    """
    Improved scoring system that heavily penalizes unrealistic scenarios
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
    time_score = np.minimum(candidates['time_gap_hours'] / 72, 1.0)     # Normalize by 3 days (stricter)
    
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
        0.3 * distance_score +      # Distance traveled
        0.2 * speed_score +         # Speed reasonableness  
        0.2 * time_score +          # Time gap
        0.2 * month_score +         # Month continuity
        0.1 * position_score        # Position logic
    )
    
    return total_score

def calculate_position_continuity_score(candidates, current_segment):
    """
    Calculate a score based on position continuity and geographical logic
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

def get_top_match_candidates(dataset, segment_id, top_n=3):
    """
    Get top N candidate matches for a given segment based on match scores.
    
    Parameters:
    - dataset: The ship tracking dataset
    - segment_id: The shipid to find candidates for
    - top_n: Number of top candidates to return (default: 3)
    
    Returns:
    - DataFrame with columns: segment_id, month, track_id, ranking
    """
    if len(dataset) == 0:
        print("Dataset is empty")
        return pd.DataFrame(columns=['segment_id', 'month', 'track_id', 'ranking'])
    
    # Clean up the data
    data = clean_data(dataset)
    
    # Get segments
    segments = get_segment_summaries(data)
    
    # Find the target segment
    target_segment = segments[segments['shipid'] == segment_id]
    if len(target_segment) == 0:
        print(f"Segment {segment_id} not found in dataset")
        return pd.DataFrame(columns=['segment_id', 'month', 'track_id', 'ranking'])
    
    target_segment = target_segment.iloc[0]
    
    # Find all potential candidates using scoring system
    candidates = get_scored_candidates(target_segment, segments)
    
    if len(candidates) == 0:
        print(f"No candidates found for segment {segment_id}")
        return pd.DataFrame(columns=['segment_id', 'month', 'track_id', 'ranking'])
    
    # Get top N candidates
    top_candidates = candidates.head(min(top_n, len(candidates)))
    
    # Create result DataFrame with only requested columns
    result_df = pd.DataFrame({
        'segment_id': top_candidates['shipid'].values,
        'month': top_candidates['month'].values,
        'track_id': None,  # Will be filled if we have track info
        'ranking': range(1, len(top_candidates) + 1)
    })
    
    print(f"Found {len(result_df)} candidates for segment {segment_id}")
    return result_df.reset_index(drop=True)

def get_scored_candidates(current_segment, all_segments):
    """
    Find all valid candidates for a given segment and return them sorted by match score.
    Uses the improved scoring system.
    """
    ship_type = current_segment['astd_cat'].lower()
    
    # Find candidates: same ship characteristics, starts after current ends
    candidates = all_segments[
        (all_segments['shipid'] != current_segment['shipid']) &  # Exclude self
        (all_segments['ship_signature'] == current_segment['ship_signature']) &
        (all_segments['start_time'] > current_segment['end_time'])
    ].copy()
    
    if len(candidates) == 0:
        return pd.DataFrame()
    
    # Calculate metrics
    candidates = calculate_candidate_metrics(current_segment, candidates)
    
    # Filter candidates based on improved constraints
    valid_candidates = apply_improved_filters(candidates, ship_type, current_segment)
    
    if len(valid_candidates) == 0:
        return pd.DataFrame()
    
    # Score candidates with improved scoring
    valid_candidates = valid_candidates.copy()
    valid_candidates['match_score'] = calculate_improved_match_score(valid_candidates, ship_type, current_segment)
    
    # Return all candidates sorted by match score (best first)
    return valid_candidates.sort_values('match_score').reset_index(drop=True)

def get_segment_summaries(df):
    """Get summary information for each ship segment (shipid)"""
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

def create_ship_signature(ship_row):
    """Create a unique signature for ship characteristics"""
    return f"{ship_row['astd_cat']}|{ship_row['flagname']}|{ship_row['iceclass']}|{ship_row['sizegroup_gt']}"

def build_tracks_flexible(segments):
    """Build tracks by connecting segments that represent the same physical ship"""
    tracks = {}
    track_num = 1
    unassigned = set(segments['shipid'].tolist())
    
    # Sort segments by start time to process chronologically
    segments_sorted = segments.sort_values('start_time').reset_index(drop=True)
    
    for _, segment in segments_sorted.iterrows():
        if segment['shipid'] not in unassigned:
            continue  # Already assigned to a track
        
        # Start a new track with this segment
        track_id = f"track_{track_num:03d}"
        current_track = [segment['shipid']]
        unassigned.remove(segment['shipid'])
        
        # Keep looking for subsequent segments that could be the same ship
        current_segment = segment
        while True:
            next_segment = find_next_segment(current_segment, segments_sorted, unassigned)
            if next_segment is not None:
                current_track.append(next_segment['shipid'])
                unassigned.remove(next_segment['shipid'])
                current_segment = next_segment
            else:
                break  # No more segments to connect
        
        tracks[track_id] = current_track
        track_num += 1
    
    return tracks

def find_next_segment(current_segment, all_segments, unassigned):
    """Find the next segment that could be the same physical ship"""
    # Dynamic thresholds based on ship type and time gap
    ship_type = current_segment['astd_cat'].lower()
    
    # Find candidates: same ship characteristics, starts after current ends, unassigned
    candidates = all_segments[
        (all_segments['shipid'].isin(unassigned)) &
        (all_segments['ship_signature'] == current_segment['ship_signature']) &
        (all_segments['start_time'] > current_segment['end_time'])
    ].copy()
    
    if len(candidates) == 0:
        return None
    
    # Calculate time gap and distance for each candidate
    current_end_time = current_segment['end_time']
    current_end_lat = current_segment['end_lat']
    current_end_lon = current_segment['end_lon']
    
    candidates['time_gap_hours'] = candidates['start_time'].apply(
        lambda x: (x - current_end_time).total_seconds() / 3600
    )
    
    candidates['distance_km'] = candidates.apply(
        lambda row: distance_between_points(
            current_end_lat, current_end_lon,
            row['start_lat'], row['start_lon']
        ), axis=1
    )
    
    # Calculate implied speed (km/h)
    candidates['implied_speed'] = candidates.apply(
        lambda row: row['distance_km'] / row['time_gap_hours'] if row['time_gap_hours'] > 0 else float('inf'),
        axis=1
    )
    
    # Filter candidates based on realistic constraints
    valid_candidates = filter_realistic_candidates(candidates, ship_type)
    
    if len(valid_candidates) == 0:
        return None
    
    # Score candidates (lower score = better match)
    valid_candidates = valid_candidates.copy()
    valid_candidates['match_score'] = calculate_match_score(valid_candidates, ship_type)
    
    # Return the best match
    best_match = valid_candidates.sort_values('match_score').iloc[0]
    return best_match

def filter_realistic_candidates(candidates, ship_type):
    """Filter candidates based on realistic constraints for the ship type"""
    
    # Ship type constraints: (max_speed_kmh, max_time_gap_hours, max_distance_km)
    ship_constraints = {
        'unknown': (50, 168, 2000),  # 1 week max gap
        'fishing vessels': (30, 72, 800),  # 3 days max gap
        'passenger ships': (60, 48, 1200),  # 2 days max gap
        'oil product tankers': (40, 168, 2500),  # 1 week max gap
        'other activities': (40, 168, 1500),  # 1 week max gap
        'general cargo ships': (35, 168, 2000),  # 1 week max gap
        'ro-ro cargo ships': (45, 72, 1500),  # 3 days max gap
        'cruise ships': (50, 48, 1000),  # 2 days max gap
        'refrigerated cargo ships': (35, 168, 2000),  # 1 week max gap
        'chemical tankers': (40, 168, 2200),  # 1 week max gap
        'bulk carriers': (30, 168, 2000),  # 1 week max gap
        'other service offshore vessels': (25, 72, 600),  # 3 days max gap
        'offshore supply ships': (25, 72, 600),  # 3 days max gap
        'crude oil tankers': (35, 168, 2500),  # 1 week max gap
        'container ships': (50, 168, 3000),  # 1 week max gap
        'gas tankers': (40, 168, 2200),  # 1 week max gap
    }
    
    # Get constraints for this ship type (default to general cargo if unknown)
    max_speed, max_time_gap, max_distance = ship_constraints.get(
        ship_type, ship_constraints['general cargo ships']
    )
    
    # Filter based on constraints
    valid = candidates[
        (candidates['time_gap_hours'] <= max_time_gap) &
        (candidates['distance_km'] <= max_distance) &
        (candidates['implied_speed'] <= max_speed)
    ]
    
    return valid

def calculate_match_score(candidates, ship_type):
    """Calculate match score for candidates (lower = better)"""
    
    # Expected typical speeds for ship types (km/h)
    typical_speeds = {
        'unknown': 15,
        'fishing vessels': 10,
        'passenger ships': 35,
        'oil product tankers': 20,
        'other activities': 15,
        'general cargo ships': 20,
        'ro-ro cargo ships': 25,
        'cruise ships': 35,
        'refrigerated cargo ships': 20,
        'chemical tankers': 20,
        'bulk carriers': 16,
        'other service offshore vessels': 12,
        'offshore supply ships': 10,
        'crude oil tankers': 18,
        'container ships': 30,
        'gas tankers': 22,
    }
    
    typical_speed = typical_speeds.get(ship_type, 20)
    
    # Calculate score components (normalized 0-1)
    distance_score = np.minimum(candidates['distance_km'] / 1000, 1.0)  # Normalize by 1000km
    time_score = np.minimum(candidates['time_gap_hours'] / 168, 1.0)     # Normalize by 1 week
    
    # Speed deviation score
    speed_deviation = np.abs(candidates['implied_speed'] - typical_speed) / typical_speed
    speed_score = np.minimum(speed_deviation, 1.0)
    
    # Weighted combination (prioritize distance and speed reasonableness)
    total_score = (0.5 * distance_score + 0.3 * speed_score + 0.2 * time_score)
    
    return total_score

def clean_data(df):
    """Clean up the ship data"""
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