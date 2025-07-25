"""
Ship Tracking and Trajectory Analysis Module

This module provides functionality for analyzing ship movement data, connecting ship segments
across time periods, and building continuous ship tracks. It includes methods for finding
candidate matches between ship segments, scoring potential connections, and debugging
the matching process.

The main workflow is:
1. Clean and preprocess ship tracking data
2. Create segment summaries for each ship appearance
3. Find candidate matches between segments using various constraints
4. Score and rank candidates based on multiple factors
5. Build continuous tracks by connecting related segments

Key concepts:
- Segment: A ship's appearance in a specific time period (identified by shipid)
- Track: A sequence of connected segments representing the same physical ship over time
- Ship signature: A combination of ship characteristics used for matching (type, flag, etc.)

Author: William Ponczak (ponczawm@dukes.jmu.edu)
"""

import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def get_top_match_candidates(dataset: pd.DataFrame, segment_id: str, top_n: int = 3) -> pd.DataFrame:
    """
    Get top N candidate matches for a given segment based on match scores.
    
    This is the main entry point for finding which other ship segments could potentially
    be the same physical ship as the input segment. Uses improved scoring algorithms
    to rank candidates by likelihood of being a match.
    
    Parameters:
    - dataset (pd.DataFrame): The ship tracking dataset with required columns
    - segment_id (str): The shipid to find candidates for
    - top_n (int): Number of top candidates to return (default: 3)
    
    Returns:
    - pd.DataFrame: DataFrame with columns: segment_id, month, track_id, ranking
                   Empty DataFrame if no candidates found or input invalid
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


def debug_candidates(dataset: pd.DataFrame, segment_id: str, month: str) -> None:
    """
    Debug method to see what's happening in the candidate filtering process.
    
    Provides detailed step-by-step output showing how candidates are filtered
    and why they might be rejected. Useful for understanding matching behavior
    and tuning parameters.
    
    Parameters:
    - dataset (pd.DataFrame): The ship tracking dataset
    - segment_id (str): The shipid to analyze
    - month (str): The month of the segment (YYYY-MM format)
    
    Returns:
    - None: Prints debugging information to console
    """
    print(f"=== DEBUGGING CANDIDATES FOR SEGMENT {segment_id} IN {month} ===")
    
    # Clean up the data
    data = clean_data(dataset)
    segments = get_segment_summaries(data)
    
    # Find the target segment
    target_segment = segments[
        (segments['shipid'] == segment_id) & 
        (segments['month'] == month)
    ]
    
    if len(target_segment) == 0:
        print(f"Target segment not found!")
        return
    
    target_segment = target_segment.iloc[0]
    print(f"Target segment signature: {target_segment['ship_signature']}")
    print(f"Target end time: {target_segment['end_time']}")
    
    # Step 1: Find ALL segments with same signature
    same_signature = segments[
        (segments['shipid'] != target_segment['shipid']) &
        (segments['ship_signature'] == target_segment['ship_signature'])
    ]
    print(f"\nStep 1 - Segments with same signature: {len(same_signature)}")
    if len(same_signature) > 0:
        print(same_signature[['shipid', 'month', 'start_time', 'end_time']].head(10))
    
    # Step 2: Filter by time (starts after current ends)
    time_filtered = same_signature[
        same_signature['start_time'] > target_segment['end_time']
    ]
    print(f"\nStep 2 - After time filtering: {len(time_filtered)}")
    if len(time_filtered) > 0:
        print(time_filtered[['shipid', 'month', 'start_time']].head(10))
    
    # Step 3: Calculate metrics
    if len(time_filtered) > 0:
        candidates = calculate_candidate_metrics(target_segment, time_filtered)
        print(f"\nStep 3 - After calculating metrics: {len(candidates)}")
        print(candidates[['shipid', 'month', 'distance_km', 'time_gap_hours', 'implied_speed', 'month_gap']].head(10))
        
        # Step 4: Apply filters
        ship_type = target_segment['astd_cat'].lower()
        valid_candidates = apply_improved_filters(candidates, ship_type, target_segment)
        print(f"\nStep 4 - After improved filtering: {len(valid_candidates)}")
        if len(valid_candidates) > 0:
            print(valid_candidates[['shipid', 'month', 'distance_km', 'time_gap_hours', 'implied_speed', 'month_gap']].head(10))
        else:
            print("All candidates were filtered out! Let's see why...")
            
            # Check what the constraints are
            ship_constraints = {
                'unknown': (30, 72, 1000), 'fishing vessels': (25, 48, 600),
                'passenger ships': (45, 36, 800), 'oil product tankers': (30, 96, 1500),
                'other activities': (25, 72, 800), 'general cargo ships': (25, 96, 1200),
                'ro-ro cargo ships': (35, 48, 1000), 'cruise ships': (40, 36, 700),
                'refrigerated cargo ships': (25, 96, 1200), 'chemical tankers': (30, 96, 1400),
                'bulk carriers': (22, 96, 1200), 'other service offshore vessels': (20, 48, 400),
                'offshore supply ships': (20, 48, 400), 'crude oil tankers': (25, 96, 1500),
                'container ships': (35, 96, 2000), 'gas tankers': (30, 96, 1400),
            }
            max_speed, max_time_gap, max_distance = ship_constraints.get(ship_type, ship_constraints['general cargo ships'])
            
            print(f"Ship type: {ship_type}")
            print(f"Constraints - Max speed: {max_speed} km/h, Max time: {max_time_gap} hrs, Max distance: {max_distance} km")
            
            # Show why each candidate failed
            for _, candidate in candidates.iterrows():
                reasons = []
                if candidate['time_gap_hours'] > max_time_gap:
                    reasons.append(f"time_gap_hours {candidate['time_gap_hours']:.1f} > {max_time_gap}")
                if candidate['distance_km'] > max_distance:
                    reasons.append(f"distance_km {candidate['distance_km']:.1f} > {max_distance}")
                if candidate['implied_speed'] > max_speed:
                    reasons.append(f"implied_speed {candidate['implied_speed']:.1f} > {max_speed}")
                if candidate['month_gap'] > 2:
                    reasons.append(f"month_gap {candidate['month_gap']} > 2")
                
                print(f"  Segment {candidate['shipid']} ({candidate['month']}): REJECTED - {', '.join(reasons)}")


def calculate_detailed_scores_debug(candidates: pd.DataFrame, ship_type: str, current_segment: pd.Series) -> pd.DataFrame:
    """
    Calculate detailed scores for debugging purposes.
    
    Breaks down the scoring algorithm into individual components to help understand
    why certain candidates are ranked higher than others.
    
    Parameters:
    - candidates (pd.DataFrame): Candidate segments to score
    - ship_type (str): Type of ship (e.g., 'container ships', 'bulk carriers')
    - current_segment (pd.Series): The source segment being matched from
    
    Returns:
    - pd.DataFrame: Candidates with detailed scoring breakdown, sorted by match_score
    """
    candidates = candidates.copy()
    
    # Get typical speed for this ship type
    typical_speeds = {
        'unknown': 12, 'fishing vessels': 8, 'passenger ships': 25,
        'oil product tankers': 15, 'other activities': 12, 'general cargo ships': 15,
        'ro-ro cargo ships': 20, 'cruise ships': 25, 'refrigerated cargo ships': 15,
        'chemical tankers': 15, 'bulk carriers': 12, 'other service offshore vessels': 10,
        'offshore supply ships': 8, 'crude oil tankers': 14, 'container ships': 22,
        'gas tankers': 16,
    }
    typical_speed = typical_speeds.get(ship_type, 15)
    
    # Calculate individual score components (all normalized 0-1, lower is better)
    candidates.loc[:, 'distance_score'] = np.minimum(candidates['distance_km'] / 500, 1.0)
    candidates.loc[:, 'time_score'] = np.minimum(candidates['time_gap_hours'] / 72, 1.0)
    
    # Speed score: penalize deviation from typical speed for this ship type
    speed_deviation = np.abs(candidates['implied_speed'] - typical_speed) / typical_speed
    candidates.loc[:, 'speed_score'] = np.minimum(speed_deviation, 2.0)
    
    # Month score: heavily favor consecutive months
    candidates.loc[:, 'month_score'] = candidates['month_gap'].apply(
        lambda x: 0.0 if x == 1 else 0.5 if x == 2 else 1.0
    )
    
    # Position score: check geographical continuity
    candidates.loc[:, 'position_score'] = calculate_position_continuity_score(candidates, current_segment)
    
    # Calculate final match score (weighted combination)
    candidates.loc[:, 'match_score'] = (
        0.3 * candidates['distance_score'] +
        0.2 * candidates['speed_score'] +
        0.2 * candidates['time_score'] +
        0.2 * candidates['month_score'] +
        0.1 * candidates['position_score']
    )
    
    return candidates.sort_values('match_score')


def get_top_candidates_with_scores_unfiltered(dataset: pd.DataFrame, segment_id: str, month: str, top_n: int = 3) -> pd.DataFrame:
    """
    Get top N candidate matches with NO hard filtering - shows all candidates ranked by score.
    
    This version bypasses all constraint filtering to show even impossible matches,
    useful for debugging when no valid candidates are found with normal filtering.
    
    Parameters:
    - dataset (pd.DataFrame): The ship tracking dataset
    - segment_id (str): The shipid to find candidates for
    - month (str): The month of the segment (YYYY-MM format, e.g., '2024-01')
    - top_n (int): Number of top candidates to return (default: 3)
    
    Returns:
    - pd.DataFrame: DataFrame with columns: ranking, segment_id, month, match_score, 
                   distance_km, time_gap_hours, implied_speed, month_gap, and scoring details
    """
    if len(dataset) == 0:
        print("Dataset is empty")
        return pd.DataFrame(columns=['ranking', 'segment_id', 'month', 'match_score'])
    
    # Clean up the data
    data = clean_data(dataset)
    segments = get_segment_summaries(data)
    
    # Find the target segment
    target_segment = segments[
        (segments['shipid'] == segment_id) & 
        (segments['month'] == month)
    ]
    if len(target_segment) == 0:
        print(f"Segment {segment_id} in month {month} not found in dataset")
        return pd.DataFrame(columns=['ranking', 'segment_id', 'month', 'match_score'])
    
    target_segment = target_segment.iloc[0]
    ship_type = target_segment['astd_cat'].lower()
    
    # Find candidates: same ship characteristics, starts after current ends
    candidates = segments[
        (segments['shipid'] != target_segment['shipid']) &
        (segments['ship_signature'] == target_segment['ship_signature']) &
        (segments['start_time'] > target_segment['end_time'])
    ].copy()
    
    if len(candidates) == 0:
        print(f"No candidates found for segment {segment_id}")
        return pd.DataFrame(columns=['ranking', 'segment_id', 'month', 'match_score'])
    
    # Calculate metrics for ALL candidates (no filtering)
    candidates = calculate_candidate_metrics(target_segment, candidates)
    
    # Calculate detailed scores for ALL candidates
    scored_candidates = calculate_detailed_scores_debug(candidates, ship_type, target_segment)
    
    # Get top N candidates (no score threshold)
    top_candidates = scored_candidates.head(min(top_n, len(scored_candidates)))
    
    # Create result DataFrame with detailed scoring information
    result_df = pd.DataFrame({
        'ranking': range(1, len(top_candidates) + 1),
        'segment_id': top_candidates['shipid'].values,
        'month': top_candidates['month'].values,
        'match_score': top_candidates['match_score'].round(4),
        'distance_km': top_candidates['distance_km'].round(2),
        'time_gap_hours': top_candidates['time_gap_hours'].round(1),
        'implied_speed': top_candidates['implied_speed'].round(2),
        'month_gap': top_candidates['month_gap'].astype(int),
        'distance_score': top_candidates['distance_score'].round(4),
        'speed_score': top_candidates['speed_score'].round(4),
        'time_score': top_candidates['time_score'].round(4),
        'month_score': top_candidates['month_score'].round(4),
        'position_score': top_candidates['position_score'].round(4)
    })
    
    print(f"Found {len(scored_candidates)} total candidates for segment {segment_id}")
    print(f"Showing top {len(result_df)} candidates (including impossible ones)")
    print(f"Score breakdown: lower scores = better matches")
    return result_df.reset_index(drop=True)


def get_top_candidates_with_scores(dataset: pd.DataFrame, segment_id: str, month: str, top_n: int = 3) -> pd.DataFrame:
    """
    Get top N candidate matches for a given segment with detailed scoring information.
    
    This is the standard method for finding candidates with full constraint filtering
    and detailed scoring breakdown. Provides insight into why candidates are ranked
    in a particular order.
    
    Parameters:
    - dataset (pd.DataFrame): The ship tracking dataset
    - segment_id (str): The shipid to find candidates for
    - month (str): The month of the segment (YYYY-MM format, e.g., '2024-01')
    - top_n (int): Number of top candidates to return (default: 3)
    
    Returns:
    - pd.DataFrame: DataFrame with columns: ranking, segment_id, month, match_score, 
                   distance_km, time_gap_hours, implied_speed, month_gap, and scoring details
    """
    if len(dataset) == 0:
        print("Dataset is empty")
        return pd.DataFrame(columns=['ranking', 'segment_id', 'month', 'match_score'])
    
    # Clean up the data
    data = clean_data(dataset)
    
    # Get segments
    segments = get_segment_summaries(data)
    
    # Find the target segment - need both shipid and month to uniquely identify
    target_segment = segments[
        (segments['shipid'] == segment_id) & 
        (segments['month'] == month)
    ]
    if len(target_segment) == 0:
        print(f"Segment {segment_id} in month {month} not found in dataset")
        available_segments = segments[segments['shipid'] == segment_id]
        if len(available_segments) > 0:
            print(f"Available months for segment {segment_id}: {available_segments['month'].tolist()}")
        return pd.DataFrame(columns=['ranking', 'segment_id', 'month', 'match_score'])
    
    target_segment = target_segment.iloc[0]
    
    # Find all potential candidates using the improved scoring system
    candidates = get_scored_candidates_detailed(target_segment, segments)
    
    if len(candidates) == 0:
        print(f"No candidates found for segment {segment_id}")
        return pd.DataFrame(columns=['ranking', 'segment_id', 'month', 'match_score'])
    
    # Get top N candidates
    top_candidates = candidates.head(min(top_n, len(candidates)))
    
    # Create result DataFrame with detailed scoring information
    result_df = pd.DataFrame({
        'ranking': range(1, len(top_candidates) + 1),
        'segment_id': top_candidates['shipid'].values,
        'month': top_candidates['month'].values,
        'match_score': top_candidates['match_score'].round(4),
        'distance_km': top_candidates['distance_km'].round(2),
        'time_gap_hours': top_candidates['time_gap_hours'].round(1),
        'implied_speed': top_candidates['implied_speed'].round(2),
        'month_gap': top_candidates['month_gap'].astype(int),
        'distance_score': top_candidates['distance_score'].round(4),
        'speed_score': top_candidates['speed_score'].round(4),
        'time_score': top_candidates['time_score'].round(4),
        'month_score': top_candidates['month_score'].round(4),
        'position_score': top_candidates['position_score'].round(4)
    })
    
    print(f"Found {len(result_df)} candidates for segment {segment_id}")
    print(f"Score breakdown: lower scores = better matches")
    return result_df.reset_index(drop=True)


def get_scored_candidates_detailed(current_segment: pd.Series, all_segments: pd.DataFrame) -> pd.DataFrame:
    """
    Find all valid candidates with detailed scoring breakdown.
    
    Similar to get_scored_candidates but returns individual score components
    for analysis and debugging purposes.
    
    Parameters:
    - current_segment (pd.Series): The segment to find matches for
    - all_segments (pd.DataFrame): All available segments to search in
    
    Returns:
    - pd.DataFrame: Valid candidates with detailed scoring, sorted by match_score (best first)
                   Empty DataFrame if no valid candidates found
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
    
    # Calculate detailed scoring with individual components
    valid_candidates = valid_candidates.copy()
    
    # Get typical speed for this ship type
    typical_speeds = {
        'unknown': 12, 'fishing vessels': 8, 'passenger ships': 25,
        'oil product tankers': 15, 'other activities': 12, 'general cargo ships': 15,
        'ro-ro cargo ships': 20, 'cruise ships': 25, 'refrigerated cargo ships': 15,
        'chemical tankers': 15, 'bulk carriers': 12, 'other service offshore vessels': 10,
        'offshore supply ships': 8, 'crude oil tankers': 14, 'container ships': 22,
        'gas tankers': 16,
    }
    typical_speed = typical_speeds.get(ship_type, 15)
    
    # Calculate individual score components
    valid_candidates['distance_score'] = np.minimum(valid_candidates['distance_km'] / 500, 1.0)
    valid_candidates['time_score'] = np.minimum(valid_candidates['time_gap_hours'] / 72, 1.0)
    
    speed_deviation = np.abs(valid_candidates['implied_speed'] - typical_speed) / typical_speed
    valid_candidates['speed_score'] = np.minimum(speed_deviation, 2.0)
    
    valid_candidates['month_score'] = valid_candidates['month_gap'].apply(
        lambda x: 0.0 if x == 1 else 0.5 if x == 2 else 1.0
    )
    
    valid_candidates['position_score'] = calculate_position_continuity_score(valid_candidates, current_segment)
    
    # Calculate final match score
    valid_candidates['match_score'] = (
        0.3 * valid_candidates['distance_score'] +
        0.2 * valid_candidates['speed_score'] +
        0.2 * valid_candidates['time_score'] +
        0.2 * valid_candidates['month_score'] +
        0.1 * valid_candidates['position_score']
    )
    
    # Return all candidates sorted by match score (best first)
    return valid_candidates.sort_values('match_score').reset_index(drop=True)


def build_track_table(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Build ship tracking table by connecting ship segments across time periods.
    
    This is the main function for creating continuous ship tracks from discrete
    segments. Works with any data density - daily, weekly, monthly, or custom periods.
    
    Parameters:
    - dataset (pd.DataFrame): Ship tracking dataset with required columns:
                             ['shipid', 'date_time_utc', 'latitude', 'longitude', 
                              'astd_cat', 'flagname', 'iceclass', 'sizegroup_gt']
    
    Returns:
    - pd.DataFrame: Track table with columns ['month', 'segment_id', 'track_id']
                   Each row represents a segment assigned to a track
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


def build_tracks_improved(segments: pd.DataFrame) -> dict:
    """
    Improved track building with stricter continuity requirements and better scoring.
    
    Builds ship tracks by connecting segments that likely represent the same physical
    ship across different time periods. Uses conservative constraints to avoid
    false connections.
    
    Parameters:
    - segments (pd.DataFrame): Segment summaries with ship characteristics and positions
    
    Returns:
    - dict: Dictionary mapping track_id (str) to list of segment_ids (list of str)
           e.g., {'track_001': ['ship_123', 'ship_456'], 'track_002': ['ship_789']}
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


def calculate_month_gap(month1: str, month2: str) -> int:
    """
    Calculate the number of months between two month strings.
    
    Parameters:
    - month1 (str): First month in YYYY-MM format (e.g., '2024-01')
    - month2 (str): Second month in YYYY-MM format (e.g., '2024-03')
    
    Returns:
    - int: Number of months between the dates (positive if month2 > month1)
           Returns infinity if parsing fails
    """
    try:
        date1 = datetime.strptime(month1, '%Y-%m')
        date2 = datetime.strptime(month2, '%Y-%m')
        
        # Calculate difference in months
        month_diff = (date2.year - date1.year) * 12 + (date2.month - date1.month)
        return month_diff
    except:
        return float('inf')  # If parsing fails, treat as infinite gap


def find_next_segment_improved(current_segment: pd.Series, all_segments: pd.DataFrame, unassigned: set) -> pd.Series:
    """
    Improved segment matching with stricter criteria and better scoring.
    
    Finds the best candidate for the next segment in a track, using conservative
    constraints to avoid false matches. Only returns matches with good confidence scores.
    
    Parameters:
    - current_segment (pd.Series): The current segment in the track
    - all_segments (pd.DataFrame): All available segments to search
    - unassigned (set): Set of segment IDs not yet assigned to tracks
    
    Returns:
    - pd.Series: The best matching next segment, or None if no good match found
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


def calculate_candidate_metrics(current_segment: pd.Series, candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all metrics needed for candidate evaluation.
    
    Computes time gaps, distances, implied speeds, and month gaps between
    the current segment and all candidate segments.
    
    Parameters:
    - current_segment (pd.Series): The source segment to calculate metrics from
    - candidates (pd.DataFrame): Candidate segments to evaluate
    
    Returns:
    - pd.DataFrame: Candidates with added columns: time_gap_hours, distance_km, 
                   implied_speed, month_gap
    """
    current_end_time = current_segment['end_time']
    current_end_lat = current_segment['end_lat']
    current_end_lon = current_segment['end_lon']
    
    # Calculate time gap in hours between segment end and candidate start
    candidates['time_gap_hours'] = candidates['start_time'].apply(
        lambda x: (x - current_end_time).total_seconds() / 3600
    )
    
    # Calculate great circle distance in kilometers
    candidates['distance_km'] = candidates.apply(
        lambda row: distance_between_points(
            current_end_lat, current_end_lon,
            row['start_lat'], row['start_lon']
        ), axis=1
    )
    
    # Calculate implied speed (km/h) - distance / time
    candidates['implied_speed'] = candidates.apply(
        lambda row: row['distance_km'] / row['time_gap_hours'] if row['time_gap_hours'] > 0 else float('inf'),
        axis=1
    )
    
    # Calculate month gap between current and candidate segments
    candidates['month_gap'] = candidates.apply(
        lambda row: calculate_month_gap(current_segment['month'], row['month']),
        axis=1
    )
    
    return candidates


def apply_improved_filters(candidates: pd.DataFrame, ship_type: str, current_segment: pd.Series) -> pd.DataFrame:
    """
    Apply improved filtering with stricter constraints.
    
    Filters candidates based on realistic physical constraints for different ship types.
    Uses conservative thresholds to reduce false positive matches.
    
    Parameters:
    - candidates (pd.DataFrame): Candidates with calculated metrics
    - ship_type (str): Type of ship (lowercase, e.g., 'container ships')
    - current_segment (pd.Series): Current segment for additional logic checks
    
    Returns:
    - pd.DataFrame: Filtered candidates that pass all constraints
    """
    
    # Stricter ship type constraints: (max_speed_kmh, max_time_gap_hours, max_distance_km)
    # Reduced from previous more lenient values based on realistic ship capabilities
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


def get_scored_candidates(current_segment: pd.Series, all_segments: pd.DataFrame) -> pd.DataFrame:
    """
    Find all valid candidates for a given segment and return them sorted by match score.
    
    Uses the improved scoring system to find and rank potential matches for a segment.
    This is the core matching function used by the track building algorithm.
    
    Parameters:
    - current_segment (pd.Series): The segment to find matches for
    - all_segments (pd.DataFrame): All available segments to search in
    
    Returns:
    - pd.DataFrame: Valid candidates sorted by match score (best first)
                   Empty DataFrame if no valid candidates found
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


def build_tracks_flexible(segments: pd.DataFrame) -> dict:
    """
    Build tracks by connecting segments that represent the same physical ship.
    
    Alternative track building method with more flexible constraints.
    Less conservative than build_tracks_improved().
    
    Parameters:
    - segments (pd.DataFrame): Segment summaries with ship characteristics and positions
    
    Returns:
    - dict: Dictionary mapping track_id (str) to list of segment_ids (list of str)
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


def find_next_segment(current_segment: pd.Series, all_segments: pd.DataFrame, unassigned: set) -> pd.Series:
    """
    Find the next segment that could be the same physical ship.
    
    More flexible version of find_next_segment_improved() with looser constraints.
    
    Parameters:
    - current_segment (pd.Series): The current segment in the track
    - all_segments (pd.DataFrame): All available segments to search
    - unassigned (set): Set of segment IDs not yet assigned to tracks
    
    Returns:
    - pd.Series: The best matching next segment, or None if no match found
    """
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


def filter_realistic_candidates(candidates: pd.DataFrame, ship_type: str) -> pd.DataFrame:
    """
    Filter candidates based on realistic constraints for the ship type.
    
    Uses more lenient constraints than apply_improved_filters().
    
    Parameters:
    - candidates (pd.DataFrame): Candidates with calculated metrics
    - ship_type (str): Type of ship (lowercase)
    
    Returns:
    - pd.DataFrame: Filtered candidates that pass basic constraints
    """
    
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


def calculate_match_score(candidates: pd.DataFrame, ship_type: str) -> np.ndarray:
    """
    Calculate match score for candidates (lower = better).
    
    Simpler scoring algorithm than calculate_improved_match_score().
    
    Parameters:
    - candidates (pd.DataFrame): Candidates with calculated metrics
    - ship_type (str): Type of ship for speed expectations
    
    Returns:
    - np.ndarray: Array of match scores (0-1 scale, lower is better)
    """
    
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


def distance_between_points(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two GPS points in kilometers using the Haversine formula.
    
    Accurately computes great circle distance accounting for Earth's curvature.
    
    Parameters:
    - lat1 (float): Latitude of first point in decimal degrees
    - lon1 (float): Longitude of first point in decimal degrees  
    - lat2 (float): Latitude of second point in decimal degrees
    - lon2 (float): Longitude of second point in decimal degrees
    
    Returns:
    - float: Distance between points in kilometers
    """
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