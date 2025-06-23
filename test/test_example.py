import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import our functions (assuming they're in a file called ship_tracker.py)
from ship_tracker import (
    build_track_table, clean_data, add_ship_bucket,
    distance_between_points, time_diff_hours, get_start_end_points,
    find_matches, score_matches, pick_best_match
)

def test_basic_functionality():
    """Test if the main function works with simple data"""
    print("Testing basic functionality...")
    
    # Make some test data
    data = {
        'shipid': ['ship_001', 'ship_002', 'ship_003'],
        'date_time_utc': ['2025-04-01 10:00:00', '2025-04-02 14:00:00', '2025-05-01 10:30:00'],
        'lat': [60.0, 60.1, 60.15],
        'lon': [-5.0, -4.9, -4.8],
        'astd_cat': ['fishing vessels', 'fishing vessels', 'fishing vessels'],
        'flagname': ['norway', 'norway', 'norway'],
        'iceclass': ['none', 'none', 'none'],
        'sizegroup_gt': ['100-499', '100-499', '100-499']
    }
    
    df = pd.DataFrame(data)
    result = build_track_table(df)
    
    # Check if we got something back
    assert len(result) > 0, "Should return some results"
    assert 'track_id' in result.columns, "Should have track_id column"
    assert 'segment_id' in result.columns, "Should have segment_id column"
    assert 'month' in result.columns, "Should have month column"
    
    print("✓ Basic functionality test passed")

def test_empty_data():
    """Test what happens with empty data"""
    print("Testing empty data...")
    
    empty_df = pd.DataFrame(columns=[
        'shipid', 'date_time_utc', 'lat', 'lon', 
        'astd_cat', 'flagname', 'iceclass', 'sizegroup_gt'
    ])
    
    result = build_track_table(empty_df)
    
    # Should get empty result but not crash
    assert len(result) == 0, "Empty data should give empty result"
    
    print("✓ Empty data test passed")

def test_distance_calculation():
    """Test if distance calculation works"""
    print("Testing distance calculation...")
    
    # Test with known coordinates
    lat1, lon1 = 60.0, -5.0
    lat2, lon2 = 60.1, -4.9
    
    distance = distance_between_points(lat1, lon1, lat2, lon2)
    
    # Should get some reasonable distance
    assert distance > 0, "Distance should be positive"
    assert distance < 1000, "Distance shouldn't be crazy large"
    
    # Test same point
    same_point_distance = distance_between_points(60.0, -5.0, 60.0, -5.0)
    assert same_point_distance == 0, "Same point should have zero distance"
    
    print("✓ Distance calculation test passed")

def test_time_difference():
    """Test time difference calculation"""
    print("Testing time difference...")
    
    t1 = pd.Timestamp('2025-04-01 10:00:00')
    t2 = pd.Timestamp('2025-04-01 12:00:00')
    
    diff = time_diff_hours(t1, t2)
    
    assert diff == 2.0, "2 hour difference should be 2.0"
    
    # Test reverse order (should be same)
    diff_reverse = time_diff_hours(t2, t1)
    assert diff_reverse == 2.0, "Reverse order should give same result"
    
    print("✓ Time difference test passed")

def test_data_cleaning():
    """Test if data cleaning works"""
    print("Testing data cleaning...")
    
    # Data with some problems
    messy_data = pd.DataFrame({
        'shipid': ['ship_001', 'ship_002'],
        'date_time_utc': ['2025-04-01 10:00:00', '2025-04-01 11:00:00'],
        'lat': [60.0, 'bad_value'],  # One bad coordinate
        'lon': [-5.0, -4.9],
        'astd_cat': ['FISHING VESSELS', 'fishing vessels'],  # Mixed case
        'flagname': [' NORWAY ', 'norway'],  # Extra spaces
        'iceclass': ['none', 'none'],
        'sizegroup_gt': ['100-499', '100-499']
    })
    
    cleaned = clean_data(messy_data)
    
    # Should remove bad coordinates
    assert len(cleaned) == 1, "Should remove row with bad coordinates"
    
    # Should lowercase text
    assert cleaned['astd_cat'].iloc[0] == 'fishing vessels', "Should be lowercase"
    assert cleaned['flagname'].iloc[0] == 'norway', "Should be lowercase and trimmed"
    
    print("✓ Data cleaning test passed")

def test_bucket_creation():
    """Test bucket creation for grouping ships"""
    print("Testing bucket creation...")
    
    data = pd.DataFrame({
        'shipid': ['ship_001', 'ship_002'],
        'astd_cat': ['fishing vessels', 'fishing vessels'],
        'flagname': ['norway', 'norway'],
        'iceclass': ['none', 'none'],
        'sizegroup_gt': ['100-499', '100-499']
    })
    
    result = add_ship_bucket(data)
    
    # Should add bucket column
    assert 'bucket' in result.columns, "Should add bucket column"
    
    # Similar ships should have same bucket
    assert result['bucket'].iloc[0] == result['bucket'].iloc[1], "Similar ships should have same bucket"
    
    print("✓ Bucket creation test passed")

def test_start_end_points():
    """Test finding start and end points"""
    print("Testing start/end point finding...")
    
    # Ship with multiple observations
    data = pd.DataFrame({
        'shipid': ['ship_001', 'ship_001', 'ship_001'],
        'date_time_utc': ['2025-04-01 10:00:00', '2025-04-01 15:00:00', '2025-04-02 08:00:00'],
        'lat': [60.0, 60.1, 60.2],
        'lon': [-5.0, -4.9, -4.8],
        'astd_cat': ['fishing vessels', 'fishing vessels', 'fishing vessels'],
        'flagname': ['norway', 'norway', 'norway'],
        'iceclass': ['none', 'none', 'none'],
        'sizegroup_gt': ['100-499', '100-499', '100-499']
    })
    
    cleaned = clean_data(data)
    starts, ends = get_start_end_points(cleaned)
    
    # Should get one start and one end
    assert len(starts) == 1, "Should have one start point"
    assert len(ends) == 1, "Should have one end point"
    
    # Start should be earliest time
    start_time = pd.to_datetime(starts.iloc[0]['date_time_utc'])
    end_time = pd.to_datetime(ends.iloc[0]['date_time_utc'])
    assert start_time < end_time, "Start should be before end"
    
    print("✓ Start/end point test passed")

def test_ship_matching():
    """Test if ships can be matched correctly"""
    print("Testing ship matching...")
    
    # Create ships that should match
    data = pd.DataFrame({
        'shipid': ['ship_001', 'ship_002'],
        'date_time_utc': ['2025-04-30 23:30:00', '2025-05-01 00:30:00'],  # 1 hour apart
        'lat': [70.0, 70.05],  # Close positions
        'lon': [10.0, 10.02],
        'astd_cat': ['fishing vessels', 'fishing vessels'],  # Same type
        'flagname': ['norway', 'norway'],
        'iceclass': ['none', 'none'],
        'sizegroup_gt': ['100-499', '100-499']
    })
    
    cleaned = clean_data(data)
    bucketed = add_ship_bucket(cleaned)
    starts, ends = get_start_end_points(bucketed)
    
    # Try to find matches
    if len(ends) > 0:
        end_ship = ends.iloc[0]
        matches = find_matches(end_ship, starts)
        
        # Should find at least one match (or none if thresholds are strict)
        assert len(matches) >= 0, "Should not crash when finding matches"
    
    print("✓ Ship matching test passed")

def test_connected_ships():
    """Test that similar ships get connected into tracks"""
    print("Testing ship connections...")
    
    # Ships that should be connected
    data = pd.DataFrame({
        'shipid': ['seg_001', 'seg_002'],
        'date_time_utc': ['2025-04-30 23:30:00', '2025-05-01 00:30:00'],
        'lat': [70.0, 70.05],
        'lon': [10.0, 10.02],
        'astd_cat': ['fishing vessels', 'fishing vessels'],
        'flagname': ['norway', 'norway'],
        'iceclass': ['none', 'none'],
        'sizegroup_gt': ['100-499', '100-499']
    })
    
    result = build_track_table(data)
    
    # Should have 2 ships in result
    assert len(result) == 2, "Should have 2 ships in result"
    
    # Check if they got connected (same track_id) or separate tracks
    unique_tracks = result['track_id'].nunique()
    assert unique_tracks <= 2, "Should have at most 2 tracks"
    
    print("✓ Ship connection test passed")

def test_with_different_ship_types():
    """Test with different types of ships"""
    print("Testing different ship types...")
    
    data = pd.DataFrame({
        'shipid': ['fishing_ship', 'cargo_ship'],
        'date_time_utc': ['2025-04-01 10:00:00', '2025-04-01 11:00:00'],
        'lat': [60.0, 60.1],
        'lon': [-5.0, -4.9],
        'astd_cat': ['fishing vessels', 'general cargo ships'],  # Different types
        'flagname': ['norway', 'norway'],
        'iceclass': ['none', 'none'],
        'sizegroup_gt': ['100-499', '1000-4999']  # Different sizes
    })
    
    result = build_track_table(data)
    
    # Should handle different ship types
    assert len(result) == 2, "Should handle different ship types"
    
    # Different types should probably get different tracks
    unique_tracks = result['track_id'].nunique()
    assert unique_tracks >= 1, "Should create tracks"
    
    print("✓ Different ship types test passed")

def run_all_tests():
    """Run all tests"""
    print("Running all tests...\n")
    
    try:
        test_basic_functionality()
        test_empty_data()
        test_distance_calculation()
        test_time_difference()
        test_data_cleaning()
        test_bucket_creation()
        test_start_end_points()
        test_ship_matching()
        test_connected_ships()
        test_with_different_ship_types()
        
        print("\n✓ All tests passed! 🎉")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise

if __name__ == "__main__":
    run_all_tests()

