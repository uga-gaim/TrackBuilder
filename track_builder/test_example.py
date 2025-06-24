import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import main

# Assuming your ship tracking module is saved as 'ship_tracker.py'
# from ship_tracker import build_track_table, distance_between_points

class TestShipTracker(unittest.TestCase):
    
    def setUp(self):
        """Set up test data for each test"""
        # Create base ship characteristics
        self.ship_attrs = {
            'astd_cat': 'container ships',
            'flagname': 'panama',
            'iceclass': 'no ice class',
            'sizegroup_gt': '10000-24999 gt'
        }
    
    def create_test_data(self, ship_tracks):
        """
        Create test dataset from ship track definitions.
        ship_tracks: list of dicts with 'shipid', 'month', 'start_pos', 'end_pos', 'attrs'
        """
        data = []
        
        for track in ship_tracks:
            shipid = track['shipid']
            month = track['month']
            start_pos = track['start_pos']  # (lat, lon)
            end_pos = track['end_pos']      # (lat, lon)
            attrs = track.get('attrs', self.ship_attrs)
            
            # Create start and end records for the month
            base_date = datetime.strptime(f"{month}-01", "%Y-%m-%d")
            
            # Start of month record
            data.append({
                'shipid': shipid,
                'date_time_utc': base_date,
                'latitude': start_pos[0],
                'longitude': start_pos[1],
                **attrs
            })
            
            # End of month record
            end_date = base_date + timedelta(days=28)  # Roughly end of month
            data.append({
                'shipid': shipid,
                'date_time_utc': end_date,
                'latitude': end_pos[0],
                'longitude': end_pos[1],
                **attrs
            })
        
        return pd.DataFrame(data)
    
    def test_single_ship_one_month(self):
        """Test: Single ship, one month should create one track"""
        test_tracks = [
            {
                'shipid': 'ship_001',
                'month': '2024-01',
                'start_pos': (60.0, -20.0),
                'end_pos': (61.0, -19.0)
            }
        ]
        
        df = self.create_test_data(test_tracks)
        result = main.build_track_table(df)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['segment_id'], 'ship_001')
        self.assertEqual(result.iloc[0]['month'], '2024-01')
        self.assertTrue(result.iloc[0]['track_id'].startswith('track_'))
    
    def test_same_ship_multiple_months(self):
        """Test: Same ship across multiple months should be one track"""
        test_tracks = [
            {
                'shipid': 'ship_jan',
                'month': '2024-01',
                'start_pos': (60.0, -20.0),
                'end_pos': (61.0, -19.0)
            },
            {
                'shipid': 'ship_feb',
                'month': '2024-02',
                'start_pos': (61.1, -18.9),  # Close to January end
                'end_pos': (62.0, -18.0)
            },
            {
                'shipid': 'ship_mar',
                'month': '2024-03',
                'start_pos': (62.1, -17.9),  # Close to February end
                'end_pos': (63.0, -17.0)
            }
        ]
        
        df = self.create_test_data(test_tracks)
        result = main.build_track_table(df)
        
        # Should have 3 segments in 1 track
        self.assertEqual(len(result), 3)
        unique_tracks = result['track_id'].unique()
        self.assertEqual(len(unique_tracks), 1)
        
        # All segments should have the same track_id
        track_id = result.iloc[0]['track_id']
        self.assertTrue(all(result['track_id'] == track_id))
    
    def test_different_ships_separate_tracks(self):
        """Test: Ships with different characteristics should get separate tracks"""
        test_tracks = [
            {
                'shipid': 'container_ship',
                'month': '2024-01',
                'start_pos': (60.0, -20.0),
                'end_pos': (61.0, -19.0),
                'attrs': {
                    'astd_cat': 'container ships',
                    'flagname': 'panama',
                    'iceclass': 'no ice class',
                    'sizegroup_gt': '10000-24999 gt'
                }
            },
            {
                'shipid': 'fishing_vessel',
                'month': '2024-01',
                'start_pos': (60.1, -19.9),  # Very close position
                'end_pos': (61.1, -18.9),
                'attrs': {
                    'astd_cat': 'fishing vessels',  # Different type
                    'flagname': 'panama',
                    'iceclass': 'no ice class',
                    'sizegroup_gt': '< 1000 gt'
                }
            }
        ]
        
        df = self.create_test_data(test_tracks)
        result = main.build_track_table(df)
        
        # Should have 2 segments in 2 different tracks
        self.assertEqual(len(result), 2)
        unique_tracks = result['track_id'].unique()
        self.assertEqual(len(unique_tracks), 2)
    
    def test_too_far_apart_separate_tracks(self):
        """Test: Ships too far apart should not be connected"""
        test_tracks = [
            {
                'shipid': 'ship_atlantic',
                'month': '2024-01',
                'start_pos': (60.0, -20.0),
                'end_pos': (61.0, -19.0)
            },
            {
                'shipid': 'ship_pacific',
                'month': '2024-02',
                'start_pos': (35.0, 140.0),  # Too far from Atlantic
                'end_pos': (36.0, 141.0)
            }
        ]
        
        df = self.create_test_data(test_tracks)
        result = main.build_track_table(df)
        
        # Should be 2 separate tracks
        self.assertEqual(len(result), 2)
        unique_tracks = result['track_id'].unique()
        self.assertEqual(len(unique_tracks), 2)
    
    def test_chronological_ordering(self):
        """Test: Tracks should connect in chronological order"""
        test_tracks = [
            {
                'shipid': 'ship_march',  # Note: March comes first in data
                'month': '2024-03',
                'start_pos': (62.0, -18.0),
                'end_pos': (63.0, -17.0)
            },
            {
                'shipid': 'ship_january',
                'month': '2024-01',
                'start_pos': (60.0, -20.0),
                'end_pos': (61.0, -19.0)
            },
            {
                'shipid': 'ship_february',
                'month': '2024-02',
                'start_pos': (61.1, -18.9),  # Connects Jan to Mar
                'end_pos': (62.1, -17.9)
            }
        ]
        
        df = self.create_test_data(test_tracks)
        result = main.build_track_table(df)
        
        # Should connect Jan->Feb->Mar even though data wasn't in order
        self.assertEqual(len(result), 3)
        unique_tracks = result['track_id'].unique()
        self.assertEqual(len(unique_tracks), 1)
    
    def test_gap_too_large_no_connection(self):
        """Test: Time gaps too large should prevent connection"""
        test_tracks = [
            {
                'shipid': 'ship_jan',
                'month': '2024-01',
                'start_pos': (60.0, -20.0),
                'end_pos': (61.0, -19.0)
            },
            {
                'shipid': 'ship_june',  # 5 month gap
                'month': '2024-06',
                'start_pos': (61.1, -18.9),
                'end_pos': (62.0, -18.0)
            }
        ]
        
        df = self.create_test_data(test_tracks)
        result = main.build_track_table(df)
        
        # Should be 2 separate tracks due to large time gap
        self.assertEqual(len(result), 2)
        unique_tracks = result['track_id'].unique()
        self.assertEqual(len(unique_tracks), 2)
    
    def test_complex_scenario(self):
        """Test: Complex scenario with multiple ships and connections"""
        test_tracks = [
            # Track 1: Container ship journey
            {'shipid': 'container_jan', 'month': '2024-01', 'start_pos': (50.0, 0.0), 'end_pos': (51.0, 1.0)},
            {'shipid': 'container_feb', 'month': '2024-02', 'start_pos': (51.1, 1.1), 'end_pos': (52.0, 2.0)},
            
            # Track 2: Different container ship (different flag)
            {'shipid': 'container2_jan', 'month': '2024-01', 'start_pos': (50.1, 0.1), 'end_pos': (51.1, 1.1),
             'attrs': {'astd_cat': 'container ships', 'flagname': 'liberia', 'iceclass': 'no ice class', 'sizegroup_gt': '10000-24999 gt'}},
            
            # Track 3: Fishing vessel
            {'shipid': 'fishing_jan', 'month': '2024-01', 'start_pos': (60.0, -10.0), 'end_pos': (60.5, -9.5),
             'attrs': {'astd_cat': 'fishing vessels', 'flagname': 'iceland', 'iceclass': 'fs ice class 1c', 'sizegroup_gt': '< 1000 gt'}},
            {'shipid': 'fishing_feb', 'month': '2024-02', 'start_pos': (60.6, -9.4), 'end_pos': (61.0, -9.0),
             'attrs': {'astd_cat': 'fishing vessels', 'flagname': 'iceland', 'iceclass': 'fs ice class 1c', 'sizegroup_gt': '< 1000 gt'}},
        ]
        
        df = self.create_test_data(test_tracks)
        result = main.build_track_table(df)
        
        # Should have 5 segments in 3 tracks
        self.assertEqual(len(result), 5)
        unique_tracks = result['track_id'].unique()
        self.assertEqual(len(unique_tracks), 3)
        
        # Check that fishing vessel segments are in same track
        fishing_segments = result[result['segment_id'].str.contains('fishing')]
        self.assertEqual(len(fishing_segments['track_id'].unique()), 1)
        
        # Check that container segments are in same track
        container_segments = result[result['segment_id'].str.contains('container_')]
        self.assertEqual(len(container_segments['track_id'].unique()), 1)
    
    def test_distance_calculation(self):
        """Test: Distance calculation function"""
        # Known distance: London to Paris is approximately 344 km
        london_lat, london_lon = 51.5074, -0.1278
        paris_lat, paris_lon = 48.8566, 2.3522
        
        distance = main.distance_between_points(london_lat, london_lon, paris_lat, paris_lon)
        
        # Should be approximately 344 km (allow 10% error)
        self.assertAlmostEqual(distance, 344, delta=34)
    
    def test_empty_dataset(self):
        """Test: Empty dataset should return empty result"""
        df = pd.DataFrame(columns=['shipid', 'date_time_utc', 'latitude', 'longitude', 
                                 'astd_cat', 'flagname', 'iceclass', 'sizegroup_gt'])
        result = main.build_track_table(df)
        
        self.assertEqual(len(result), 0)
        self.assertListEqual(list(result.columns), ['month', 'segment_id', 'track_id'])
    
    def test_missing_columns(self):
        """Test: Missing required columns should return empty result"""
        df = pd.DataFrame({
            'shipid': ['ship1'],
            'date_time_utc': [datetime.now()]
            # Missing other required columns
        })
        
        result = main.build_track_table(df)
        self.assertEqual(len(result), 0)
    
    def test_invalid_coordinates(self):
        """Test: Invalid coordinates should be filtered out"""
        test_tracks = [
            {
                'shipid': 'valid_ship',
                'month': '2024-01',
                'start_pos': (60.0, -20.0),
                'end_pos': (61.0, -19.0)
            },
            {
                'shipid': 'invalid_ship',
                'month': '2024-01',
                'start_pos': (999.0, -999.0),  # Invalid coordinates
                'end_pos': (61.0, -19.0)
            }
        ]
        
        df = self.create_test_data(test_tracks)
        result = main.build_track_table(df)
        
        # Should only have the valid ship
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['segment_id'], 'valid_ship')


class TestShipTrackerValidation(unittest.TestCase):
    """Additional validation tests to ensure correctness"""
    
    def test_track_continuity(self):
        """Validate that tracks are actually continuous in space and time"""
        # Create a known continuous journey
        test_data = []
        
        # Ship travels from Iceland to UK over 3 months
        positions = [
            (64.0, -22.0),  # Iceland
            (62.0, -15.0),  # Mid-Atlantic
            (60.0, -8.0),   # North Sea
            (58.0, -3.0)    # UK
        ]
        
        for i, pos in enumerate(positions[:-1]):
            month = f"2024-{i+1:02d}"
            test_data.extend([
                {
                    'shipid': f'ship_{month}',
                    'date_time_utc': datetime.strptime(f"{month}-01", "%Y-%m-%d"),
                    'latitude': pos[0],
                    'longitude': pos[1],
                    'astd_cat': 'container ships',
                    'flagname': 'iceland',
                    'iceclass': 'no ice class',
                    'sizegroup_gt': '10000-24999 gt'
                },
                {
                    'shipid': f'ship_{month}',
                    'date_time_utc': datetime.strptime(f"{month}-28", "%Y-%m-%d"),
                    'latitude': positions[i+1][0],
                    'longitude': positions[i+1][1],
                    'astd_cat': 'container ships',
                    'flagname': 'iceland',
                    'iceclass': 'no ice class',
                    'sizegroup_gt': '10000-24999 gt'
                }
            ])
        
        df = pd.DataFrame(test_data)
        result = main.build_track_table(df)
        
        # Should connect all segments into one track
        unique_tracks = result['track_id'].unique()
        self.assertEqual(len(unique_tracks), 1, 
                        f"Expected 1 track, got {len(unique_tracks)}. Tracks: {unique_tracks}")
        
        # All segments should be in chronological order
        track_segments = result.sort_values('month')
        expected_months = ['2024-01', '2024-02', '2024-03']
        actual_months = track_segments['month'].tolist()
        self.assertEqual(actual_months, expected_months)


def run_validation_report(test_df):
    """
    Generate a validation report for a real dataset
    """
    print("=== SHIP TRACKING VALIDATION REPORT ===\n")
    
    print(f"Dataset size: {len(test_df)} records")
    print(f"Unique ships: {test_df['shipid'].nunique()}")
    print(f"Time range: {test_df['date_time_utc'].min()} to {test_df['date_time_utc'].max()}")
    print(f"Ship types: {test_df['astd_cat'].nunique()} unique types")
    print()
    
    # Run the tracking algorithm
    result = main.build_track_table(test_df)
    
    print(f"RESULTS:")
    print(f"- Total tracks created: {result['track_id'].nunique()}")
    print(f"- Total segments: {len(result)}")
    print(f"- Average segments per track: {len(result) / result['track_id'].nunique():.2f}")
    print()
    
    # Analyze track sizes
    track_sizes = result.groupby('track_id').size()
    print(f"TRACK SIZE DISTRIBUTION:")
    print(f"- Single-segment tracks: {(track_sizes == 1).sum()}")
    print(f"- Multi-segment tracks: {(track_sizes > 1).sum()}")
    print(f"- Largest track: {track_sizes.max()} segments")
    print()
    
    # Sample some multi-segment tracks for manual inspection
    multi_segment_tracks = track_sizes[track_sizes > 1].head(3)
    print(f"SAMPLE MULTI-SEGMENT TRACKS (for manual validation):")
    for track_id in multi_segment_tracks.index:
        track_data = result[result['track_id'] == track_id].sort_values('month')
        print(f"\n{track_id}:")
        for _, row in track_data.iterrows():
            segment_info = test_df[test_df['shipid'] == row['segment_id']].iloc[0]
            print(f"  {row['month']}: {row['segment_id']} - "
                  f"{segment_info['astd_cat']}, {segment_info['flagname']}")
    
    return result


if __name__ == "__main__":
    print("Running Ship Tracker Test Suite...")
    print("=" * 50)
    
    # Run the unit tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 50)
    print("Test suite completed!")
    print("\nTo run validation on your real data, use:")
    print("result = run_validation_report(your_dataframe)")