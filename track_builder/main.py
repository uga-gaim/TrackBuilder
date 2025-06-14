import pandas as pd
from sklearn import preprocessing
import numpy as np

def build_track_table(dataset):
    """
    This will be the main function users call when building the map.
    The purpose of this function is to build a dictionary table with
    the keys being ["month", "segment_id", "track_id"]. The "track_id"
    key is just a placeholder for the track.

    Args:

    Returns: Outputs a dictionary in this format: 
    
    {
        month: ['2025-04', '2025-05', '2025-05']
        segment_id: ['ship_001', s'hip_003', 'ship_002']
        track_id: ['track_001', 'track_002', 'track_003']
    }

    #method for getting the months
    # method for getting the segment id
    # method for creating/getting the track id

    """
    pass

def preprocess_attributes(df):
    df = df.copy()

    for col in ['astd_cat', 'flagname', 'fuel_quality', 'ice_class']:
        df[col] = df[col].astype(str).str.lower().str.strip()

    return df


def bucket_creation(df):
    """
    This method will create buckets for ships
    to simplify comparisons

    Args:
        dataset (dataframe): the dataframe of all ship data being used.

    Returns:
        List: a list of all of the preprocessed ship data.
    """
    
    # This is creating the buckets
    df['bucket'] = (
        df['astd_cat'] + '_' +
        df['flagname'] + '_' +
        df['sizegroup_gt'].astype(str)
    )
    return df




    




