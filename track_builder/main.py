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

def bucket_creation(dataset):
    """
    This method will create buckets for ships
    to simplify comparisons

    Args:
        dataset (dataframe): the dataframe of all ship data being used.

    Returns:
        List: a list of all of the preprocessed ship data.
    """
    
    # This is creating the buckets
    grouped = dataset.groupby(['flagname', 'fuelquality', 'iceclass', 'astd_cat', 'sizegroup_gt'])
    return grouped


def normalize_group(df_group, columns_to_normalize):
    for col in columns_to_normalize:
        std = df_group[col].std()
        mean = df_group[col].mean()
        df_group[col + '_zscore'] = (df_group[col] - mean) / std
        return df_group

def normalize_all_groups(grouped, cols_to_normalize):
    normalized = []

    for _, df_group in grouped:
        norm = normalize_group(df_group.copy(), cols_to_normalize)
        normalized.append(norm)
    return pd.concat(normalized, ignore_index=True)

# might not need to normalize data



    




