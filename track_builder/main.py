import math
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

    for col in ['astd_cat', 'flagname', 'iceclass', 'sizegroup_gt']:
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
        df['iceclass'] + '_' +
        df['sizegroup_gt'].astype(str)
    )
    return df

def haversine_distance(lat1, long1, lat2, long2):
    """
    Calculates the distance between 2 GPS points in kilometers.

    Args:
        lat1 (_type_): _description_
        long1 (_type_): _description_
        lat2 (_type_): _description_
        long2 (_type_): _description_
    """
    R = 6371 # Earth radius in km

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    deltaPhi = math.radians(lat2 - lat1)
    deltaLambda = math.radians(long2 - long1)

    a = math.sin(deltaPhi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(deltaLambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    
    return distance


def calculate_time_diff_hours(t1, t2):
    """
    Calculates the time difference between 2 timestamps in hours.

    Args:
        t1 (_type_): _description_
        t2 (_type_): _description_
    """








    




