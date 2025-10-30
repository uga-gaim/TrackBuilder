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
