from typing import Dict

ASTD_USEFUL_COLS = [
    'shipid', 
    'date_time_utc', 
    'astd_cat', 
    'dist_nextpoint', 
    'sec_nextpoint', 
    'longitude', 
    'latitude',
    'flagname',
    'iceclass',
    'sizegroup_gt'
]

ASTD_DTYPE_MAP = {
    'shipid': 'int32',
    'astd_cat': 'category',
    'dist_nextpoint': 'float32',
    'sec_nextpoint': 'int32',
    'longitude': 'float32',
    'latitude': 'float32',
    'flagname': 'category',
    'iceclass': 'category',
    'sizegroup_gt': 'category',
}

# ---------- IO HELPERS Constants ----------

# --- Minimal column normalization based on use-example.ipynb ---
COLMAP: Dict[str, str] = {
    # Time columns
    "date_time_utc": "date_time_utc",
    "datetime_utc":  "date_time_utc",
    "timestamp":     "date_time_utc",
    # Position columns
    "lat":        "latitude",
    "latitude":   "latitude",
    "lon":        "longitude",
    "long":       "longitude",
    "longitude":  "longitude",
    # Identifier and ship attributes
    "shipid":       "shipid",
    "astd_cat":     "astd_cat",
    "flagname":     "flagname",
    "iceclass":     "iceclass",
    "sizegroup_gt": "sizegroup_gt",
}


STR_COLS = ("astd_cat", "flagname", "iceclass", "sizegroup_gt")
DATE_CANDS = ("date_time_utc", "datetime_utc", "timestamp")


# For visualization defaults

MAPBOX_STYLE_ALIASES: Dict[str, str] = {
    "satellite": "satellite-streets",
    "streets": "streets",
    "light": "light",
    "dark": "dark",
    "outdoors": "outdoors",
    "terrain": "outdoors",
    "osm": "open-street-map",
    "openstreetmap": "open-street-map",
}

MAPBOX_TOKEN = "dhdddddd"