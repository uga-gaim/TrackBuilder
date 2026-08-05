from typing import Dict, Literal, Tuple

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

# ---------------- Track-building Constants ----------------

# =====================================================================
# Conservative caps (km/h) — safety-only thresholds
# =====================================================================
# Sources (consulted for reference, 1 knot = 1.852 km/h):
#
# 1. Container Ships: 36.5 knots (68 km/h)
#    - Reference to fast containerships (e.g. Maersk Boston)
#    - Sources: [Sahyog Freight](https://sahyogfreight.com/blog/cargo-ship-speed/), [SlashGear](https://www.slashgear.com/1791884/container-ships-top-speed/)
#
# 2. Bulk & Tankers (Bulk, Crude, Product, Chemical): ~16-17 knots (30-32 km/h)
#    - Typical caps for large bulk and oil tankers.
#    - Sources: [Maritime Page](https://maritimepage.com/the-speed-of-a-cargo-ship-at-sea-compare-top-10-types/), [Points East Magazine](https://www.pointseast.com/tankers-are-fast/)
#
# 3. Refrigerated ships (Reefers): ~25-27 knots (46-50 km/h)
#    - Reefers are structurally faster than standard cargo vessels.
#    - Source: [Jack Cooper Transport Services](https://www.jackcooper.com/the-speed-of-cargo-ships-secrets-you-should-know/)
#
# 4. Passenger & Cruise ships: ~30 knots (56 km/h)
#    - Typical top speed for large modern cruise ships.
#    - Source: [Betts Boat Repair Blog](https://bettsboatrepair.com/blog/can-a-cruise-ship-reach-50-knots/)
#
# 5. Ro-Ro Cargo: ~25 knots (47 km/h)
#    - Typical speed for Ro-Ro ferries and large Ro-Ro cargo vessels.
#    - Sources: [Shippax](https://www.shippax.com/en/news/gnv-virgo-delivered-as-gnvs-first-lng-powered-ship.aspx), [StayVista](https://www.stayvista.com/blog/mumbai-to-sindhudurg-roro-ferry-route-timings-fares/)
#
# 6. Offshore service vessels: ~25 knots (47 km/h)
#    - Based on Fast Supply/Intervention Vessel classes.
#    - Source: [Wikipedia "Platform supply vessel"](https://en.wikipedia.org/wiki/Platform_supply_vessel)
#
# 7. Fishing vessels: ~15 knots (28 km/h)
#    - Cap for large commercial trawlers (not small recreational boats).
#    - Source: industry discussions (e.g. r/deadliestcatch)
# =====================================================================
_LIT_CAPS_KMH: Dict[str, float] = {
    # Fast vessels (container ships, reefers, ferries)
    'container ships': 68.0,          # (~36.5 knots)
    'refrigerated cargo ships': 50.0, # (~27 knots)
    'passenger ships': 56.0,          # (~30 knots)
    'cruise ships': 56.0,             # (same as passenger)
    'ro-ro cargo ships': 47.0,        # (~25 knots)

    # Specialized vessels (medium-high speed)
    'gas tankers': 41.0,              # (~22 knots, LNG carriers)
    'general cargo ships': 34.0,      # (~18 knots)

    # Slow vessels (bulk, tankers)
    'bulk carriers': 30.0,            # (~16 knots)
    'crude oil tankers': 32.0,        # (~17 knots, VLCC/Supertankers)
    'oil product tankers': 32.0,      # (~17 knots)
    'chemical tankers': 32.0,         # (~17 knots)

    # Service and fishing vessels
    'fishing vessels': 28.0,          # (~15 knots, large trawlers)
    'other service offshore vessels': 47.0, # (~25 knots, e.g. FSIV)
    'offshore supply ships': 47.0,    # (same as service)

    # Default categories
    'other activities': 37.0,         # (conservative default ~20 knots)
    'unknown': 37.0,                  # (conservative default ~20 knots)
}

# =====================================================================
# Options & Stratégies
# =====================================================================
MatchingStrategy = Literal["conservative", "balanced", "aggressive"]

# remove this one later
# _SCORE_THRESHOLDS: Dict[MatchingStrategy, float] = {
#     "conservative": 0.40,
#     "balanced": 0.55,
#     "aggressive": 0.70,
# }

_LIMIT_MULTIPLIERS: Dict[MatchingStrategy, Tuple[float, float, float]] = {
    # (time_gap, distance, speed)
    "conservative": (0.9, 0.9, 1.0),
    "balanced": (1.0, 1.0, 1.0),
    "aggressive": (1.2, 1.2, 1.15),
}

# ---------------------------
# Typical speed parameters
# ---------------------------
SPEED_TYPICALS_N_PER_DAY = 4        # n speed samples per day
SPEED_TYPICALS_CLIP_KMH  = 80.0     # clip unrealistic speeds
SPEED_TYP_MIN_POINTS     = 5        # min samples kept per ship
SPEED_TYP_MIN_SHIPS      = 50       # minimum number of ship to compute typ speed

# =====================================================================
# Geographic Zones (Bounding Boxes)
# Format: (min_lon, max_lon, min_lat, max_lat)
# Note: If min_lon > max_lon, it indicates an International Date Line crossing.
# =====================================================================
ARCTIC_ZONES: Dict[str, Tuple[float, float, float, float]] = {
    "canada": (-141.0, -50.0, 60.0, 85.0),   # Northwest Passage coverage
    "norway": (5.0, 35.0, 60.0, 82.0),       # Barents Sea / Svalbard area
    "russia": (50.0, -168.0, 65.0, 85.0),    # Northern Sea Route (crosses 180/-180)
    "usa": (-170.0, -140.0, 60.0, 75.0),     # Alaska (Arctic region)
    "iceland": (-25.0, -12.0, 63.0, 67.0),   # Iceland EEZ approx
}