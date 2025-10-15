"""
track_builder Public API
========================

High-level functions for loading and preprocessing ASTD datasets.

Typical usage
-------------
>>> import track_builder as tb
>>> df = tb.load_astd_data('ASTD_area_level3_201907.csv')
>>> df_monthly = tb.load_astd_monthly('/data/astd/', 2019, months=[7, 8, 9])

Notes
-----
This module exposes the top-level public I/O functions.
Internal helpers (under track_builder.core) are not meant to be imported directly.
"""

from __future__ import annotations

# Import only the public-facing wrappers
from track_builder.io.astd_loader import (
    load_astd_data,
    load_astd_monthly,
)

from track_builder.visualization.visualizer import (
    plot_ship_tracks,
    plot_individual_track
)

from track_builder import main

__all__ = [
    "load_astd_data",
    "load_astd_monthly",
    "plot_ship_tracks",
    "plot_individual_track",
    "main",
]
