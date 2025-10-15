# visualization/visualizer.py
from __future__ import annotations
from typing import Iterable, Optional, Union
import pandas as pd

from plotly import graph_objects as go

from track_builder.core.vis_helpers import (
    resolve_geo_time_cols,
    first_present,
    TRACK_ID_CANDS,
    SEGMENT_ID_CANDS,
    MONTH_CANDS,
)

POSITION_SEGMENT_CANDS = ("shipid", "segment_id")


def plot_ship_tracks(
        data: pd.DataFrame,
        track_ids: Optional[Iterable[Union[str, int]]] = None,
        *,
        show_points: bool = True,
        line_width: float = 2.0,
        map_style: str = "open-street-map",
        title: Optional[str] = None,
        height: int = 720,
        zoom: Optional[float] = None,
) -> go.Figure:
    """
    Plots ship tracks and positions on an interactive map using Plotly.
        Args:
            data (pd.DataFrame): Input DataFrame containing ship position data. Must include latitude, longitude, and timestamp columns.
            track_ids (Optional[Iterable[Union[str, int]]], optional): Specific track IDs to filter and plot. If None, all tracks are plotted.
            show_points (bool, optional): If True, displays individual position points on the map. Defaults to True.
            line_width (float, optional): Width of the track lines. Defaults to 2.0.
            map_style (str, optional): Mapbox style to use for the background. Defaults to "open-street-map".
            title (Optional[str], optional): Title of the plot. If None, a default title is used.
            height (int, optional): Height of the figure in pixels. Defaults to 720.
            zoom (Optional[float], optional): Initial zoom level for the map. If None, a default zoom is used.
        Returns:
            go.Figure: A Plotly Figure object displaying the ship tracks and positions.
        Notes:
            - The function automatically detects latitude, longitude, and timestamp columns using helper functions.
            - If track IDs are provided, only the corresponding tracks are plotted.
            - Each track is plotted as a separate line, and positions are shown as markers if `show_points` is True.
        
    """
    cols = resolve_geo_time_cols(data)
    df = cols["_df"]  # cleaned DataFrame with standard columns
    lat, lon, tcol = cols["lat"], cols["lon"], cols["time"]

    # optional filtering by track_id if present
    track_col = first_present(df, TRACK_ID_CANDS)
    if track_col and track_ids is not None:
        df = df[df[track_col].isin(set(track_ids))].copy()

    fig = go.Figure()

    if show_points:
        fig.add_trace(go.Scattermapbox(
            lat=df[lat],
            lon=df[lon],
            mode="markers",
            marker={"size": 5},
            text=df[tcol].dt.strftime("%Y-%m-%d %H:%M:%S"),
            hovertemplate="<b>%{text}</b><br>Lat: %{lat:.4f}  Lon: %{lon:.4f}<extra></extra>",
            name="Positions",
        ))

    # lines by track if we have the column
    if track_col:
        df_sorted = df.sort_values([track_col, tcol])
        for track, g in df_sorted.groupby(track_col):
            fig.add_trace(go.Scattermapbox(
                lat=g[lat],
                lon=g[lon],
                mode="lines",
                line={"width": line_width},
                name=f"Track {track}",
                hoverinfo="skip",
            ))

    fig.update_layout(
        mapbox_style=map_style,
        mapbox_zoom=(zoom if zoom is not None else 2.7),
        mapbox_center={"lat": float(df[lat].median()), "lon": float(df[lon].median())},
        height=height,
        title=title or "ASTD Ship Positions / Tracks",
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
    )
    return fig


def plot_individual_track(
    track_id,
    track_table,
    astd_data,
    *,
    show_segments: bool = True,
    map_style: str = "open-street-map",
    height: int = 700,
    title=None,
) -> go.Figure:
    """
    Detailed view of a single track, compatible with the table produced by main.build_track_table().
    Assumptions (from previous main.py):
      - track_table contains at least: 'track_id' and 'segment_id' (or equivalents)
      - astd_data (positions) often contains 'shipid' instead of 'segment_id'
    Therefore, 'segment_id' (track_table) is aligned with 'shipid' (positions) without assuming the same column name.

    Parameters
    ----------
    track_id : str|int
        Identifier of the track to visualize.
    track_table : pd.DataFrame
        Table of tracks (e.g., columns ['track_id','segment_id','month']).
    astd_data : pd.DataFrame
        Position data already loaded/standardized by Part 1.
    show_segments : bool
        If True, draws one polyline per segment (legend = identifier from positions).
    map_style : str
        Mapbox/OSM tile style ('open-street-map' does not require a token).
    height : int
        Height of the Plotly figure.
    title : str|None
        Custom title.

    Returns
    -------
    plotly.graph_objects.Figure

    """

    # Geo + time columns (reuse standardization from Part 1; no I/O re-validations here)
    cols = resolve_geo_time_cols(astd_data)
    df = cols["_df"]
    lat, lon, tcol = cols["lat"], cols["lon"], cols["time"]

    # Separate resolution of 'segment' columns for each DataFrame
    seg_col_track = first_present(track_table, SEGMENT_ID_CANDS)       # e.g. 'segment_id' from track_table
    seg_col_pos = first_present(df, POSITION_SEGMENT_CANDS)          #  e.g. 'shipid' from astd_data
    track_col = first_present(track_table, TRACK_ID_CANDS)         # e.g. 'track_id' from track_table

    if seg_col_track is None or seg_col_pos is None or track_col is None:
        raise KeyError(
            "Missing expected columns: "
            f"track_table[{SEGMENT_ID_CANDS + TRACK_ID_CANDS}] et "
            f"astd_data[{POSITION_SEGMENT_CANDS}]."
        )

    # Sélection des segments pour le track demandé (tri par mois si présent)
    tt = track_table[track_table[track_col] == track_id].copy()
    month_col = first_present(track_table, MONTH_CANDS)
    if month_col:
        tt = tt.sort_values(month_col)

    if tt.empty:
        raise ValueError(f"Track '{track_id}' not found in track_table.")

    # List of segments from track_table (e.g., values from 'segment_id')
    segment_ids = set(tt[seg_col_track].dropna().unique().tolist())

    # Filter positions with the equivalent column from positions (e.g., 'shipid')
    pos = df[df[seg_col_pos].isin(segment_ids)].copy()
    if pos.empty:
        raise ValueError(
            "No positions found for this track/segments in astd_data. "
            f"Compared via astd_data['{seg_col_pos}'] ∈ track_table['{seg_col_track}']."
        )

    # Plotly figure
    fig = go.Figure()

    if show_segments:
        # One polyline per 'segment' as named in positions (often 'shipid')
        pos_sorted = pos.sort_values([seg_col_pos, tcol])
        for seg, g in pos_sorted.groupby(seg_col_pos):
            fig.add_trace(go.Scattermapbox(
                lat=g[lat],
                lon=g[lon],
                mode="lines+markers",
                marker={"size": 4},
                name=str(seg),
                text=g[tcol].dt.strftime("%Y-%m-%d %H:%M:%S"),
                hovertemplate=(
                        "<b>Segment:</b> " + str(seg) +
                        "<br><b>Date/Heure:</b> %{text}"
                        "<br>Lat: %{lat:.4f}  Lon: %{lon:.4f}"
                        "<extra></extra>"
                ),
            ))

    else:
        # One polyline for the entire track
        g = pos.sort_values(tcol)
        fig.add_trace(go.Scattermapbox(
            lat=g[lat],
            lon=g[lon],
            mode="lines+markers",
            marker={"size": 4},
            name=f"Track {track_id}",
            hoverinfo="skip",
        ))

    # Layout / centering
    fig.update_layout(
        mapbox_style=map_style,
        mapbox_zoom=3,
        mapbox_center={
            "lat": float(pos[lat].median()),
            "lon": float(pos[lon].median()),
        },
        height=height,
        title=title or f"Track {track_id} – detailed view",
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
    )

    return fig
