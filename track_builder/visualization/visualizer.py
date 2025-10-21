from __future__ import annotations
from typing import Iterable, Optional, Union, Literal
import pandas as pd

from plotly import graph_objects as go

from track_builder.core.vis_helpers import (
    resolve_geo_time_cols,
    first_present,
    TRACK_ID_CANDS,
    SEGMENT_ID_CANDS,
    MONTH_CANDS, resolve_map_style, discrete_palette, is_continuous,
)

POSITION_SEGMENT_CANDS = ("shipid", "segment_id")


def plot_ship_tracks(
        data: pd.DataFrame,
        track_ids: Optional[Iterable[Union[str, int]]] = None,
        *,

        color_by: Optional[str] = None,
        color_mode: Literal["auto","categorical","continuous"] = "auto",  # NEW
        color_lines: bool = False,
        date_from: Optional[pd.Timestamp] = None,
        date_to: Optional[pd.Timestamp] = None,
        ship_types: Optional[Iterable[str]] = None,
        flags: Optional[Iterable[str]] = None,

        show_points: bool = True,
        line_width: float = 2.0,
        map_style: str = "open-street-map",
        title: Optional[str] = None,
        height: int = 720,
        zoom: Optional[float] = None,
) -> go.Figure:
    """
    Plots ship tracks and positions on an interactive map using Plotly.

    Key options now include:
      - color_by: any column in `data` (e.g., 'ship_type', 'flagname', 'track_id', 'time_bin')
      - date_from/date_to: inclusive datetime range filters
      - ship_types/flags: categorical filters (if columns exist)
    """
    # --- Standardization : geo + time columns ---
    cols = resolve_geo_time_cols(data)
    df = cols["_df"]
    lat, lon, tcol = cols["lat"], cols["lon"], cols["time"]

    # date_from/date_to conversion if necessary
    if date_from is not None and not pd.api.types.is_datetime64_any_dtype(pd.Series([date_from])):
        date_from = pd.to_datetime(date_from, errors="coerce")
    if date_to is not None and not pd.api.types.is_datetime64_any_dtype(pd.Series([date_to])):
        date_to = pd.to_datetime(date_to, errors="coerce")

    # --- Filtering ---
    # 1) Track IDs
    track_col = first_present(df, TRACK_ID_CANDS)
    if track_col and track_ids is not None:
        df = df[df[track_col].isin(set(track_ids))].copy()

    # 2) Date range
    if date_from is not None:
        df = df[df[tcol] >= date_from]
    if date_to is not None:
        df = df[df[tcol] <= date_to]

    # 3) Ship types
    if ship_types is not None and "ship_type" in df.columns:
        df = df[df["ship_type"].isin(set(ship_types))]

    # 4) Flags
    if flags is not None and "flagname" in df.columns:
        df = df[df["flagname"].isin(set(flags))]

    # if no data after filtering : empty figure with message
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            mapbox_style=resolve_map_style(map_style),
            title=title or "ASTD Ship Positions / Tracks (no data after filtering)",
            height=height,
            margin=dict(l=0, r=0, t=60, b=0),
        )
        return fig

    # --- Logic for color_by ---
    # Default: use color_by if present in df.columns
    use_color = color_by if (color_by is not None and color_by in df.columns) else None

    # Special case: time_bin
    if use_color is None and color_by is not None:
        # color_by by time bins
        if color_by == "time_bin":
            # simple trim to hourly bins
            df = df.copy()
            df["time_bin"] = df[tcol].dt.strftime("%Y-%m-%d %H:00")
            use_color = "time_bin"
        # else : skip silently

    # --- Figure ---
    fig = go.Figure()

    # Points avec horodatage
    if show_points:
        if use_color is not None and use_color in df.columns:
            series = df[use_color]
            if is_continuous(series):
                # Single trace with coloraxis and a colorbar
                fig.add_trace(go.Scattermapbox(
                    lat=df[lat], lon=df[lon],
                    mode="markers",
                    marker={"size": 6, "color": series, "coloraxis": "coloraxis"},
                    text=df[tcol].dt.strftime("%Y-%m-%d %H:%M:%S"),
                    hovertemplate=(
                        "<b>Date/Heure:</b> %{text}"
                        "<br>Lat: %{lat:.4f}  Lon: %{lon:.4f}"
                        f"<br><b>{use_color}:</b> %{{marker.color}}"
                        "<extra></extra>"
                    ),
                    name="Positions",
                ))
                fig.update_layout(coloraxis=dict(colorbar_title=use_color))
            else:
                # Discrete categories: one trace per category with fixed color and legend entries
                cats = [c for c in series.astype('category').cat.categories.tolist() if pd.notna(c)]
                palette = discrete_palette(len(cats))
                for cat, col in zip(cats, palette):
                    mask = series.astype('object') == cat
                    if not mask.any():
                        continue
                    df_cat = df[mask]
                    fig.add_trace(go.Scattermapbox(
                        lat=df_cat[lat], lon=df_cat[lon],
                        mode="markers",
                        marker={"size": 6, "color": col},
                        text=df_cat[tcol].dt.strftime("%Y-%m-%d %H:%M:%S"),
                        hovertemplate=(
                            "<b>Date/Heure:</b> %{text}"
                            "<br>Lat: %{lat:.4f}  Lon: %{lon:.4f}"
                            f"<br><b>{use_color}:</b> {cat}"
                            "<extra></extra>"
                        ),
                        name=f"{use_color} = {cat}",
                    ))
        else:
            # No color_by provided: single neutral layer
            fig.add_trace(go.Scattermapbox(
                lat=df[lat], lon=df[lon],
                mode="markers",
                marker={"size": 5},
                text=df[tcol].dt.strftime("%Y-%m-%d %H:%M:%S"),
                hovertemplate=(
                    "<b>Date/Heure:</b> %{text}"
                    "<br>Lat: %{lat:.4f}  Lon: %{lon:.4f}"
                    "<extra></extra>"
                ),
                name="Positions",
            ))

    # line for tracks if track_id is present
    if track_col:
        df_sorted = df.sort_values([track_col, tcol])
        # one trace per track
        for track, g in df_sorted.groupby(track_col):
            fig.add_trace(go.Scattermapbox(
                lat=g[lat], lon=g[lon],
                mode="lines",
                line={"width": line_width},
                name=f"Track {track}",
                hoverinfo="skip",
            ))

    # Layout / centering
    fig.update_layout(
        coloraxis=dict(
            colorbar_title=use_color,
            colorscale="Viridis"
        ),
        mapbox_style=resolve_map_style(map_style),
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
    seg_col_track = first_present(track_table, SEGMENT_ID_CANDS)  # e.g. 'segment_id' from track_table
    seg_col_pos = first_present(df, POSITION_SEGMENT_CANDS)  # e.g. 'shipid' from astd_data
    track_col = first_present(track_table, TRACK_ID_CANDS)  # e.g. 'track_id' from track_table

    if seg_col_track is None or seg_col_pos is None or track_col is None:
        raise KeyError(
            "Missing expected columns: "
            f"track_table[{SEGMENT_ID_CANDS + TRACK_ID_CANDS}] et "
            f"astd_data[{POSITION_SEGMENT_CANDS}]."
        )

    # Filter track_table for the given track_id
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
        pos_sorted = pos.sort_values([seg_col_pos, tcol])
        for seg, g in pos_sorted.groupby(seg_col_pos):
            fig.add_trace(go.Scattermapbox(
                lat=g[lat],
                lon=g[lon],
                mode="lines",  # just lines, no markers
                line={"width": 2},  # fixed line width
                name=f"Segment {seg}",  # legend entry
                hoverinfo="skip",
            ))

    # Layout / centering
    fig.update_layout(
        mapbox_style=resolve_map_style(map_style),
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
