from __future__ import annotations
from typing import Iterable, Optional, Union, Literal, Dict, Any, Sequence
import pandas as pd
import numpy as np

from plotly import graph_objects as go
import plotly.express as px

from track_builder.core.vis_helpers import (
    resolve_geo_time_cols,
    first_present,
    TRACK_ID_CANDS,
    SEGMENT_ID_CANDS,
    MONTH_CANDS, resolve_map_style,
    build_color_spec,
    get_lat_lon_time_cols,
    get_track_col,
    ensure_datetime, get_mapbox_token, build_hover_customdata
)

POSITION_SEGMENT_CANDS = ("shipid", "segment_id")


def plot_ship_tracks(
    df: pd.DataFrame,
    track_ids: Optional[Sequence[str]] = None,
    *,
    color_by: Optional[str] = None,
    color_mode: Literal["auto", "categorical", "continuous"] = "categorical",
    color_lines: bool = False,
    max_categories: int = 20,
    show_points: bool = False,
    map_style: Optional[str] = None,
    title: Optional[str] = None,
    date_from: Optional[pd.Timestamp] = None,
    date_to: Optional[pd.Timestamp] = None,
    ship_types: Optional[Sequence[str]] = None,
    flags: Optional[Sequence[str]] = None,
    height: int = 650,
    zoom: Optional[float] = None,
    center: Optional[Dict[str, float]] = None,
    extra_cols_priority: Optional[Sequence[str]] = None,
) -> go.Figure:
    """
    Visualization of ASTD trajectories/positions.

    Main parameters:
      - color_by: column used for coloring
      - color_mode: 'auto' | 'categorical' | 'continuous'
      - color_lines: also color lines when categorical
      - max_categories: upper bound on number of categories (the rest → 'Other')
      - extra_cols_priority: columns to include in the hover (desired order),
        e.g. ('astd_cat','astd_cat','flagname','shipid').
    """
    if df is None or len(df) == 0:
        return go.Figure()

    # columns lat/lon/time
    lat, lon, tcol = get_lat_lon_time_cols(df)
    df = ensure_datetime(df, tcol)

    # Simple filters
    work = df
    if date_from is not None:
        work = work[work[tcol] >= pd.to_datetime(date_from, utc=True)]
    if date_to is not None:
        work = work[work[tcol] <= pd.to_datetime(date_to, utc=True)]
    if ship_types:
        if "astd_cat" in work.columns:
            work = work[work["astd_cat"].isin(ship_types)]
    if flags:
        for flag_col in ("flagname", "flag", "flag_name"):
            if flag_col in work.columns:
                flags_lower = {f.lower() for f in flags}
                work = work[work[flag_col].astype(str).str.lower().isin(flags_lower)]
                break
    track_col = get_track_col(work)
    if track_ids and track_col:
        work = work[work[track_col].isin(track_ids)]

    # TTemporal sorting (useful for lines)
    work = work.sort_values(tcol)

    # Color specification
    color_spec = build_color_spec(
        work, color_by, mode=color_mode, max_categories=max_categories, colorscale="Viridis"
    )
    use_color = color_by if color_spec["enabled"] else None

    # delete rows where color_by is NaN
    work = work[work[use_color].notna()] if use_color else work

    fig = go.Figure()

    # --------- LINES ----------
    if track_col and track_col in work.columns:
        color_map = color_spec["color_map"] if (
            color_spec["enabled"] and not color_spec["is_cont"] and color_lines
        ) else None

        for track_id, grp in work.groupby(track_col, sort=False):
            line_kwargs: Dict[str, Any] = {}
            if color_map and use_color in grp.columns:
                cat_val = str(color_spec["series"].loc[grp.index].iloc[0])
                line_kwargs = {"line": {"color": color_map.get(cat_val, "#444")}}
            fig.add_trace(go.Scattermapbox(
                lat=grp[lat],
                lon=grp[lon],
                mode="lines",
                name=f"Track {track_id}",
                hoverinfo="skip",
                **line_kwargs
            ))

    # --------- POINTS ----------
    if show_points:
        # Prepare hover info (customdata + suffix) based on actual columns present
        cdata_all, suffix_all, _ = build_hover_customdata(work, extra_cols_priority, color_by=color_by)

        if color_spec["enabled"]:
            if color_spec["is_cont"]:
                fig.add_trace(go.Scattermapbox(
                    lat=work[lat], lon=work[lon],
                    mode="markers",
                    marker={"size": 6, "color": color_spec["series"], "coloraxis": "coloraxis"},
                    text=work[tcol].dt.strftime("%Y-%m-%d %H:%M:%S"),
                    customdata=cdata_all,
                    hovertemplate=(
                        "<b>Date/Hour:</b> %{text}"
                        "<br>Lat: %{lat:.4f}  Lon: %{lon:.4f}"
                        f"<br><b>{use_color}:</b> %{{marker.color}}"
                        f"{suffix_all}"
                        "<extra></extra>"
                    ),
                    name="Positions",
                ))
                fig.update_layout(coloraxis=color_spec["coloraxis_kwargs"])
            else:
                cats = color_spec["cats"]
                color_map = color_spec["color_map"]
                s = color_spec["series"]  # other col with category per point

                # loop by category → customdata/suffix calculated on the subsample
                for cat in cats:
                    mask = (s == cat)
                    if not mask.any():
                        continue
                    df_cat = work.loc[mask]
                    cdata, suffix, _ = build_hover_customdata(df_cat, extra_cols_priority, color_by=color_by)

                    fig.add_trace(go.Scattermapbox(
                        lat=df_cat[lat], lon=df_cat[lon],
                        mode="markers",
                        marker={"size": 6, "color": color_map[cat]},
                        text=df_cat[tcol].dt.strftime("%Y-%m-%d %H:%M:%S"),
                        customdata=cdata,
                        hovertemplate=(
                            "<b>Date/Hour:</b> %{text}"
                            "<br>Lat: %{lat:.4f}  Lon: %{lon:.4f}"
                            f"<br><b>{use_color}:</b> {cat}"
                            f"{suffix}"
                            "<extra></extra>"
                        ),
                        name=f"{use_color} = {cat}",
                    ))
        else:
            # Without color_by → a single layer of points
            fig.add_trace(go.Scattermapbox(
                lat=work[lat], lon=work[lon],
                mode="markers",
                marker={"size": 5},
                text=work[tcol].dt.strftime("%Y-%m-%d %H:%M:%S"),
                customdata=cdata_all,
                hovertemplate=(
                    "<b>Date/Hour:</b> %{text}"
                    "<br>Lat: %{lat:.4f}  Lon: %{lon:.4f}"
                    f"{suffix_all}"
                    "<extra></extra>"
                ),
                name="Positions",
            ))

    # render map 
    if center is None:
        center = {"lat": float(work[lat].median()), "lon": float(work[lon].median())}
    if zoom is None:
        zoom = 2.5

    style = resolve_map_style(map_style)
    layout_kwargs = dict(
        mapbox_style=style,
        mapbox_zoom=zoom,
        mapbox_center=center,
        height=height,
        title=title or "ASTD Ship Positions / Tracks",
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    if style != "open-street-map":
        tok = get_mapbox_token()
        if tok:
            layout_kwargs["mapbox_accesstoken"] = tok
        else:
            layout_kwargs["mapbox_style"] = "open-street-map"

    fig.update_layout(**layout_kwargs)
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
        extra_cols_priority=None,
        color_by=None,
) -> go.Figure:
    # Visualize a single track with positions from astd_data.
    cols = resolve_geo_time_cols(astd_data)
    df = cols["_df"].copy()
    lat, lon, tcol = cols["lat"], cols["lon"], cols["time"]

    # --- Key columns available
    seg_col_track = first_present(track_table, SEGMENT_ID_CANDS)
    seg_col_pos = first_present(df, POSITION_SEGMENT_CANDS)
    track_col = first_present(track_table, TRACK_ID_CANDS)
    if seg_col_track is None or seg_col_pos is None or track_col is None:
        raise KeyError(
            "Missing expected columns: "
            f"track_table[{SEGMENT_ID_CANDS + TRACK_ID_CANDS}] and "
            f"astd_data[{POSITION_SEGMENT_CANDS}]."
        )

    # --- Sub-table of the track and normalization of the month -> yyyymm
    tt = track_table[track_table[track_col] == track_id].copy()
    if tt.empty:
        raise ValueError(f"Track '{track_id}' not found in track_table.")
    month_col = first_present(tt, MONTH_CANDS)
    if month_col is None:
        raise KeyError(f"track_table must have a month column in {MONTH_CANDS}.")
    # sort and yyyymm
    tt = tt.sort_values(month_col)
    if tt[month_col].dtype == "O":
        tt["_yyyymm"] = pd.to_datetime(tt[month_col]).dt.strftime("%Y%m")
    else:
        tt["_yyyymm"] = pd.to_datetime(tt[month_col]).dt.strftime("%Y%m")

    # --- Positions side: ensure yyyymm
    if "yyyymm" not in df.columns:
        df["yyyymm"] = pd.to_datetime(df[tcol]).dt.strftime("%Y%m")

    # --- Align type for segment key (int vs str)
    seg_dtype = tt[seg_col_track].dtype
    try:
        df["_segkey"] = df[seg_col_pos].astype(seg_dtype)
        tt_seg = tt[seg_col_track]
    except Exception:
        # fallback: string on both if direct cast is impossible
        df["_segkey"] = df[seg_col_pos].astype(str)
        tt_seg = tt[seg_col_track].astype(str)

    #  Strict join (segment_id, yyyymm)
    pos = df.merge(
        pd.DataFrame({seg_col_track: tt_seg, "_yyyymm": tt["_yyyymm"]}),
        left_on=["_segkey", "yyyymm"],
        how="inner",
    ).copy()

    if pos.empty:
        raise ValueError(
            "No positions found for this track/segments in astd_data after strict (segment, month) filtering."
        )

    # --- Colors by month (sorted)
    sorted_months = sorted(pos["yyyymm"].dropna().unique().tolist())
    color_palette = px.colors.qualitative.Plotly
    month_color_map = {m: color_palette[i % len(color_palette)] for i, m in enumerate(sorted_months)}

    # --- Auto title if not provided
    if title is None:
        flags = ", ".join(sorted([x for x in pos.get("flagname", pd.Series(dtype=str)).dropna().unique().tolist()]))
        title = f"Track {track_id} – Flag: {flags or 'n/a'}"

    legend_months_added = set()
    fig = go.Figure()

    pos_sorted = pos.sort_values([seg_col_pos, tcol])

    for seg, g in pos_sorted.groupby(seg_col_pos):
        if g.empty:
            continue
        g = g.sort_values(tcol)

        sub_g = g

        if len(sub_g) < 2:
            continue

        current_month = sub_g["yyyymm"].iloc[0]
        current_color = month_color_map.get(current_month, "#808080")
        show_legend_for_this = (current_month not in legend_months_added)
        if show_legend_for_this:
            legend_months_added.add(current_month)

        cdata, suffix, used_cols = build_hover_customdata(
            sub_g,
            extra_cols_priority=extra_cols_priority,
            color_by=(color_by if color_by is not None else "yyyymm"),
        )

        if show_segments:
            fig.add_trace(go.Scattermapbox(
                lat=sub_g[lat],
                lon=sub_g[lon],
                mode="lines+markers",
                marker={"size": 4, "color": current_color},
                line={"color": current_color},
                name=current_month,
                legendgroup=current_month,
                showlegend=show_legend_for_this,
                text=sub_g[tcol].dt.strftime("%Y-%m-%d %H:%M:%S"),
                customdata=cdata,
                hovertemplate=(
                    "<b>Month:</b> " + current_month +
                    "<br><b>Segment:</b> " + str(seg) +
                    "<br><b>Date/hour:</b> %{text}"
                    "<br>Lat: %{lat:.4f}  Lon: %{lon:.4f}"
                    f"{suffix}"
                    "<extra></extra>"
                ),
            ))
        else:
            fig.add_trace(go.Scattermapbox(
                lat=sub_g[lat],
                lon=sub_g[lon],
                mode="lines",
                line={"width": 2, "color": current_color},
                name=current_month,
                legendgroup=current_month,
                showlegend=show_legend_for_this,
                hoverinfo="skip",
            ))

    fig.update_layout(
        mapbox_style=resolve_map_style(map_style),
        mapbox_zoom=3,
        mapbox_center={"lat": float(pos[lat].median()), "lon": float(pos[lon].median())},
        height=height,
        title=title,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01, title="Month (yyyymm)"),
    )


    order_index = {m: i for i, m in enumerate(sorted_months)}
    traces = list(fig.data)
    traces.sort(key=lambda tr: order_index.get(tr.name, 10**9))
    fig.data = tuple(traces)

    return fig

