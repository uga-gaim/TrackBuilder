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
    Visualisation simple des positions et, si disponible, des lignes par track_id.
    - Réutilise la standardisation de colonnes (Partie 1).
    - N’effectue pas de vérifs d’I/O (déjà faites en amont par load_astd_data).
    """
    cols = resolve_geo_time_cols(data)
    df = cols["_df"]  # df standardisé (copie légère fournie par le helper)
    lat, lon, tcol = cols["lat"], cols["lon"], cols["time"]

    # filtre optionnel par track_id si présent
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

    # lignes par track si on a la colonne
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
    Vue détaillée d’un track unique, compatible avec la table émise par main.build_track_table().
    Hypothèses (issues du main.py précédent) :
      - track_table contient au moins: 'track_id' et 'segment_id' (ou équivalents)
      - astd_data (positions) contient souvent 'shipid' plutôt que 'segment_id'
    On aligne donc 'segment_id' (track_table) <-> 'shipid' (positions) sans supposer le même nom.

    Paramètres
    ----------
    track_id : str|int
        Identifiant du track à visualiser.
    track_table : pd.DataFrame
        Table des tracks (ex: colonnes ['track_id','segment_id','month']).
    astd_data : pd.DataFrame
        Données positions déjà chargées/standardisées par la Partie 1.
    show_segments : bool
        Si True, trace une polyline par segment (légende = identifiant côté positions).
    map_style : str
        Style des tuiles Mapbox/OSM ('open-street-map' ne nécessite pas de token).
    height : int
        Hauteur de la figure Plotly.
    title : str|None
        Titre personnalisé.

    Retour
    ------
    plotly.graph_objects.Figure
    """

    # Colonnes géo + temps (réutilise la standardisation Partie 1 ; pas de re-validations I/O ici)
    cols = resolve_geo_time_cols(astd_data)
    df = cols["_df"]
    lat, lon, tcol = cols["lat"], cols["lon"], cols["time"]

    # Résolution séparée des colonnes 'segment' pour chaque DataFrame
    seg_col_track = first_present(track_table, SEGMENT_ID_CANDS)       # p.ex. 'segment_id' côté track_table
    seg_col_pos = first_present(df, POSITION_SEGMENT_CANDS)          # p.ex. 'shipid'      côté positions
    track_col = first_present(track_table, TRACK_ID_CANDS)         # p.ex. 'track_id'

    if seg_col_track is None or seg_col_pos is None or track_col is None:
        raise KeyError(
            "Colonnes attendues manquantes : "
            f"track_table[{SEGMENT_ID_CANDS + TRACK_ID_CANDS}] et "
            f"astd_data[{POSITION_SEGMENT_CANDS}]."
        )

    # Sélection des segments pour le track demandé (tri par mois si présent)
    tt = track_table[track_table[track_col] == track_id].copy()
    month_col = first_present(track_table, MONTH_CANDS)
    if month_col:
        tt = tt.sort_values(month_col)

    if tt.empty:
        raise ValueError(f"Track '{track_id}' introuvable dans track_table.")

    # Liste des segments côté track_table (ex: valeurs de 'segment_id')
    segment_ids = set(tt[seg_col_track].dropna().unique().tolist())

    # On filtre les positions avec la colonne équivalente côté positions (ex: 'shipid')
    pos = df[df[seg_col_pos].isin(segment_ids)].copy()
    if pos.empty:
        raise ValueError(
            "Aucune position trouvée pour ce track/segments dans astd_data. "
            f"Comparé via astd_data['{seg_col_pos}'] ∈ track_table['{seg_col_track}']."
        )

    # Construction de la figure
    fig = go.Figure()

    if show_segments:
        # Une polyline par 'segment' tel que nommé côté positions (souvent 'shipid')
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
        # Une seule polyline pour tout le track
        g = pos.sort_values(tcol)
        fig.add_trace(go.Scattermapbox(
            lat=g[lat],
            lon=g[lon],
            mode="lines+markers",
            marker={"size": 4},
            name=f"Track {track_id}",
            hoverinfo="skip",
        ))

    # Mise en page / centrage
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
