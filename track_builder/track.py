"""
Public API — Partie 3 : Track Matching & Building (thin‑wrapper + day_gap + logs)
===============================================================================

Objectif
--------
Refactor pédagogique en couches claires au‑dessus de `track_builder.main` (alias `_core`).
On isole explicitement :
  (i) préparation des segments → (ii) génération des candidats → (iii) scoring → (iv) sélection greedy
et on **journalise** chaque décision/filtrage avec des logs structurés (`match_id`, `from`, `to`, `stage`, `reason`).

Points clés
-----------
- **day_gap** (écart réel en jours) remplace les heuristiques `month_gap`.
- **Vitesses typiques data‑driven** par catégorie (`astd_cat`) via **quantile 90%** des vitesses de segments,
  **bornées** par des plafonds littérature prudents (km/h) → pas de liste inventée.
- **Score simple** et lisible : a·Δt_norm + b·Δd_norm + c·ratio_vitesse + pénalité(day_gap>seuil).
- API compatible cahier :
    - build_ship_tracks(astd_data, **options) → DataFrame['month','segment_id','track_id']
    - find_track_candidates(segment_id, month, astd_data, top_n=5, **options)
    - get_track_statistics(track_table, astd_data)
- Options supplémentaires : `return_logs: bool=False` pour récupérer un DataFrame `logs` de traçabilité.

Dépendances internes :
- `_core.clean_data`, `_core.get_segment_summaries`, `_core.calculate_candidate_metrics` (pour cohérence projet)
- `_core.haversine_km` si disponible ; sinon fallback Haversine local.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

# On ré‑utilise les briques éprouvées de main.py
from track_builder import main as _core

# =====================================================================
# Utilitaires
# =====================================================================

def _to_ts(x):
    return pd.to_datetime(x, utc=True, errors="coerce")


def _haversine_km(lat1, lon1, lat2, lon2):
    """Fallback Haversine si _core.haversine_km n'existe pas."""
    try:
        return _core.haversine_km(lat1, lon1, lat2, lon2)
    except Exception:
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        c = 2*np.arcsin(np.sqrt(a))
        return R * c

# =====================================================================
# Plafonds prudents (km/h) — uniquement garde‑fous
# =====================================================================
_LIT_CAPS_KMH: Dict[str, float] = {
    'container ships': 46.0,
    'bulk carriers': 28.0,
    'crude oil tankers': 28.0,
    'oil product tankers': 30.0,
    'chemical tankers': 30.0,
    'passenger ships': 37.0,
    'cruise ships': 37.0,
    'fishing vessels': 28.0,
    'refrigerated cargo ships': 46.0,
    'general cargo ships': 28.0,
    'gas tankers': 31.0,
    'ro-ro cargo ships': 33.0,
    'other service offshore vessels': 24.0,
    'offshore supply ships': 24.0,
    'other activities': 24.0,
    'unknown': 24.0,
}

# =====================================================================
# Options & Stratégies
# =====================================================================
MatchingStrategy = Literal["conservative", "balanced", "aggressive"]

_SCORE_THRESHOLDS: Dict[MatchingStrategy, float] = {
    "conservative": 0.40,
    "balanced": 0.55,
    "aggressive": 0.70,
}

_LIMIT_MULTIPLIERS: Dict[MatchingStrategy, Tuple[float, float, float]] = {
    # (time_gap, distance, speed)
    "conservative": (0.9, 0.9, 0.85),
    "balanced": (1.0, 1.0, 1.0),
    "aggressive": (1.2, 1.2, 1.15),
}

@dataclass
class BuildOptions:
    max_time_gap_hours: int = 96      # fenêtre temporelle (heures)
    max_distance_km: int = 1200       # fenêtre spatiale (km)
    min_track_length: int = 1         # #segments min pour garder un track
    matching_strategy: MatchingStrategy = "conservative"
    # paramètres du score
    w_time: float = 0.4
    w_dist: float = 0.4
    w_speed: float = 0.2
    gap_days_no_penalty: float = 3.0
    gap_penalty_per_day: float = 0.05
    return_logs: bool = False         # si True, on renvoie (result_df, logs_df)

# =====================================================================
# (i) Préparation des segments
# =====================================================================

def _prepare_segments(astd_data: pd.DataFrame) -> pd.DataFrame:
    """Nettoie et résume les points ASTD en segments mensuels via `_core`.
    Exige que `get_segment_summaries` retourne au minimum les colonnes:
      ['shipid','month','start_time','end_time','start_lat','start_lon','end_lat','end_lon','astd_cat']
    Si `month` n'est pas présent en entrée brute, `_core.clean_data` / `_core.get_segment_summaries`
    doivent l'ajouter; sinon on peut l'inférer depuis `date_time_utc` (AAAA‑MM).
    """
    data = _core.clean_data(astd_data)
    if len(data) == 0:
        return pd.DataFrame()
    segs = _core.get_segment_summaries(data)
    # normalisation minimale
    needed = {"shipid","month","start_time","end_time","start_lat","start_lon","end_lat","end_lon"}
    missing = needed - set(segs.columns)
    if missing:
        raise ValueError(f"Colonnes segments manquantes (résumé): {missing}")
    # cast temps
    segs = segs.copy()
    segs['start_time'] = _to_ts(segs['start_time'])
    segs['end_time']   = _to_ts(segs['end_time'])
    return segs.sort_values('start_time').reset_index(drop=True)

# =====================================================================
# Vitesses typiques (data‑driven)
# =====================================================================

def _compute_typical_speeds_from_data(segments: pd.DataFrame) -> Dict[str, float]:
    """Q90 des vitesses de segments par `astd_cat`, borné par _LIT_CAPS_KMH.
    Si `astd_cat` manquant → retourne une clé `_global`.
    """
    if segments.empty:
        return {}
    s = segments.copy()
    # distance/durée approximatives du segment (bout‑à‑bout)
    if 'distance_km' not in s.columns:
        s['distance_km'] = _haversine_km(s['start_lat'], s['start_lon'], s['end_lat'], s['end_lon'])
    if 'duration_h' not in s.columns:
        s['duration_h'] = (s['end_time'] - s['start_time']).dt.total_seconds() / 3600.0
    s = s.dropna(subset=['distance_km','duration_h'])
    s = s[s['duration_h'] > 0]
    if s.empty:
        return {}
    s['seg_v_kmh'] = s['distance_km'] / s['duration_h']
    s = s[s['seg_v_kmh'] <= 110]  # coupe valeurs absurdes

    if 'astd_cat' not in s.columns:
        q90 = float(s['seg_v_kmh'].quantile(0.90))
        return {'_global': float(min(max(8.0, q90), 35.0))}

    out: Dict[str, float] = {}
    grp = s.groupby(s['astd_cat'].astype(str).str.lower())['seg_v_kmh'].quantile(0.90)
    for cat, v in grp.items():
        cap = _LIT_CAPS_KMH.get(cat, 30.0)
        out[cat] = float(min(max(8.0, v), cap))
    return out

# =====================================================================
# (ii) Génération des candidats + (iii) Scoring
# =====================================================================

def _generate_and_score_candidates(cur: pd.Series,
                                   pool: pd.DataFrame,
                                   opts: BuildOptions,
                                   speed_lookup: Dict[str,float],
                                   multipliers: Tuple[float,float,float],
                                   logs: List[Dict]) -> pd.DataFrame:
    """Filtre les candidats en appliquant (temps, distance, vitesse implicite),
    puis calcule un score simple. Journalise les rejets pour traçabilité.
    Retourne un DataFrame trié par `score` croissant.
    """
    tg_mul, dist_mul, spd_mul = multipliers

    # 1) Calculs de base
    c = pool.copy()
    c['dt_hours'] = (c['start_time'] - cur['end_time']).dt.total_seconds() / 3600.0
    c['day_gap']  = c['dt_hours'] / 24.0
    # Dist fin→début
    c['distance_km_fd'] = _haversine_km(cur['end_lat'], cur['end_lon'], c['start_lat'], c['start_lon'])

    # 2) Filtres + logs
    def _log(row, stage, reason):
        logs.append({
            'match_id': f"{cur['shipid']}→{row.get('shipid', row.get('segment_id','?'))}",
            'from_shipid': cur['shipid'],
            'to_shipid': row.get('shipid', row.get('segment_id','?')),
            'from_month': cur['month'],
            'to_month': row.get('month','?'),
            'stage': stage,
            'reason': reason,
            'dt_hours': row.get('dt_hours', np.nan),
            'distance_km_fd': row.get('distance_km_fd', np.nan),
            'implied_v_kmh': row.get('implied_v_kmh', np.nan),
        })

    # a) temps non négatif et dans la fenêtre
    bad_time = (c['dt_hours'] < 0) | (c['dt_hours'] > opts.max_time_gap_hours * tg_mul)
    for _, r in c[bad_time].iterrows():
        _log(r, 'filter', 'time_window')
    c = c[~bad_time]
    if c.empty:
        return c

    # b) distance dans la fenêtre
    bad_dist = c['distance_km_fd'] > (opts.max_distance_km * dist_mul)
    for _, r in c[bad_dist].iterrows():
        _log(r, 'filter', 'distance_window')
    c = c[~bad_dist]
    if c.empty:
        return c

    # c) vitesse implicite plausible
    # éviter division par 0
    dt_h = c['dt_hours'].replace(0, np.finfo(float).eps)
    c['implied_v_kmh'] = c['distance_km_fd'] / dt_h
    ship_type = str(cur.get('astd_cat','unknown')).lower()
    typical = speed_lookup.get(ship_type, _LIT_CAPS_KMH.get(ship_type, 24.0))
    max_v = typical * 1.3 * spd_mul

    bad_speed = c['implied_v_kmh'] > max_v
    for _, r in c[bad_speed].iterrows():
        _log(r, 'filter', 'speed_cap')
    c = c[~bad_speed]
    if c.empty:
        return c

    # 3) Scoring (plus petit = meilleur)
    dt_norm = (c['dt_hours'] / (opts.max_time_gap_hours * max(tg_mul, 1e-9))).clip(upper=1.0)
    dd_norm = (c['distance_km_fd'] / (opts.max_distance_km * max(dist_mul, 1e-9))).clip(upper=1.0)
    vr = (c['implied_v_kmh'] / max(1.0, typical)).clip(upper=2.0)
    penalty = (c['day_gap'] - opts.gap_days_no_penalty).clip(lower=0) * opts.gap_penalty_per_day

    c['match_score_simple'] = opts.w_time*dt_norm + opts.w_dist*dd_norm + opts.w_speed*vr + penalty

    # 4) Score "amélioré" (si dispo dans _core) — on le considère en *secondaire*
    try:
        c2 = _core.calculate_improved_match_score(c.copy(), ship_type, cur)
        c['match_score_core'] = c2['match_score'] if 'match_score' in c2 else np.nan
    except Exception:
        c['match_score_core'] = np.nan

    # 5) Seuil par stratégie (sur le score amélioré s'il existe, sinon le simple)
    return c.sort_values(['match_score_simple','dt_hours','distance_km_fd']).reset_index(drop=True)

# =====================================================================
# (iv) Sélection greedy + API publique
# =====================================================================

def build_ship_tracks(
    astd_data: pd.DataFrame,
    *,
    max_time_gap_hours: int = 96,
    max_distance_km: int = 1200,
    min_track_length: int = 1,
    matching_strategy: MatchingStrategy = "conservative",
    w_time: float = 0.4,
    w_dist: float = 0.4,
    w_speed: float = 0.2,
    gap_days_no_penalty: float = 3.0,
    gap_penalty_per_day: float = 0.05,
    return_logs: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """Connecte des segments en trajectoires continues (day_gap + logs).

    Retour: DataFrame ['month','segment_id','track_id']
    Si `return_logs=True`, retourne `(result_df, logs_df)`.
    """
    # 1) Segments
    segs = _prepare_segments(astd_data)
    if segs.empty:
        res = pd.DataFrame(columns=["month","segment_id","track_id"])
        return (res, pd.DataFrame()) if return_logs else res

    # 2) Paramètres & vitesses typiques
    opts = BuildOptions(
        max_time_gap_hours=max_time_gap_hours,
        max_distance_km=max_distance_km,
        min_track_length=min_track_length,
        matching_strategy=matching_strategy,
        w_time=w_time, w_dist=w_dist, w_speed=w_speed,
        gap_days_no_penalty=gap_days_no_penalty,
        gap_penalty_per_day=gap_penalty_per_day,
        return_logs=return_logs,
    )
    score_threshold = _SCORE_THRESHOLDS[opts.matching_strategy]
    multipliers = _LIMIT_MULTIPLIERS[opts.matching_strategy]

    speed_lookup = _compute_typical_speeds_from_data(segs)

    # 3) Indexation par mois
    def _mkey(m: str) -> int:
        y, M = str(m).split('-')
        return int(y)*12 + int(M)
    months = sorted(segs['month'].unique(), key=_mkey)
    by_month = {m: segs[segs['month']==m].copy() for m in months}

    # 4) Greedy chronologique
    logs: List[Dict] = []
    track_id = 0
    assigned: Dict[Tuple[str,str], int] = {}

    for mi, m in enumerate(months):
        cur_month = by_month[m]
        for _, cur in cur_month.iterrows():
            key = (cur['month'], cur['shipid'])
            if key in assigned:
                continue
            track_id += 1
            assigned[key] = track_id
            tail = cur

            # extension
            for nxt in months[mi+1:]:
                pool = by_month[nxt]
                # priorité douce même catégorie en tête
                if 'astd_cat' in pool.columns:
                    same = pool['astd_cat'].astype(str).str.lower() == str(tail.get('astd_cat','')).lower()
                    pool = pd.concat([pool.loc[same], pool.loc[~same]], ignore_index=True)

                # génération + score
                cands = _generate_and_score_candidates(tail, pool, opts, speed_lookup, multipliers, logs)
                if cands.empty:
                    break

                # seuil stratégie — utiliser score_core s'il existe sinon simple
                if 'match_score_core' in cands and cands['match_score_core'].notna().any():
                    cands_ok = cands[(cands['match_score_core'] <= score_threshold) | (cands['match_score_core'].isna())]
                else:
                    cands_ok = cands[cands['match_score_simple'] <= score_threshold]
                if cands_ok.empty:
                    break

                # choisir le 1er non assigné
                chosen = None
                for _, r in cands_ok.iterrows():
                    k2 = (r['month'], r['shipid'])
                    if k2 not in assigned:
                        chosen = r
                        break
                    else:
                        logs.append({'match_id': f"{tail['shipid']}→{r['shipid']}", 'stage':'skip', 'reason':'already_assigned'})
                if chosen is None:
                    break

                assigned[(chosen['month'], chosen['shipid'])] = track_id
                tail = chosen

    # 5) Sortie normalisée + filtrage longueur
    out = pd.DataFrame([
        {'month': k[0], 'segment_id': k[1], 'track_id': tid} for k, tid in assigned.items()
    ])
    if out.empty:
        res = out.reindex(columns=['month','segment_id','track_id'])
        return (res, pd.DataFrame(logs)) if return_logs else res

    sizes = out.groupby('track_id').size()
    keep_ids = sizes[sizes >= opts.min_track_length].index
    out = out[out['track_id'].isin(keep_ids)].sort_values(['track_id','month']).reset_index(drop=True)

    return (out, pd.DataFrame(logs)) if return_logs else out


def find_track_candidates(
    segment_id: str,
    month: str,
    astd_data: pd.DataFrame,
    *,
    top_n: int = 5,
    matching_strategy: MatchingStrategy = "conservative",
    max_time_gap_hours: int = 96,
    max_distance_km: int = 1200,
    w_time: float = 0.4,
    w_dist: float = 0.4,
    w_speed: float = 0.2,
    gap_days_no_penalty: float = 3.0,
    gap_penalty_per_day: float = 0.05,
    return_logs: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """Renvoie les meilleurs candidats pour un segment (avec logs optionnels)."""
    segs = _prepare_segments(astd_data)
    segment_id_str = str(segment_id)
    segs = segs.copy()
    segs['shipid_str'] = segs['shipid'].astype(str)
    this = segs[(segs['month'] == month) & (segs['shipid_str'] == segment_id_str)]
    if this.empty:
        raise ValueError("Segment introuvable pour ce mois.")
    cur = this.iloc[0]

    if segs.empty:
        res = pd.DataFrame()
        return (res, pd.DataFrame()) if return_logs else res

    this = segs[(segs['month']==month) & (segs['shipid']==segment_id)]
    if this.empty:
        raise ValueError("Segment introuvable pour ce mois.")
    cur = this.iloc[0]

    opts = BuildOptions(
        max_time_gap_hours=max_time_gap_hours,
        max_distance_km=max_distance_km,
        matching_strategy=matching_strategy,
        w_time=w_time, w_dist=w_dist, w_speed=w_speed,
        gap_days_no_penalty=gap_days_no_penalty,
        gap_penalty_per_day=gap_penalty_per_day,
        return_logs=return_logs,
    )
    multipliers = _LIMIT_MULTIPLIERS[opts.matching_strategy]
    speed_lookup = _compute_typical_speeds_from_data(segs)

    # pool = tous segments dont le start est après fin du courant et dans la fenêtre d'heures
    segs = segs.copy()
    segs['dt_hours'] = (segs['start_time'] - cur['end_time']).dt.total_seconds() / 3600.0
    pool = segs[(segs['dt_hours'] >= 0) & (segs['dt_hours'] <= opts.max_time_gap_hours * multipliers[0])]

    logs: List[Dict] = []
    cands = _generate_and_score_candidates(cur, pool, opts, speed_lookup, multipliers, logs)

    # tri final et colonnes utiles (uniformiser en 'segment_id')
    cands = cands.rename(columns={'shipid': 'segment_id'})
    use_cols = ['month', 'segment_id', 'match_score_simple', 'match_score_core',
                'distance_km_fd', 'implied_v_kmh', 'dt_hours']
    # garder seulement les colonnes qui existent (selon score_core dispo ou non)
    use_cols = [c for c in use_cols if c in cands.columns]
    cands = cands[use_cols].head(top_n).reset_index(drop=True)

    return (cands, pd.DataFrame(logs)) if return_logs else cands



def get_track_statistics(track_table: pd.DataFrame, astd_data: pd.DataFrame) -> Dict[str, object]:
    """Analyse synthétique des tracks (inchangé)."""
    if track_table is None or track_table.empty:
        return {
            "n_tracks": 0,
            "n_segments": 0,
            "avg_length": 0.0,
            "max_length": 0,
            "lengths": pd.Series(dtype=int),
            "by_month": pd.Series(dtype=int),
            "by_ship_type": pd.Series(dtype=int),
        }

    lengths = track_table.groupby('track_id').size()
    n_tracks = lengths.size
    n_segments = int(lengths.sum())
    avg_len = float(lengths.mean()) if n_tracks else 0.0
    max_len = int(lengths.max()) if n_tracks else 0

    by_month = track_table.groupby('month').size().sort_index()

    by_ship_type = pd.Series(dtype=int)
    required_cols = {"shipid", "astd_cat"}
    if isinstance(astd_data, pd.DataFrame) and required_cols.issubset(set(astd_data.columns)):
        seg2type = astd_data.drop_duplicates('shipid')[['shipid','astd_cat']].set_index('shipid')['astd_cat'].str.lower()
        tmp = track_table.merge(seg2type.rename_axis('segment_id'), left_on='segment_id', right_index=True, how='left')
        by_ship_type = tmp['astd_cat'].fillna('unknown').value_counts().sort_values(ascending=False)

    return {
        "n_tracks": int(n_tracks),
        "n_segments": int(n_segments),
        "avg_length": float(avg_len),
        "max_length": int(max_len),
        "lengths": lengths,
        "by_month": by_month,
        "by_ship_type": by_ship_type,
    }

