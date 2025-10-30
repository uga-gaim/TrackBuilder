"""
Objective
---------
Pedagogical refactor into clear layers on top of `track_builder.main` (alias `_core`).
We explicitly isolate:
    (i) segment preparation → (ii) candidate generation → (iii) scoring → (iv) greedy selection
and we **log** every decision/filter with structured logs (`match_id`, `from`, `to`, `stage`, `reason`).

Key points
----------
- **day_gap** (actual gap in days) replaces heuristic `month_gap`.
- **Data-driven typical speeds** by category (`astd_cat`) using the 90th percentile of segment speeds,
    **bounded** by conservative literature caps (km/h) → not an invented list.
- Simple, readable score: a·Δt_norm + b·Δd_norm + c·speed_ratio + penalty(day_gap > threshold).
- API compatible with the notebook:
        - build_ship_tracks(astd_data, **options) → DataFrame['month','segment_id','track_id']
        - find_track_candidates(segment_id, month, astd_data, top_n=5, **options)
        - get_track_statistics(track_table, astd_data)
- Additional option: `return_logs: bool=False` to retrieve a `logs` DataFrame for traceability.

Internal dependencies:
- `_core.clean_data`, `_core.get_segment_summaries`, `_core.calculate_candidate_metrics` (for project consistency)
- `_core.haversine_km` if available; otherwise a local Haversine fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Reuse proven building blocks from track_v0.py
from track_builder import track_v0 as _core
from track_builder.core.track_helpers import haversine_km, to_ts
from track_builder.config import _LIT_CAPS_KMH, MatchingStrategy, _SCORE_THRESHOLDS, _LIMIT_MULTIPLIERS


@dataclass
class BuildOptions:
    max_time_gap_hours: int = 96      # temporal window (hours)
    max_distance_km: int = 1200       # spatial window (km)
    min_track_length: int = 1         # min #segments to keep a track
    matching_strategy: MatchingStrategy = "conservative"
    # scoring parameters
    w_time: float = 0.4
    w_dist: float = 0.4
    w_speed: float = 0.2
    gap_days_no_penalty: float = 3.0
    gap_penalty_per_day: float = 0.05
    return_logs: bool = False         # if True, return (result_df, logs_df)
    speed_margin: float = 1.3        # margin on typical speed for filtering
# =====================================================================
# (i) Segment preparation
# =====================================================================

def _prepare_segments(astd_data: pd.DataFrame) -> pd.DataFrame:
    """Cleans and summarizes ASTD points into monthly segments via `_core`.
    Requires `get_segment_summaries` to return at least the following columns:
      ['shipid','month','start_time','end_time','start_lat','start_lon','end_lat','end_lon','astd_cat']
    If `month` is not present in raw input, `_core.clean_data` / `_core.get_segment_summaries`
    must add it; otherwise it can be inferred from `date_time_utc` (YYYY-MM).
    """
    data = _core.clean_data(astd_data)
    if len(data) == 0:
        return pd.DataFrame()
    segs = _core.get_segment_summaries(data)
    # normalize and checks
    needed = {"shipid","month","start_time","end_time","start_lat","start_lon","end_lat","end_lon"}
    missing = needed - set(segs.columns)
    if missing:
        raise ValueError(f"Missing segment columns (summary): {missing}")
    # cast temps
    segs = segs.copy()
    segs['start_time'] = to_ts(segs['start_time'])
    segs['end_time']   = to_ts(segs['end_time'])
    return segs.sort_values('start_time').reset_index(drop=True)

# =====================================================================
# Typical speeds (data‑driven)
# =====================================================================

def _compute_typical_speeds_from_data(segments: pd.DataFrame) -> Dict[str, float]:
    """
    Compute typical (90th-percentile) segment speeds from raw segment data.
    This function derives representative speeds (in km/h) from a table of segments by
    computing the 90th percentile (Q90) of per-segment speeds and applying sensible
    clipping rules. It is robust to missing precomputed distance/duration columns and
    performs several filters to remove invalid or absurd values.
    Parameters
    ----------
    segments : pandas.DataFrame
        DataFrame containing segment records. Expected columns (at minimum):
          - start_lat, start_lon, end_lat, end_lon
          - start_time (dtype datetime-like), end_time (dtype datetime-like)
        Optional columns that will be used if present:
          - distance_km : float — precomputed segment distance in kilometres
          - duration_h  : float — precomputed segment duration in hours
          - astd_cat    : category or string — activity/category used to group speeds
        If distance_km is missing, distance is computed with haversine_km on the
        start/end coordinates. If duration_h is missing, it is computed from
        (end_time - start_time) in hours.
    Returns
    -------
    dict[str, float]
        Mapping from category (lowercased string) to a typical speed in km/h.
        - If the input has no 'astd_cat' column, returns a single entry with key
          '_global' containing the clipped global Q90 speed.
        - If 'astd_cat' is present, returns one key per distinct category
          (converted to str and lowercased).
        - If there are no valid segments after filtering, returns an empty dict.
    Behavior and filtering
    ----------------------
    1. A working copy of the input DataFrame is used (original is not modified).
    2. distance_km and duration_h are computed if absent (see Parameters).
    3. Rows with NaN distance_km or duration_h, or with duration_h <= 0, are dropped.
    4. Per-segment speed seg_v_kmh = distance_km / duration_h is computed.
    5. Speeds above 110 km/h are discarded as absurd.
    6. The 90th percentile (quantile 0.90) of seg_v_kmh is computed per category.
       - When grouping by astd_cat, categories are converted to string and lowercased
         before grouping, so returned keys are lowercase strings.
    7. Clipping rules:
       - For per-category values: result = min(max(8.0, q90), cap), where cap =
         _LIT_CAPS_KMH.get(category, 30.0).
       - For the global value (no astd_cat column): result = min(max(8.0, q90), 35.0).
    Notes
    -----
    - The function returns floats (km/h) and uses a lower bound of 8.0 km/h to avoid
      unrealistically low typical speeds.
    - Category-specific caps come from the module-level mapping _LIT_CAPS_KMH;
      categories not present in that mapping default to a 30 km/h cap (global uses 35).
    - No exceptions are raised for missing optional columns; lack of valid data
      simply results in an empty dict.
    - The function is robust to empty input DataFrames.
    Q90 of segment speeds by `astd_cat`, capped by _LIT_CAPS_KMH.
    If `astd_cat` is missing → returns a `_global` key.
    """
    if segments.empty:
        return {}
    s = segments.copy()
    # Compute missing distance/duration if needed
    if 'distance_km' not in s.columns:
        s['distance_km'] = haversine_km(s['start_lat'], s['start_lon'], s['end_lat'], s['end_lon'])
    if 'duration_h' not in s.columns:
        s['duration_h'] = (s['end_time'] - s['start_time']).dt.total_seconds() / 3600.0
    s = s.dropna(subset=['distance_km','duration_h'])
    s = s[s['duration_h'] > 0]
    if s.empty:
        return {}
    s['seg_v_kmh'] = s['distance_km'] / s['duration_h']
    s = s[s['seg_v_kmh'] <= 110]  # filter absurd speeds

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
# (ii) Candidate generation + (iii) scoring
# =====================================================================

def _generate_and_score_candidates(cur: pd.Series,
                                   pool: pd.DataFrame,
                                   opts: BuildOptions,
                                   speed_lookup: Dict[str,float],
                                   multipliers: Tuple[float,float,float],
                                   logs: List[Dict]) -> pd.DataFrame:
    """Filter candidates by (time, distance, implied speed), then compute a simple score.
    Returns a DataFrame sorted by ascending `match_score_simple`.
    """
    tg_mul, dist_mul, spd_mul = multipliers

    # 1) base calculations
    c = pool.copy()
    c['dt_hours'] = (c['start_time'] - cur['end_time']).dt.total_seconds() / 3600.0
    c['day_gap']  = c['dt_hours'] / 24.0
    # Dist front‑to‑back
    c['distance_km_fd'] = haversine_km(cur['end_lat'], cur['end_lon'], c['start_lat'], c['start_lon'])

    # 2) Filtering + logging
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

    # a) no negative time or too large time gap
    bad_time = (c['dt_hours'] < 0) | (c['dt_hours'] > opts.max_time_gap_hours * tg_mul)
    for _, r in c[bad_time].iterrows():
        _log(r, 'filter', 'time_window')
    c = c[~bad_time]
    if c.empty:
        return c

    # b) distance within limit
    bad_dist = c['distance_km_fd'] > (opts.max_distance_km * dist_mul)
    for _, r in c[bad_dist].iterrows():
        _log(r, 'filter', 'distance_window')
    c = c[~bad_dist]
    if c.empty:
        return c

    # c) plausible implied speed
    # avoid division by 0
    dt_h = c['dt_hours'].replace(0, np.finfo(float).eps)
    c['implied_v_kmh'] = c['distance_km_fd'] / dt_h
    ship_type = str(cur.get('astd_cat','unknown')).lower()
    typical = speed_lookup.get(ship_type, _LIT_CAPS_KMH.get(ship_type, 24.0))
    max_v = typical * opts.speed_margin * spd_mul

    bad_speed = c['implied_v_kmh'] > max_v
    for _, r in c[bad_speed].iterrows():
        _log(r, 'filter', 'speed_cap')
    c = c[~bad_speed]
    if c.empty:
        return c

    # 3) Scoring (lower is better)
    dt_norm = (c['dt_hours'] / (opts.max_time_gap_hours * max(tg_mul, 1e-9))).clip(upper=1.0)
    dd_norm = (c['distance_km_fd'] / (opts.max_distance_km * max(dist_mul, 1e-9))).clip(upper=1.0)
    vr = (c['implied_v_kmh'] / max(1.0, typical)).clip(upper=2.0)
    penalty = (c['day_gap'] - opts.gap_days_no_penalty).clip(lower=0) * opts.gap_penalty_per_day

    c['match_score_simple'] = opts.w_time*dt_norm + opts.w_dist*dd_norm + opts.w_speed*vr + penalty

    # 4) Optional: improved score from _core (may fail)
    try:
        c2 = _core.calculate_improved_match_score(c.copy(), ship_type, cur)
        c['match_score_core'] = c2['match_score'] if 'match_score' in c2 else np.nan
    except Exception:
        c['match_score_core'] = np.nan

    # 5) Strategy threshold: prefer the improved core score when available, otherwise fall back to the simple score
    return c.sort_values(['match_score_simple','dt_hours','distance_km_fd']).reset_index(drop=True)

# =====================================================================
# (iv) Greedy track building
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
    """Connect segments into continuous tracks using day gaps and structured logs.

    Returns a DataFrame with columns ['month','segment_id','track_id'].
    If `return_logs=True`, returns a tuple: (result_df, logs_df).
    """
    # 1) Segments
    segs = _prepare_segments(astd_data)
    if segs.empty:
        res = pd.DataFrame(columns=["month","segment_id","track_id"])
        return (res, pd.DataFrame()) if return_logs else res

    # 2) Options + typical speeds
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

    # 3) Organize by month
    def _mkey(m: str) -> int:
        y, M = str(m).split('-')
        return int(y)*12 + int(M)
    months = sorted(segs['month'].unique(), key=_mkey)
    by_month = {m: segs[segs['month']==m].copy() for m in months}

    # 4) Greedy linking
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
            for nxt in months[mi+1:mi+2]:  # limit to immediate next month
                pool = by_month[nxt]
                # soft priority same category first
                if 'astd_cat' in pool.columns:
                    same = pool['astd_cat'].astype(str).str.lower() == str(tail.get('astd_cat','')).lower()
                    pool = pd.concat([pool.loc[same], pool.loc[~same]], ignore_index=True)

                # generation + score
                cands = _generate_and_score_candidates(tail, pool, opts, speed_lookup, multipliers, logs)
                if cands.empty:
                    break

                # strategy threshold — use match_score_core if available, otherwise fall back to match_score_simple
                if 'match_score_core' in cands and cands['match_score_core'].notna().any():
                    cands_ok = cands[(cands['match_score_core'] <= score_threshold) | (cands['match_score_core'].isna())]
                else:
                    cands_ok = cands[cands['match_score_simple'] <= score_threshold]
                if cands_ok.empty:
                    break

                # choose best candidate not already assigned
                # sort before choosing
                cands_ok = cands_ok.sort_values(['match_score_simple','dt_hours','distance_km_fd']).reset_index(drop=True)
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

    # 5) Output assembly + filtering short tracks
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
    max_time_gap_hours: int = 48,
    max_distance_km: int = 600,
    w_time: float = 0.4,
    w_dist: float = 0.4,
    w_speed: float = 0.2,
    gap_days_no_penalty: float = 2.0,
    gap_penalty_per_day: float = 0.05,
    return_logs: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """Find possible next segments for a given segment in a given month.
    Returns a DataFrame with columns:"""
    segs = _prepare_segments(astd_data)
    segment_id_str = str(segment_id)
    segs = segs.copy()
    segs['shipid_str'] = segs['shipid'].astype(str)
    this = segs[(segs['month'] == month) & (segs['shipid_str'] == segment_id_str)]
    if this.empty:
        raise ValueError("Segment ID not found for the specified month.")
    cur = this.iloc[0]

    if segs.empty:
        res = pd.DataFrame()
        return (res, pd.DataFrame()) if return_logs else res

    this = segs[(segs['month']==month) & (segs['shipid']==segment_id)]
    if this.empty:
        raise ValueError("Segment ID not found for the specified month.")
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

    # pool = all segments whose start is after the end of the current one and within the time window
    segs = segs.copy()
    segs['dt_hours'] = (segs['start_time'] - cur['end_time']).dt.total_seconds() / 3600.0
    pool = segs[(segs['dt_hours'] >= 0) & (segs['dt_hours'] <= opts.max_time_gap_hours * multipliers[0])]

    logs: List[Dict] = []
    cands = _generate_and_score_candidates(cur, pool, opts, speed_lookup, multipliers, logs)

    # final formatting
    cands = cands.rename(columns={'shipid': 'segment_id'})
    use_cols = ['month', 'segment_id', 'match_score_simple', 'match_score_core',
                'distance_km_fd', 'implied_v_kmh', 'dt_hours']
    # keep only columns that exist (depending on whether score_core is available or not)
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

