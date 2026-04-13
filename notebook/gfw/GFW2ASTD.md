# GFW to ASTD - Matching rules
This file present the rules used to match GFW (mmsi) to ASTD (shipid) fishing ships.

## 1. Description

### GFW
[Global Fishing Watch. 2025. Global Apparent Fishing Effort Dataset, Version 3.0. doi:10.5281/zenodo.14982712](https://globalfishingwatch.org/data-download/datasets/public-fishing-effort)

`mmsi` : ship identifier - over its full "lifecycle" \
`longitude / latitude` : bottom left corner of the boat position grid with a resolution of 0.1*0.1 \
`hours` : time spent in the position grid \
`day` : the day of the record

### ASTD
`shipid` : ship identifier - only over each month \
`longitude / latitude` : precise position of the boat \
`date time` : the exact moment the position was saved - year/month/day-hours:minutes:seconds

## 2. Objective
Marching GFW ("ground truth") to ASTD monthly segments will help evaluate TrackBuilder performances to accurately build tracks, while analyzing score calculations' correctness.

### Expected output :

```json
{
  "{mmsi}": {
    "{month}": "{shipid}",
    "{month2}": "{shipid2}",
    ...
  }
}
```
```json
{
  54321 : {
    "2020-01": 789,
    "2020-02": 456,
    "2020-03": 123
  },
  98765 : {
    "2020-05": 321,
    "2020-07": 654,
    "2020-12": 987
  }
}
```

## 3. Pre-processing
### Aggregation
> Aggregating by day and by grid helps lighten the number of rows in the data by grouping all unique positions in the same area into one.

#### Per days
For both datasets, we group the data points by day.
We then filter the ships using a date-based reference with daily resolution.
This means that all records for a given ship within the same day, regardless of the exact time, are represented by that single day in the timeline.


#### Per grids
In ASTD, we convert positions into grid cells using the same resolution.
Each duplicate grid cells during a day are removed to keep only one reference of the position during the day.

#### Example :

| Ship  | Timestamp        | Latitude | Longitude |
|-------|------------------|----------|-----------|
| 54321 | 2024-01-01 08:15 | 45.501   | -73.567   |
| 54321 | 2024-01-01 12:30 | 45.502   | -73.568   |
| 54321 | 2024-01-01 13:34 | 44.562   | -73.234   |
| 54321 | 2024-01-02 03:45 | 42.499   | -72.445   |

becomes:

| ship  | Day        | Grid Lat | Grid Lon |
|-------|------------|----------|----------|
| 54321 | 2024-01-01 | 45.5     | -73.6    |
| 54321 | 2024-01-01 | 44.5     | -73.3    |
| 54321 | 2024-01-02 | 42.4     | -72.5    |

[//]: # (___)

[//]: # (### Sampling)

[//]: # (> Sampling reduces GFW data points to make the matching process faster. For each mmsi, per month, we only keep a % subset of the per day & per grid records.)

[//]: # (Sampling is optional, and by default we don't use it. All GFW data for a month are kept during the matching process. \\)

[//]: # (If enabled, sampling is performed by selecting, for each MMSI and for each month, a defined fraction of the data randomly. )

[//]: # (Alternatively, a subset of the data can be selected for each MMSI, per month, using evenly spaced intervals.)

___
### Join
> Ths join (merge) gives us all the matching ship between GFW and ASTD that have been in the same area (grid) the same day.

We do an inner join on :
- the day
- the grid latitude
- the grid longitude
___
### Scores

We compute scores on both datasets : 
* GFW
  * for each mmsi, per month
    * `gfw_n_mmsi` (or `sample_n_mmsi` for the GFW samples) : the total number of per day & per grid records
* ASTD
  * for each shipid, per month
    * `astd_n_ship` : the total number of per day & per grid records
* Joined data
  * for each mmsi, per month, per matched shipid
    * `match_n_mmsi` : the total number of per day & per grid match


For the joined data, we then compute for each mmsi, per month:
#### Ratio_GFW
> The ratio between `match_n_mmsi` and `gfw_n_mmsi` (proportion of matched points relative to total number of GFW points, for each mmsi, per month):
$$
ratio_{GFW} = \frac{merged\_n\_mmsi}{gfw\_n\_mmsi}
$$

#### Ratio_ASTD
> The ratio between `match_n_mmsi` and `astd_n_ship` (proportion of matched points relative to total number of ASTD points for each shipid, per month):
$$
ratio_{ASTD} = \frac{merged\_n\_mmsi}{astd\_n\_ship}
$$
For example, for a given shipid and month, `ratio_ASTD` measures the proportion of daily and grids records that fall within the matched mmsi grids for that month.
> - If `ratio_ASTD = 1`, the match is perfect.
> - If `ratio_ASTD > 1`, there are more records than expected.
> - If `ratio_ASTD < 1`, there are fewer records than expected.


#### Score
> The score between both computed ratio:
$$
score = ratio_{GFW} + ratio_{ASTD}
$$

## 4. Matching Thresholds
The ratios and score computed earlier help define the tolerance over how much missing data we allow for a match to be considered valid.

```python
mask = (
    # 1. GFW coverage constraint (adaptive)
    (
        ((match_n_mmsi < n_small) & (ratio_GFW >= t_gfw_high)) |
        ((match_n_mmsi >= n_small) & (ratio_GFW >= t_gfw_low))
    )

    # 2. ASTD coverage constraint (strict)
    & (ratio_ASTD >= t_astd)

    # 3. Best match selection
    & (score == max_score)
)
```
Where:

- `t_astd` : strict threshold ensuring high coverage of ASTD points  
- `t_gfw_high` : stricter threshold applied to small tracks  
- `t_gfw_low` : more flexible threshold applied to larger tracks  
- `n_small` : cutoff defining small vs large tracks  

> `ratio_ASTD` must be above a strict threshold (`t_astd`) (e.g., 0.9): this constraint is intentionally strict (high) because we do not expect the GFW dataset to contain many missing data points. 
> As a result, nearly all ASTD points should be represented in the matched MMSI points (if no sample is selected).

> `ratio_GFW` must satisfy an adaptive threshold that depends on track size: a stricter threshold `t_gfw_high` (e.g., 0.7) is applied for small tracks, while a more flexible threshold `t_gfw_low` (e.g., 0.1) is used for larger tracks*. This reflects the expectation that ASTD data may contain missing observations, meaning that not all GFW points are necessarily expected to have a match.
> 
> *smaller tracks have a more stricts threshold (higher) as ships tend to stay longer in the same area so it's less possible that they have missing points (e.g. if it stays at bay for many days)

> We retain only the match(es) with the highest score within each `(mmsi, month)` group. If multiple candidates share the same maximum score, they are dismissed.

> We only keep (mmsi, month) with a unique shipid, and shipid with unique (mmsi, month) : 1-1 link