"""
Load and transform PSLM Lahore.dta for simulation validation.
All rows in the file are school-age children (ages 5-16), one per child.
Aggregated to household level for model inference.
"""

import warnings
import numpy as np
import pandas as pd

from lahore_simulation import PSLM_QUINTILES, _road_km_to_dist_cat
from inference_utils import INCOME_STATS

PSLM_DTA = "Lahore.dta"

# PSLM distance_to_school codes → approximate km midpoints
_DIST_CODE_KM = {
    1: 0.5,
    2: 1.5,
    3: 2.5,
    4: 4.0,
    5: 7.5,
    6: 12.0,
    7: 20.0,
}

# PSLM transport_mode → model travel_mode (0=foot,1=bike,2=moto,3=van,4=bus)
_TRANSPORT_MAP = {0: 0, 1: 4, 2: 3}


def load_pslm_households(seed: int = 42) -> pd.DataFrame:
    """
    Aggregate PSLM child-level rows to household level.

    Returns a DataFrame with all columns required by run_batch_inference()
    plus `actual_enrollment_prob` (fraction of children enrolled per household).

    route_safe is forced to 1 for all households.
    lat/lon and x/y are randomly assigned within Lahore bounds.
    """
    warnings.filterwarnings("ignore", category=UnicodeWarning)
    rng = np.random.default_rng(seed)
    stats = INCOME_STATS["urban"]

    df = pd.read_stata(PSLM_DTA, convert_categoricals=False)

    # Actual enrollment probability per household
    actual = (
        df.groupby("hhcode")["education_access"]
        .apply(lambda x: x.fillna(0).astype(float).mean())
        .reset_index(name="actual_enrollment_prob")
    )

    # Head-level attributes (same for every child row in the household)
    head = (
        df.groupby("hhcode")
        .first()[["head_edu_level_num", "total_household_income"]]
        .reset_index()
    )

    # Distance: map PSLM codes to km, then take min/max per household
    df["_dist_km"] = pd.to_numeric(df["distance_to_school"], errors="coerce").map(_DIST_CODE_KM)
    dist_agg = (
        df.groupby("hhcode")["_dist_km"]
        .agg(min_dist_km="min", max_dist_km="max")
        .reset_index()
    )

    # Travel mode: most common non-null value per household
    def _modal_mode(s):
        s = s.dropna()
        if s.empty:
            return 0
        return _TRANSPORT_MAP.get(int(s.value_counts().index[0]), 0)

    travel = (
        df.groupby("hhcode")["transport_mode"]
        .apply(_modal_mode)
        .reset_index(name="travel_mode")
    )

    # Merge all household-level pieces
    hh = (
        actual
        .merge(head, on="hhcode")
        .merge(dist_agg, on="hhcode")
        .merge(travel, on="hhcode")
    )
    n = len(hh)

    # read_write: head has attended any school (edu_level_num >= 2)
    edu = pd.to_numeric(hh["head_edu_level_num"], errors="coerce").fillna(0)
    hh["read_write"] = (edu >= 2).astype(int)

    # solve_math: head completed primary (edu_level_num >= 4)
    hh["solve_math"] = (edu >= 4).astype(int)

    # Income: annual → monthly; impute missing by education-stratified quintile
    hh["monthly_income_raw"] = (
        pd.to_numeric(hh["total_household_income"], errors="coerce") / 12.0
    )
    missing = hh["monthly_income_raw"].isna()
    if missing.any():
        q_idx = (
            pd.cut(edu[missing], bins=[-1, 0, 2, 5, 9, 99], labels=[0, 1, 2, 3, 4])
            .astype(float).fillna(2).astype(int)
        )
        hh.loc[missing, "monthly_income_raw"] = np.array([
            rng.uniform(PSLM_QUINTILES[min(qi, 4)][1], PSLM_QUINTILES[min(qi, 4)][2])
            for qi in q_idx
        ])
    hh["monthly_income_raw"] = np.round(
        np.clip(hh["monthly_income_raw"], stats["raw_min"], stats["raw_max"]), 0
    )
    hh["monthly_income_norm"] = (
        (hh["monthly_income_raw"] - stats["mean"]) / stats["std"]
    ).round(4)

    # Distance category bounds from min/max km across children
    min_km = hh["min_dist_km"].fillna(0.5).values
    max_km = hh["max_dist_km"].fillna(1.5).values

    dist_cats, min_dists, max_dists, min_times, max_times = [], [], [], [], []
    for lo, hi in zip(min_km, max_km):
        _, mn_d, _, mn_t, _ = _road_km_to_dist_cat(float(lo))
        label, _, mx_d, _, mx_t = _road_km_to_dist_cat(float(hi))
        dist_cats.append(label)
        min_dists.append(mn_d)
        max_dists.append(mx_d)
        min_times.append(mn_t)
        max_times.append(mx_t)

    hh["distance_cat"] = dist_cats
    hh["min_distance"] = min_dists
    hh["max_distance"] = max_dists
    hh["min_time"] = min_times
    hh["max_time"] = max_times

    # route_safe: forced to 1 for all
    hh["route_safe"] = 1

    # school_facilities: no PSLM equivalent — random Beta(2,3)*5
    hh["school_facilities"] = np.round(rng.beta(2, 3, n) * 5, 1)

    hh["travel_mode"] = hh["travel_mode"].fillna(0).astype(int)

    # Geo: random within Lahore bounding box (for map display)
    hh["lat"] = np.round(rng.uniform(31.40, 31.70, n), 6)
    hh["lon"] = np.round(rng.uniform(74.20, 74.55, n), 6)

    # Abstract x/y coordinates (for Urban/Rural ABM map pages)
    hh["x"] = np.round(rng.uniform(55.0, 95.0, n), 2)
    hh["y"] = np.round(rng.uniform(5.0, 95.0, n), 2)

    hh["hh_id"] = np.arange(1, n + 1)
    hh["preset"] = "urban"
    hh["enrollment_prob"] = np.nan

    return hh.reset_index(drop=True)
