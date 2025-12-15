#!/usr/bin/env python3
"""
Household data engineering pipeline with CLI.

Main flow (default / full pipeline):
    1. build-subset: from long_df.csv (+ completed surveys + area lists)
    2. impute-income: KNN on monthly income (NaN/0)
    3. impute-facilities-distance-time: KNN on school_facilities (for a subset),
       and median imputation of distance/time.
    4. Save data-distribution plots to plots/
    5. Flag remaining NaNs in final CSV.
    6. Split into urban/rural, optionally normalise only urban/rural outputs.

KNN for income:
    X = [school_facilities (weighted index), travel_mode_max, highest_grade_max]
    y = monthly_income

KNN for school_facilities:
    X = [monthly_income]
    y = school_facilities
    applied only when school_facilities == 0 and distance/time were missing,
    and monthly_income is available.

Area type:
    Derived from hh_urban.txt and hh_rural.txt:
        1 = urban, 2 = rural

Normalisation:
    - subset_final.csv is kept RAW (not normalised).
    - urban.csv and rural.csv are normalised if --norm is given:
        --norm z   → z-normalisation
        --norm 0-5 → min–max scaling to [0, 5]
    - Normalisation is only for continuous numeric columns, excluding:
        hh_id, travel_mode, route_safe, school_facilities, read_write, solve_math, area_type.
"""

import argparse
import os
import sys
from typing import List, Tuple, Optional

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def print_banner(msg: str):
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70 + "\n")



# CONSTANT MAPS


INCOME_MAP = {
    "Less than Rs. 5,000": 2500,
    "Rs. 5,000 – 9,999": 7500,
    "Rs. 10,000 – 14,999": 12500,
    "Rs. 15,000 – 19,999": 17500,
    "Rs. 20,000 -- 24,999": 22500,
    "Rs. 25,000 -- 29,999": 27500,
    "Rs. 30,000 -- 34,999": 32500,
    "Rs. 35,000 -- 39,999": 37500,
    "Rs. 40,000 -- 44,999": 42500,
    "Rs. 45,000 -- 49,999": 47500,
    "Rs. 50,000 or more": 55000,
    "Don’t know / Refused": 0,
    "0.0": 0,
    np.nan: 0,
}

CODED_INCOME_MAP = {
    0: 0,
    1: 2500,
    2: 7500,
    3: 12500,
    4: 17500,
    5: 22500,
    6: 27500,
    7: 32500,
    8: 37500,
    9: 42500,
    10: 47500,
    11: 55000,
    12: 60000,
}

DISTANCE_MAP = {0: 0.25, 1: 0.75, 2: 1.5, 3: 2.5, 4: 3.5, 5: 4.5, 6: 7.5, 7: 10.0}
TIME_MAP = {0: 2.5, 1: 7.5, 2: 15.0, 3: 25.0, 4: 38.0, 5: 50.0}

NORMALIZE_MAP = {
    "Very important": "Very Important",
    "Extremely important": "Very Important",
    "Important": "Important",
    "Important,": "Important",
    "Somewhat important": "Slightly Important",
    "Slightly important": "Slightly Important",
    "Slightly important,": "Slightly Important",
    "Not at all important": "Not Important",
    "Not at all important,": "Not Important",
    "Not important": "Not Important",
    "Very helpful": "Very Helpful",
    "Moderately helpful": "Helpful",
    "Helpful": "Helpful",
    "Slightly helpful": "Slightly Helpful",
    "Not at all helpful": "Not Helpful",
}

ENCODE_MAP = {
    "Very Important": 3,
    "Important": 2,
    "Slightly Important": 1,
    "Not Important": 0,
    "Very Helpful": 3,
    "Helpful": 2,
    "Slightly Helpful": 1,
    "Not Helpful": 0,
}

FACILITY_WEIGHTS = {
    "school_facilities_1": 5,
    "school_facilities_2": 5,  # girls-only, adjusted per gender
    "school_facilities_3": 5,
    "school_facilities_4": 4,
    "school_facilities_5": 3,
    "school_facilities_6": 3,
    "school_facilities_7": 2,
    "school_facilities_8": 2,
    "school_facilities_9": 1,
    "school_facilities_10": 3,
    "school_facilities_11": 3,
}

CHILD_REL_VALUES = {"Son", "Daughter", "Son/Daughter", 3, 3.0, "3", "3.0"}



# HELPERS


def load_hh_area_sets(urban_list_path: str, rural_list_path: str) -> Tuple[set, set]:
    def _load(path: str) -> set:
        with open(path) as f:
            return {line.strip() for line in f if line.strip()}

    urban_ids = _load(urban_list_path)
    rural_ids = _load(rural_list_path)
    return urban_ids, rural_ids


def assign_area_type(series_hh_id: pd.Series, urban_ids: set, rural_ids: set) -> pd.Series:
    hh_str = series_hh_id.astype(str)
    area = pd.Series(np.nan, index=series_hh_id.index, dtype="float64")
    area[hh_str.isin(urban_ids)] = 1
    area[hh_str.isin(rural_ids)] = 2
    return area


def identify_children(df: pd.DataFrame) -> pd.Series:
    rel = df["relation_head"].astype(str).str.strip()
    return rel.isin({str(v) for v in CHILD_REL_VALUES})


def clean_income_column(df: pd.DataFrame, col: str = "monthly_income") -> None:
    df[col] = df[col].replace(INCOME_MAP)
    df[col] = pd.to_numeric(df[col], errors="coerce")
    coded = df[col].map(CODED_INCOME_MAP)
    df[col] = np.where(coded.notna(), coded, df[col])
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)


def map_distance_time_mode_route(df: pd.DataFrame) -> None:
    df["distance_km"] = (
        pd.to_numeric(df.get("distance_km", 0), errors="coerce")
        .map(DISTANCE_MAP)
        .fillna(0)
    )
    df["time_minutes"] = (
        pd.to_numeric(df.get("time_minutes", 0), errors="coerce")
        .map(TIME_MAP)
        .fillna(0)
    )
    df["travel_mode"] = (
        pd.to_numeric(df.get("travel_mode", 0), errors="coerce")
        .fillna(0)
        .astype(int)
    )
    df["route_safe"] = (
        pd.to_numeric(df.get("route_safe", 1), errors="coerce")
        .fillna(1)
        .astype(int)
    )


def normalize_yes_no(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .replace(
            {
                "Yes": 1,
                "No": 0,
                "1": 1,
                "2": 0,
                "1.0": 1,
                "2.0": 0,
                "nan": 0,
                np.nan: 0,
            }
        )
        .astype(float)
        .fillna(0)
        .astype(int)
    )



# NORMALISATION OF CONTINUOUS COLUMNS

def normalize_continuous_columns(df: pd.DataFrame, method: str = None) -> pd.DataFrame:
    """
    method: None | 'z' | '0-5'
    Applies ONLY to continuous numeric columns.
    Excludes: travel_mode, route_safe, school_facilities, read_write, solve_math, hh_id
    """
    if method is None:
        return df

    protected = {
        "travel_mode", "route_safe", "school_facilities",
        "read_write", "solve_math", "hh_id"
    }

    # Identify continuous numeric columns
    cont_cols = [
        c for c in df.columns
        if c not in protected
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    df = df.copy()

    # Z-normalisation
    if method == "z":
        for col in cont_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std
            else:
                df[col] = 0.0

    # 0-5 scaling
    elif method == "0-5":
        for col in cont_cols:
            mn = df[col].min()
            mx = df[col].max()
            if mx > mn:
                df[col] = 5 * (df[col] - mn) / (mx - mn)
            else:
                df[col] = 0.0

    return df



# TRAVEL MODE / ROUTE SAFE / HOUSEHOLD AGGREGATION


def compute_travel_mode_and_route_safe(df_children: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    df_children is assumed to already be restricted to school-going children.
    """
    if df_children.empty:
        travel_mode_by_hh = pd.DataFrame(columns=["hh_id", "travel_mode"])
        route_safe_by_hh = pd.DataFrame(columns=["hh_id", "route_safe"])
        return travel_mode_by_hh, route_safe_by_hh

    tm_counts = (
        df_children.groupby(["hh_id", "travel_mode"])
        .size()
        .reset_index(name="count")
    )
    tm_counts.sort_values(
        ["hh_id", "count", "travel_mode"],
        ascending=[True, False, True],
        inplace=True,
    )
    travel_mode_by_hh = tm_counts.drop_duplicates("hh_id")[["hh_id", "travel_mode"]]
    route_safe_by_hh = (
        df_children.groupby("hh_id")["route_safe"].min().reset_index()
    )
    return travel_mode_by_hh, route_safe_by_hh


def aggregate_household(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to household level:
      - monthly_income: sum of all members
      - min/max distance & time: ONLY among school-going children
      - travel_mode & route_safe: ONLY from school-going children
    """
    base = (
        df.groupby("hh_id")
        .agg(
            monthly_income=("monthly_income", "sum"),
        )
        .reset_index()
    )

    child_mask = identify_children(df)
    child_df = df[child_mask].copy()

    if child_df.empty:
        base["min_distance"] = 0
        base["max_distance"] = 0
        base["min_time"] = 0
        base["max_time"] = 0
        travel_mode_by_hh = pd.DataFrame(columns=["hh_id", "travel_mode"])
        route_safe_by_hh = pd.DataFrame(columns=["hh_id", "route_safe"])
    else:
        child_group = child_df.groupby("hh_id")
        base = base.merge(
            child_group["distance_km"].agg(min_distance="min", max_distance="max"),
            on="hh_id",
            how="left",
        )
        base = base.merge(
            child_group["time_minutes"].agg(min_time="min", max_time="max"),
            on="hh_id",
            how="left",
        )

        travel_mode_by_hh, route_safe_by_hh = compute_travel_mode_and_route_safe(child_df)

    for col in ["min_distance", "max_distance", "min_time", "max_time"]:
        if col not in base.columns:
            base[col] = 0
        base[col] = base[col].fillna(0)

    subset = base.merge(travel_mode_by_hh, on="hh_id", how="left")
    subset = subset.merge(route_safe_by_hh, on="hh_id", how="left")

    subset["travel_mode"] = subset["travel_mode"].fillna(0).astype(int)
    subset["route_safe"] = subset["route_safe"].fillna(1).astype(int)
    return subset



# FACILITY PROCESSING AND SCHOOL SCORE


def get_facility_columns(df: pd.DataFrame) -> List[str]:
    cols = [f"school_facilities_{i}" for i in range(1, 12)]
    return [c for c in cols if c in df.columns]


def compute_school_facilities_score(df: pd.DataFrame, facility_cols: List[str]) -> pd.DataFrame:
    """
    Compute weighted, gender-adjusted school facilities score per household (0–5).
    Girls-only facility (2) is only counted for girls, not boys.
    """
    is_child = identify_children(df)
    children_df = df[is_child].copy()

    if children_df.empty or not facility_cols:
        return pd.DataFrame(columns=["hh_id", "school_facilities"])

    for col in facility_cols:
        children_df[col] = (
            pd.to_numeric(children_df[col], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    children_df["child_idx"] = children_df.groupby("hh_id").cumcount()
    agg_dict = {col: "max" for col in facility_cols}
    agg_dict["gender"] = "first"

    child_fac = (
        children_df.groupby(["hh_id", "child_idx"])
        .agg(agg_dict)
        .reset_index()
    )

    w = pd.Series(FACILITY_WEIGHTS)
    weights_no_f2 = w.copy()
    if "school_facilities_2" in weights_no_f2:
        weights_no_f2["school_facilities_2"] = 0

    fac_matrix = child_fac[facility_cols].values
    weight_vec = np.array([weights_no_f2.get(col, 0) for col in facility_cols])
    base_scores = fac_matrix.dot(weight_vec)

    gender = pd.to_numeric(child_fac["gender"], errors="coerce")
    facility2_weight = np.where(gender == 2, 5, 0)

    if "school_facilities_2" in facility_cols:
        idx_f2 = facility_cols.index("school_facilities_2")
        facility2_values = fac_matrix[:, idx_f2]
        score_f2 = facility2_values * facility2_weight
    else:
        score_f2 = np.zeros(len(child_fac))

    raw_scores = base_scores + score_f2
    max_without_f2 = weights_no_f2.sum()
    max_scores = max_without_f2 + facility2_weight
    max_scores = np.where(max_scores <= 0, np.nan, max_scores)

    child_norm_scores = (raw_scores / max_scores) * 5
    child_norm_scores = np.clip(child_norm_scores, 0, 5)

    child_fac["school_score_child"] = child_norm_scores

    school_score = (
        child_fac.groupby("hh_id")["school_score_child"]
        .mean()
        .round(1)
        .reset_index()
        .rename(columns={"school_score_child": "school_facilities"})
    )
    return school_score



# MERGE COMPLETED SURVEYS


def merge_completed_surveys(subset: pd.DataFrame, completed_path: str) -> pd.DataFrame:
    completed = pd.read_csv(completed_path)

    if "read_write_h" in completed.columns:
        completed["read_write_h"] = normalize_yes_no(completed["read_write_h"])
    else:
        completed["read_write_h"] = 0

    if "solve_math_h" in completed.columns:
        completed["solve_math_h"] = normalize_yes_no(completed["solve_math_h"])
    else:
        completed["solve_math_h"] = 0

    completed = completed[["hh_id", "read_write_h", "solve_math_h"]].rename(
        columns={"read_write_h": "read_write", "solve_math_h": "solve_math"}
    )

    subset = subset.merge(completed, on="hh_id", how="left")
    subset[["read_write", "solve_math"]] = subset[["read_write", "solve_math"]].fillna(0).astype(int)
    return subset



# INCOME KNN IMPUTATION


def detect_grade_column(df: pd.DataFrame) -> str:
    if "highest_grade" in df.columns:
        return "highest_grade"
    if "highest_grade_h" in df.columns:
        return "highest_grade_h"
    raise ValueError("Could not find 'highest_grade' or 'highest_grade_h' in long_df.")

def knn_impute_income(
    subset: pd.DataFrame,
    long_df: pd.DataFrame,
    urban_ids: set,
    rural_ids: set,
    k: int = 5,
    z_norm: bool = True,
    knn_norm_income: bool = False,
):
    grade_col = detect_grade_column(long_df)
    long_df["travel_mode"] = pd.to_numeric(long_df.get("travel_mode", np.nan), errors="coerce")
    long_df[grade_col] = pd.to_numeric(long_df[grade_col], errors="coerce")

    travel_mode_max = (
        long_df.groupby("hh_id")["travel_mode"]
        .max()
        .rename("travel_mode_max")
        .reset_index()
    )
    highest_grade_max = (
        long_df.groupby("hh_id")[grade_col]
        .max()
        .rename("highest_grade_max")
        .reset_index()
    )

    hh_subset = (
        subset.groupby("hh_id")
        .agg(
            monthly_income=("monthly_income", "first"),
            school_facilities=("school_facilities", "first"),
        )
        .reset_index()
    )

    hh_subset["monthly_income"] = pd.to_numeric(hh_subset["monthly_income"], errors="coerce")
    hh_subset["school_facilities"] = pd.to_numeric(hh_subset["school_facilities"], errors="coerce")

    hh = (
        hh_subset.merge(travel_mode_max, on="hh_id", how="left")
        .merge(highest_grade_max, on="hh_id", how="left")
    )

    hh["area_type"] = assign_area_type(hh["hh_id"], urban_ids, rural_ids)

    feature_cols = ["school_facilities", "travel_mode_max", "highest_grade_max"]
    for c in feature_cols:
        hh[c] = pd.to_numeric(hh[c], errors="coerce")

    missing_mask = hh["monthly_income"].isna() | (hh["monthly_income"] <= 0)
    train_hh = hh[~missing_mask].copy()
    pred_hh = hh[missing_mask].copy()

    def _train_predict(area_value: int) -> pd.Series:
        train_area = train_hh[train_hh["area_type"] == area_value].copy()
        pred_area = pred_hh[pred_hh["area_type"] == area_value].copy()
        if len(train_area) == 0 or len(pred_area) == 0:
            return pd.Series(dtype=float)

        X_train = train_area[feature_cols]
        y_train = train_area["monthly_income"]
        X_pred = pred_area[feature_cols]

        medians = X_train.median()
        X_train = X_train.fillna(medians).fillna(0)
        X_pred = X_pred.fillna(medians).fillna(0)

        # ✔ FIXED NORMALISATION LOGIC
        if z_norm or knn_norm_income:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_pred = scaler.transform(X_pred)

        n_neighbors = min(k, len(train_area))
        knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
        knn.fit(X_train, y_train)

        preds = knn.predict(X_pred)
        return pd.Series(preds, index=pred_area.index, dtype=float)

    preds_urban = _train_predict(1)
    preds_rural = _train_predict(2)

    hh["monthly_income_pred"] = np.nan
    hh.loc[preds_urban.index, "monthly_income_pred"] = preds_urban
    hh.loc[preds_rural.index, "monthly_income_pred"] = preds_rural

    hh["monthly_income_filled"] = hh["monthly_income"].astype(float)
    hh.loc[missing_mask, "monthly_income_filled"] = hh.loc[missing_mask, "monthly_income_pred"]

    changed_hh_ids = (
        hh.loc[missing_mask & hh["monthly_income_filled"].notna(), "hh_id"]
        .astype(str)
        .unique()
        .tolist()
    )

    subset = subset.merge(hh[["hh_id", "monthly_income_filled"]], on="hh_id", how="left")
    subset["monthly_income"] = subset["monthly_income_filled"].where(
        subset["monthly_income_filled"].notna(), subset["monthly_income"]
    )
    subset.drop(columns=["monthly_income_filled"], inplace=True)

    return subset, changed_hh_ids



# DISTANCE/TIME AND SCHOOL FACILITIES IMPUTATION


def knn_impute_school_facilities_for_missing_distance_time(
    subset: pd.DataFrame,
    urban_ids: set,
    rural_ids: set,
    k: int = 5,
    z_norm: bool = True,
) -> pd.DataFrame:
    required_cols = [
        "hh_id",
        "monthly_income",
        "min_distance",
        "max_distance",
        "min_time",
        "max_time",
        "school_facilities",
    ]
    for c in required_cols:
        if c not in subset.columns:
            raise ValueError(f"Column '{c}' not found in subset for distance/time imputation.")

    df = subset.copy()
    df["area_type"] = assign_area_type(df["hh_id"], urban_ids, rural_ids)

    numeric_cols = ["min_distance", "max_distance", "min_time", "max_time"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["school_facilities"] = pd.to_numeric(df["school_facilities"], errors="coerce")
    df["monthly_income"] = pd.to_numeric(df["monthly_income"], errors="coerce")

    missing_any = df[numeric_cols].isna().any(axis=1)

    # Treat zeros as missing for purpose of median calculation
    df_nozero = df.copy()
    for c in numeric_cols:
        df_nozero[c] = df_nozero[c].replace(0, np.nan)

    medians = (
        df_nozero.groupby("area_type")[numeric_cols]
        .median()
        .rename_axis("area_type")
    )

    for area_value in [1, 2]:
        if area_value not in medians.index:
            continue
        area_mask = df["area_type"] == area_value
        for c in numeric_cols:
            median_val = medians.loc[area_value, c]
            if np.isnan(median_val):
                continue
            df.loc[area_mask & df[c].isna(), c] = median_val

    still_missing_any = df[numeric_cols].isna().any(axis=1)
    if still_missing_any.any():
        global_medians = df_nozero[numeric_cols].median()
        for c in numeric_cols:
            df[c] = df[c].fillna(global_medians[c])

    target_mask = missing_any & (df["school_facilities"].fillna(0) == 0)

    for area_value in [1, 2]:
        area_mask = df["area_type"] == area_value
        train_mask = area_mask & (df["school_facilities"] > 0) & df["monthly_income"].notna()
        pred_mask = area_mask & target_mask & df["monthly_income"].notna()

        if not train_mask.any() or not pred_mask.any():
            continue

        X_train = df.loc[train_mask, ["monthly_income"]]
        y_train = df.loc[train_mask, "school_facilities"]
        X_pred = df.loc[pred_mask, ["monthly_income"]]

        if z_norm:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_pred = scaler.transform(X_pred)

        n_neighbors = min(k, train_mask.sum())
        knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
        knn.fit(X_train, y_train)

        preds = knn.predict(X_pred)
        preds = np.clip(preds, 0, 5)
        preds = np.round(preds, 1)
        df.loc[pred_mask, "school_facilities"] = preds

    subset[numeric_cols] = df[numeric_cols]
    subset["school_facilities"] = df["school_facilities"]
    return subset



# LIKERT ENCODING


def encode_likert_file(infile: str, outfile: str) -> None:
    df = pd.read_excel(infile)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace(NORMALIZE_MAP)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(ENCODE_MAP)

    df.to_excel(outfile, index=False)



# MISSING DISTANCE EXPORT


def export_missing_distance(subset_path: str, out_path: str) -> None:
    df = pd.read_csv(subset_path)
    df["distance_km"] = pd.to_numeric(df.get("distance_km", 0), errors="coerce").fillna(0)
    result = df.loc[df["distance_km"] == 0, "hh_id"].astype(str).unique()
    with open(out_path, "w") as f:
        for hh_id in sorted(result):
            f.write(hh_id + "\n")



# PLOTS


def save_distribution_plots(df: pd.DataFrame, out_dir: str = "plots") -> None:
    os.makedirs(out_dir, exist_ok=True)

    plot_specs = [
        ("monthly_income", "Histogram of Monthly Income"),
        ("school_facilities", "Histogram of School Facilities Score"),
        ("min_distance", "Histogram of Min Distance"),
        ("max_distance", "Histogram of Max Distance"),
        ("min_time", "Histogram of Min Time"),
        ("max_time", "Histogram of Max Time"),
    ]

    for col, title in plot_specs:
        if col not in df.columns:
            continue
        data = pd.to_numeric(df[col], errors="coerce").dropna()
        if data.empty:
            continue

        plt.figure()
        plt.hist(data, bins=30)
        plt.title(title)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{col}_hist.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()



# NAN FLAGGING


def flag_nans_in_csv(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    nan_counts = df.isna().sum()
    total_rows = len(df)

    print(f"NaN summary for {csv_path}:")
    any_nans = False
    for col, cnt in nan_counts.items():
        if cnt > 0:
            any_nans = True
            print(f"  {col}: {cnt} NaNs ({cnt / total_rows:.2%})")
    if not any_nans:
        print("  No NaNs found.")



# BUILD SUBSET


def build_subset(
    long_path: str,
    completed_path: Optional[str],
    urban_list_path: str,
    rural_list_path: str,
    out_path: str,
) -> pd.DataFrame:
    print_banner("STEP 1: BUILDING HOUSEHOLD SUBSET")
    df = pd.read_csv(long_path)

    clean_income_column(df, "monthly_income")
    map_distance_time_mode_route(df)

    subset = aggregate_household(df)

    facility_cols = get_facility_columns(df)
    if facility_cols:
        school_score_df = compute_school_facilities_score(df, facility_cols)
        subset = subset.merge(school_score_df, on="hh_id", how="left")
        subset["school_facilities"] = subset["school_facilities"].fillna(0)
    else:
        subset["school_facilities"] = 0.0

    if completed_path is not None:
        subset = merge_completed_surveys(subset, completed_path)

    urban_ids, rural_ids = load_hh_area_sets(urban_list_path, rural_list_path)
    subset["area_type"] = assign_area_type(subset["hh_id"], urban_ids, rural_ids)

    # Ensure no stray facility_i columns
    subset = subset[[col for col in subset.columns if not col.startswith("facility")]]

    subset.to_csv(out_path, index=False)
    print(f"Subset saved to {out_path}")
    return subset



# STATS (RAW + NORMALISED)


def compute_and_save_stats(
    df_raw: pd.DataFrame,
    urban_ids: set,
    rural_ids: set,
    out_dir: str,
    urban_norm: Optional[pd.DataFrame] = None,
    rural_norm: Optional[pd.DataFrame] = None,
):
    os.makedirs(out_dir, exist_ok=True)

    stats = {"raw": {}, "normalised": {}}

    # RAW STATS
    df = df_raw.copy()
    df["hh_id_str"] = df["hh_id"].astype(str)
    urban_mask = df["hh_id_str"].isin(urban_ids)
    rural_mask = df["hh_id_str"].isin(rural_ids)

    n_urban = int(urban_mask.sum())
    n_rural = int(rural_mask.sum())

    print_banner("HOUSEHOLD AREA DISTRIBUTION (RAW)")
    print(f"Urban households: {n_urban}")
    print(f"Rural households: {n_rural}")

    stats["raw"]["urban_count"] = n_urban
    stats["raw"]["rural_count"] = n_rural

    def income_stats(sub):
        inc = pd.to_numeric(sub["monthly_income"], errors="coerce").dropna()
        if len(inc) == 0:
            return {}
        return {
            "mean": float(inc.mean()),
            "median": float(inc.median()),
            "min": float(inc.min()),
            "max": float(inc.max()),
        }

    stats["raw"]["income_overall"] = income_stats(df)
    stats["raw"]["income_urban"] = income_stats(df[urban_mask])
    stats["raw"]["income_rural"] = income_stats(df[rural_mask])

    print_banner("INCOME STATISTICS (RAW)")
    print("Overall:")
    print(json.dumps(stats["raw"]["income_overall"], indent=4))
    print("\nUrban:")
    print(json.dumps(stats["raw"]["income_urban"], indent=4))
    print("\nRural:")
    print(json.dumps(stats["raw"]["income_rural"], indent=4))

    if "route_safe" in df.columns:
        safe_rate = float(pd.to_numeric(df["route_safe"], errors="coerce").mean())
        stats["raw"]["route_safety_rate"] = safe_rate

        print_banner("ROUTE SAFETY (RAW)")
        print(f"Households with safe routes: {safe_rate * 100:.2f}%")

    if "school_facilities" in df.columns:
        fac = pd.to_numeric(df["school_facilities"], errors="coerce")
        stats["raw"]["school_facilities"] = {
            "mean": float(fac.mean()),
            "median": float(fac.median()),
            "min": float(fac.min()),
            "max": float(fac.max()),
        }

        print_banner("SCHOOL FACILITY SCORE SUMMARY (RAW)")
        print(json.dumps(stats["raw"]["school_facilities"], indent=4))

    missing_dist = df[(df["min_distance"].isna()) | (df["max_distance"].isna())]
    stats["raw"]["missing_distance_count"] = int(len(missing_dist))

    missing_time = df[(df["min_time"].isna()) | (df["max_time"].isna())]
    stats["raw"]["missing_time_count"] = int(len(missing_time))

    print_banner("MISSING VALUES SUMMARY (RAW)")
    print(f"Missing distance households: {stats['raw']['missing_distance_count']}")
    print(f"Missing time households: {stats['raw']['missing_time_count']}")

    # NORMALISED STATS (if provided)
    if urban_norm is not None and rural_norm is not None:
        stats_norm = {}

        def income_stats_norm(sub):
            if "monthly_income" not in sub.columns:
                return {}
            inc = pd.to_numeric(sub["monthly_income"], errors="coerce").dropna()
            if len(inc) == 0:
                return {}
            return {
                "mean": float(inc.mean()),
                "median": float(inc.median()),
                "min": float(inc.min()),
                "max": float(inc.max()),
            }

        stats_norm["income_urban"] = income_stats_norm(urban_norm)
        stats_norm["income_rural"] = income_stats_norm(rural_norm)

        print_banner("INCOME STATISTICS (NORMALISED)")
        print("Urban (normalised):")
        print(json.dumps(stats_norm["income_urban"], indent=4))
        print("\nRural (normalised):")
        print(json.dumps(stats_norm["income_rural"], indent=4))

        stats["normalised"] = stats_norm

    out_path = os.path.join(out_dir, "stats.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=4)

    print_banner(f"STATS SAVED → {out_path}")



# FULL PIPELINE


def full_pipeline(
    long_path: str,
    completed_path: str,
    urban_list_path: str,
    rural_list_path: str,
    out_path: str,
    modified_ids_path: str,
    k_income: int = 5,
    k_fac: int = 5,
    z_norm: bool = True,
    knn_norm_income: bool = False,
    norm_method: Optional[str] = None,
) -> None:

    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir == "":
        out_dir = "."
    os.makedirs(out_dir, exist_ok=True)

    # === 1) BUILD SUBSET ===
    print_banner("STEP 1: BUILDING BASE SUBSET")
    subset = build_subset(
        long_path, completed_path, urban_list_path, rural_list_path, out_path
    )

    # Load long_df and area sets
    long_df = pd.read_csv(long_path)
    urban_ids, rural_ids = load_hh_area_sets(urban_list_path, rural_list_path)

    # === 2) KNN IMPUTATION OF INCOME ===
    print_banner("STEP 2: KNN IMPUTATION OF INCOME")
    subset_imputed_income, changed_ids = knn_impute_income(
        subset,
        long_df,
        urban_ids,
        rural_ids,
        k=k_income,
        z_norm=z_norm,
        knn_norm_income=knn_norm_income,   # NEW FLAG
    )
    print(f"Households with imputed income: {len(changed_ids)}")

    # === 3) KNN IMPUTATION OF DISTANCE/TIME + SCHOOL FACILITIES ===
    print_banner("STEP 3: KNN IMPUTATION OF SCHOOL FACILITIES + DISTANCE/TIME")
    subset_imputed = knn_impute_school_facilities_for_missing_distance_time(
        subset_imputed_income, urban_ids, rural_ids, k=k_fac, z_norm=z_norm
    )

    # Save RAW full dataset (NOT NORMALISED)
    subset_imputed.to_csv(out_path, index=False)
    print(f"Final subset (RAW) saved to {out_path}")

    # Save modified household IDs
    modified_path = modified_ids_path
    if not os.path.isabs(modified_path):
        modified_path = os.path.join(out_dir, os.path.basename(modified_ids_path))
    with open(modified_path, "w") as f:
        for hh_id in sorted(changed_ids):
            f.write(str(hh_id) + "\n")
    print(f"Modified household IDs saved to {modified_path}")

    # === 4) BUILD URBAN/RURAL SPLITS ===
    urban_mask = subset_imputed["hh_id"].astype(str).isin(urban_ids)
    rural_mask = subset_imputed["hh_id"].astype(str).isin(rural_ids)

    urban_raw = subset_imputed[urban_mask].copy().drop(columns=["area_type"], errors="ignore")
    rural_raw = subset_imputed[rural_mask].copy().drop(columns=["area_type"], errors="ignore")

    # === NORMALISATION APPLIED ONLY IF REQUESTED, AND ONLY TO URBAN/RURAL ===
    if norm_method is not None:
        urban_norm = normalize_continuous_columns(urban_raw, method=norm_method)
        rural_norm = normalize_continuous_columns(rural_raw, method=norm_method)
    else:
        urban_norm = urban_raw.copy()
        rural_norm = rural_raw.copy()

    # Save the split files
    urban_out = os.path.join(out_dir, "urban.csv")
    rural_out = os.path.join(out_dir, "rural.csv")

    urban_norm.to_csv(urban_out, index=False)
    rural_norm.to_csv(rural_out, index=False)

    print_banner("SAVED AREA-SPLIT DATASETS")
    print(f"Urban file → {urban_out}")
    print(f"Rural file → {rural_out}")

    # === 5) PLOTS + MISSING VALUE REPORT ===
    print_banner("STEP 4: PLOTS AND NAN REPORT")
    save_distribution_plots(subset_imputed, out_dir=os.path.join(out_dir, "plots"))
    flag_nans_in_csv(out_path)

    # === 6) RAW + NORMALISED STATS ===
    print_banner("STEP 5: SUMMARY STATS (RAW + NORMALISED)")
    compute_and_save_stats(
        df_raw=subset_imputed,
        urban_ids=urban_ids,
        rural_ids=rural_ids,
        out_dir=out_dir,
        urban_norm=urban_norm if norm_method is not None else None,
        rural_norm=rural_norm if norm_method is not None else None,
    )


# CLI


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Household data engineering pipeline."
    )

    parser.add_argument("--long", help="Path to long_df.csv")
    parser.add_argument("--completed", help="Path to completed_surveys.csv")
    parser.add_argument("--urban-list", help="Path to hh_urban.txt")
    parser.add_argument("--rural-list", help="Path to hh_rural.txt")
    parser.add_argument("--out", help="Output path for main subset CSV")
    parser.add_argument("--modified", help="Path to modified_households.txt")
    parser.add_argument("--k-income", type=int, default=5)
    parser.add_argument("--k-fac", type=int, default=5)
    parser.add_argument(
        "--z-norm",
        action="store_true",
        help="Use Z-normalization inside KNN models.",
    )
    parser.add_argument(
        "--norm",
        choices=["z", "0-5"],
        help="Normalise continuous columns in urban/rural outputs (z or 0-5).",
    )

    subparsers = parser.add_subparsers(dest="command")

    p_build = subparsers.add_parser("build-subset", help="Build subset from long_df.")
    p_build.add_argument("--long", required=True)
    p_build.add_argument("--completed", required=False, default=None)
    p_build.add_argument("--urban-list", required=True)
    p_build.add_argument("--rural-list", required=True)
    p_build.add_argument("--out", required=True)

    p_impute_inc = subparsers.add_parser(
        "impute-income", help="Impute household income with KNN."
    )
    p_impute_inc.add_argument("--subset", required=True)
    p_impute_inc.add_argument("--long", required=True)
    p_impute_inc.add_argument("--urban-list", required=True)
    p_impute_inc.add_argument("--rural-list", required=True)
    p_impute_inc.add_argument("--out", required=True)
    p_impute_inc.add_argument("--modified", required=True)
    p_impute_inc.add_argument("--k", type=int, default=5)
    p_impute_inc.add_argument("--z-norm", action="store_true")

    p_impute_fac = subparsers.add_parser(
        "impute-facilities-distance-time",
        help="Impute min/max distance/time and school_facilities using KNN and medians.",
    )
    p_impute_fac.add_argument("--subset", required=True)
    p_impute_fac.add_argument("--urban-list", required=True)
    p_impute_fac.add_argument("--rural-list", required=True)
    p_impute_fac.add_argument("--out", required=True)
    p_impute_fac.add_argument("--k", type=int, default=5)
    p_impute_fac.add_argument("--z-norm", action="store_true")

    p_likert = subparsers.add_parser("encode-likert", help="Encode Likert in Excel file.")
    p_likert.add_argument("--infile", required=True)
    p_likert.add_argument("--outfile", required=True)

    p_flag = subparsers.add_parser("flag-nans", help="Flag NaNs in a CSV.")
    p_flag.add_argument("--csv", required=True)

    p_missing_dist = subparsers.add_parser(
        "missing-distance", help="Export hh_ids with distance_km == 0."
    )
    p_missing_dist.add_argument("--subset", required=True)
    p_missing_dist.add_argument("--out", required=True)

    p_plots = subparsers.add_parser("plots", help="Generate distribution plots from CSV.")
    p_plots.add_argument("--csv", required=True)
    p_plots.add_argument("--out-dir", default="plots")

    return parser.parse_args(argv)


def main(argv: List[str] = None) -> None:
    args = parse_args(argv or sys.argv[1:])

    if args.command is None:
        if not (args.long and args.completed and args.urban_list and args.rural_list and args.out and args.modified):
            raise SystemExit(
                "Default mode runs full pipeline. Provide --long, --completed, "
                "--urban-list, --rural-list, --out, and --modified."
            )
        full_pipeline(
            long_path=args.long,
            completed_path=args.completed,
            urban_list_path=args.urban_list,
            rural_list_path=args.rural_list,
            out_path=args.out,
            modified_ids_path=args.modified,
            k_income=args.k_income,
            k_fac=args.k_fac,
            z_norm=args.z_norm,
            norm_method=args.norm,
        )
        return

    if args.command == "build-subset":
        build_subset(
            long_path=args.long,
            completed_path=args.completed,
            urban_list_path=args.urban_list,
            rural_list_path=args.rural_list,
            out_path=args.out,
        )

    elif args.command == "impute-income":
        subset = pd.read_csv(args.subset)
        long_df = pd.read_csv(args.long)
        urban_ids, rural_ids = load_hh_area_sets(args.urban_list, args.rural_list)

        subset_imputed, changed_ids = knn_impute_income(
            subset,
            long_df,
            urban_ids,
            rural_ids,
            k=args.k,
            z_norm=args.z_norm,
        )
        subset_imputed.to_csv(args.out, index=False)
        with open(args.modified, "w") as f:
            for hh_id in sorted(changed_ids):
                f.write(str(hh_id) + "\n")

    elif args.command == "impute-facilities-distance-time":
        subset = pd.read_csv(args.subset)
        urban_ids, rural_ids = load_hh_area_sets(args.urban_list, args.rural_list)
        subset_imputed = knn_impute_school_facilities_for_missing_distance_time(
            subset,
            urban_ids,
            rural_ids,
            k=args.k,
            z_norm=args.z_norm,
        )
        subset_imputed.to_csv(args.out, index=False)

    elif args.command == "encode-likert":
        encode_likert_file(args.infile, args.outfile)

    elif args.command == "flag-nans":
        flag_nans_in_csv(args.csv)

    elif args.command == "missing-distance":
        export_missing_distance(args.subset, args.out)

    elif args.command == "plots":
        df = pd.read_csv(args.csv)
        save_distribution_plots(df, out_dir=args.out_dir)

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

