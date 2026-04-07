"""
Population generation and batch inference logic for the ABM simulation.
Used by both Rural and Urban Community pages.
"""
from typing import Optional

import numpy as np
import pandas as pd

from inference_utils import (
    DISTANCE_CATEGORIES,
    DISTANCE_CAT_LIST,
    FEATURE_COLUMNS,
    INCOME_STATS,
    _phase2_predict,
    run_autoencoder,
    run_phase1,
)

# Geographic x-ranges per preset
GEO_X_RANGE = {
    "rural": (5.0, 45.0),
    "urban": (55.0, 95.0),
}

# Distance category sampling weights per preset
DISTANCE_WEIGHTS = {
    "rural":  [0.15, 0.40, 0.35, 0.10],   # Near, Moderate, Far, Very Far
    "urban":  [0.55, 0.35, 0.08, 0.02],
}

# Travel mode sampling probabilities per preset
TRAVEL_MODE_WEIGHTS = {
    "rural":  [0.40, 0.20, 0.20, 0.10, 0.10],  # foot, bike, moto, van, bus
    "urban":  [0.25, 0.20, 0.25, 0.15, 0.15],
}


def _distance_cat_for_row(row: pd.Series) -> str:
    """Infer the closest distance category for a given min_distance value."""
    mid_distances = {
        "Near (0–2 km)": 1.0,
        "Moderate (2–6 km)": 4.0,
        "Far (6–15 km)": 9.5,
        "Very Far (15+ km)": 21.0,
    }
    best = min(mid_distances, key=lambda k: abs(mid_distances[k] - float(row.get("min_distance", 1.0))))
    return best


def generate_population(n: int, preset: str, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generate n synthetic household agents with realistic feature distributions.

    Returns a DataFrame with columns:
        hh_id, x, y, preset,
        monthly_income_raw, monthly_income_norm,
        travel_mode, route_safe, read_write, solve_math,
        school_facilities, distance_cat,
        min_distance, max_distance, min_time, max_time,
        enrollment_prob
    """
    rng = np.random.default_rng(seed)
    stats = INCOME_STATS[preset]

    # Income
    income_raw = rng.normal(stats["mean"], stats["std"], n)
    income_raw = np.clip(income_raw, stats["raw_min"], stats["raw_max"])
    income_norm = (income_raw - stats["mean"]) / stats["std"]

    # Categorical features
    travel_mode = rng.choice(5, size=n, p=TRAVEL_MODE_WEIGHTS[preset])
    route_safe_p = 0.62 if preset == "rural" else 0.78
    route_safe = rng.binomial(1, route_safe_p, n)
    read_write = rng.binomial(1, 0.62, n)
    solve_math = rng.binomial(1, 0.54, n)

    # School facilities: Beta(2,3) mapped to 0–5, rounded to 1 decimal
    facilities = np.round(rng.beta(2, 3, n) * 5, 1)

    # Distance categories
    dist_cat_indices = rng.choice(4, size=n, p=DISTANCE_WEIGHTS[preset])
    dist_cats = [DISTANCE_CAT_LIST[i] for i in dist_cat_indices]

    min_distances = np.array([DISTANCE_CATEGORIES[c]["min_distance"] for c in dist_cats])
    max_distances = np.array([DISTANCE_CATEGORIES[c]["max_distance"] for c in dist_cats])
    min_times = np.array([DISTANCE_CATEGORIES[c]["min_time"] for c in dist_cats])
    max_times = np.array([DISTANCE_CATEGORIES[c]["max_time"] for c in dist_cats])

    # Geographic positions
    x_lo, x_hi = GEO_X_RANGE[preset]
    x_pos = rng.uniform(x_lo, x_hi, n)
    y_pos = rng.uniform(5.0, 95.0, n)

    df = pd.DataFrame({
        "hh_id": np.arange(1, n + 1),
        "x": np.round(x_pos, 2),
        "y": np.round(y_pos, 2),
        "preset": preset,
        "monthly_income_raw": np.round(income_raw, 0),
        "monthly_income_norm": np.round(income_norm, 4),
        "travel_mode": travel_mode.astype(int),
        "route_safe": route_safe.astype(int),
        "read_write": read_write.astype(int),
        "solve_math": solve_math.astype(int),
        "school_facilities": facilities,
        "distance_cat": dist_cats,
        "min_distance": min_distances,
        "max_distance": max_distances,
        "min_time": min_times,
        "max_time": max_times,
        "enrollment_prob": np.nan,
    })

    return df


def generate_school(preset: str) -> dict:
    """Return school location for the given preset."""
    if preset == "rural":
        return {"x": 25.0, "y": 50.0, "name": "Rural School"}
    return {"x": 75.0, "y": 50.0, "name": "Urban School"}


def _agents_to_features(agents_df: pd.DataFrame) -> pd.DataFrame:
    """Convert agents DataFrame to the format expected by inference functions."""
    features = pd.DataFrame({
        "hh_id": agents_df["hh_id"].values,
        "monthly_income": agents_df["monthly_income_norm"].values,
        "travel_mode": agents_df["travel_mode"].values.astype(float),
        "route_safe": agents_df["route_safe"].values.astype(float),
        "read_write": agents_df["read_write"].values.astype(float),
        "solve_math": agents_df["solve_math"].values.astype(float),
        "school_facilities": agents_df["school_facilities"].values.astype(float),
        "min_distance": agents_df["min_distance"].values.astype(float),
        "max_distance": agents_df["max_distance"].values.astype(float),
        "min_time": agents_df["min_time"].values.astype(float),
        "max_time": agents_df["max_time"].values.astype(float),
    })
    return features


def run_batch_inference(agents_df: pd.DataFrame, preset: str) -> pd.Series:
    """
    Run the full inference pipeline on all agents at once (vectorized).

    Returns a pd.Series of clipped enrollment probabilities,
    indexed to match agents_df's positional index.
    """
    features_df = _agents_to_features(agents_df)

    # Phase 1 — forward pass on all N rows at once
    p1_preds = run_phase1(features_df, preset)

    # Autoencoder — encode all N rows at once
    z_df = run_autoencoder(p1_preds, preset)
    z_df.index = features_df.index

    # Phase 2 — predict on all N rows at once
    merged = pd.concat(
        [features_df.reset_index(drop=True), z_df.reset_index(drop=True)],
        axis=1
    )
    feature_df = merged.drop(columns=["hh_id"]).astype(float)
    probs, _ = _phase2_predict(feature_df, preset)
    probs = np.array(probs).flatten()

    # Route safety adjustment — vectorized
    rs = agents_df["route_safe"].fillna(1).astype(int).values
    probs = np.where(rs == 0, probs * 0.8, probs)

    # Clip to [0, 1]
    probs = np.clip(probs, 0.0, 1.0)

    return pd.Series(probs, index=agents_df.index)


def apply_global_adjustment(
    agents_df: pd.DataFrame,
    income_pct: float = 0.0,
    force_route_safe: bool = False,
) -> pd.DataFrame:
    """
    Apply global parameter shifts to all agents.

    Args:
        agents_df: Current agent DataFrame.
        income_pct: Fraction to shift income, e.g. 0.2 = +20%, -0.3 = -30%.
        force_route_safe: If True, set route_safe=1 for all agents.

    Returns:
        Updated copy of agents_df with enrollment_prob reset to NaN.
    """
    df = agents_df.copy()
    preset = df["preset"].iloc[0]
    stats = INCOME_STATS[preset]

    if income_pct != 0.0:
        df["monthly_income_raw"] = np.clip(
            df["monthly_income_raw"] * (1.0 + income_pct),
            stats["raw_min"],
            stats["raw_max"],
        )
        df["monthly_income_norm"] = (df["monthly_income_raw"] - stats["mean"]) / stats["std"]

    if force_route_safe:
        df["route_safe"] = 1

    df["enrollment_prob"] = np.nan
    return df


def load_population_from_csv(file, preset: str) -> pd.DataFrame:
    """
    Load agents from an uploaded CSV (test.csv format) and add simulation columns.

    The CSV must have at minimum: hh_id, monthly_income, travel_mode, route_safe,
    read_write, solve_math, school_facilities, min_distance, max_distance, min_time, max_time.

    monthly_income in the CSV is assumed to be already z-score normalised.
    """
    df = pd.read_csv(file)
    required = set(["hh_id"] + FEATURE_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    n = len(df)
    stats = INCOME_STATS[preset]

    # Reverse z-score to get raw income (approximate, for display)
    df["monthly_income_norm"] = df["monthly_income"].astype(float)
    df["monthly_income_raw"] = np.round(
        df["monthly_income_norm"] * stats["std"] + stats["mean"], 0
    )
    df["monthly_income_raw"] = np.clip(df["monthly_income_raw"], stats["raw_min"], stats["raw_max"])

    # Geographic positions within preset's zone
    rng = np.random.default_rng()
    x_lo, x_hi = GEO_X_RANGE[preset]
    df["x"] = np.round(rng.uniform(x_lo, x_hi, n), 2)
    df["y"] = np.round(rng.uniform(5.0, 95.0, n), 2)

    # Distance category (inferred from min_distance)
    df["distance_cat"] = df.apply(_distance_cat_for_row, axis=1)

    df["preset"] = preset
    df["enrollment_prob"] = np.nan

    # Reset hh_id to be sequential if needed
    df["hh_id"] = np.arange(1, n + 1)

    return df
