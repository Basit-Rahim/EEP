"""
Lahore School Simulation helpers.

Pipeline:
1. Load all ~3000 schools from POIs_Schools.csv
2. Generate N households within 2 km of any school (random)
3. Vectorized Haversine (numpy, chunked) → nearest school per household  — instant
4. OSRM Table API in chunks of 100 households → road distance + travel time
   (one request per chunk, ~100 requests for 10k households)
5. Plotly Scattermapbox on OpenStreetMap tiles
"""

import math
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCHOOLS_CSV = "POIs_Schools.csv"
EARTH_R_KM  = 6371.0

_OSRM_TABLE = "http://router.project-osrm.org/table/v1/driving"
_HH_CHUNK   = 100   # households per OSRM Table request

# Distance category boundaries (road km → model inputs)
_DIST_CAT_BOUNDS = [
    ("Near (0–2 km)",      0.0,  2.0,  0.5,  1.5,  5.0,  15.0),
    ("Moderate (2–6 km)",  2.0,  6.0,  2.5,  5.5, 20.0,  45.0),
    ("Far (6–15 km)",      6.0, 15.0,  7.0, 12.0, 50.0,  90.0),
    ("Very Far (15+ km)", 15.0, 9999, 17.0, 25.0,100.0, 180.0),
]


def _road_km_to_dist_cat(km: float):
    for label, lo, hi, mn_d, mx_d, mn_t, mx_t in _DIST_CAT_BOUNDS:
        if lo <= km < hi:
            return label, mn_d, mx_d, mn_t, mx_t
    return _DIST_CAT_BOUNDS[-1][0], *_DIST_CAT_BOUNDS[-1][2:]


# ---------------------------------------------------------------------------
# Load schools
# ---------------------------------------------------------------------------
def load_all_schools() -> pd.DataFrame:
    """Load every school from the CSV, drop rows with missing coordinates."""
    df = pd.read_csv(SCHOOLS_CSV)
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Lattitude"] = pd.to_numeric(df["Lattitude"], errors="coerce")
    return df.dropna(subset=["Longitude", "Lattitude"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Vectorized Haversine
# ---------------------------------------------------------------------------
def _haversine_vectorized(
    hh_lats: np.ndarray,
    hh_lons: np.ndarray,
    sc_lats: np.ndarray,
    sc_lons: np.ndarray,
    hh_chunk: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each household find the index and distance (km) of the nearest school.
    Processes households in chunks to keep memory under ~25 MB per chunk.

    Returns:
        nearest_idx  shape (N,) — index into sc_lats/sc_lons arrays
        nearest_dist shape (N,) — Haversine distance in km
    """
    n = len(hh_lats)
    nearest_idx  = np.zeros(n, dtype=np.int32)
    nearest_dist = np.full(n, np.inf)

    hl  = np.radians(np.asarray(hh_lats, dtype=np.float64))
    hln = np.radians(np.asarray(hh_lons, dtype=np.float64))
    sl  = np.radians(np.asarray(sc_lats, dtype=np.float64))
    sln = np.radians(np.asarray(sc_lons, dtype=np.float64))

    for i in range(0, n, hh_chunk):
        end = min(i + hh_chunk, n)
        chl  = hl[i:end, None]   # (chunk, 1)
        chln = hln[i:end, None]

        dphi    = sl[None, :] - chl
        dlambda = sln[None, :] - chln
        a = np.sin(dphi / 2) ** 2 + np.cos(chl) * np.cos(sl[None, :]) * np.sin(dlambda / 2) ** 2
        dists = EARTH_R_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))  # (chunk, M)

        idx  = np.argmin(dists, axis=1)
        dist = dists[np.arange(end - i), idx]

        nearest_idx[i:end]  = idx
        nearest_dist[i:end] = dist

    return nearest_idx, nearest_dist


# ---------------------------------------------------------------------------
# Household generation
# ---------------------------------------------------------------------------
def generate_households(schools_df: pd.DataFrame, n: int = 10_000, max_km: float = 2.0) -> pd.DataFrame:
    """
    Scatter n households uniformly inside a disk of radius max_km around
    a randomly chosen school from schools_df.
    """
    sc_lats = schools_df["Lattitude"].astype(float).values
    sc_lons = schools_df["Longitude"].astype(float).values
    sc_fids = schools_df["fid"].values
    sc_names = schools_df["Name"].values
    m = len(schools_df)

    rng = np.random.default_rng()
    sc_idx  = rng.integers(0, m, size=n)
    radii   = max_km * np.sqrt(rng.random(n))        # uniform on disk
    thetas  = rng.uniform(0, 2 * math.pi, n)

    slats = sc_lats[sc_idx]
    slons = sc_lons[sc_idx]

    dlat = (radii / 111.0) * np.cos(thetas)
    dlon = (radii / (111.0 * np.cos(np.radians(slats)))) * np.sin(thetas)

    return pd.DataFrame({
        "hh_id":          np.arange(1, n + 1),
        "lat":            slats + dlat,
        "lon":            slons + dlon,
        "src_school_fid": sc_fids[sc_idx],
        "src_school_name": sc_names[sc_idx],
    })


# ---------------------------------------------------------------------------
# OSRM Table API helper
# ---------------------------------------------------------------------------
def _osrm_table(hh_coords: list, school_coords: list) -> tuple[np.ndarray, np.ndarray]:
    """
    One OSRM Table API call.

    hh_coords     : [(lon, lat), ...]  N rows
    school_coords : [(lon, lat), ...]  M rows

    Returns (durations_sec, distances_m) each shaped (N, M).
    Unreachable pairs → np.nan.
    """
    n, m = len(hh_coords), len(school_coords)
    all_coords = hh_coords + school_coords
    coord_str  = ";".join(f"{lon},{lat}" for lon, lat in all_coords)
    sources    = ";".join(str(i) for i in range(n))
    dests      = ";".join(str(n + j) for j in range(m))

    url  = f"{_OSRM_TABLE}/{coord_str}?sources={sources}&destinations={dests}&annotations=duration,distance"
    resp = requests.get(url, timeout=30)
    data = resp.json()

    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM error: {data.get('message', data.get('code'))}")

    def _arr(matrix):
        a = np.array(matrix, dtype=float)
        a[a == 0] = np.nan   # OSRM returns 0 for unreachable pairs
        return a

    return _arr(data["durations"]), _arr(data["distances"])


# ---------------------------------------------------------------------------
# Route fetching (chunked Table API)
# ---------------------------------------------------------------------------
def fetch_routes(
    households_df: pd.DataFrame,
    schools_df: pd.DataFrame,
    progress_cb=None,
) -> pd.DataFrame:
    """
    Compute road distance + travel time for every household.

    Strategy:
      1. Vectorized Haversine → nearest school index for all households (instant).
      2. Process households in chunks of _HH_CHUNK.
         For each chunk, collect the unique nearest-school coords as destinations
         and fire one OSRM Table API request.
      Total OSRM requests ≈ ceil(N / _HH_CHUNK).
    """
    hh_lats = households_df["lat"].values
    hh_lons = households_df["lon"].values
    sc_lats = schools_df["Lattitude"].astype(float).values
    sc_lons = schools_df["Longitude"].astype(float).values
    sc_fids  = schools_df["fid"].values
    sc_names = schools_df["Name"].values
    n_hh = len(hh_lats)

    # Step 1 — vectorized nearest school
    nearest_sc_idx, straight_km_arr = _haversine_vectorized(hh_lats, hh_lons, sc_lats, sc_lons)

    # Step 2 — chunked OSRM Table API
    road_km_arr    = np.full(n_hh, np.nan)
    travel_min_arr = np.full(n_hh, np.nan)

    for start in range(0, n_hh, _HH_CHUNK):
        end = min(start + _HH_CHUNK, n_hh)

        chunk_lons  = hh_lons[start:end]
        chunk_lats  = hh_lats[start:end]
        chunk_sc    = nearest_sc_idx[start:end]

        unique_sc   = np.unique(chunk_sc)
        sc_col_map  = {int(si): col for col, si in enumerate(unique_sc)}

        hh_coords = list(zip(chunk_lons, chunk_lats))
        sc_coords = [(sc_lons[j], sc_lats[j]) for j in unique_sc]

        try:
            dur, dst = _osrm_table(hh_coords, sc_coords)
            for k in range(end - start):
                col = sc_col_map[int(chunk_sc[k])]
                if not np.isnan(dst[k, col]):
                    road_km_arr[start + k]    = dst[k, col] / 1000.0
                    travel_min_arr[start + k] = dur[k, col] / 60.0
        except Exception:
            pass   # OSRM unavailable for this chunk → keep NaN, fall back to straight_km

        if progress_cb:
            progress_cb(end, n_hh)

    # Round
    road_km_arr    = np.where(np.isnan(road_km_arr),    np.nan, np.round(road_km_arr,    3))
    travel_min_arr = np.where(np.isnan(travel_min_arr), np.nan, np.round(travel_min_arr, 1))

    return pd.DataFrame({
        "hh_id":               households_df["hh_id"].values,
        "lat":                 hh_lats,
        "lon":                 hh_lons,
        "nearest_school_fid":  sc_fids[nearest_sc_idx],
        "nearest_school_name": sc_names[nearest_sc_idx],
        "straight_km":         np.round(straight_km_arr, 3),
        "road_km":             road_km_arr,
        "travel_min":          travel_min_arr,
    })


# ---------------------------------------------------------------------------
# Plotly map
# ---------------------------------------------------------------------------
def build_lahore_map(
    schools_df: pd.DataFrame,
    households_df: pd.DataFrame | None = None,
) -> go.Figure:
    """
    Scattermapbox on OpenStreetMap tiles.
    Schools — small blue markers (no labels at scale).
    Households — coloured by enrollment prob > travel time > purple (no data).
    Income and all available fields shown in hover.
    """
    fig = go.Figure()

    # --- Schools ---
    sc_hover = [
        f"<b>{row['Name']}</b><br>FID: {row['fid']}<extra></extra>"
        for _, row in schools_df.iterrows()
    ]
    fig.add_trace(go.Scattermapbox(
        lat=schools_df["Lattitude"].astype(float).tolist(),
        lon=schools_df["Longitude"].astype(float).tolist(),
        mode="markers",
        marker=dict(size=6, color="#3b82f6", opacity=0.7),
        hovertemplate=sc_hover,
        name=f"Schools ({len(schools_df):,})",
    ))

    # --- Households ---
    if households_df is not None and len(households_df) > 0:
        has_routes = "road_km" in households_df.columns and households_df["road_km"].notna().any()
        has_probs  = "enrollment_prob" in households_df.columns and households_df["enrollment_prob"].notna().any()

        if has_probs:
            color_vals  = households_df["enrollment_prob"].fillna(0.5).tolist()
            color_label = "Enrollment Prob"
            colorscale  = "RdYlGn"
            fixed_color = None
        elif has_routes:
            color_vals  = households_df["travel_min"].fillna(0).tolist()
            color_label = "Travel Time (min)"
            colorscale  = "Plasma"
            fixed_color = None
        else:
            color_vals  = None
            color_label = None
            colorscale  = None
            fixed_color = "#a855f7"

        def _hh_hover(row):
            lines = [f"<b>Household #{int(row.hh_id)}</b>"]
            if "nearest_school_name" in row.index:
                lines.append(f"School: {row.nearest_school_name}")
            if "monthly_income_raw" in row.index and pd.notna(row.monthly_income_raw):
                q = f"  {row.income_quintile}" if "income_quintile" in row.index and pd.notna(row.income_quintile) else ""
                lines.append(f"Income: PKR {row.monthly_income_raw:,.0f}{q}")
            if "straight_km" in row.index and pd.notna(row.straight_km):
                lines.append(f"Straight dist: {row.straight_km:.3f} km")
            if "road_km" in row.index and pd.notna(row.road_km):
                lines.append(f"Road dist: {row.road_km:.3f} km")
            if "travel_min" in row.index and pd.notna(row.travel_min):
                lines.append(f"Travel time: {row.travel_min:.1f} min")
            if "distance_cat" in row.index and pd.notna(row.distance_cat):
                lines.append(f"Category: {row.distance_cat}")
            if "enrollment_prob" in row.index and pd.notna(row.enrollment_prob):
                lines.append(f"Model predicted: <b>{row.enrollment_prob:.0%}</b>")
            if "actual_enrollment_prob" in row.index and pd.notna(row.actual_enrollment_prob):
                lines.append(f"Actual (PSLM): <b>{row.actual_enrollment_prob:.0%}</b>")
            return "<br>".join(lines) + "<extra></extra>"

        hover_hh = households_df.apply(_hh_hover, axis=1).tolist()

        fig.add_trace(go.Scattermapbox(
            lat=households_df["lat"].tolist(),
            lon=households_df["lon"].tolist(),
            mode="markers",
            marker=dict(
                size=7,
                color=fixed_color if fixed_color else color_vals,
                colorscale=colorscale,
                cmin=min((v for v in color_vals if v is not None), default=0) if color_vals else None,
                cmax=max((v for v in color_vals if v is not None), default=1) if color_vals else None,
                showscale=fixed_color is None,
                colorbar=dict(title=color_label, thickness=14, len=0.5) if not fixed_color else None,
                opacity=0.75,
            ),
            hovertemplate=hover_hh,
            name=f"Households ({len(households_df):,})",
        ))

    # Fixed Lahore city center — user pans/zooms to explore areas
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=31.5204, lon=74.3587),   # Lahore city centre
            zoom=11,                                   # city-wide overview; zoom in to see dots
        ),
        height=700,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            bgcolor="rgba(255,255,255,0.88)",
            font=dict(color="#0f172a", size=12),
            bordercolor="#cbd5e1",
            borderwidth=1,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# PSLM quintile income sampling
# ---------------------------------------------------------------------------
# Approximate Lahore urban monthly income bounds per quintile (PKR).
# SOURCE: Estimated from PSLM/HIES published summaries — replace lo/hi with
#         real microdata quintile cut-points when available.
PSLM_QUINTILES = [
    # label,           lo_pkr,  hi_pkr
    ("Q1 (0–20%)",     5_000,   18_000),
    ("Q2 (20–40%)",   18_000,   32_000),
    ("Q3 (40–60%)",   32_000,   50_000),
    ("Q4 (60–80%)",   50_000,   80_000),
    ("Q5 (80–100%)",  80_000,  200_000),
]

# Reference stats for z-score normalisation (used by the inference model)
_URBAN_INCOME   = {"mean": 48_412.3, "std": 13_308.6}
_TRAVEL_WEIGHTS = [0.25, 0.20, 0.25, 0.15, 0.15]   # foot/bike/moto/van/bus (urban)


def _sample_pslm_income(n: int, rng: np.random.Generator):
    """
    Sample n incomes with exactly 20 % allocated to each PSLM quintile.
    Within each quintile income is drawn uniformly from [lo, hi].
    Returns (income_raw ndarray, quintile_label list).
    """
    per_q     = n // 5
    remainder = n % 5
    incomes, labels = [], []
    for i, (label, lo, hi) in enumerate(PSLM_QUINTILES):
        count = per_q + (1 if i < remainder else 0)
        incomes.append(rng.uniform(lo, hi, count))
        labels.extend([label] * count)

    income_arr = np.concatenate(incomes)
    label_arr  = np.array(labels)

    # Shuffle so quintile ordering is random across the DataFrame
    idx = rng.permutation(n)
    return income_arr[idx], label_arr[idx]


# ---------------------------------------------------------------------------
# Prepare households for enrollment inference
# ---------------------------------------------------------------------------
def prepare_for_inference(households_df: pd.DataFrame, preset: str = "urban") -> pd.DataFrame:
    """
    Add socioeconomic features needed by run_batch_inference.

    Income: 20 % per PSLM quintile, uniform within each quintile's [lo, hi] band.
    z-score normalised with Lahore-urban reference stats for model input.
    Other features use realistic Lahore-urban probability distributions.
    road_km (or straight_km if OSRM failed) is mapped to a distance category.
    """
    rng = np.random.default_rng(42)
    n   = len(households_df)

    income_raw, income_quintile = _sample_pslm_income(n, rng)
    stats      = _URBAN_INCOME
    income_norm = (income_raw - stats["mean"]) / stats["std"]

    travel_mode = rng.choice(5, size=n, p=_TRAVEL_WEIGHTS)
    route_safe  = rng.binomial(1, 0.78, n)
    read_write  = rng.binomial(1, 0.62, n)
    solve_math  = rng.binomial(1, 0.54, n)
    facilities  = np.round(rng.beta(2, 3, n) * 5, 1)

    km_series = households_df["road_km"].fillna(
        households_df["straight_km"] if "straight_km" in households_df.columns else pd.Series(1.0, index=households_df.index)
    ).fillna(1.0)

    dist_cats, min_dists, max_dists, min_times, max_times = [], [], [], [], []
    for km in km_series:
        label, mn_d, mx_d, mn_t, mx_t = _road_km_to_dist_cat(float(km))
        dist_cats.append(label)
        min_dists.append(mn_d)
        max_dists.append(mx_d)
        min_times.append(mn_t)
        max_times.append(mx_t)

    df = households_df.copy().reset_index(drop=True)
    df["preset"]             = preset
    df["monthly_income_raw"] = np.round(income_raw, 0)
    df["monthly_income_norm"]= np.round(income_norm, 4)
    df["income_quintile"]    = income_quintile
    df["travel_mode"]        = travel_mode.astype(int)
    df["route_safe"]         = route_safe.astype(int)
    df["read_write"]         = read_write.astype(int)
    df["solve_math"]         = solve_math.astype(int)
    df["school_facilities"]  = facilities
    df["distance_cat"]       = dist_cats
    df["min_distance"]       = min_dists
    df["max_distance"]       = max_dists
    df["min_time"]           = min_times
    df["max_time"]           = max_times
    df["enrollment_prob"]    = np.nan
    return df
