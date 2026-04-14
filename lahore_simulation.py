"""
Lahore School Simulation helpers.
Includes routing via OSRM and enrollment probability inference.

Steps:
1. Load POIs_Schools.csv
2. Find 5 schools nearest to a reference school (fid=1314) using Haversine
3. Randomly generate 100 households within 2 km of any of those schools
4. For each household query OSRM (free, no API key) for road distance + travel time
5. Build a Plotly Scattermapbox figure on an OpenStreetMap base
"""

import math
import random
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests

# Distance category boundaries (road km)
_DIST_CAT_BOUNDS = [
    ("Near (0–2 km)",       0.0,  2.0,  0.5,  1.5,  5.0,  15.0),
    ("Moderate (2–6 km)",   2.0,  6.0,  2.5,  5.5, 20.0,  45.0),
    ("Far (6–15 km)",       6.0, 15.0,  7.0, 12.0, 50.0,  90.0),
    ("Very Far (15+ km)",  15.0, 9999, 17.0, 25.0,100.0, 180.0),
]

def _road_km_to_dist_cat(km: float):
    """Map a road distance in km to (category_label, min_dist, max_dist, min_time, max_time)."""
    for label, lo, hi, mn_d, mx_d, mn_t, mx_t in _DIST_CAT_BOUNDS:
        if lo <= km < hi:
            return label, mn_d, mx_d, mn_t, mx_t
    # fallback: Very Far
    return _DIST_CAT_BOUNDS[-1][0], *_DIST_CAT_BOUNDS[-1][2:]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCHOOLS_CSV = "POIs_Schools.csv"
EARTH_R_KM = 6371.0


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in km between two (lat, lon) points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return EARTH_R_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# School selection
# ---------------------------------------------------------------------------
def find_nearest_schools(fid: int = 1314, n: int = 5) -> tuple[pd.Series, pd.DataFrame]:
    """
    Returns (reference_school_row, nearest_n_schools_df).
    Distances are added as column 'dist_km'.
    """
    df = pd.read_csv(SCHOOLS_CSV)
    ref = df[df["fid"] == fid].iloc[0]
    ref_lat, ref_lon = float(ref["Lattitude"]), float(ref["Longitude"])

    others = df[df["fid"] != fid].copy()
    others["dist_km"] = others.apply(
        lambda r: haversine(ref_lat, ref_lon, float(r["Lattitude"]), float(r["Longitude"])),
        axis=1,
    )
    nearest = others.nsmallest(n, "dist_km").reset_index(drop=True)
    return ref, nearest


# ---------------------------------------------------------------------------
# Household generation
# ---------------------------------------------------------------------------
def generate_households(schools_df: pd.DataFrame, n: int = 100, max_km: float = 2.0) -> pd.DataFrame:
    """
    Scatter n households uniformly within a disk of radius max_km around
    any of the given schools.
    """
    rows = []
    for i in range(n):
        school = schools_df.sample(1).iloc[0]
        slat, slon = float(school["Lattitude"]), float(school["Longitude"])

        # Uniform distribution inside a disk of radius max_km
        r = max_km * math.sqrt(random.random())
        theta = random.uniform(0, 2 * math.pi)

        # Convert km offset to degrees
        dlat = (r / 111.0) * math.cos(theta)
        dlon = (r / (111.0 * math.cos(math.radians(slat)))) * math.sin(theta)

        rows.append(
            {
                "hh_id": i + 1,
                "lat": slat + dlat,
                "lon": slon + dlon,
                "src_school_fid": int(school["fid"]),
                "src_school_name": school["Name"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# OSRM routing (public demo, no API key required)
# ---------------------------------------------------------------------------
_OSRM_BASE = "http://router.project-osrm.org/route/v1/driving"


def _osrm_route(
    hh_lat: float, hh_lon: float, school_lat: float, school_lon: float
) -> tuple[float | None, float | None]:
    """
    Query OSRM for driving distance (km) and duration (min).
    Returns (None, None) on any failure.
    Note: OSRM uses lon,lat order.
    """
    url = f"{_OSRM_BASE}/{hh_lon},{hh_lat};{school_lon},{school_lat}?overview=false"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("code") == "Ok":
            route = data["routes"][0]
            return route["distance"] / 1000.0, route["duration"] / 60.0
    except Exception:
        pass
    return None, None


def fetch_routes(
    households_df: pd.DataFrame,
    schools_df: pd.DataFrame,
    progress_cb=None,
) -> pd.DataFrame:
    """
    For every household find the nearest school (among schools_df) and fetch
    the OSRM road distance + travel time.

    progress_cb: optional callable(done, total) for progress reporting.
    """
    total = len(households_df)
    results = []

    for i, (_, hh) in enumerate(households_df.iterrows()):
        # Nearest school by straight-line distance
        best_dist = float("inf")
        nearest = None
        for _, school in schools_df.iterrows():
            d = haversine(hh["lat"], hh["lon"], float(school["Lattitude"]), float(school["Longitude"]))
            if d < best_dist:
                best_dist = d
                nearest = school

        road_km, time_min = _osrm_route(
            hh["lat"], hh["lon"],
            float(nearest["Lattitude"]), float(nearest["Longitude"]),
        )

        results.append(
            {
                "hh_id": hh["hh_id"],
                "lat": hh["lat"],
                "lon": hh["lon"],
                "nearest_school_fid": int(nearest["fid"]),
                "nearest_school_name": nearest["Name"],
                "straight_km": round(best_dist, 3),
                "road_km": round(road_km, 3) if road_km is not None else None,
                "travel_min": round(time_min, 1) if time_min is not None else None,
            }
        )

        if progress_cb:
            progress_cb(i + 1, total)

        time.sleep(0.08)  # polite delay for public OSRM server

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Plotly map
# ---------------------------------------------------------------------------
def build_lahore_map(
    ref_school: pd.Series,
    nearest_schools: pd.DataFrame,
    households_df: pd.DataFrame | None = None,
) -> go.Figure:
    """
    Build a Scattermapbox figure on an OpenStreetMap base tile.
    Households are coloured by travel time (or straight-line distance if routes
    have not been fetched yet).
    """
    fig = go.Figure()

    # --- Reference school (amber star) ---
    fig.add_trace(
        go.Scattermapbox(
            lat=[float(ref_school["Lattitude"])],
            lon=[float(ref_school["Longitude"])],
            mode="markers+text",
            marker=dict(size=22, color="#f59e0b"),
            text=[f"REF: {ref_school['Name']}"],
            textposition="top right",
            hovertemplate=(
                f"<b>{ref_school['Name']}</b><br>"
                f"FID: {ref_school['fid']}<br>"
                f"Ref school<extra></extra>"
            ),
            name="Reference School (FID 1314)",
        )
    )

    # --- Nearest 5 schools (blue markers) ---
    hover_schools = [
        f"<b>{row['Name']}</b><br>FID: {row['fid']}<br>Dist from ref: {row['dist_km']:.2f} km<extra></extra>"
        for _, row in nearest_schools.iterrows()
    ]
    fig.add_trace(
        go.Scattermapbox(
            lat=nearest_schools["Lattitude"].astype(float).tolist(),
            lon=nearest_schools["Longitude"].astype(float).tolist(),
            mode="markers+text",
            marker=dict(size=18, color="#3b82f6"),
            text=nearest_schools["Name"].tolist(),
            textposition="top right",
            hovertemplate=hover_schools,
            name="Nearest 5 Schools",
        )
    )

    # --- Households ---
    if households_df is not None and len(households_df) > 0:
        has_routes = households_df["road_km"].notna().any() if "road_km" in households_df.columns else False

        has_probs = "enrollment_prob" in households_df.columns and households_df["enrollment_prob"].notna().any()

        if has_probs:
            color_vals = households_df["enrollment_prob"].fillna(0.5).tolist()
            color_label = "Enrollment Probability"
            colorscale = "RdYlGn"
            fixed_color = None
        elif has_routes:
            color_vals = households_df["travel_min"].fillna(0).tolist()
            color_label = "Travel Time (min)"
            colorscale = "Plasma"
            fixed_color = None
        else:
            color_vals = None
            color_label = None
            colorscale = None
            fixed_color = "#a855f7"   # purple

        def _hh_hover(row):
            lines = [f"<b>Household #{int(row.hh_id)}</b>", f"Nearest school: {row.nearest_school_name}"]
            if "straight_km" in row.index and pd.notna(row.straight_km):
                lines.append(f"Straight dist: {row.straight_km:.3f} km")
            if "road_km" in row.index and pd.notna(row.road_km):
                lines.append(f"Road dist: {row.road_km:.3f} km")
            if "travel_min" in row.index and pd.notna(row.travel_min):
                lines.append(f"Travel time: {row.travel_min:.1f} min")
            if "enrollment_prob" in row.index and pd.notna(row.enrollment_prob):
                lines.append(f"Enrollment prob: <b>{row.enrollment_prob:.0%}</b>")
            return "<br>".join(lines) + "<extra></extra>"

        hover_hh = households_df.apply(_hh_hover, axis=1).tolist()

        fig.add_trace(
            go.Scattermapbox(
                lat=households_df["lat"].tolist(),
                lon=households_df["lon"].tolist(),
                mode="markers",
                marker=dict(
                    size=9,
                    color=fixed_color if fixed_color else color_vals,
                    colorscale=colorscale,
                    cmin=min((v for v in color_vals if v is not None), default=0) if color_vals else None,
                    cmax=max((v for v in color_vals if v is not None), default=1) if color_vals else None,
                    showscale=fixed_color is None,
                    colorbar=dict(title=color_label, thickness=14, len=0.5) if not fixed_color else None,
                    opacity=0.85,
                ),
                hovertemplate=hover_hh,
                name="Households",
            )
        )

    # Compute bounds from ALL points (schools + households + ref)
    all_lats = nearest_schools["Lattitude"].astype(float).tolist() + [float(ref_school["Lattitude"])]
    all_lons = nearest_schools["Longitude"].astype(float).tolist() + [float(ref_school["Longitude"])]
    if households_df is not None and len(households_df) > 0:
        all_lats += households_df["lat"].tolist()
        all_lons += households_df["lon"].tolist()

    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    # Approximate zoom from lat/lon span
    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon
    max_span = max(lat_span, lon_span) or 0.01
    zoom = max(1, min(13, round(math.log2(0.5 / max_span) + 9)))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
        ),
        height=540,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            font=dict(color="#0f172a", size=12),
            bordercolor="#cbd5e1",
            borderwidth=1,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Prepare households for enrollment inference
# ---------------------------------------------------------------------------
_URBAN_INCOME = {"mean": 48412.3, "std": 13308.6, "raw_min": 2500.0, "raw_max": 120000.0}
_TRAVEL_WEIGHTS = [0.25, 0.20, 0.25, 0.15, 0.15]   # foot, bike, moto, van, bus (urban)


def prepare_for_inference(households_df: pd.DataFrame, preset: str = "urban") -> pd.DataFrame:
    """
    Enrich a routed households DataFrame with the socioeconomic features
    required by run_batch_inference, then return a DataFrame ready for it.

    Uses 'road_km' to assign distance category / min-max bounds.
    Randomly samples income, travel_mode, route_safe, read_write,
    solve_math, school_facilities using realistic Lahore-urban distributions.
    """
    rng = np.random.default_rng(42)
    n = len(households_df)
    stats = _URBAN_INCOME

    # Income
    income_raw = rng.normal(stats["mean"], stats["std"], n)
    income_raw = np.clip(income_raw, stats["raw_min"], stats["raw_max"])
    income_norm = (income_raw - stats["mean"]) / stats["std"]

    # Socioeconomic features
    travel_mode = rng.choice(5, size=n, p=_TRAVEL_WEIGHTS)
    route_safe = rng.binomial(1, 0.78, n)
    read_write = rng.binomial(1, 0.62, n)
    solve_math = rng.binomial(1, 0.54, n)
    facilities = np.round(rng.beta(2, 3, n) * 5, 1)

    # Distance/time bounds from road_km
    dist_cats, min_dists, max_dists, min_times, max_times = [], [], [], [], []
    road_kms = households_df["road_km"].fillna(households_df["straight_km"].fillna(1.0))
    for km in road_kms:
        label, mn_d, mx_d, mn_t, mx_t = _road_km_to_dist_cat(float(km))
        dist_cats.append(label)
        min_dists.append(mn_d)
        max_dists.append(mx_d)
        min_times.append(mn_t)
        max_times.append(mx_t)

    df = households_df.copy().reset_index(drop=True)
    df["preset"] = preset
    df["monthly_income_raw"] = np.round(income_raw, 0)
    df["monthly_income_norm"] = np.round(income_norm, 4)
    df["travel_mode"] = travel_mode.astype(int)
    df["route_safe"] = route_safe.astype(int)
    df["read_write"] = read_write.astype(int)
    df["solve_math"] = solve_math.astype(int)
    df["school_facilities"] = facilities
    df["distance_cat"] = dist_cats
    df["min_distance"] = min_dists
    df["max_distance"] = max_dists
    df["min_time"] = min_times
    df["max_time"] = max_times
    df["enrollment_prob"] = np.nan

    return df
