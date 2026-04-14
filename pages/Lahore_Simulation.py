"""
Lahore School Simulation — real map, real school coordinates, OSRM routing.
Accessible from the Streamlit sidebar navigation.
"""
import pandas as pd
import streamlit as st

from lahore_simulation import (
    build_lahore_map,
    fetch_routes,
    find_nearest_schools,
    generate_households,
    prepare_for_inference,
)
from inference_utils import APP_CSS
from population import run_batch_inference

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Lahore School Simulation", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "lhr_ref_school": None,
        "lhr_nearest_schools": None,
        "lhr_households": None,
        "lhr_routes_fetched": False,
        "lhr_inference_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="title-pill">LAHORE — School Proximity & Household Simulation</div>',
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#94a3b8;margin-top:6px;'>"
    "Real school coordinates from POIs_Schools.csv · "
    "Haversine nearest-school search · "
    "Random households within 2 km · "
    "Road distance &amp; travel time via OSRM"
    "</p>",
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
col_btn, col_info = st.columns([1, 3])

with col_btn:
    generate_clicked = st.button(
        "Generate Simulation for Lahore",
        type="primary",
        use_container_width=True,
    )

with col_info:
    if st.session_state.lhr_nearest_schools is not None:
        n_hh = len(st.session_state.lhr_households) if st.session_state.lhr_households is not None else 0
        routed = st.session_state.lhr_routes_fetched
        inferred = st.session_state.lhr_inference_done
        st.markdown(
            f"<div style='padding-top:10px;color:#94a3b8;'>"
            f"5 nearest schools found &nbsp;|&nbsp; "
            f"{n_hh} households generated"
            f"{'&nbsp;|&nbsp; Road routes loaded' if routed else ''}"
            f"{'&nbsp;|&nbsp; <b style=\"color:#22c55e\">Enrollment probs computed</b>' if inferred else ''}"
            f"</div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Step 1 — find schools & scatter households
# ---------------------------------------------------------------------------
if generate_clicked:
    with st.spinner("Finding 5 nearest schools to FID 1314 …"):
        ref, nearest = find_nearest_schools(fid=1314, n=5)
        st.session_state.lhr_ref_school = ref
        st.session_state.lhr_nearest_schools = nearest
        st.session_state.lhr_routes_fetched = False
        st.session_state.lhr_inference_done = False

    with st.spinner("Generating 100 households within 2 km of schools …"):
        hh_df = generate_households(nearest, n=100, max_km=2.0)
        # Add placeholder columns so build_lahore_map works before routing
        hh_df["nearest_school_fid"] = hh_df["src_school_fid"]
        hh_df["nearest_school_name"] = hh_df["src_school_name"]
        hh_df["straight_km"] = None
        hh_df["road_km"] = None
        hh_df["travel_min"] = None
        st.session_state.lhr_households = hh_df

    st.rerun()

# ---------------------------------------------------------------------------
# Map — shown as soon as schools are available
# ---------------------------------------------------------------------------
if st.session_state.lhr_nearest_schools is not None:
    ref = st.session_state.lhr_ref_school
    nearest = st.session_state.lhr_nearest_schools
    hh_df = st.session_state.lhr_households

    # --- School table ---
    with st.expander("5 Nearest Schools (click to expand)", expanded=False):
        display_cols = ["fid", "Name", "Address", "Longitude", "Lattitude", "dist_km"]
        st.dataframe(
            nearest[display_cols].rename(columns={"dist_km": "Dist from Ref (km)"}),
            use_container_width=True,
            hide_index=True,
        )

    # --- Map ---
    fig = build_lahore_map(ref, nearest, hh_df if hh_df is not None and len(hh_df) > 0 else None)
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------------------------
    # Step 2 — fetch OSRM routes (only if not already done)
    # ---------------------------------------------------------------------------
    if hh_df is not None and not st.session_state.lhr_routes_fetched:
        st.info(
            "Households are placed on the map. "
            "Click below to fetch real road distances and travel times from OSRM "
            "(free, no API key — ~15–20 s for 100 households)."
        )
        if st.button("Fetch Road Distances & Travel Times", use_container_width=False):
            progress_bar = st.progress(0, text="Fetching routes from OSRM …")

            def _progress(done, total):
                progress_bar.progress(done / total, text=f"OSRM: {done}/{total} households …")

            with st.spinner("Querying OSRM for each household …"):
                routed_df = fetch_routes(hh_df, nearest, progress_cb=_progress)

            progress_bar.empty()
            st.session_state.lhr_households = routed_df
            st.session_state.lhr_routes_fetched = True
            st.rerun()

    # ---------------------------------------------------------------------------
    # Results table + summary — only after routes are fetched
    # ---------------------------------------------------------------------------
    if st.session_state.lhr_routes_fetched and hh_df is not None:
        st.markdown(
            '<div class="section-header">Household Route Results</div>',
            unsafe_allow_html=True,
        )

        # Summary metrics
        valid = hh_df.dropna(subset=["road_km"])
        has_probs = "enrollment_prob" in hh_df.columns and hh_df["enrollment_prob"].notna().any()

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Households", len(hh_df))
        m2.metric("Routes fetched", len(valid))
        m3.metric("Avg road dist", f"{valid['road_km'].mean():.2f} km" if len(valid) else "—")
        m4.metric("Avg travel time", f"{valid['travel_min'].mean():.1f} min" if len(valid) else "—")
        if has_probs:
            avg_prob = hh_df["enrollment_prob"].mean()
            pct_enrolled = (hh_df["enrollment_prob"] >= 0.5).mean()
            m5.metric("Avg enrollment prob", f"{avg_prob:.0%} ({pct_enrolled:.0%} ≥50%)")

        # ---------------------------------------------------------------------------
        # Step 3 — Run enrollment inference
        # ---------------------------------------------------------------------------
        if not st.session_state.lhr_inference_done:
            st.info("Routes loaded. Click below to run the enrollment probability model on all 100 households.")
            if st.button("Run Enrollment Inference", type="primary"):
                with st.spinner("Preparing features and running inference …"):
                    enriched = prepare_for_inference(hh_df, preset="urban")
                    probs = run_batch_inference(enriched, preset="urban")
                    enriched["enrollment_prob"] = probs.values
                st.session_state.lhr_households = enriched
                st.session_state.lhr_inference_done = True
                st.rerun()

        # Full table
        table_cols = ["hh_id", "lat", "lon", "nearest_school_name", "straight_km", "road_km", "travel_min"]
        col_rename = {
            "hh_id": "HH ID", "lat": "Latitude", "lon": "Longitude",
            "nearest_school_name": "Nearest School",
            "straight_km": "Straight (km)", "road_km": "Road (km)",
            "travel_min": "Travel Time (min)",
        }
        if has_probs:
            table_cols.append("enrollment_prob")
            col_rename["enrollment_prob"] = "Enrollment Prob"

        display_df = hh_df[table_cols].rename(columns=col_rename)

        if has_probs:
            # Format enrollment prob as percentage string for display
            display_df = display_df.copy()
            display_df["Enrollment Prob"] = display_df["Enrollment Prob"].map(
                lambda v: f"{v:.0%}" if pd.notna(v) else "—"
            )

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Enrollment summary chart — only when probs available
        if has_probs:
            import plotly.graph_objects as _go
            st.markdown('<div class="section-header">Enrollment Probability Summary</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                fig_hist = _go.Figure(_go.Histogram(
                    x=hh_df["enrollment_prob"],
                    nbinsx=20,
                    marker_color="#3b82f6",
                    opacity=0.85,
                ))
                fig_hist.update_layout(
                    height=220, margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                    title=dict(text="Distribution of Enrollment Probability", font=dict(size=12), x=0.5),
                    xaxis=dict(tickformat=".0%", title="Probability", color="#0f172a"),
                    yaxis=dict(title="# Households", color="#0f172a"),
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            with c2:
                cat_avg = (
                    hh_df.groupby("distance_cat")["enrollment_prob"].mean()
                    if "distance_cat" in hh_df.columns else None
                )
                if cat_avg is not None and len(cat_avg):
                    fig_bar = _go.Figure(_go.Bar(
                        x=cat_avg.index, y=cat_avg.values,
                        marker_color=["#22c55e" if v >= 0.5 else "#ef4444" for v in cat_avg.values],
                        text=[f"{v:.0%}" for v in cat_avg.values],
                        textposition="outside",
                    ))
                    fig_bar.update_layout(
                        height=220, margin=dict(l=10, r=10, t=30, b=10),
                        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                        title=dict(text="Avg Prob by Distance Category", font=dict(size=12), x=0.5),
                        xaxis=dict(color="#0f172a", tickangle=-15),
                        yaxis=dict(tickformat=".0%", range=[0, 1.1], color="#0f172a"),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

        # Download
        csv_bytes = hh_df.to_csv(index=False).encode()
        st.download_button(
            "Download Results as CSV",
            data=csv_bytes,
            file_name="lahore_simulation_results.csv",
            mime="text/csv",
        )

else:
    st.info("Click **Generate Simulation for Lahore** to begin.")
