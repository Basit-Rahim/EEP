"""
Lahore School Simulation — all ~3000 schools, 10k households, real map, OSRM routing.
"""
import math

import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from lahore_simulation import (
    build_lahore_map,
    fetch_routes,
    generate_households,
    load_all_schools,
    prepare_for_inference,
)
from inference_utils import APP_CSS
from population import run_batch_inference

st.set_page_config(page_title="Lahore School Simulation", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state():
    for k, v in {
        "lhr_schools":        None,
        "lhr_households":     None,
        "lhr_routes_fetched": False,
        "lhr_inference_done": False,
    }.items():
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
    "All schools from POIs_Schools.csv · Vectorized Haversine nearest-school · "
    "10,000 households within 2 km · OSRM Table API (chunked) · Enrollment inference"
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Controls row
# ---------------------------------------------------------------------------
col_btn, col_status = st.columns([1, 3])

with col_btn:
    generate_clicked = st.button(
        "Generate Simulation for Lahore",
        type="primary",
        use_container_width=True,
    )

with col_status:
    if st.session_state.lhr_schools is not None:
        n_sc  = len(st.session_state.lhr_schools)
        n_hh  = len(st.session_state.lhr_households) if st.session_state.lhr_households is not None else 0
        parts = [f"{n_sc:,} schools loaded", f"{n_hh:,} households generated"]
        if st.session_state.lhr_routes_fetched:
            parts.append("Road routes loaded")
        if st.session_state.lhr_inference_done:
            parts.append('<b style="color:#22c55e">Enrollment probs computed</b>')
        st.markdown(
            f"<div style='padding-top:10px;color:#94a3b8;'>"
            + " &nbsp;|&nbsp; ".join(parts)
            + "</div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Step 1 — load schools + scatter households
# ---------------------------------------------------------------------------
if generate_clicked:
    with st.spinner("Loading all schools from CSV …"):
        schools = load_all_schools()
        st.session_state.lhr_schools        = schools
        st.session_state.lhr_routes_fetched = False
        st.session_state.lhr_inference_done = False

    with st.spinner(f"Generating 10,000 households near {len(schools):,} schools …"):
        hh_df = generate_households(schools, n=10_000, max_km=2.0)
        hh_df["nearest_school_fid"]  = hh_df["src_school_fid"]
        hh_df["nearest_school_name"] = hh_df["src_school_name"]
        hh_df["straight_km"]  = None
        hh_df["road_km"]      = None
        hh_df["travel_min"]   = None
        st.session_state.lhr_households = hh_df

    st.rerun()

# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
if st.session_state.lhr_schools is not None:
    schools = st.session_state.lhr_schools
    hh_df   = st.session_state.lhr_households

    fig = build_lahore_map(
        schools,
        hh_df if hh_df is not None and len(hh_df) > 0 else None,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------------------------
    # Step 2 — OSRM routes
    # ---------------------------------------------------------------------------
    if hh_df is not None and not st.session_state.lhr_routes_fetched:
        n_chunks = math.ceil(len(hh_df) / 100)
        st.info(
            f"10,000 households placed. Click below to fetch road distances via OSRM "
            f"Table API (~{n_chunks} requests, typically 30–60 s)."
        )
        if st.button("Fetch Road Distances & Travel Times"):
            prog = st.progress(0, text="Vectorizing nearest-school search …")

            def _cb(done, total):
                prog.progress(done / total, text=f"OSRM Table API: {done:,}/{total:,} households …")

            with st.spinner("Running OSRM Table API in chunks …"):
                routed_df = fetch_routes(hh_df, schools, progress_cb=_cb)

            prog.empty()
            st.session_state.lhr_households = routed_df
            st.session_state.lhr_routes_fetched = True
            st.rerun()

    # ---------------------------------------------------------------------------
    # Results + Step 3 — inference
    # ---------------------------------------------------------------------------
    if st.session_state.lhr_routes_fetched and hh_df is not None:
        st.markdown('<div class="section-header">Household Route Results</div>', unsafe_allow_html=True)

        valid     = hh_df.dropna(subset=["road_km"])
        has_probs = "enrollment_prob" in hh_df.columns and hh_df["enrollment_prob"].notna().any()

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Households",    f"{len(hh_df):,}")
        m2.metric("Routes fetched",f"{len(valid):,}")
        m3.metric("Avg road dist", f"{valid['road_km'].mean():.2f} km"   if len(valid) else "—")
        m4.metric("Avg travel time",f"{valid['travel_min'].mean():.1f} min" if len(valid) else "—")
        if has_probs:
            avg_p = hh_df["enrollment_prob"].mean()
            pct   = (hh_df["enrollment_prob"] >= 0.5).mean()
            m5.metric("Avg enrollment prob", f"{avg_p:.0%}  ({pct:.0%} ≥50%)")

        # Step 3 — inference button
        if not st.session_state.lhr_inference_done:
            st.info(
                "Routes loaded. Income is sampled from Normal(mean=PKR 48,412, std=13,308) "
                "clipped to Lahore-urban bounds [2,500–120,000 PKR]. "
                "Click below to run enrollment inference on all households."
            )
            if st.button("Run Enrollment Inference", type="primary"):
                with st.spinner("Preparing features and running inference …"):
                    enriched = prepare_for_inference(hh_df, preset="urban")
                    probs    = run_batch_inference(enriched, preset="urban")
                    enriched["enrollment_prob"] = probs.values
                st.session_state.lhr_households = enriched
                st.session_state.lhr_inference_done = True
                st.rerun()

        # Table (sample of 500 for display speed)
        table_cols = ["hh_id", "lat", "lon", "nearest_school_name", "straight_km", "road_km", "travel_min"]
        col_rename = {
            "hh_id": "HH ID", "lat": "Latitude", "lon": "Longitude",
            "nearest_school_name": "Nearest School",
            "straight_km": "Straight (km)", "road_km": "Road (km)",
            "travel_min": "Travel Time (min)",
        }
        if has_probs:
            table_cols += ["monthly_income_raw", "income_quintile", "enrollment_prob"]
            col_rename["monthly_income_raw"] = "Income (PKR)"
            col_rename["income_quintile"]    = "PSLM Quintile"
            col_rename["enrollment_prob"]    = "Enrollment Prob"

        display_df = hh_df[table_cols].rename(columns=col_rename)
        if has_probs:
            display_df = display_df.copy()
            display_df["Enrollment Prob"] = display_df["Enrollment Prob"].map(
                lambda v: f"{v:.0%}" if pd.notna(v) else "—"
            )
            display_df["Income (PKR)"] = display_df["Income (PKR)"].map(
                lambda v: f"{v:,.0f}" if pd.notna(v) else "—"
            )

        st.caption(f"Showing first 500 of {len(display_df):,} rows")
        st.dataframe(display_df.head(500), use_container_width=True, hide_index=True)

        # Charts
        if has_probs:
            st.markdown('<div class="section-header">Enrollment Probability Summary</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                fig_hist = go.Figure(go.Histogram(
                    x=hh_df["enrollment_prob"], nbinsx=25,
                    marker_color="#3b82f6", opacity=0.85,
                ))
                fig_hist.update_layout(
                    height=240, margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                    title=dict(text="Enrollment Probability Distribution", font=dict(size=12), x=0.5),
                    xaxis=dict(tickformat=".0%", title="Probability", color="#0f172a"),
                    yaxis=dict(title="# Households", color="#0f172a"),
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            with c2:
                if "distance_cat" in hh_df.columns:
                    cat_avg = hh_df.groupby("distance_cat")["enrollment_prob"].mean().dropna()
                    if len(cat_avg):
                        fig_bar = go.Figure(go.Bar(
                            x=cat_avg.index, y=cat_avg.values,
                            marker_color=["#22c55e" if v >= 0.5 else "#ef4444" for v in cat_avg.values],
                            text=[f"{v:.0%}" for v in cat_avg.values],
                            textposition="outside",
                        ))
                        fig_bar.update_layout(
                            height=240, margin=dict(l=10, r=10, t=30, b=10),
                            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                            title=dict(text="Avg Prob by Distance Category", font=dict(size=12), x=0.5),
                            xaxis=dict(color="#0f172a", tickangle=-15),
                            yaxis=dict(tickformat=".0%", range=[0, 1.1], color="#0f172a"),
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
            with c3:
                if "income_quintile" in hh_df.columns:
                    from lahore_simulation import PSLM_QUINTILES
                    q_order = [q[0] for q in PSLM_QUINTILES]
                    q_avg = hh_df.groupby("income_quintile")["enrollment_prob"].mean().reindex(q_order).dropna()
                    if len(q_avg):
                        fig_q = go.Figure(go.Bar(
                            x=q_avg.index, y=q_avg.values,
                            marker_color=["#22c55e" if v >= 0.5 else "#ef4444" for v in q_avg.values],
                            text=[f"{v:.0%}" for v in q_avg.values],
                            textposition="outside",
                        ))
                        fig_q.update_layout(
                            height=240, margin=dict(l=10, r=10, t=30, b=10),
                            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                            title=dict(text="Avg Prob by Income Quintile (PSLM)", font=dict(size=12), x=0.5),
                            xaxis=dict(color="#0f172a", tickangle=-20),
                            yaxis=dict(tickformat=".0%", range=[0, 1.1], color="#0f172a"),
                        )
                        st.plotly_chart(fig_q, use_container_width=True)

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
