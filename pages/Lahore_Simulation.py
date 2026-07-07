"""
Lahore School Simulation — all ~3000 schools, 10k households, real map, OSRM routing.
"""
import math

import numpy as np
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
from pslm_lahore import load_pslm_households

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
        "lhr_pslm_mode":      False,
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
col_btn, col_btn2, col_status = st.columns([1, 1, 2])

with col_btn:
    generate_clicked = st.button(
        "Generate Simulation for Lahore",
        type="primary",
        use_container_width=True,
    )

with col_btn2:
    pslm_clicked = st.button(
        "Validate with PSLM Data",
        type="secondary",
        use_container_width=True,
    )

with col_status:
    if st.session_state.get("lhr_pslm_mode"):
        st.markdown(
            "<div style='padding-top:10px;color:#94a3b8;'>"
            "<b style='color:#f59e0b'>PSLM Validation Mode</b> &nbsp;|&nbsp; "
            f"{len(st.session_state.lhr_households):,} households &nbsp;|&nbsp; "
            "Route safety forced to 1 &nbsp;|&nbsp; Geo randomly assigned"
            "</div>",
            unsafe_allow_html=True,
        )
    elif st.session_state.lhr_schools is not None:
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
# PSLM Validation — load households directly from Lahore.dta
# ---------------------------------------------------------------------------
if pslm_clicked:
    with st.spinner("Loading PSLM Lahore households …"):
        pslm_df = load_pslm_households()
    with st.spinner(f"Running enrollment inference on {len(pslm_df):,} PSLM households …"):
        probs = run_batch_inference(pslm_df, preset="urban")
        pslm_df["enrollment_prob"] = probs.values
    schools = load_all_schools()
    st.session_state.lhr_schools        = schools
    st.session_state.lhr_households     = pslm_df
    st.session_state.lhr_routes_fetched = True
    st.session_state.lhr_inference_done = True
    st.session_state.lhr_pslm_mode      = True
    st.rerun()

# ---------------------------------------------------------------------------
# Step 1 — load schools + scatter households
# ---------------------------------------------------------------------------
if generate_clicked:
    with st.spinner("Loading all schools from CSV …"):
        schools = load_all_schools()
        st.session_state.lhr_schools        = schools
        st.session_state.lhr_routes_fetched = False
        st.session_state.lhr_inference_done = False
        st.session_state.lhr_pslm_mode      = False

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
        is_pslm = st.session_state.get("lhr_pslm_mode", False)
        has_probs = "enrollment_prob" in hh_df.columns and hh_df["enrollment_prob"].notna().any()

        if is_pslm:
            st.markdown('<div class="section-header">PSLM Validation Results</div>', unsafe_allow_html=True)
            actual_rate = hh_df["actual_enrollment_prob"].mean() if "actual_enrollment_prob" in hh_df.columns else None
            model_rate  = hh_df["enrollment_prob"].mean() if has_probs else None
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Households (PSLM)", f"{len(hh_df):,}")
            if actual_rate is not None:
                m2.metric("Actual Enrollment Rate", f"{actual_rate:.1%}")
            if model_rate is not None:
                m3.metric("Model Predicted Rate", f"{model_rate:.1%}")
            if actual_rate is not None and model_rate is not None:
                diff = model_rate - actual_rate
                m4.metric("Difference (Model − Actual)", f"{diff:+.1%}")
            st.caption("Route safety forced to 1 for all households. Geo coordinates randomly assigned within Lahore.")
        else:
            st.markdown('<div class="section-header">Household Route Results</div>', unsafe_allow_html=True)
            valid = hh_df.dropna(subset=["road_km"])
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
        if is_pslm:
            table_cols = ["hh_id", "lat", "lon", "distance_cat", "monthly_income_raw"]
            col_rename = {
                "hh_id": "HH ID", "lat": "Latitude", "lon": "Longitude",
                "distance_cat": "Distance Category", "monthly_income_raw": "Income (PKR)",
            }
            if has_probs:
                table_cols += ["enrollment_prob", "actual_enrollment_prob"]
                col_rename["enrollment_prob"]        = "Model Prob"
                col_rename["actual_enrollment_prob"] = "Actual (PSLM)"
        else:
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

        table_cols = [c for c in table_cols if c in hh_df.columns]
        display_df = hh_df[table_cols].rename(columns=col_rename)
        if has_probs:
            display_df = display_df.copy()
            for prob_col in ["Enrollment Prob", "Model Prob", "Actual (PSLM)"]:
                if prob_col in display_df.columns:
                    display_df[prob_col] = display_df[prob_col].map(
                        lambda v: f"{v:.0%}" if pd.notna(v) else "—"
                    )
            if "Income (PKR)" in display_df.columns:
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
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=hh_df["enrollment_prob"], nbinsx=25,
                    marker_color="#3b82f6", opacity=0.85, name="Model",
                ))
                if is_pslm and "actual_enrollment_prob" in hh_df.columns:
                    fig_hist.add_trace(go.Histogram(
                        x=hh_df["actual_enrollment_prob"], nbinsx=25,
                        marker_color="#f59e0b", opacity=0.6, name="Actual (PSLM)",
                    ))
                    fig_hist.update_layout(barmode="overlay")
                fig_hist.update_layout(
                    height=240, margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                    title=dict(text="Enrollment Probability Distribution", font=dict(size=12), x=0.5),
                    xaxis=dict(tickformat=".0%", title="Probability", color="#0f172a"),
                    yaxis=dict(title="# Households", color="#0f172a"),
                    legend=dict(font=dict(color="#0f172a"), bgcolor="rgba(0,0,0,0)"),
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
                            title=dict(text="Model Prob by Distance Category", font=dict(size=12), x=0.5),
                            xaxis=dict(color="#0f172a", tickangle=-15),
                            yaxis=dict(tickformat=".0%", range=[0, 1.1], color="#0f172a"),
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
            with c3:
                if is_pslm and "actual_enrollment_prob" in hh_df.columns:
                    # Actual vs Model by distance category
                    from lahore_simulation import PSLM_QUINTILES
                    cat_order = ["Near (0–2 km)", "Moderate (2–6 km)", "Far (6–15 km)", "Very Far (15+ km)"]
                    grp = hh_df.groupby("distance_cat")[["enrollment_prob", "actual_enrollment_prob"]].mean().reindex(cat_order).dropna(how="all")
                    if len(grp):
                        fig_cmp = go.Figure()
                        fig_cmp.add_trace(go.Bar(
                            x=grp.index, y=grp["enrollment_prob"],
                            name="Model", marker_color="#3b82f6",
                            text=[f"{v:.0%}" for v in grp["enrollment_prob"]], textposition="outside",
                        ))
                        fig_cmp.add_trace(go.Bar(
                            x=grp.index, y=grp["actual_enrollment_prob"],
                            name="Actual (PSLM)", marker_color="#f59e0b",
                            text=[f"{v:.0%}" for v in grp["actual_enrollment_prob"]], textposition="outside",
                        ))
                        fig_cmp.update_layout(
                            barmode="group",
                            height=240, margin=dict(l=10, r=10, t=30, b=10),
                            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                            title=dict(text="Actual vs Model by Distance", font=dict(size=12), x=0.5),
                            xaxis=dict(color="#0f172a", tickangle=-15),
                            yaxis=dict(tickformat=".0%", range=[0, 1.2], color="#0f172a"),
                            legend=dict(font=dict(color="#0f172a"), bgcolor="rgba(0,0,0,0)"),
                        )
                        st.plotly_chart(fig_cmp, use_container_width=True)
                elif "income_quintile" in hh_df.columns:
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

        # -----------------------------------------------------------------------
        # PSLM Model Evaluation (ROC, Threshold Analysis, Confusion Matrix)
        # -----------------------------------------------------------------------
        if is_pslm and has_probs and "actual_enrollment_prob" in hh_df.columns:
            st.markdown('<div class="section-header">Model Evaluation — PSLM Validation</div>', unsafe_allow_html=True)
            st.caption("Binary ground truth: household is 'enrolled' if ≥50% of its children are in school (actual_enrollment_prob ≥ 0.5).")

            y_true = (hh_df["actual_enrollment_prob"] >= 0.5).astype(int).values
            y_score = hh_df["enrollment_prob"].values
            n_pos = y_true.sum()
            n_neg = len(y_true) - n_pos

            # --- ROC curve ---
            thresholds_roc = np.linspace(0.0, 1.0, 200)
            tprs, fprs = [], []
            for t in thresholds_roc:
                pred = (y_score >= t).astype(int)
                tp = ((pred == 1) & (y_true == 1)).sum()
                fp = ((pred == 1) & (y_true == 0)).sum()
                tpr = tp / n_pos if n_pos > 0 else 0
                fpr = fp / n_neg if n_neg > 0 else 0
                tprs.append(tpr)
                fprs.append(fpr)

            # AUC via trapezoidal rule on sorted FPR
            sorted_pairs = sorted(zip(fprs, tprs))
            s_fprs = [p[0] for p in sorted_pairs]
            s_tprs = [p[1] for p in sorted_pairs]
            auc = float(np.trapz(s_tprs, s_fprs))

            # --- Threshold sensitivity table ---
            eval_thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
            rows = []
            for t in eval_thresholds:
                pred = (y_score >= t).astype(int)
                tp = int(((pred == 1) & (y_true == 1)).sum())
                fp = int(((pred == 1) & (y_true == 0)).sum())
                tn = int(((pred == 0) & (y_true == 0)).sum())
                fn = int(((pred == 0) & (y_true == 1)).sum())
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                accuracy  = (tp + tn) / len(y_true)
                rows.append({"threshold": t, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                             "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy})
            eval_df = pd.DataFrame(rows)

            # Row 1: ROC + Threshold line chart
            re1, re2 = st.columns(2)

            with re1:
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=s_fprs, y=s_tprs, mode="lines",
                    line=dict(color="#3b82f6", width=2.5),
                    name=f"ROC (AUC = {auc:.3f})",
                    hovertemplate="FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra></extra>",
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(color="#94a3b8", width=1, dash="dash"),
                    name="Random", showlegend=True,
                ))
                fig_roc.update_layout(
                    height=300, margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                    title=dict(text=f"ROC Curve  (AUC = {auc:.3f})", font=dict(size=13, color="#0f172a"), x=0.5),
                    xaxis=dict(title="False Positive Rate", color="#0f172a", range=[0, 1]),
                    yaxis=dict(title="True Positive Rate",  color="#0f172a", range=[0, 1]),
                    legend=dict(font=dict(color="#0f172a"), bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_roc, use_container_width=True)

            with re2:
                fig_thr = go.Figure()
                thr_labels = [f"{int(t*100)}%" for t in eval_df["threshold"]]
                fig_thr.add_trace(go.Scatter(
                    x=thr_labels, y=eval_df["precision"], mode="lines+markers",
                    name="Precision", line=dict(color="#3b82f6", width=2),
                    marker=dict(size=7),
                ))
                fig_thr.add_trace(go.Scatter(
                    x=thr_labels, y=eval_df["recall"], mode="lines+markers",
                    name="Recall", line=dict(color="#f59e0b", width=2),
                    marker=dict(size=7),
                ))
                fig_thr.add_trace(go.Scatter(
                    x=thr_labels, y=eval_df["f1"], mode="lines+markers",
                    name="F1 Score", line=dict(color="#22c55e", width=2),
                    marker=dict(size=7),
                ))
                fig_thr.add_trace(go.Scatter(
                    x=thr_labels, y=eval_df["accuracy"], mode="lines+markers",
                    name="Accuracy", line=dict(color="#a855f7", width=2, dash="dot"),
                    marker=dict(size=7),
                ))
                fig_thr.update_layout(
                    height=300, margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                    title=dict(text="Precision / Recall / F1 by Threshold", font=dict(size=13, color="#0f172a"), x=0.5),
                    xaxis=dict(title="Threshold", color="#0f172a"),
                    yaxis=dict(title="Score", color="#0f172a", range=[0, 1.05]),
                    legend=dict(font=dict(color="#0f172a"), bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.25),
                )
                st.plotly_chart(fig_thr, use_container_width=True)

            # Row 2: Correct predictions bar + Confusion matrix
            re3, re4 = st.columns(2)

            with re3:
                fig_correct = go.Figure()
                fig_correct.add_trace(go.Bar(
                    x=thr_labels,
                    y=eval_df["tp"] + eval_df["tn"],
                    name="Correct", marker_color="#22c55e",
                    text=(eval_df["tp"] + eval_df["tn"]).astype(str),
                    textposition="outside",
                ))
                fig_correct.add_trace(go.Bar(
                    x=thr_labels,
                    y=eval_df["fp"] + eval_df["fn"],
                    name="Incorrect", marker_color="#ef4444",
                    text=(eval_df["fp"] + eval_df["fn"]).astype(str),
                    textposition="outside",
                ))
                fig_correct.update_layout(
                    barmode="stack",
                    height=300, margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                    title=dict(text="Correct vs Incorrect Households by Threshold", font=dict(size=13, color="#0f172a"), x=0.5),
                    xaxis=dict(title="Threshold", color="#0f172a"),
                    yaxis=dict(title="# Households", color="#0f172a"),
                    legend=dict(font=dict(color="#0f172a"), bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_correct, use_container_width=True)

            with re4:
                # Confusion matrix at threshold 0.50
                row50 = eval_df[eval_df["threshold"] == 0.50].iloc[0]
                tp50, fp50, tn50, fn50 = int(row50.tp), int(row50.fp), int(row50.tn), int(row50.fn)
                cm_z    = [[tn50, fp50], [fn50, tp50]]
                cm_text = [
                    [f"<b>{tn50:,}</b><br>True Negative<br>Model: No  | Actual: No",
                     f"<b>{fp50:,}</b><br>False Positive<br>Model: Yes | Actual: No"],
                    [f"<b>{fn50:,}</b><br>False Negative<br>Model: No  | Actual: Yes",
                     f"<b>{tp50:,}</b><br>True Positive<br>Model: Yes | Actual: Yes"],
                ]
                fig_cm = go.Figure(go.Heatmap(
                    z=cm_z,
                    text=cm_text,
                    texttemplate="%{text}",
                    colorscale=[[0, "#fef2f2"], [0.5, "#bfdbfe"], [1, "#1d4ed8"]],
                    showscale=False,
                    hoverinfo="text",
                ))
                fig_cm.update_layout(
                    height=300, margin=dict(l=10, r=10, t=40, b=40),
                    paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                    title=dict(text="Confusion Matrix  (threshold = 50%)", font=dict(size=13, color="#0f172a"), x=0.5),
                    xaxis=dict(tickvals=[0, 1], ticktext=["Predicted: No", "Predicted: Yes"], color="#0f172a"),
                    yaxis=dict(tickvals=[0, 1], ticktext=["Actual: No", "Actual: Yes"], color="#0f172a"),
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            # Summary stats table
            st.markdown("**Metrics by Threshold**")
            display_eval = eval_df.copy()
            display_eval["threshold"] = display_eval["threshold"].map(lambda v: f"{v:.0%}")
            display_eval["precision"] = display_eval["precision"].map(lambda v: f"{v:.1%}")
            display_eval["recall"]    = display_eval["recall"].map(lambda v: f"{v:.1%}")
            display_eval["f1"]        = display_eval["f1"].map(lambda v: f"{v:.3f}")
            display_eval["accuracy"]  = display_eval["accuracy"].map(lambda v: f"{v:.1%}")
            display_eval = display_eval.rename(columns={
                "threshold": "Threshold", "tp": "TP", "fp": "FP",
                "tn": "TN", "fn": "FN",
                "precision": "Precision", "recall": "Recall",
                "f1": "F1", "accuracy": "Accuracy",
            })
            st.dataframe(display_eval, use_container_width=True, hide_index=True)

        # Download
        csv_bytes = hh_df.to_csv(index=False).encode()
        st.download_button(
            "Download Results as CSV",
            data=csv_bytes,
            file_name="lahore_simulation_results.csv",
            mime="text/csv",
        )

else:
    st.info("Click **Generate Simulation for Lahore** to run the proximity simulation, or **Validate with PSLM Data** to load real survey households.")
