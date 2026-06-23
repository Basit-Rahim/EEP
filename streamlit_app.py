"""
Rural Community — ABM Population Simulation
Main entry point. Run with: streamlit run streamlit_app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from inference_utils import (
    APP_CSS,
    DISTANCE_CATEGORIES,
    DISTANCE_CAT_LIST,
    FEATURE_COLUMNS,
    INCOME_STATS,
    TRAVEL_MODE_MAP,
    TRAVEL_MODE_UI,
    YES_NO_UI,
    run_full_inference,
)
from population import (
    apply_global_adjustment,
    generate_population,
    generate_school,
    load_population_from_csv,
    run_batch_inference,
)

PRESET = "rural"

# -----------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------
st.set_page_config(page_title="ABM — Rural Community", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------
def _init_state():
    if "agents_df" not in st.session_state:
        st.session_state.agents_df = None
    if "school" not in st.session_state:
        st.session_state.school = generate_school(PRESET)
    if "selected_agent_id" not in st.session_state:
        st.session_state.selected_agent_id = None
    if "inference_done" not in st.session_state:
        st.session_state.inference_done = False
    if "pslm_mode" not in st.session_state:
        st.session_state.pslm_mode = False


# -----------------------------------------------------------------------
# MAP BUILDER
# -----------------------------------------------------------------------
def build_map(agents_df: pd.DataFrame, school: dict, selected_id) -> go.Figure:
    fig = go.Figure()

    # School marker
    fig.add_trace(go.Scatter(
        x=[school["x"]], y=[school["y"]],
        mode="markers+text",
        marker=dict(symbol="square", size=22, color="#1d4ed8", line=dict(color="#ffffff", width=2)),
        text=[school["name"]],
        textposition="top center",
        textfont=dict(color="#ffffff", size=11),
        name="School",
        hovertemplate=f"<b>{school['name']}</b><extra></extra>",
    ))

    if agents_df is None or len(agents_df) == 0:
        fig.update_layout(**_map_layout())
        return fig

    has_probs = agents_df["enrollment_prob"].notna().any()

    if has_probs:
        color_vals = agents_df["enrollment_prob"].fillna(0.5).values
        colorscale = "RdYlGn"
        marker_color = color_vals
        showscale = True
    else:
        marker_color = "#94a3b8"
        colorscale = None
        showscale = False

    def hover_text(row):
        prob_str = f"{row.enrollment_prob:.0%}" if pd.notna(row.enrollment_prob) else "—"
        lines = [
            f"<b>Agent #{int(row.hh_id)}</b>",
            f"Income: {row.monthly_income_raw:,.0f} PKR",
            f"Distance: {row.distance_cat}",
            f"Route: {'Safe' if row.route_safe == 1 else 'Unsafe'}",
            f"Model Predicted: {prob_str}",
        ]
        if "actual_enrollment_prob" in row.index and pd.notna(row.actual_enrollment_prob):
            lines.append(f"Actual (PSLM): <b>{row.actual_enrollment_prob:.0%}</b>")
        return "<br>".join(lines)

    hover_texts = agents_df.apply(hover_text, axis=1).tolist()

    # Non-selected agents
    non_selected = agents_df[agents_df["hh_id"] != selected_id] if selected_id is not None else agents_df
    if len(non_selected) > 0:
        if has_probs:
            mc = non_selected["enrollment_prob"].fillna(0.5).values
        else:
            mc = "#94a3b8"

        fig.add_trace(go.Scatter(
            x=non_selected["x"].values,
            y=non_selected["y"].values,
            mode="markers",
            marker=dict(
                color=mc,
                colorscale=colorscale if has_probs else None,
                cmin=0.0, cmax=1.0,
                size=8,
                opacity=0.85,
                line=dict(color="rgba(0,0,0,0.1)", width=0.5),
                colorbar=dict(
                    title="Enrollment<br>Probability",
                    tickformat=".0%",
                    thickness=15,
                    len=0.6,
                ) if showscale else None,
                showscale=showscale,
            ),
            text=[hover_texts[i] for i in non_selected.index],
            hovertemplate="%{text}<extra></extra>",
            name="Households",
        ))

    # Selected agent (highlighted)
    if selected_id is not None:
        sel = agents_df[agents_df["hh_id"] == selected_id]
        if len(sel) > 0:
            fig.add_trace(go.Scatter(
                x=sel["x"].values,
                y=sel["y"].values,
                mode="markers",
                marker=dict(
                    color="#f59e0b",
                    size=16,
                    symbol="circle",
                    line=dict(color="#ffffff", width=3),
                ),
                hovertemplate=f"<b>SELECTED: Agent #{int(selected_id)}</b><extra></extra>",
                name=f"Selected #{int(selected_id)}",
                showlegend=True,
            ))

    fig.update_layout(**_map_layout())
    return fig


def _map_layout():
    return dict(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        xaxis=dict(
            range=[0, 50], showgrid=True, gridcolor="#334155",
            zeroline=False, title="",
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            range=[0, 100], showgrid=True, gridcolor="#334155",
            zeroline=False, title="",
            tickfont=dict(color="#94a3b8"),
        ),
        legend=dict(
            bgcolor="rgba(30,41,59,0.9)",
            font=dict(color="#ffffff"),
            bordercolor="#334155",
            borderwidth=1,
        ),
        clickmode="event+select",
    )


# -----------------------------------------------------------------------
# AGENT EDIT PANEL
# -----------------------------------------------------------------------
def render_agent_panel(agents_df: pd.DataFrame):
    sel_id = st.session_state.selected_agent_id
    agent_row = agents_df[agents_df["hh_id"] == sel_id]
    if len(agent_row) == 0:
        return

    agent = agent_row.iloc[0]
    prob_str = f"{agent.enrollment_prob:.0%}" if pd.notna(agent.enrollment_prob) else "Not run"
    prob_color = "#22c55e" if pd.notna(agent.enrollment_prob) and agent.enrollment_prob >= 0.5 else "#ef4444"

    st.markdown(
        f"""<div class="css-card">
        <span style="font-weight:800;font-size:1.1rem;color:#0f172a;">Agent #{int(sel_id)}</span>
        &nbsp;&nbsp;
        <span style="font-weight:800;font-size:1.4rem;color:{prob_color};">{prob_str}</span>
        &nbsp;enrollment probability
        </div>""",
        unsafe_allow_html=True,
    )

    stats = INCOME_STATS[PRESET]
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        income_raw = st.slider(
            "Monthly Income (PKR)",
            float(stats["raw_min"]), float(stats["raw_max"]),
            float(agent.monthly_income_raw), step=500.0,
            key="panel_income",
        )
    with c2:
        route_label = "Safe" if agent.route_safe == 1 else "Unsafe"
        route_str = st.selectbox("Route Safety", ["Safe", "Unsafe"],
                                 index=0 if route_label == "Safe" else 1, key="panel_route")
        route_val = 1 if route_str == "Safe" else 0

    with c3:
        dist_cat = agent.get("distance_cat", DISTANCE_CAT_LIST[0])
        if dist_cat not in DISTANCE_CAT_LIST:
            dist_cat = DISTANCE_CAT_LIST[0]
        dist_str = st.selectbox("Distance to School", DISTANCE_CAT_LIST,
                                index=DISTANCE_CAT_LIST.index(dist_cat), key="panel_dist")

    with c4:
        t_label = TRAVEL_MODE_MAP.get(int(agent.travel_mode), "On foot")
        travel_str = st.selectbox("Travel Mode", list(TRAVEL_MODE_UI.keys()),
                                  index=list(TRAVEL_MODE_UI.keys()).index(t_label)
                                  if t_label in TRAVEL_MODE_UI else 0,
                                  key="panel_travel")
        travel_val = TRAVEL_MODE_UI[travel_str]

    c5, c6, c7 = st.columns(3)
    with c5:
        rw_str = st.radio("Head can Read/Write?", ["Yes", "No"],
                          index=0 if int(agent.read_write) == 1 else 1,
                          horizontal=True, key="panel_rw")
        rw_val = YES_NO_UI[rw_str]
    with c6:
        math_str = st.radio("Head can Solve Math?", ["Yes", "No"],
                            index=0 if int(agent.solve_math) == 1 else 1,
                            horizontal=True, key="panel_math")
        math_val = YES_NO_UI[math_str]
    with c7:
        fac_val = st.slider("School Facilities (0–5)", 0.0, 5.0,
                            float(agent.school_facilities), step=0.5, key="panel_fac")

    if st.button("Re-run Inference for This Agent", key="panel_run"):
        dist_vals = DISTANCE_CATEGORIES[dist_str]
        income_norm = (income_raw - stats["mean"]) / stats["std"]

        model_inputs = {
            "hh_id": int(sel_id),
            "monthly_income": income_norm,
            "travel_mode": float(travel_val),
            "route_safe": float(route_val),
            "read_write": float(rw_val),
            "solve_math": float(math_val),
            "school_facilities": float(fac_val),
            "min_distance": dist_vals["min_distance"],
            "max_distance": dist_vals["max_distance"],
            "min_time": dist_vals["min_time"],
            "max_time": dist_vals["max_time"],
        }
        new_prob = run_full_inference(model_inputs, PRESET)

        idx = agents_df.index[agents_df["hh_id"] == sel_id][0]
        st.session_state.agents_df.at[idx, "enrollment_prob"] = new_prob
        st.session_state.agents_df.at[idx, "monthly_income_raw"] = income_raw
        st.session_state.agents_df.at[idx, "monthly_income_norm"] = income_norm
        st.session_state.agents_df.at[idx, "route_safe"] = route_val
        st.session_state.agents_df.at[idx, "travel_mode"] = travel_val
        st.session_state.agents_df.at[idx, "read_write"] = rw_val
        st.session_state.agents_df.at[idx, "solve_math"] = math_val
        st.session_state.agents_df.at[idx, "school_facilities"] = fac_val
        st.session_state.agents_df.at[idx, "distance_cat"] = dist_str
        st.session_state.agents_df.at[idx, "min_distance"] = dist_vals["min_distance"]
        st.session_state.agents_df.at[idx, "max_distance"] = dist_vals["max_distance"]
        st.session_state.agents_df.at[idx, "min_time"] = dist_vals["min_time"]
        st.session_state.agents_df.at[idx, "max_time"] = dist_vals["max_time"]
        st.rerun()


# -----------------------------------------------------------------------
# SUMMARY CHARTS
# -----------------------------------------------------------------------
def render_summary(agents_df: pd.DataFrame):
    valid = agents_df.dropna(subset=["enrollment_prob"])
    if len(valid) == 0:
        return

    avg_prob = valid["enrollment_prob"].mean()
    enrolled_pct = (valid["enrollment_prob"] >= 0.5).mean()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""<div class="css-card prob-display">
            <div class="prob-number">{avg_prob:.0%}</div>
            <div class="prob-label">Avg Enrollment Probability</div>
            <div style="color:#64748b;font-size:0.9rem;margin-top:8px;">
            {enrolled_pct:.0%} of agents above 50%
            </div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col2:
        fig_hist = go.Figure(go.Histogram(
            x=valid["enrollment_prob"],
            nbinsx=20,
            marker_color="#3b82f6",
            opacity=0.85,
        ))
        fig_hist.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            title=dict(text="Enrollment Probability Distribution", font=dict(size=12, color="#0f172a"), x=0.5),
            xaxis=dict(tickformat=".0%", title="Probability", color="#0f172a"),
            yaxis=dict(title="# Agents", color="#0f172a"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col3:
        dist_order = DISTANCE_CAT_LIST
        dist_avgs = (
            valid.groupby("distance_cat")["enrollment_prob"]
            .mean()
            .reindex(dist_order)
            .dropna()
        )
        fig_bar = go.Figure(go.Bar(
            x=dist_avgs.index,
            y=dist_avgs.values,
            marker_color=["#22c55e" if v >= 0.5 else "#ef4444" for v in dist_avgs.values],
            text=[f"{v:.0%}" for v in dist_avgs.values],
            textposition="outside",
        ))
        fig_bar.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            title=dict(text="Avg Prob by Distance Category", font=dict(size=12, color="#0f172a"), x=0.5),
            xaxis=dict(color="#0f172a", tickangle=-15),
            yaxis=dict(tickformat=".0%", range=[0, 1.05], color="#0f172a"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
def main():
    _init_state()

    st.markdown(
        '<div class="title-pill">RURAL COMMUNITY — Agent Based Modeling for Out-of-School Children</div>',
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------------------
    # TOP CONTROLS
    # ----------------------------------------------------------------
    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5, ctrl6 = st.columns([1, 1, 1, 1, 1, 2])

    with ctrl1:
        n_agents = st.number_input("Number of agents", min_value=50, max_value=2000,
                                   value=400, step=50, key="n_agents_rural")
    with ctrl2:
        st.write("")
        st.write("")
        if st.button("Generate Population", key="gen_btn"):
            st.session_state.agents_df = generate_population(int(n_agents), PRESET)
            st.session_state.selected_agent_id = None
            st.session_state.inference_done = False
            st.session_state.pslm_mode = False
            st.rerun()

    with ctrl3:
        st.write("")
        st.write("")
        if st.button("Validate with PSLM", key="pslm_btn", type="secondary"):
            from pslm_lahore import load_pslm_households
            with st.spinner("Loading PSLM Lahore households …"):
                pslm_df = load_pslm_households()
            with st.spinner(f"Running inference on {len(pslm_df):,} households …"):
                probs = run_batch_inference(pslm_df, preset="urban")
                pslm_df["enrollment_prob"] = probs.values
            st.session_state.agents_df = pslm_df
            st.session_state.selected_agent_id = None
            st.session_state.inference_done = True
            st.session_state.pslm_mode = True
            st.rerun()

    with ctrl4:
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload_rural",
                                    label_visibility="collapsed")
        if uploaded is not None:
            try:
                st.session_state.agents_df = load_population_from_csv(uploaded, PRESET)
                st.session_state.selected_agent_id = None
                st.session_state.inference_done = False
                st.session_state.pslm_mode = False
                st.success(f"Loaded {len(st.session_state.agents_df)} agents from CSV.")
            except ValueError as e:
                st.error(str(e))

    with ctrl5:
        st.write("")
        st.write("")
        run_disabled = st.session_state.agents_df is None
        if st.button("Run Batch Inference", disabled=run_disabled, type="primary", key="run_btn"):
            with st.spinner("Running inference on all agents..."):
                probs = run_batch_inference(st.session_state.agents_df, PRESET)
                st.session_state.agents_df["enrollment_prob"] = probs.values
                st.session_state.inference_done = True
            st.rerun()

    with ctrl6:
        if st.session_state.agents_df is not None:
            n = len(st.session_state.agents_df)
            done = st.session_state.agents_df["enrollment_prob"].notna().sum()
            if done > 0:
                avg = st.session_state.agents_df["enrollment_prob"].mean()
                st.markdown(
                    f"<div style='padding-top:16px;color:#94a3b8;'>"
                    f"{n} agents &nbsp;|&nbsp; {done} inferred &nbsp;|&nbsp; "
                    f"Avg: <b style='color:#22c55e'>{avg:.0%}</b></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='padding-top:16px;color:#94a3b8;'>{n} agents generated — run inference to score.</div>",
                    unsafe_allow_html=True,
                )

    # ----------------------------------------------------------------
    # MAP
    # ----------------------------------------------------------------
    if st.session_state.agents_df is not None:
        fig = build_map(
            st.session_state.agents_df,
            st.session_state.school,
            st.session_state.selected_agent_id,
        )
        event = st.plotly_chart(fig, on_select="rerun", key="sim_map_rural", use_container_width=True)

        # Handle click — "agent" trace is index 1 (index 0 = school)
        if event and hasattr(event, "selection") and event.selection.points:
            pt = event.selection.points[0]
            # point_index maps into the non-selected trace; find actual hh_id
            trace_idx = pt.get("curve_number", 1)
            point_idx = pt.get("point_index", 0)
            agents_df = st.session_state.agents_df

            if trace_idx == 1 and len(agents_df) > point_idx:
                # The non-selected scatter is sorted by agents_df order (excluding selected)
                sel_id_now = st.session_state.selected_agent_id
                if sel_id_now is not None:
                    non_sel = agents_df[agents_df["hh_id"] != sel_id_now]
                else:
                    non_sel = agents_df
                if point_idx < len(non_sel):
                    clicked_hh_id = int(non_sel.iloc[point_idx]["hh_id"])
                    if clicked_hh_id != st.session_state.selected_agent_id:
                        st.session_state.selected_agent_id = clicked_hh_id
                        st.rerun()
    else:
        st.info("Generate or upload a population to see the simulation map.")

    # ----------------------------------------------------------------
    # AGENT EDIT PANEL
    # ----------------------------------------------------------------
    if (
        st.session_state.selected_agent_id is not None
        and st.session_state.agents_df is not None
    ):
        st.markdown('<div class="section-header">Edit Selected Agent</div>', unsafe_allow_html=True)
        render_agent_panel(st.session_state.agents_df)

        if st.button("Deselect Agent", key="deselect_btn"):
            st.session_state.selected_agent_id = None
            st.rerun()

    # ----------------------------------------------------------------
    # SUMMARY STATISTICS
    # ----------------------------------------------------------------
    if (
        st.session_state.agents_df is not None
        and st.session_state.agents_df["enrollment_prob"].notna().any()
    ):
        if st.session_state.get("pslm_mode") and "actual_enrollment_prob" in st.session_state.agents_df.columns:
            st.markdown('<div class="section-header">PSLM Validation Summary</div>', unsafe_allow_html=True)
            df_v = st.session_state.agents_df
            actual_rate = df_v["actual_enrollment_prob"].mean()
            model_rate  = df_v["enrollment_prob"].mean()
            diff = model_rate - actual_rate
            vm1, vm2, vm3, vm4 = st.columns(4)
            vm1.metric("Actual Enrollment Rate", f"{actual_rate:.1%}")
            vm2.metric("Model Predicted Rate",   f"{model_rate:.1%}")
            vm3.metric("Difference (Model − Actual)", f"{diff:+.1%}")
            vm4.metric("N Households (PSLM)",    f"{len(df_v):,}")
            st.caption("Route safety forced to 1. School facilities randomly assigned. Geo coordinates random within Lahore.")
        else:
            st.markdown('<div class="section-header">Population Summary</div>', unsafe_allow_html=True)
        render_summary(st.session_state.agents_df)

    # ----------------------------------------------------------------
    # GLOBAL ADJUSTMENTS
    # ----------------------------------------------------------------
    if st.session_state.agents_df is not None:
        st.markdown('<div class="section-header">Global Parameter Adjustments</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            ga1, ga2, ga3 = st.columns([2, 1, 1])
            with ga1:
                income_shift = st.slider(
                    "Income shift for all agents (%)",
                    -50, 50, 0, step=5, key="global_income",
                    format="%d%%",
                )
            with ga2:
                force_safe = st.checkbox("Force all routes safe", key="global_safe")
            with ga3:
                st.write("")
                st.write("")
                if st.button("Apply & Re-run All", key="global_apply"):
                    df_adj = apply_global_adjustment(
                        st.session_state.agents_df,
                        income_pct=income_shift / 100.0,
                        force_route_safe=force_safe,
                    )
                    with st.spinner("Re-running inference on adjusted population..."):
                        probs = run_batch_inference(df_adj, PRESET)
                        df_adj["enrollment_prob"] = probs.values
                    st.session_state.agents_df = df_adj
                    st.session_state.inference_done = True
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------------------------------------------
    # DOWNLOAD
    # ----------------------------------------------------------------
    if (
        st.session_state.agents_df is not None
        and st.session_state.agents_df["enrollment_prob"].notna().any()
    ):
        export_cols = ["hh_id", "monthly_income_raw", "travel_mode", "route_safe",
                       "read_write", "solve_math", "school_facilities",
                       "distance_cat", "min_distance", "max_distance",
                       "min_time", "max_time", "enrollment_prob"]
        export_df = st.session_state.agents_df[export_cols].copy()
        csv_bytes = export_df.to_csv(index=False).encode()
        st.download_button(
            "Download Results as CSV",
            data=csv_bytes,
            file_name="rural_simulation_results.csv",
            mime="text/csv",
            key="download_btn",
        )


if __name__ == "__main__":
    main()
