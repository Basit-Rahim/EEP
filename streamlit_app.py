import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

# --- ASSUMED LOCAL IMPORTS ---
# (Ensure these files exist in your directory)
from inference_autoencoder import Autoencoder, encode_with_autoencoder
from inference_phase1 import get_p1_models
from inference_phase2 import load_model
from preprocess import FACILITY_WEIGHTS


# ---------------------------------------------------------------------
# CONFIG & STYLE
# ---------------------------------------------------------------------
st.set_page_config(page_title="ABM Dashboard", layout="wide")

# Polished Color Scheme & CSS
st.markdown(
    """
    <style>
    /* Global Background - Deep Professional Blue */
    .stApp {
        background-color: #1e293b;
    }
    
    /* Typography */
    h1, h2, h3, p, div, label, span, button {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    }
    /* Default text to white on dark background */
    .stApp, .stApp p, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #ffffff;
    }
    /* Inside white cards, revert text to dark for readability */
    .css-card p,
    .css-card label,
    .css-card span,
    .css-card h3,
    .css-card h4,
    .css-card h5,
    .css-card h6 {
        color: #0f172a !important;
    }
    
/* Main Title Pill */
.title-pill {
    background-color: #ffffff;
    color: #0f172a;
    padding: 16px;
    border-radius: 12px;
    text-align: center;
    font-weight: 800;
    font-size: 1.2rem;
    letter-spacing: 1px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.18);
    margin-bottom: 20px;
}

    /* Cards (White Containers) */
    .css-card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }

    /* Probability Big Number */
    .prob-display {
        text-align: center;
    }
    .prob-number {
        font-size: 5rem;
        font-weight: 800;
        color: #1e293b;
        line-height: 1;
        margin-bottom: 10px;
    }
    .prob-label {
        font-size: 1.1rem;
        color: #0f172a; 
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Metric Mini-Cards */
    .metric-container {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 15px 10px;
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
        border: 1px solid #e2e8f0;
    }
    .metric-container:hover {
        transform: translateY(-2px);
        border-color: #cbd5e1;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: 800;
        color: #0f172a;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #0f172a;
        margin-top: 4px;
        font-weight: 700;
    }

    /* Input Section Header */
    .section-header {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 15px;
        border-left: 5px solid #3b82f6;
        padding-left: 10px;
    }

    /* History Section */
    .history-card {
        background-color: #ffffff; 
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 20px;
        margin-top: 30px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.12);
    }
    .history-title {
        color: #be185d;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 10px;
    }

    /* Custom Button */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        padding: 0.6rem 1rem;
        transition: background-color 0.2s;
    }
    .stButton > button:hover {
        background-color: #2563eb;
    }

    /* Fix Streamlit input labels contrast on dark bg if needed, 
       though they usually sit on white cards here. */
    div[data-testid="stMarkdownContainer"] > p {
         font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# SHARED CONSTANTS & MAPPINGS
# ---------------------------------------------------------------------
FEATURE_COLUMNS: List[str] = [
    "monthly_income", "travel_mode", "route_safe", "read_write",
    "solve_math", "school_facilities", "min_distance", "max_distance",
    "min_time", "max_time",
]

FACILITY_LABELS_ORDERED = [
    "Functional toilet for girls and boys",
    "Separate toilets for girls",
    "Drinking water",
    "Handwashing facilities",
    "Boundary wall",
    "Gate remains locked during school hours",
    "Security guard or watchman",
    "Garden or open play area",
    "Boys and girls attend classes together",
    "Boys and girls attend classes separately",
    "Enough classrooms (not overcrowded)",
]
FACILITY_KEYS_ORDERED = [f"school_facilities_{i}" for i in range(1, 12)]
FACILITY_LABEL_TO_KEY = dict(zip(FACILITY_LABELS_ORDERED, FACILITY_KEYS_ORDERED))
FACILITY_KEY_TO_LABEL = dict(zip(FACILITY_KEYS_ORDERED, FACILITY_LABELS_ORDERED))

# Internal Logic Maps (Value -> Label)
TRAVEL_MODE_MAP = {
    0: "On foot",
    1: "Bicycle",
    2: "Motorcycle",
    3: "Van/rickshaw",
    4: "Public transport",
}

# Reverse Maps for UI (Label -> Value)
TRAVEL_MODE_UI = {v: k for k, v in TRAVEL_MODE_MAP.items()}
YES_NO_UI = {"Yes": 1, "No": 0}
# FIXED: Changed to Male/Female
GENDER_UI = {"Male": 1, "Female": 2}

INCOME_STATS = {
    "urban": {"mean": 48412.3, "std": 13308.6, "raw_min": 2500.0, "raw_max": 120000.0},
    "rural": {"mean": 42790.0, "std": 18465.2, "raw_min": 7500.0, "raw_max": 167500.0},
}

# ---------------------------------------------------------------------
# LOADERS (UNCHANGED)
# ---------------------------------------------------------------------
@st.cache_resource
def load_phase1_state(preset: str):
    path = f"w_p1/{preset}.pt"
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except:
        ckpt = torch.load(path, map_location="cpu", pickle_module=torch.serialization.pickle, weights_only=False)
        return ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt

@st.cache_resource
def load_autoencoder_model(preset: str):
    model = Autoencoder(input_dim=5, latent_dim=3)
    state = torch.load(f"w_ae/autoencoder_{preset}.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

@st.cache_resource
def load_phase2_model(preset: str, input_dim: int):
    return load_model(f"w_p2/{preset}.joblib", input_dim)

# ---------------------------------------------------------------------
# INFERENCE PIPELINE (UNCHANGED)
# ---------------------------------------------------------------------
def run_phase1(features_df: pd.DataFrame, preset: str) -> pd.DataFrame:
    state = load_phase1_state(preset)
    X = features_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    tensor_x = torch.tensor(X, dtype=torch.float32)
    preds: Dict[str, np.ndarray] = {}
    model_map = get_p1_models(preset)
    for out_id, model_cls in model_map.items():
        model = model_cls(tensor_x.shape[1])
        try:
            model.load_state_dict(state[out_id])
        except:
            model.load_state_dict(state[out_id], strict=False)
        model.eval()
        with torch.no_grad():
            preds[f"pred_{out_id}"] = model(tensor_x).argmax(dim=1).numpy()
    return pd.DataFrame(preds)

def run_autoencoder(p1_preds: pd.DataFrame, preset: str) -> pd.DataFrame:
    model = load_autoencoder_model(preset)
    z = encode_with_autoencoder(model, p1_preds.to_numpy(dtype=np.float32), latent_dim=3)
    return pd.DataFrame(z, columns=["z1", "z2", "z3"])

def _phase2_predict(feature_df: pd.DataFrame, preset: str):
    model_type, model = load_phase2_model(preset, feature_df.shape[1])

    if model_type == "sklearn_dict":
        ordered = feature_df[model["feature_names"]].to_numpy(dtype=np.float32)
        mean, std = np.asarray(model["mean"], dtype=np.float32), np.asarray(model["std"], dtype=np.float32)
        std[std == 0] = 1.0
        normed = (ordered - mean) / std
        mdl = model["model"]
        if hasattr(mdl, "predict_proba"):
            pred = mdl.predict_proba(normed)[:, 1]
        else:
            pred = mdl.predict(normed)
        meta = {"model_type": model_type, "feature_names": model.get("feature_names"), "ordered": ordered, "normed": normed}
    elif model_type == "sklearn":
        data = feature_df.to_numpy(dtype=np.float32)
        if hasattr(model, "predict_proba"):
            pred = model.predict_proba(data)[:, 1]
        else:
            pred = model.predict(data)
        meta = {"model_type": model_type, "feature_names": list(feature_df.columns), "ordered": data}
    else:
        data = feature_df.to_numpy(dtype=np.float32)
        with torch.no_grad():
            pred = model(torch.tensor(data)).numpy()
        meta = {"model_type": model_type, "feature_names": list(feature_df.columns), "ordered": data}
    return pred, meta


def run_phase2(features_df: pd.DataFrame, z_df: pd.DataFrame, preset: str) -> float:
    merged = pd.concat([features_df, z_df], axis=1)
    feature_df = merged.drop(columns=["hh_id"]).astype(np.float32)
    pred, _ = _phase2_predict(feature_df, preset)
    prob = float(pred.flatten()[0])
    return prob

def run_full_inference(inputs: Dict[str, float], preset: str) -> float:
    features_df = pd.DataFrame([inputs], columns=["hh_id"] + FEATURE_COLUMNS)
    p1_preds = run_phase1(features_df, preset)
    z_df = run_autoencoder(p1_preds, preset)
    raw_prob = run_phase2(features_df, z_df, preset)
    clipped = float(np.clip(raw_prob, 0.0, 1.0))
    return clipped


def run_full_inference_debug(inputs: Dict[str, float], preset: str):
    """Return intermediate outputs for debugging."""
    features_df = pd.DataFrame([inputs], columns=["hh_id"] + FEATURE_COLUMNS)
    p1_preds = run_phase1(features_df, preset)
    z_df = run_autoencoder(p1_preds, preset)
    merged = pd.concat([features_df, z_df], axis=1)
    feature_df = merged.drop(columns=["hh_id"]).astype(np.float32)
    pred, meta = _phase2_predict(feature_df, preset)
    raw_prob = float(pred.flatten()[0])
    clipped = float(np.clip(raw_prob, 0.0, 1.0))
    return features_df, p1_preds, z_df, raw_prob, clipped, meta

# ---------------------------------------------------------------------
# LOGIC & HELPERS
# ---------------------------------------------------------------------
@dataclass
class HistoryItem:
    timestamp: float
    preset: str
    prob: float
    features: Dict[str, float]

def child_defaults() -> Dict:
    return {"gender": 1, "travel_mode": 0, "route_safe": 1, "distance": 1.0, "time": 10.0, "facilities": []}

def compute_school_facility_score(children: List[Dict]) -> float:
    if not children: return 0.0
    weights = dict(FACILITY_WEIGHTS)
    # Ensure facility 2 weight is accounted for manually based on gender
    if "school_facilities_2" in weights: weights["school_facilities_2"] = 0
    
    base_max_score = sum(weights.values())
    scores = []
    
    for child in children:
        selected_facilities = set(child.get("facilities", []))
        gender = child.get("gender", 1)
        
        # Sum weights of selected facilities (excluding fac_2)
        current_base_score = sum(weights.get(f, 0) for f in selected_facilities)
        
        # Logic for facility 2 (Separate toilets for girls)
        # It only adds to the score if the child is female (gender 2) AND it is selected.
        facility2_weight_if_applicable = 5 if gender == 2 else 0
        score_from_facility2 = facility2_weight_if_applicable if "school_facilities_2" in selected_facilities else 0
        
        total_raw_score = current_base_score + score_from_facility2
        total_max_possible = base_max_score + facility2_weight_if_applicable

        # Normalize to 0-5 range
        norm_score = (total_raw_score / total_max_possible) * 5 if total_max_possible > 0 else 0
        scores.append(float(np.clip(norm_score, 0, 5)))
        
    return round(float(np.mean(scores)), 1)

def aggregate_children(children: List[Dict]) -> Dict:
    if not children:
        return {"travel_mode": 0, "route_safe": 1, "min_distance": 0.0, "max_distance": 0.0, 
                "min_time": 0.0, "max_time": 0.0, "school_facilities": 0.0}
    
    t_modes = [c.get("travel_mode", 0) for c in children]
    travel_mode = Counter(t_modes).most_common(1)[0][0] if t_modes else 0
    
    route_safes = [c.get("route_safe", 1) for c in children]
    route_safe = min(route_safes) if route_safes else 1
    
    dists = [c.get("distance", 0.0) for c in children]
    times = [c.get("time", 0.0) for c in children]
    
    return {
        "travel_mode": travel_mode,
        "route_safe": route_safe,
        "min_distance": float(np.min(dists)) if dists else 0.0,
        "max_distance": float(np.max(dists)) if dists else 0.0,
        "min_time": float(np.min(times)) if times else 0.0,
        "max_time": float(np.max(times)) if times else 0.0,
        "school_facilities": compute_school_facility_score(children)
    }

def plot_history(history: List[HistoryItem]):
    if not history:
        st.info("Run inference to see history.")
        return
    df = pd.DataFrame({"run": range(1, len(history) + 1), "prob": [h.prob for h in history]})
    fig = go.Figure(go.Scatter(x=df["run"], y=df["prob"], mode="lines+markers", marker=dict(size=10, color='#be185d'), line=dict(color='#be185d', width=3)))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(showgrid=False, title="Run ID"), yaxis=dict(showgrid=True, gridcolor='#e5e7eb', title="Probability"))
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------
def main():
    # --- State Initialization ---
    if "children" not in st.session_state: st.session_state.children = [child_defaults()]
    if "prob_history" not in st.session_state: st.session_state.prob_history = []
    if "last_prob" not in st.session_state: st.session_state.last_prob = 0.5
    if "debug_info" not in st.session_state: st.session_state.debug_info = None

    # --- Header ---
    st.markdown('<div class="title-pill">AGENT BASED MODELING FOR OUT OF SCHOOL CHILDREN</div>', unsafe_allow_html=True)

    # --- Region Selector (no expander to avoid overlap) ---
    st.markdown('<div class="section-header">Configuration & Region</div>', unsafe_allow_html=True)
    preset = st.radio("Region", ["rural", "urban"], index=0, horizontal=True, key="region_radio")
    stats = INCOME_STATS[preset]

    # -----------------------------------------------------------------
    # LAYOUT PLACEHOLDERS
    # -----------------------------------------------------------------
    prob_container = st.container()
    metrics_container = st.container()
    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # INPUTS (Bottom Section)
    # -----------------------------------------------------------------
    
    # --- Household Section ---
    st.markdown('<div class="section-header">Household Details</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        col_hh_1, col_hh_2, col_hh_3 = st.columns([2, 1, 1])
        with col_hh_1:
            monthly_income_raw = st.slider(
                "Monthly Income (PKR)",
                0.0,
                200_000.0,
                float(stats["mean"]),
                step=500.0,
            )
            monthly_income_norm = (monthly_income_raw - stats["mean"]) / stats["std"]
            monthly_income_for_model = monthly_income_norm
            display_income = f"{monthly_income_raw:,.0f} PKR"
        with col_hh_2:
            rw_str = st.radio("Head can Read/Write?", ["Yes", "No"], horizontal=True)
        with col_hh_3:
            math_str = st.radio("Head can Solve Math?", ["Yes", "No"], horizontal=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Children Section ---
    st.markdown('<div class="section-header">Children Details</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        
        b_col1, b_col2 = st.columns([1, 5])
        with b_col1:
            if st.button("Add Child +"): st.session_state.children.append(child_defaults())
        with b_col2:
            if st.button("Reset List"): st.session_state.children = [child_defaults()]

        updated_children = []
        for idx, c in enumerate(st.session_state.children):
            st.markdown(f"#### Child {idx+1}")
            c1, c2, c3, c4, c5 = st.columns([1, 2, 1, 1, 1])
            
            with c1:
                g_idx = 0 if c["gender"] == 1 else 1
                g_str = st.selectbox("Gender", ["Male", "Female"], index=g_idx, key=f"g{idx}")
                gender_val = GENDER_UI[g_str]
            with c2:
                t_str = st.selectbox("Travel Mode", list(TRAVEL_MODE_UI.keys()), index=c["travel_mode"], key=f"t{idx}")
                travel_val = TRAVEL_MODE_UI[t_str]
            with c3:
                r_idx = 0 if c["route_safe"] == 0 else 1
                r_str = st.selectbox("Route Safe?", ["No", "Yes"], index=r_idx, key=f"r{idx}")
                route_val = YES_NO_UI[r_str]
            with c4:
                dist_val = st.number_input("Distance to school (km)", 0.0, 200.0, float(c.get("distance", 1.0)), step=0.5, key=f"d{idx}")
            with c5:
                time_val = st.number_input("Travel time (minutes)", 0.0, 240.0, float(c.get("time", 10.0)), step=1.0, key=f"tm{idx}")
            
            # FIXED: Added Facilities Multiselect
            # Convert any stored keys back to labels for display; unknowns are dropped
            existing_labels = [
                FACILITY_KEY_TO_LABEL.get(f, f) for f in c.get("facilities", [])
                if FACILITY_KEY_TO_LABEL.get(f, f) in FACILITY_LABELS_ORDERED
            ]
            fac_selected = st.multiselect(
                "School Facilities Available at Destination", 
                FACILITY_LABELS_ORDERED, 
                default=existing_labels,
                key=f"fac{idx}"
            )

            st.divider()
            
            updated_children.append({
                "gender": gender_val,
                "travel_mode": travel_val,
                "route_safe": route_val,
                "distance": dist_val,
                "time": time_val,
                # store model-facing facility keys
                "facilities": [FACILITY_LABEL_TO_KEY.get(lbl, lbl) for lbl in fac_selected]
            })

        st.session_state.children = updated_children
        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # LOGIC & DATA PREP
    # -----------------------------------------------------------------
    agg = aggregate_children(updated_children)
    
    read_write_val = YES_NO_UI[rw_str]
    solve_math_val = YES_NO_UI[math_str]
    model_inputs = {
        "hh_id": 1,
        "monthly_income": monthly_income_for_model,
        "travel_mode": float(agg['travel_mode']),
        "route_safe": float(agg['route_safe']),
        "read_write": float(read_write_val),
        "solve_math": float(solve_math_val),
        "school_facilities": float(agg['school_facilities']),
        "min_distance": agg['min_distance'],
        "max_distance": agg['max_distance'],
        "min_time": agg['min_time'],
        "max_time": agg['max_time'],
    }

    # -----------------------------------------------------------------
    # POPULATE TOP VISUALS
    # -----------------------------------------------------------------
    with prob_container:
        st.markdown(
            f"""
            <div class="css-card prob-display">
                <div class="prob-number">{st.session_state.last_prob:.0%}</div>
                <div class="prob-label">Household Enrollment Probability</div>
            </div>
            """, 
            unsafe_allow_html=True
        )

    # Aggregated Metrics Display
    disp_income = display_income
    disp_safe = "Safe" if agg['route_safe'] == 1 else "Unsafe"
    disp_travel = TRAVEL_MODE_MAP.get(agg['travel_mode'], "Unknown")
    disp_score = f"{agg['school_facilities']:.1f} / 5.0"
    disp_time = f"{agg['min_time']:.0f} - {agg['max_time']:.0f}"
    disp_dist = f"{agg['min_distance']:.1f} - {agg['max_distance']:.1f}"

    with metrics_container:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        def render_metric(col, val, label):
            col.markdown(f"""<div class="metric-container"><div class="metric-value">{val}</div><div class="metric-title">{label}</div></div>""", unsafe_allow_html=True)

        render_metric(m1, disp_income, "Monthly Income")
        render_metric(m2, disp_safe, "Route Safety")
        render_metric(m3, disp_travel, "Pref. Travel Mode")
        render_metric(m4, disp_score, "Facility Score")
        render_metric(m5, disp_time, "Time (min)")
        render_metric(m6, disp_dist, "Distance (km)")

    # -----------------------------------------------------------------
    # ACTION & HISTORY
    # -----------------------------------------------------------------
    st.write("")
    show_debug = st.checkbox("Show inference details", value=False)

    if st.button("RUN INFERENCE MODEL", type="primary", width="stretch"):
        if show_debug:
            features_df, p1_preds, z_df, raw_prob, clipped_prob, meta = run_full_inference_debug(model_inputs, preset)
            st.session_state.debug_info = {
                "features": features_df,
                "p1_preds": p1_preds,
                "ae_latent": z_df,
                "raw_prob": raw_prob,
                "clipped_prob": clipped_prob,
                "meta": meta,
                "model_inputs": model_inputs,
                "monthly_income_for_model": monthly_income_for_model,
                "monthly_income_raw": monthly_income_raw,
            }
            new_prob = clipped_prob
            # Persist latest prob/history without forcing a rerun so debug stays visible
            st.session_state.last_prob = new_prob
            st.session_state.prob_history.append(HistoryItem(time.time(), preset, new_prob, model_inputs))
            st.success(f"Phase 2 raw prob: {raw_prob:.4f} (clipped: {clipped_prob:.4f})")
        else:
            new_prob = run_full_inference(model_inputs, preset)
            st.session_state.last_prob = new_prob
            st.session_state.prob_history.append(HistoryItem(time.time(), preset, new_prob, model_inputs))
            st.rerun()

    if show_debug and st.session_state.debug_info:
        dbg = st.session_state.debug_info
        st.markdown("#### Debug details")
        st.write("Model inputs", dbg["features"])
        st.write("Phase 1 predictions", dbg["p1_preds"])
        st.write("Autoencoder latent", dbg["ae_latent"])
        st.write(f"Phase 2 raw prob: {dbg['raw_prob']:.4f} (clipped: {dbg['clipped_prob']:.4f})")
        st.write("Phase 2 meta", dbg.get("meta"))
        st.write("Income sent to model", {"raw": dbg.get("monthly_income_raw"), "z_norm": dbg.get("monthly_income_for_model")})

    st.markdown('<div class="history-card">', unsafe_allow_html=True)
    st.markdown('<div class="history-title">Probability History</div>', unsafe_allow_html=True)
    plot_history(st.session_state.prob_history)
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
