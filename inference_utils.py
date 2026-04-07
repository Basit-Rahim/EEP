"""
Shared inference functions, model loaders, and constants.
Extracted from streamlit_app.py for reuse across multiple simulation pages.
"""
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import torch

from inference_autoencoder import Autoencoder, encode_with_autoencoder
from inference_phase1 import get_p1_models
from inference_phase2 import load_model
from preprocess import FACILITY_WEIGHTS

# ---------------------------------------------------------------------
# SHARED CSS
# ---------------------------------------------------------------------
APP_CSS = """
<style>
/* Global Background - Deep Professional Blue */
.stApp {
    background-color: #1e293b;
}

/* Typography */
h1, h2, h3, p, div, label, span, button {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
}
.stApp, .stApp p, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: #ffffff;
}
.css-card p, .css-card label, .css-card span, .css-card h3, .css-card h4, .css-card h5, .css-card h6 {
    color: #0f172a !important;
}

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

.css-card {
    background-color: #ffffff;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    margin-bottom: 20px;
}

.prob-display { text-align: center; }
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

.section-header {
    color: #ffffff;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 15px;
    border-left: 5px solid #3b82f6;
    padding-left: 10px;
}

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

div[data-testid="stMarkdownContainer"] > p {
    font-weight: 500;
}
</style>
"""

# ---------------------------------------------------------------------
# CONSTANTS & MAPPINGS
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

TRAVEL_MODE_MAP = {
    0: "On foot",
    1: "Bicycle",
    2: "Motorcycle",
    3: "Van/rickshaw",
    4: "Public transport",
}
TRAVEL_MODE_UI = {v: k for k, v in TRAVEL_MODE_MAP.items()}
YES_NO_UI = {"Yes": 1, "No": 0}
GENDER_UI = {"Male": 1, "Female": 2}

INCOME_STATS = {
    "urban": {"mean": 48412.3, "std": 13308.6, "raw_min": 2500.0, "raw_max": 120000.0},
    "rural": {"mean": 42790.0, "std": 18465.2, "raw_min": 7500.0, "raw_max": 167500.0},
}

# Distance/time categories — replaces raw km sliders
DISTANCE_CATEGORIES = {
    "Near (0–2 km)": {
        "min_distance": 0.5, "max_distance": 1.5,
        "min_time": 5.0, "max_time": 15.0,
    },
    "Moderate (2–6 km)": {
        "min_distance": 2.5, "max_distance": 5.5,
        "min_time": 20.0, "max_time": 45.0,
    },
    "Far (6–15 km)": {
        "min_distance": 7.0, "max_distance": 12.0,
        "min_time": 50.0, "max_time": 90.0,
    },
    "Very Far (15+ km)": {
        "min_distance": 17.0, "max_distance": 25.0,
        "min_time": 100.0, "max_time": 180.0,
    },
}
DISTANCE_CAT_LIST = list(DISTANCE_CATEGORIES.keys())


# ---------------------------------------------------------------------
# MODEL LOADERS
# ---------------------------------------------------------------------
@st.cache_resource
def load_phase1_state(preset: str):
    path = f"w_p1/{preset}.pt"
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(
            path, map_location="cpu",
            pickle_module=torch.serialization.pickle, weights_only=False
        )
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
# INFERENCE PIPELINE
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
        except Exception:
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
        mean = np.asarray(model["mean"], dtype=np.float32)
        std = np.asarray(model["std"], dtype=np.float32)
        std[std == 0] = 1.0
        normed = (ordered - mean) / std
        mdl = model["model"]
        if hasattr(mdl, "predict_proba"):
            pred = mdl.predict_proba(normed)[:, 1]
        else:
            pred = mdl.predict(normed)
        meta = {
            "model_type": model_type,
            "feature_names": model.get("feature_names"),
            "ordered": ordered,
            "normed": normed,
        }
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
    if "route_safe" in features_df.columns:
        rs = pd.to_numeric(features_df["route_safe"], errors="coerce").fillna(1).astype(int)
        if int(rs.iloc[0]) == 0:
            prob *= 0.8
    return prob


def run_full_inference(inputs: Dict[str, float], preset: str) -> float:
    features_df = pd.DataFrame([inputs], columns=["hh_id"] + FEATURE_COLUMNS)
    p1_preds = run_phase1(features_df, preset)
    z_df = run_autoencoder(p1_preds, preset)
    raw_prob = run_phase2(features_df, z_df, preset)
    return float(np.clip(raw_prob, 0.0, 1.0))


def run_full_inference_debug(inputs: Dict[str, float], preset: str):
    features_df = pd.DataFrame([inputs], columns=["hh_id"] + FEATURE_COLUMNS)
    p1_preds = run_phase1(features_df, preset)
    z_df = run_autoencoder(p1_preds, preset)
    merged = pd.concat([features_df, z_df], axis=1)
    feature_df = merged.drop(columns=["hh_id"]).astype(np.float32)
    pred, meta = _phase2_predict(feature_df, preset)
    raw_prob = float(pred.flatten()[0])
    if "route_safe" in features_df.columns:
        rs = pd.to_numeric(features_df["route_safe"], errors="coerce").fillna(1).astype(int)
        if int(rs.iloc[0]) == 0:
            raw_prob *= 0.8
    clipped = float(np.clip(raw_prob, 0.0, 1.0))
    return features_df, p1_preds, z_df, raw_prob, clipped, meta


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
@dataclass
class HistoryItem:
    timestamp: float
    preset: str
    prob: float
    features: Dict[str, float]


def child_defaults() -> Dict:
    return {
        "gender": 1, "travel_mode": 0, "route_safe": 1,
        "distance": 0.5, "time": 2.0, "facilities": [],
    }


def compute_school_facility_score(children: List[Dict]) -> float:
    if not children:
        return 0.0
    weights = dict(FACILITY_WEIGHTS)
    if "school_facilities_2" in weights:
        weights["school_facilities_2"] = 0
    base_max_score = sum(weights.values())
    scores = []
    for child in children:
        selected_facilities = set(child.get("facilities", []))
        gender = child.get("gender", 1)
        current_base_score = sum(weights.get(f, 0) for f in selected_facilities)
        facility2_weight_if_applicable = 5 if gender == 2 else 0
        score_from_facility2 = (
            facility2_weight_if_applicable if "school_facilities_2" in selected_facilities else 0
        )
        total_raw_score = current_base_score + score_from_facility2
        total_max_possible = base_max_score + facility2_weight_if_applicable
        norm_score = (total_raw_score / total_max_possible) * 5 if total_max_possible > 0 else 0
        scores.append(float(np.clip(norm_score, 0, 5)))
    return round(float(np.mean(scores)), 1)


def aggregate_children(children: List[Dict]) -> Dict:
    if not children:
        return {
            "travel_mode": 0, "route_safe": 1,
            "min_distance": 0.5, "max_distance": 0.5,
            "min_time": 2.0, "max_time": 2.0,
            "school_facilities": 0.0,
        }
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
        "school_facilities": compute_school_facility_score(children),
    }
