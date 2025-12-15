#!/usr/bin/env python3
# ======================================================================
# SYNTHETIC HOUSEHOLD GENERATOR – PHASE 2 (URBAN)
# Fully corrected version that properly merges:
#   - original X data
#   - autoencoder (concept) data
#   - synthetic X
#   - synthetic autoencoder
# And outputs correct p2x_urban.csv and p2y_urban.csv
# ======================================================================

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# GMM support
try:
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    GaussianMixture = None

# COLOR OUTPUT
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"

def color(txt, c, enable=True):
    return f"{c}{txt}{RESET}" if enable else txt

def bold(txt, enable=True):
    return f"{BOLD}{txt}{RESET}" if enable else txt

# DATA CLASSES
@dataclass
class SyntheticMeta:
    synthetic_hh_id: int
    base_hh_id: Any
    categories: List[str]
    modified_fields: Dict[str, Dict[str, Any]]

# GMM SAMPLING HELPERS
def fit_gmm(values: np.ndarray, components: int, seed: int):
    values = values[~np.isnan(values)]
    if len(values) == 0 or not SKLEARN_AVAILABLE:
        return None

    gmm = GaussianMixture(
        n_components=components,
        random_state=seed,
        covariance_type="full",
        max_iter=300
    )
    gmm.fit(values.reshape(-1, 1))
    return gmm

def sample_gmm_component(gmm, comp_idx, n, rng):
    mean = gmm.means_[comp_idx][0]
    var = gmm.covariances_[comp_idx][0][0]
    std = max(np.sqrt(var), 1e-6)
    return rng.normal(mean, std, size=n)

def sample_tail(values, tail, n, use_gmm, comps, rng):
    """Lower or upper tail sampling with optional spikes"""
    values = np.array(values)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.zeros(n)

    if use_gmm and SKLEARN_AVAILABLE:
        gmm = fit_gmm(values, comps, seed=rng.randint(99999))
    else:
        gmm = None

    if gmm is not None:
        means = gmm.means_.flatten()
        comp = int(np.argmin(means)) if tail == "lower" else int(np.argmax(means))
        s = sample_gmm_component(gmm, comp, n, rng)
    else:
        mu, sd = values.mean(), values.std()
        sd = sd if sd > 0 else 1.0
        s = rng.normal(mu, sd, n)

    # enforce tail direction
    if tail == "lower":
        max_val = values.mean()
        s = np.where(s <= max_val, s, rng.choice(values[values <= max_val], n))
    else:
        min_val = values.mean()
        s = np.where(s >= min_val, s, rng.choice(values[values >= min_val], n))
        # add spikes
        idx = rng.choice(n, size=max(1, n // 5))
        s[idx] *= rng.uniform(1.3, 1.8, size=len(idx))

    return s

# CATEGORY ASSIGNMENT
def assign_categories(n_total, n_income_tail, n_route_unsafe, n_distance_tail, rng):
    labels = (["income_tail"] * n_income_tail +
              ["route_unsafe"] * n_route_unsafe +
              ["distance_tail"] * n_distance_tail)

    if len(labels) < n_total:
        labels += ["neutral"] * (n_total - len(labels))

    rng.shuffle(labels)

    categories = []
    for lab in labels:
        if lab == "neutral":
            # maybe mixed
            picks = []
            if rng.rand() < 0.3: picks.append("income_tail")
            if rng.rand() < 0.3: picks.append("distance_tail")
            if rng.rand() < 0.2: picks.append("route_unsafe")
            categories.append(picks)
        else:
            # one or two categories
            picks = [lab]
            if rng.rand() < 0.25:
                other = rng.choice(["income_tail", "distance_tail", "route_unsafe"])
                if other not in picks:
                    picks.append(other)
            categories.append(picks)

    # enforce route_unsafe count
    current = sum("route_unsafe" in c for c in categories)
    deficit = n_route_unsafe - current
    idx = 0
    while deficit > 0 and idx < n_total:
        if "route_unsafe" not in categories[idx]:
            categories[idx].append("route_unsafe")
            deficit -= 1
        idx += 1

    return categories

# SYNTHETIC GENERATOR
def generate_synthetic(
    x, ae, y,
    n_total=200,
    n_income_tail=40,
    n_route_unsafe=40,
    n_distance_tail=40,
    start_id=9001,
    use_gmm=True,
    gmm_components=2,
    seed=42,
):
    rng = np.random.RandomState(seed)

    income_vals = x["monthly_income"].astype(float).values
    dist_vals   = x["min_distance"].astype(float).values
    time_vals   = x["min_time"].astype(float).values

    # thresholds
    low_income_thr = np.quantile(income_vals, 0.3)
    high_dist_thr  = np.quantile(dist_vals, 0.7)
    high_time_thr  = np.quantile(time_vals, 0.7)

    categories = assign_categories(
        n_total, n_income_tail, n_route_unsafe, n_distance_tail, rng
    )

    # pre-sampling for tails
    income_tail_s  = sample_tail(income_vals, "lower", sum("income_tail" in c for c in categories),
                                 use_gmm=use_gmm, comps=gmm_components, rng=rng)
    dist_tail_s    = sample_tail(dist_vals,   "upper", sum("distance_tail" in c for c in categories),
                                 use_gmm=use_gmm, comps=gmm_components, rng=rng)
    time_tail_s    = sample_tail(time_vals,   "upper", sum("distance_tail" in c for c in categories),
                                 use_gmm=use_gmm, comps=gmm_components, rng=rng)

    idx_income = 0
    idx_dist   = 0

    synthetic_x_rows = []
    synthetic_ae_rows = []
    synthetic_y_rows = []
    meta_list = []

    for i in range(n_total):
        cats = categories[i]

        # pick a real household as base
        base_row = x.sample(1, random_state=rng).iloc[0].copy()
        base_id = base_row["hh_id"]

        base_y = y.loc[y["hh_id"] == base_id]
        if base_y.empty:
            base_y = pd.Series({"hh_id": base_id, "prob": 1.0})
        else:
            base_y = base_y.iloc[0].copy()

        base_ae = ae.loc[ae["hh_id"] == base_id]
        if base_ae.empty:
            base_ae = ae.sample(1, random_state=rng)
        base_ae = base_ae.iloc[0].copy()

        new_id = start_id + i
        mods = {}

        # income
        if "income_tail" in cats:
            new_inc = income_tail_s[idx_income]
            mods["monthly_income"] = {"old": float(base_row["monthly_income"]), "new": float(new_inc)}
            base_row["monthly_income"] = max(new_inc, 0.0)
            idx_income += 1

        # route safety
        if "route_unsafe" in cats:
            mods["route_safe"] = {"old": int(base_row["route_safe"]), "new": 0}
            base_row["route_safe"] = 0

        # distance + time
        if "distance_tail" in cats:
            new_d = dist_tail_s[idx_dist]
            new_t = time_tail_s[idx_dist]
            idx_dist += 1

            mods["min_distance"] = {"old": float(base_row["min_distance"]), "new": float(new_d)}
            mods["min_time"]     = {"old": float(base_row["min_time"]),     "new": float(new_t)}

            base_row["min_distance"] = max(new_d, 0.0)
            base_row["min_time"]     = max(new_t, 0.0)

            # adjust max_distance/time
            if "max_distance" in base_row:
                if pd.notna(base_row["max_distance"]) and base_row["max_distance"] < base_row["min_distance"]:
                    mods["max_distance"] = {"old": float(base_row["max_distance"]), "new": float(base_row["min_distance"])}
                    base_row["max_distance"] = base_row["min_distance"]

            if "max_time" in base_row:
                if pd.notna(base_row["max_time"]) and base_row["max_time"] < base_row["min_time"]:
                    mods["max_time"] = {"old": float(base_row["max_time"]), "new": float(base_row["min_time"])}
                    base_row["max_time"] = base_row["min_time"]

        # assign new id
        base_row["hh_id"] = new_id
        base_ae["hh_id"]  = new_id

        # probability
        prob = float(rng.beta(0.5, 5.0))

        if "income_tail" in cats and base_row["monthly_income"] <= low_income_thr:
            prob *= 0.5

        if "route_unsafe" in cats:
            prob *= 0.5

        if "distance_tail" in cats:
            if base_row["min_distance"] >= high_dist_thr or base_row["min_time"] >= high_time_thr:
                prob *= 0.7

        prob = 0.0 if prob < 0.02 else min(1.0, prob)

        new_y = base_y.copy()
        new_y["hh_id"] = new_id
        new_y["prob"] = prob

        synthetic_x_rows.append(base_row)
        synthetic_ae_rows.append(base_ae)
        synthetic_y_rows.append(new_y)
        meta_list.append(SyntheticMeta(
            synthetic_hh_id=int(new_id),
            base_hh_id=base_id,
            categories=cats,
            modified_fields=mods
        ))

    return (
        pd.DataFrame(synthetic_x_rows),
        pd.DataFrame(synthetic_ae_rows),
        pd.DataFrame(synthetic_y_rows),
        meta_list
    )

# PLOTTING
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def plot_hist(df, cols, title, out_png, out_pdf):
    df = df.copy()
    fig, axes = plt.subplots(len(cols), 1, figsize=(7, len(cols)*2.5))
    if len(cols) == 1: axes = [axes]
    for ax, col in zip(axes, cols):
        if col not in df.columns:
            continue
        data = df[col].dropna()
        ax.hist(data, bins=40)
        ax.set_title(col)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)

# SUMMARY + JSON SAVE
def save_meta(meta_list, txt_path, json_path):
    with open(txt_path, "w") as f:
        for m in meta_list:
            f.write(f"{m.synthetic_hh_id},{m.base_hh_id},{m.categories},{m.modified_fields}\n")

    with open(json_path, "w") as f:
        json.dump([asdict(m) for m in meta_list], f, indent=2)

# CLI + MAIN
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--x-path",  default="x_urban.csv")
    p.add_argument("--ae-path", default="urban_autoencoder.csv")
    p.add_argument("--y-path",  default="y_urban.csv")

    p.add_argument("--output-root", default="synthetic")

    p.add_argument("--use-gmm", action="store_true")
    p.add_argument("--gmm-components", type=int, default=2)

    p.add_argument("--n-total", type=int, default=200)
    p.add_argument("--n-income-tail", type=int, default=40)
    p.add_argument("--n-route-unsafe", type=int, default=40)
    p.add_argument("--n-distance-tail", type=int, default=40)

    p.add_argument("--start-id", type=int, default=9001)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--no-color", action="store_true")

    return p.parse_args()

def main():
    args = parse_args()
    color_on = not args.no_color

    print(bold("\n=== SYNTHETIC DATA GENERATOR (PHASE 2, URBAN) ===\n", color_on))

    # -------------------------------
    # LOAD ORIGINAL DATA
    # -------------------------------
    x = pd.read_csv(args.x_path)
    ae = pd.read_csv(args.ae_path)
    y  = pd.read_csv(args.y_path)

    print("Loaded:")
    print(f"  X rows:  {len(x)}")
    print(f"  AE rows: {len(ae)}")
    print(f"  Y rows:  {len(y)}")

    # -------------------------------
    # GENERATE SYNTHETIC DATA
    # -------------------------------
    print(bold("\nGenerating synthetic households...", color_on))

    syn_x, syn_ae, syn_y, meta_list = generate_synthetic(
        x, ae, y,
        n_total=args.n_total,
        n_income_tail=args.n_income_tail,
        n_route_unsafe=args.n_route_unsafe,
        n_distance_tail=args.n_distance_tail,
        start_id=args.start_id,
        use_gmm=args.use_gmm,
        gmm_components=args.gmm_components,
        seed=args.seed
    )

    print(color(f"Generated {len(syn_x)} synthetic households.", CYAN, color_on))

    # -------------------------------
    # MERGE ORIGINAL X + AE
    # -------------------------------
    print(bold("\nMerging original inputs...", color_on))
    original_inputs = x.merge(ae, on="hh_id", how="left")
    print(f"Original merged shape: {original_inputs.shape}")

    # -------------------------------
    # MERGE SYNTHETIC X + AE
    # -------------------------------
    print(bold("Merging synthetic inputs...", color_on))
    synthetic_inputs = syn_x.merge(syn_ae, on="hh_id", how="left")
    print(f"Synthetic merged shape: {synthetic_inputs.shape}")

    # -------------------------------
    # BUILD EXTENDED DATASETS
    # -------------------------------
    print(bold("\nBuilding extended datasets...", color_on))

    p2x = pd.concat([original_inputs, synthetic_inputs], ignore_index=True)

    p2y = pd.concat([y, syn_y], ignore_index=True)

    # OUTPUT DIRS
    out_inputs = os.path.join(args.output_root, "inputs")
    out_plots  = os.path.join(args.output_root, "plots")
    ensure_dir(out_inputs)
    ensure_dir(out_plots)

    p2x_path = os.path.join(out_inputs, "p2x_urban.csv")
    p2y_path = os.path.join(out_inputs, "p2y_urban.csv")

    p2x.to_csv(p2x_path, index=False)
    p2y.to_csv(p2y_path, index=False)

    print(color(f"Saved p2x: {p2x_path}", GREEN, color_on))
    print(color(f"Saved p2y: {p2y_path}", GREEN, color_on))

    # -------------------------------
    # PLOTS
    # -------------------------------
    cols = ["monthly_income", "min_distance", "min_time", "route_safe"]
    plot_hist(original_inputs, cols, "Original Inputs",
              os.path.join(out_plots, "orig_inputs.png"),
              os.path.join(out_plots, "orig_inputs.pdf"))

    plot_hist(p2x, cols, "Extended Inputs",
              os.path.join(out_plots, "extended_inputs.png"),
              os.path.join(out_plots, "extended_inputs.pdf"))

    plot_hist(p2y, ["prob"], "Extended Output Prob",
              os.path.join(out_plots, "extended_prob.png"),
              os.path.join(out_plots, "extended_prob.pdf"))

    print(color("\nPlots saved.", CYAN, color_on))

    # -------------------------------
    # SAVE META LOGS
    # -------------------------------
    txt_path  = os.path.join(args.output_root, "modified_households.txt")
    json_path = os.path.join(args.output_root, "modified_households.json")
    save_meta(meta_list, txt_path, json_path)

    print(color("Saved metadata logs.", CYAN, color_on))

    print(bold("\nDone.\n", color_on))


if __name__ == "__main__":
    main()

