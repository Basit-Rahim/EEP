# Agent Body Modeling – Inference Pipeline

This repo contains a two‑phase inference pipeline with an intermediate autoencoder. You provide a household feature CSV, the pipeline runs Phase 1 classifiers, compresses their outputs via the autoencoder, and finishes with a stacked ensemble / neural model for Phase 2.

## Requirements
- Python 3.10+
- Dependencies: `torch`, `pandas`, `numpy`, `joblib`, `scikit-learn`, `xgboost` (and their transitive deps).
- Weights on disk:
  - Phase 1: `w_p1/{rural,urban}.pt`
  - Autoencoder: `w_ae/autoencoder_{rural,urban}.pt`
  - Phase 2 stacked ensemble: `w_p2/{rural,urban}.joblib` (dict with `model`, `feature_names`, `mean`, `std`)

## Key Scripts
- `run_inference_pipeline.py`: orchestrates Phase 1 → Autoencoder → Phase 2.
- `inference_phase1.py`: runs the 5 Phase 1 classifiers and writes predictions.
- `inference_autoencoder.py`: encodes Phase 1 outputs into a 3‑D latent.
- `inference_phase2.py`: loads either a Torch model (`.pt`) or the stacked ensemble (`.joblib`) and outputs the final probability.
- `preprocess.py`: data engineering CLI to build cleaned/normalized urban/rural datasets.
- `phase1.py` / `phase1_{rural,urban}.py`: training scripts for Phase 1 models.
- `phase2.py` / `phase2_{rural,urban}.py`: training scripts for Phase 2 (includes stacked ensemble).
- `synthetic_{rural,urban}.py`: scripts to generate synthetic data variants.
- `pdf_inputs.py`: helper to parse PDF inputs (if needed).

## Input Format
Phase 1 expects a CSV with `hh_id` plus the feature columns used during training (e.g., `monthly_income, travel_mode, route_safe, read_write, solve_math, school_facilities, min_distance, max_distance, min_time, max_time`). A minimal single‑row example is provided at `test.csv`:
```
hh_id,monthly_income,travel_mode,route_safe,read_write,solve_math,school_facilities,min_distance,max_distance,min_time,max_time
1001,3.31,0,1,0,1,1,1.11,1.11,1.42,1.42
```

## Running Inference
Example commands (choose preset `rural` or `urban`):
```
python run_inference_pipeline.py --preset rural --xfile test.csv --output_dir infer_test_rural
python run_inference_pipeline.py --preset urban --xfile test.csv --output_dir infer_test_urban
```

Outputs (written to `--output_dir`):
- `p1_pred.csv` — Phase 1 class predictions per target.
- `ae_latent.csv` — Autoencoder latent features (`z1`, `z2`, `z3`).
- `final_p2_pred.csv` — Final Phase 2 probability per `hh_id`.

## Notes
- `inference_phase2.py` auto‑detects weight type: `.joblib/.pkl` → stacked ensemble with saved normalization; `.pt` → Torch `Phase2Net`.
- Ensure all weight files exist for the chosen preset; otherwise the pipeline will fail at the missing stage.

## Preprocessing
`preprocess.py` builds cleaned datasets from raw surveys:
```
python preprocess.py --help
```
Typical flow: build subset from `long_df.csv` + `completed_surveys.csv` + area lists, impute income/facilities, optionally normalize. Outputs include `urban.csv`, `rural.csv`, and diagnostic plots/JSONs. Use `--norm z` or `--norm 0-5` to normalize numeric features; `hh_id` and key categorical flags are excluded from normalization.

## Training Phase 1
- Generic entry: `python phase1.py --preset rural --xfile <train_csv> --yfile <labels_csv>`
- Preset-specific convenience: `python phase1_rural.py ...` or `phase1_urban.py ...`
Produces weights at `w_p1/{rural,urban}.pt`. Labels for Phase 1 targets live in `p1y_rural.csv` / `p1y_urban.csv` (5 target columns with `hh_id`).

## Training Phase 2
- Generic entry: `python phase2.py --preset rural --xfile <features_csv> --yfile <target_csv> --outdir <dir>`
- Preset-specific: `python phase2_rural.py ...` or `phase2_urban.py ...`
Training builds a neural net, XGBoost, Random Forest, Linear Regression, and a stacked ensemble. The ensemble plus normalization info is saved as a `.joblib` dict under `w_p2/{preset}.joblib` (contains `model`, `feature_names`, `mean`, `std`). Plots (ROC, confusion matrices) and metrics JSONs are written to the chosen outdir.

## Synthetic Data
- `synthetic_rural.py` / `synthetic_urban.py`: generate synthetic training/inference data for experimentation. Check the script docstrings/CLI for available knobs; outputs usually land under `synthetic/` or a specified directory.
