import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns


# DATASET WRAPPER
class DatasetWrapper(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# NEURAL NETWORK MODEL
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class Phase2Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.start = nn.Sequential(nn.Linear(input_dim, 128), nn.GELU())
        self.blocks = nn.ModuleList([ResidualBlock(128, 0.05) for _ in range(3)])
        self.out = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.start(x)
        for blk in self.blocks:
            x = blk(x)
        return self.out(x)


# TRAIN NEURAL NETWORK
def train_nn(model, X_train, Y_train, lr=0.001, batch=16, epochs=120):
    train_ds = DatasetWrapper(X_train, Y_train)
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True)

    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        for Xb, Yb in loader:
            optim.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, Yb)
            loss.backward()
            optim.step()

        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  [NN] Epoch {ep:4d}/{epochs} - MSE loss: {loss.item():.6f}")

    return model


# EVALUATION HELPERS
def evaluate_regression(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    metrics = dict(
        model=name,
        MAE=float(mean_absolute_error(y_true, y_pred)),
        MSE=float(mse),
        RMSE=float(np.sqrt(mse)),
        R2=float(r2_score(y_true, y_pred)),
    )
    return metrics


def evaluate_classification_from_prob(y_true, y_prob, threshold=0.5):
    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin)
    cm = confusion_matrix(y_true_bin, y_pred_bin)

    return acc, f1, cm, y_true_bin, y_pred_bin


def find_optimal_threshold(y_true, y_prob):
    """
    Finds:
    - threshold that maximizes F1
    - threshold that maximizes Youden J = TPR - FPR
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    best_f1 = -1
    best_f1_thr = 0.5
    best_j = -1
    best_j_thr = 0.5

    y_true_bin = (y_true >= 0.5).astype(int)

    for thr in thresholds:
        y_pred_bin = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true_bin, y_pred_bin)
        fpr, tpr, _ = roc_curve(y_true_bin, y_pred_bin)
        J = tpr[1] - fpr[1]  # Youden index

        if f1 > best_f1:
            best_f1, best_f1_thr = f1, thr
        if J > best_j:
            best_j, best_j_thr = J, thr

    return best_f1_thr, best_j_thr


# TRAINING PIPELINE
def main(args):
    print("\n=== PHASE 2 URBAN – ENSEMBLE TRAINING ===\n")

    # Load Data
    X_df = pd.read_csv(args.xfile)
    Y_df = pd.read_csv(args.yfile)

    df = X_df.merge(Y_df, on="hh_id")
    target_col = "prob"

    feature_cols = [c for c in X_df.columns if c != "hh_id"]
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32).reshape(-1, 1)

    print(f"Loaded merged data: {len(df)} rows, {X.shape[1]} features.")
    print(f"Target column: {target_col}\n")

    hh_ids = df["hh_id"].values

    # Train/test split
    X_train, X_test, y_train, y_test, hh_train, hh_test = train_test_split(
        X, y, hh_ids, test_size=0.2, random_state=42, shuffle=True
    )

    # Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Save normalization
    norm_info = {
        "feature_names": feature_cols,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    with open(os.path.join(outdir, "normalisation.json"), "w") as f:
        json.dump(norm_info, f, indent=4)

    # -----------------------------------------------------------
    # Train Models
    # -----------------------------------------------------------
    print("Training Neural Network...")
    nn_model = Phase2Net(X_train.shape[1])
    nn_model = train_nn(nn_model, X_train, y_train)

    nn_model.eval()
    with torch.no_grad():
        nn_preds_test = nn_model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()

    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds_test = xgb_model.predict(X_test)

    print("Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=400, max_depth=12)
    rf_model.fit(X_train, y_train.flatten())
    rf_preds_test = rf_model.predict(X_test)

    print("Training Linear Regression...")
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    lin_preds_test = lin_model.predict(X_test).flatten()

    print("Training Stacked Ensemble...")
    stack = StackingRegressor(
        estimators=[
            ("xgb", xgb_model),
            ("rf", rf_model),
            ("lin", lin_model),
        ],
        final_estimator=LinearRegression(),
    )
    stack.fit(X_train, y_train.flatten())
    stack_preds_test = stack.predict(X_test)

    # -----------------------------------------------------------
    # SAVE TRAINED MODELS
    # -----------------------------------------------------------
    torch.save(
        {"state_dict": nn_model.state_dict(), "input_dim": X_train.shape[1]},
        os.path.join(outdir, "nn_model.pt"),
    )
    xgb_model.save_model(os.path.join(outdir, "xgb_model.json"))
    joblib.dump(rf_model, os.path.join(outdir, "rf_model.joblib"))
    joblib.dump(lin_model, os.path.join(outdir, "lin_model.joblib"))
    joblib.dump(stack, os.path.join(outdir, "stack_model.joblib"))
    joblib.dump(
        {
            "model": stack,
            "mean": mean,
            "std": std,
            "feature_names": feature_cols,
        },
        os.path.join(outdir, "stack_inference_bundle.joblib"),
    )

    y_true = y_test.flatten()

    # -----------------------------------------------------------
    # METRICS (Regression + Classification)
    # -----------------------------------------------------------
    metrics_list = []

    def eval_all(name, preds):
        metrics = evaluate_regression(name, y_true, preds)
        acc, f1, cm, yt, yp = evaluate_classification_from_prob(y_true, preds)
        metrics["Accuracy"] = acc
        metrics["F1"] = f1
        return metrics, cm

    m_nn, cm_nn = eval_all("Neural Network", nn_preds_test)
    m_xgb, cm_xgb = eval_all("XGBoost", xgb_preds_test)
    m_rf, cm_rf = eval_all("Random Forest", rf_preds_test)
    m_lin, cm_lin = eval_all("Linear Regression", lin_preds_test)
    m_stack, cm_stack = eval_all("Stacked Ensemble", stack_preds_test)

    metrics_list = [m_nn, m_xgb, m_rf, m_lin, m_stack]

    # -----------------------------------------------------------
    # FIND OPTIMAL THRESHOLDS (F1 and Youden J)
    # -----------------------------------------------------------
    best_thr_f1, best_thr_j = find_optimal_threshold(y_true, stack_preds_test)

    thr_info = {
        "best_f1_threshold": best_thr_f1,
        "best_youden_j_threshold": best_thr_j,
    }

    with open(os.path.join(outdir, "optimal_thresholds.json"), "w") as f:
        json.dump(thr_info, f, indent=4)

    # -----------------------------------------------------------
    # SAVE METRICS SUMMARY
    # -----------------------------------------------------------
    metrics_dict = {m["model"]: m for m in metrics_list}
    with open(os.path.join(outdir, "metrics_summary.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print("\nFINAL TEST METRICS (80/20 split)\n")

    from tabulate import tabulate
    table = []
    for m in metrics_list:
        table.append([
            m["model"],
            f"{m['MAE']:.4f}",
            f"{m['RMSE']:.4f}",
            f"{m['R2']:.4f}",
            f"{m['Accuracy']:.4f}",
            f"{m['F1']:.4f}",
        ])

    print(
        tabulate(
            table,
            headers=["Model", "MAE", "RMSE", "R2", "Accuracy", "F1"],
            tablefmt="github",
        )
    )

    # -----------------------------------------------------------
    # ROC CURVE (FOR STACKED ENSEMBLE)
    # -----------------------------------------------------------
    fpr, tpr, thr = roc_curve((y_true >= 0.5).astype(int), stack_preds_test)
    auc_value = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    plt.scatter(
        fpr[np.argmax(tpr - fpr)],
        tpr[np.argmax(tpr - fpr)],
        color="red",
        label=f"Optimal J Threshold = {best_thr_j:.2f}",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Stacked Ensemble")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_curve_stack.png"))
    plt.close()

    # -----------------------------------------------------------
    # CONFUSION MATRICES
    # -----------------------------------------------------------
    cms = {
        "nn": cm_nn,
        "xgb": cm_xgb,
        "rf": cm_rf,
        "lin": cm_lin,
        "stack": cm_stack,
    }

    for name, cm in cms.items():
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix ({name.upper()})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"cm_{name}.png"))
        plt.close()

    print("\nAll plots and metrics saved.\nDone.\n")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--xfile", type=str, required=True)
    parser.add_argument("--yfile", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="synthetic/models_phase2")

    args = parser.parse_args()
    main(args)
