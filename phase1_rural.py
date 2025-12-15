import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    confusion_matrix,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import random



# SET SEED


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# DATASET


class SingleOutputDataset(Dataset):
    def __init__(self, X, y_single):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_single, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



# CLASS WEIGHT COMPUTATION (Output 2 only)


def compute_class_weights(y):
    """
    Always return a vector of length 4 (for 4 classes).
    Missing classes get weight = 0.
    """
    weights = np.zeros(4, dtype=np.float32)
    classes, counts = np.unique(y, return_counts=True)

    inv = 1.0 / counts
    inv = inv / inv.sum()

    for cls, w in zip(classes, inv):
        weights[cls] = w

    if len(classes) == 1:
        weights[:] = 1.0 / 4

    return torch.tensor(weights, dtype=torch.float32)



# MODELS (5 outputs)


class Model1(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 4)
        )
    def forward(self, x): return self.net(x)


class Model3(nn.Module):  # internal for output 2 baseline
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 4)
        )
    def forward(self, x): return self.net(x)


class Model4(nn.Module):  # output 3
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 80), nn.ReLU(),
            nn.Linear(80, 40), nn.ReLU(),
            nn.Linear(40, 4)
        )
    def forward(self, x): return self.net(x)


class Model5(nn.Module):  # output 4
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48), nn.ReLU(),
            nn.Linear(48, 24), nn.ReLU(),
            nn.Linear(24, 4)
        )
    def forward(self, x): return self.net(x)


class Model6(nn.Module):  # output 5
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 70), nn.ReLU(),
            nn.Linear(70, 35), nn.ReLU(),
            nn.Linear(35, 4)
        )
    def forward(self, x): return self.net(x)


# strong model for output 2
class Model2Strong(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 4)
        )
    def forward(self, x): return self.net(x)


MODEL_FACTORY = {
    1: Model1,
    2: Model3,   # unless strong model applied
    3: Model4,
    4: Model5,
    5: Model6,
}

STRONG_MODELS = {2: Model2Strong}

EPOCHS_MAP = {5: 80}  # Output 5 gets more epochs



# TRAINING


def train_model(model, X_train, y_train, output_id, epochs=40, lr=0.001, batch=32):
    ds = SingleOutputDataset(X_train, y_train)
    loader = DataLoader(ds, batch_size=batch, shuffle=True)

    if output_id == 2:
        class_weights = compute_class_weights(y_train)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()

    return model



# EVALUATION


def evaluate(preds, y_true):
    acc = accuracy_score(y_true, preds)
    mae = mean_absolute_error(y_true, preds)
    mse = np.mean((y_true - preds)**2)
    rmse = np.sqrt(mse)
    f1 = f1_score(y_true, preds, average="weighted")
    return acc, mae, mse, rmse, f1



# MAIN ROUTINE (TRAIN + TEST ONLY)


def run(args):

    set_seed(42)

    # load
    X_df = pd.read_csv(args.xfile)
    Y_df = pd.read_csv(args.yfile)
    df = X_df.merge(Y_df, on="hh_id")

    feature_cols = [c for c in X_df.columns if c != "hh_id"]
    target_cols = [c for c in Y_df.columns if c != "hh_id"]

    X = df[feature_cols].values.astype(np.float32)
    Y = df[target_cols].values.astype(np.int64)

    indices = np.arange(len(df))

    # SIMPLE TRAIN/TEST SPLIT
    X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(
        X, Y, indices, test_size=0.20, random_state=42
    )

    hh_test = df.loc[idx_test, "hh_id"].values

    # normalise
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    if args.output_id == "all":
        output_ids = [1,2,3,4,5]
    else:
        output_ids = [int(x) for x in args.output_id.split(",")]

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    all_metrics = {}
    combined_preds = {}
    all_true = []
    all_pred = []
    saved_models = {}

    print("\n TRAINING MODELS ")

    # TRAIN EACH OUTPUT
    for out_id in output_ids:

        input_dim = X_train.shape[1]

        if out_id in STRONG_MODELS:
            model = STRONG_MODELS[out_id](input_dim)
        else:
            model = MODEL_FACTORY[out_id](input_dim)

        epochs = EPOCHS_MAP.get(out_id, args.epochs)

        model = train_model(
            model,
            X_train,
            Y_train[:, out_id - 1],
            output_id=out_id,
            epochs=epochs,
            lr=args.lr,
            batch=args.batch
        )

        saved_models[out_id] = model.state_dict()
        torch.save(model.state_dict(), f"{outdir}/model_{out_id}.pt")

        # TEST
        with torch.no_grad():
            logits = model(torch.tensor(X_test, dtype=torch.float32))
            preds = logits.argmax(dim=1).numpy()

        true_col = Y_test[:, out_id - 1]

        pd.DataFrame({
            "pred": preds,
            "true": true_col
        }).to_csv(f"{outdir}/preds_model_{out_id}.csv", index=False)

        combined_preds[f"pred_{out_id}"] = preds
        combined_preds[f"true_{out_id}"] = true_col

        acc, mae, mse, rmse, f1 = evaluate(preds, true_col)

        all_metrics[out_id] = {
            "accuracy": float(acc),
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "f1": float(f1)
        }

        all_pred.append(preds)
        all_true.append(true_col)

    # Save y_predicted
    pd.DataFrame(combined_preds).to_csv(f"{outdir}/y_predicted.csv", index=False)

    # rural predictions
    rural_df = pd.DataFrame({"hh_id": hh_test})
    for out_id in output_ids:
        rural_df[f"true_{out_id}"] = combined_preds[f"true_{out_id}"]
        rural_df[f"pred_{out_id}"] = combined_preds[f"pred_{out_id}"]
    rural_df.to_csv(f"{outdir}/rural_predictions_vs_ground_truth.csv", index=False)

    # save all models
    torch.save(saved_models, f"{outdir}/all_models.pt")

    # combined confusion matrix
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    cm = confusion_matrix(all_true, all_pred, labels=[0,1,2,3])

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.tight_layout()
    plt.savefig(f"{outdir}/confusion_combined.png")
    plt.close()

    # save json summary
    with open(f"{outdir}/metrics_summary.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    # TERMINAL METRICS SUMMARY
    print("\n FINAL TEST METRICS SUMMARY ")
    from tabulate import tabulate

    table = []
    for out_id, m in all_metrics.items():
        table.append([
            out_id,
            f"{m['accuracy']:.4f}",
            f"{m['mae']:.4f}",
            f"{m['mse']:.4f}",
            f"{m['rmse']:.4f}",
            f"{m['f1']:.4f}"
        ])

    print(tabulate(
        table,
        headers=["Output", "Accuracy", "MAE", "MSE", "RMSE", "F1"],
        tablefmt="github"
    ))

    print("\n DONE \n")



# CLI


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--xfile", type=str, required=True)
    parser.add_argument("--yfile", type=str, required=True)

    parser.add_argument("--output_id", type=str, default="all")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--outdir", type=str, default="results")

    args = parser.parse_args()
    run(args)

