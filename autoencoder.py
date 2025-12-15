import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.ReLU(),
            nn.Linear(4, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4),
            nn.ReLU(),
            nn.Linear(4, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


def load_data(path: str, id_col: str | None):
    df = pd.read_csv(path)

    ids = None
    if id_col is not None and id_col in df.columns:
        ids = df[id_col].copy()
        df = df.drop(columns=[id_col])

    # make sure everything is numeric (avoids numpy.object_ issues)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(df.mean(numeric_only=True))

    data = df.to_numpy(dtype=np.float32)
    return data, ids, df.columns.tolist()


def train_model(
    x: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    save_model_path: str | None,
):
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = x.shape[1]
    model = Autoencoder(input_dim=input_dim, latent_dim=3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training autoencoder on {x.shape[0]} rows, {input_dim} features")
    print(f"Learning rate: {lr}, epochs: {epochs}, batch size: {batch_size}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch}/{epochs} Loss={epoch_loss:.6f}")

    # final evaluation
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        recon = model(x_tensor)
        mse = criterion(recon, x_tensor).item()

        # rounded reconstruction accuracy (assuming discrete values like 1,2,3)
        rounded_recon = torch.round(recon).cpu().numpy()
        rounded_orig = torch.round(x_tensor).cpu().numpy()
        accuracy = (rounded_recon == rounded_orig).mean() * 100.0

    print(f"\nFinal reconstruction MSE: {mse:.6f}")
    print(f"Reconstruction accuracy (rounded match): {accuracy:.2f}%")

    if save_model_path is not None:
        model_dir = os.path.dirname(save_model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        torch.save(model.state_dict(), save_model_path)
        print(f"Saved model weights to {save_model_path}")

    return model, recon.detach().cpu().numpy(), mse, accuracy


def save_encoded_csv(
    model: Autoencoder,
    x: np.ndarray,
    ids: pd.Series | None,
    output_path: str,
    latent_dim: int = 3,
):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        encoded = model.encoder(x_tensor).cpu().numpy()

    encoded_df = pd.DataFrame(
        encoded, columns=[f"z{i+1}" for i in range(latent_dim)]
    )

    if ids is not None:
        out_df = pd.concat(
            [ids.reset_index(drop=True), encoded_df], axis=1
        )
    else:
        out_df = encoded_df

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    out_df.to_csv(output_path, index=False)
    print(f"Saved encoded CSV to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="5->3 autoencoder for CSV data"
    )
    parser.add_argument(
        "--preset",
        choices=["none", "rural", "urban"],
        default="none",
        help=(
            "Preset paths and default learning rates. "
            "rural: lr=0.1, urban: lr=0.01. "
            "If 'none', you must provide --input and --output."
        ),
    )
    parser.add_argument(
        "--input",
        help="Input CSV path. Optional if preset is rural/urban (defaults used).",
    )
    parser.add_argument(
        "--output",
        help="Output CSV path for encoded data. Optional if preset is rural/urban (defaults used).",
    )
    parser.add_argument(
        "--id-col",
        default="hh_id",
        help="ID column name to keep (default: hh_id). Use '' to disable.",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate. If omitted, preset-specific default is used.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size."
    )
    parser.add_argument(
        "--save-model",
        default=None,
        help=(
            "Path to save model weights. If omitted, a preset-specific "
            "default under models/ is used."
        ),
    )
    return parser.parse_args()


def resolve_paths_and_lr(args):
    # defaults you said you are using
    rural_input_default = "Phase 1/rural/p1y_rural.csv"
    rural_output_default = "Phase 2/Rural/p1y_rural.csv"

    urban_input_default = "Phase 1/urban/p1y_urban.csv"
    urban_output_default = "Phase 2/Urban/p1y_urban.csv"

    if args.preset == "rural":
        input_path = args.input or rural_input_default
        output_path = args.output or rural_output_default
        lr = args.lr if args.lr is not None else 0.1
        model_path = (
            args.save_model or "models/autoencoder_rural.pt"
        )
    elif args.preset == "urban":
        input_path = args.input or urban_input_default
        output_path = args.output or urban_output_default
        lr = args.lr if args.lr is not None else 0.01
        model_path = (
            args.save_model or "models/autoencoder_urban.pt"
        )
    else:
        # no preset; user must specify paths
        if not args.input or not args.output:
            raise SystemExit(
                "When --preset none, you must specify both --input and --output."
            )
        input_path = args.input
        output_path = args.output
        lr = args.lr if args.lr is not None else 0.01
        model_path = args.save_model or "models/autoencoder_custom.pt"

    return input_path, output_path, lr, model_path


def main():
    args = parse_args()
    input_path, output_path, lr, save_model_path = resolve_paths_and_lr(args)

    id_col = args.id_col if args.id_col != "" else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    data, ids, feature_cols = load_data(input_path, id_col=id_col)
    print(f"Loaded {input_path}")
    print(f"Using features: {feature_cols}\n")

    model, recon, mse, acc = train_model(
        x=data,
        epochs=args.epochs,
        lr=lr,
        batch_size=args.batch_size,
        device=device,
        save_model_path=save_model_path,
    )

    save_encoded_csv(
        model=model,
        x=data,
        ids=ids,
        output_path=output_path,
        latent_dim=3,
    )


if __name__ == "__main__":
    main()

