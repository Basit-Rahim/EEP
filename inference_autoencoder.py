#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn


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
        return z, x_hat


# LOAD + CLEAN CSV
def load_data(path: str, id_col: str | None):
    df = pd.read_csv(path)

    ids = None
    if id_col and id_col in df.columns:
        ids = df[id_col].copy()
        df = df.drop(columns=[id_col])

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(df.mean(numeric_only=True))
    data = df.to_numpy(dtype=np.float32)

    return data, ids, df.columns.tolist()


# RUN INFERENCE
def encode_with_autoencoder(model, x: np.ndarray, latent_dim: int = 3):
    device = next(model.parameters()).device
    model.eval()

    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        z, _ = model(x_tensor)

    return z.cpu().numpy()


# MAIN PIPELINE
def main():
    parser = argparse.ArgumentParser(description="Autoencoder Inference Pipeline")

    parser.add_argument("--input", required=True, help="Input CSV path (5-dim)")
    parser.add_argument("--weights", required=True, help="Trained AE .pt weights")
    parser.add_argument("--output", required=True, help="Output latent CSV path")
    parser.add_argument("--id-col", default="hh_id", help="ID column to preserve")

    args = parser.parse_args()

    # load input
    data, ids, cols = load_data(args.input, args.id_col)
    input_dim = data.shape[1]

    # load AE
    model = Autoencoder(input_dim=input_dim, latent_dim=3)
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state)

    print(f"Loaded autoencoder with input_dim={input_dim}")

    # run encoding
    latent = encode_with_autoencoder(model, data, latent_dim=3)

    # save output
    latent_df = pd.DataFrame(latent, columns=["z1", "z2", "z3"])
    if ids is not None:
        latent_df = pd.concat([ids.reset_index(drop=True), latent_df], axis=1)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    latent_df.to_csv(args.output, index=False)
    print(f"Saved latent representation to {args.output}")


if __name__ == "__main__":
    main()

