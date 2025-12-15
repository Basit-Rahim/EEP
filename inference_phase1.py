import argparse
import torch
import numpy as np
import pandas as pd
import torch.nn as nn


# LOAD MODELS (same as training)
class Model1(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 4)
        )
    def forward(self, x): return self.net(x)


class Model2Strong(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 4)
        )
    def forward(self, x): return self.net(x)


class Model3(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 80), nn.ReLU(),
            nn.Linear(80, 40), nn.ReLU(),
            nn.Linear(40, 4)
        )
    def forward(self, x): return self.net(x)


class Model4(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48), nn.ReLU(),
            nn.Linear(48, 24), nn.ReLU(),
            nn.Linear(24, 4)
        )
    def forward(self, x): return self.net(x)


class Model5(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 70), nn.ReLU(),
            nn.Linear(70, 35), nn.ReLU(),
            nn.Linear(35, 4)
        )
    def forward(self, x): return self.net(x)


# Urban strong variants (mirrors phase1_urban training)
class UrbanStrongModel4(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden = 256
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.block3 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.out = nn.Linear(hidden, 4)

    def forward(self, x):
        x = self.block1(x)
        x = x + self.block2(x)
        x = x + self.block3(x)
        return self.out(x)


class UrbanStrongModel5(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden = 320
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.block3 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.block4 = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.out = nn.Linear(hidden // 2, 4)
        self.block5 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.15),
        )

    def forward(self, x):
        x = self.block1(x)
        res = x
        x = self.block2(x)
        x = x + res
        res = x
        x = self.block3(x)
        x = x + res
        res = x
        x = self.block5(x)
        x = x + res
        x = self.block4(x)
        return self.out(x)


P1_MODELS_RURAL = {
    1: Model1,
    2: Model2Strong,
    3: Model3,
    4: Model4,
    5: Model5
}

P1_MODELS_URBAN = {
    1: Model1,
    2: Model2Strong,
    3: Model3,
    4: UrbanStrongModel4,
    5: UrbanStrongModel5
}


def get_p1_models(preset: str):
    return P1_MODELS_URBAN if preset == "urban" else P1_MODELS_RURAL


# RUN INFERENCE
def run_inference(xfile, weights_file, output_file, preset="rural"):
    df = pd.read_csv(xfile)
    hh_ids = df["hh_id"]
    X = df.drop(columns=["hh_id"]).values.astype(np.float32)
    input_dim = X.shape[1]

    tensor_x = torch.tensor(X, dtype=torch.float32)

    # load all 5 models (same as training)
    state = torch.load(weights_file, map_location="cpu")

    preds = {}

    model_map = get_p1_models(preset)

    for out_id in range(1, 6):
        model = model_map[out_id](input_dim)
        model.load_state_dict(state[out_id])
        model.eval()
        with torch.no_grad():
            logits = model(tensor_x)
            pred = logits.argmax(dim=1).numpy()
        preds[f"pred_{out_id}"] = pred

    # save
    out_df = pd.DataFrame({"hh_id": hh_ids})
    for k, v in preds.items():
        out_df[k] = v

    out_df.to_csv(output_file, index=False)
    print(f"Saved Phase 1 inference: {output_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["rural","urban"], required=True)
    p.add_argument("--xfile", required=True)
    p.add_argument("--weights", required=True)   # w_p1/rural.pt or w_p1/urban.pt
    p.add_argument("--output", required=True)
    args = p.parse_args()

    run_inference(args.xfile, args.weights, args.output, preset=args.preset)


if __name__ == "__main__":
    main()
