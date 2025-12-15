import argparse
import subprocess
import sys
import os


def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["rural","urban"], required=True)
    p.add_argument("--xfile", required=True)
    p.add_argument("--output_dir", default="infer_out")
    args = p.parse_args()

    preset = args.preset
    out = args.output_dir

    os.makedirs(out, exist_ok=True)

    # weight paths
    w_p1 = f"w_p1/{preset}.pt"
    # autoencoder weights stored under w_ae
    w_ae = f"w_ae/autoencoder_{preset}.pt"
    # phase 2 stacked ensemble weights are stored as .joblib
    w_p2 = f"w_p2/{preset}.joblib"

    # intermediate paths
    p1_out = f"{out}/p1_pred.csv"
    ae_out = f"{out}/ae_latent.csv"
    p2_out = f"{out}/final_p2_pred.csv"

    # Phase 1
    run([
        sys.executable, "inference_phase1.py",
        "--preset", preset,
        "--xfile", args.xfile,
        "--weights", w_p1,
        "--output", p1_out
    ])

    # Autoencoder
    run([
        sys.executable, "inference_autoencoder.py",
        "--input", p1_out,
        "--weights", w_ae,
        "--output", ae_out
    ])

    # Phase 2
    run([
        sys.executable, "inference_phase2.py",
        "--xfile", args.xfile,
        "--latent", ae_out,
        "--weights", w_p2,
        "--output", p2_out
    ])

    print("\nInference pipeline complete.")
    print(f"Final output: {p2_out}")


if __name__ == "__main__":
    main()
