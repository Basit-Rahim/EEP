#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys


def run(cmd, env):
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise SystemExit(f"Missing {label}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end training pipeline: preprocess -> phase1 -> autoencoder -> synthetic -> phase2."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-dir", default="train")
    parser.add_argument("--norm", choices=["z", "0-5"], default=None)
    parser.add_argument("--k-income", type=int, default=5)
    parser.add_argument("--k-fac", type=int, default=5)
    parser.add_argument("--z-norm", action="store_true")
    parser.add_argument("--route-unsafe-keep-prob", type=float, default=0.3)
    parser.add_argument("--route-unsafe-penalty", type=float, default=0.7)
    args = parser.parse_args()

    data_dir = args.data_dir
    train_dir = args.train_dir

    # Inputs
    long_path = os.path.join(data_dir, "long_df.csv")
    completed_path = os.path.join(data_dir, "completed_surveys.csv")
    urban_list = os.path.join(data_dir, "hh_urban.txt")
    rural_list = os.path.join(data_dir, "hh_rural.txt")
    p1y_urban = os.path.join(data_dir, "p1y_urban.csv")
    p1y_rural = os.path.join(data_dir, "p1y_rural.csv")
    y_urban = os.path.join(data_dir, "y_urban.csv")
    y_rural = os.path.join(data_dir, "y_rural.csv")

    for label, path in [
        ("long_df.csv", long_path),
        ("completed_surveys.csv", completed_path),
        ("hh_urban.txt", urban_list),
        ("hh_rural.txt", rural_list),
        ("p1y_urban.csv", p1y_urban),
        ("p1y_rural.csv", p1y_rural),
        ("y_urban.csv", y_urban),
        ("y_rural.csv", y_rural),
    ]:
        require_file(path, label)

    # Output dirs
    preprocess_dir = os.path.join(train_dir, "preprocess")
    phase1_dir = os.path.join(train_dir, "phase1")
    ae_dir = os.path.join(train_dir, "autoencoder")
    synth_dir = os.path.join(train_dir, "synthetic")
    phase2_dir = os.path.join(train_dir, "phase2")

    ensure_dir(preprocess_dir)
    ensure_dir(phase1_dir)
    ensure_dir(ae_dir)
    ensure_dir(synth_dir)
    ensure_dir(phase2_dir)

    # Weights stay under repo root
    w_p1 = "w_p1"
    w_ae = "w_ae"
    w_p2 = "w_p2"
    ensure_dir(w_p1)
    ensure_dir(w_ae)
    ensure_dir(w_p2)

    # Matplotlib cache under train/ to avoid permission issues
    mpl_dir = os.path.join(train_dir, ".mplconfig")
    ensure_dir(mpl_dir)
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", mpl_dir)

    # ---------------------------
    # 1) Preprocess
    # ---------------------------
    subset_out = os.path.join(preprocess_dir, "subset_final.csv")
    modified_out = os.path.join(preprocess_dir, "modified_households.txt")
    preprocess_cmd = [
        sys.executable, "preprocess.py",
        "--long", long_path,
        "--completed", completed_path,
        "--urban-list", urban_list,
        "--rural-list", rural_list,
        "--out", subset_out,
        "--modified", modified_out,
        "--k-income", str(args.k_income),
        "--k-fac", str(args.k_fac),
    ]
    if args.z_norm:
        preprocess_cmd.append("--z-norm")
    if args.norm:
        preprocess_cmd.extend(["--norm", args.norm])
    run(preprocess_cmd, env=env)

    urban_csv = os.path.join(preprocess_dir, "urban.csv")
    rural_csv = os.path.join(preprocess_dir, "rural.csv")
    require_file(urban_csv, "preprocess urban.csv")
    require_file(rural_csv, "preprocess rural.csv")

    # ---------------------------
    # 2) Phase 1 training
    # ---------------------------
    phase1_urban_out = os.path.join(phase1_dir, "urban")
    phase1_rural_out = os.path.join(phase1_dir, "rural")
    ensure_dir(phase1_urban_out)
    ensure_dir(phase1_rural_out)

    run([
        sys.executable, "phase1_urban.py",
        "--xfile", urban_csv,
        "--yfile", p1y_urban,
        "--outdir", phase1_urban_out,
    ], env=env)
    run([
        sys.executable, "phase1_rural.py",
        "--xfile", rural_csv,
        "--yfile", p1y_rural,
        "--outdir", phase1_rural_out,
    ], env=env)

    # Copy Phase 1 weights to main folder
    shutil.copy2(
        os.path.join(phase1_urban_out, "all_models.pt"),
        os.path.join(w_p1, "urban.pt"),
    )
    shutil.copy2(
        os.path.join(phase1_rural_out, "all_models.pt"),
        os.path.join(w_p1, "rural.pt"),
    )

    # ---------------------------
    # 3) Autoencoder training
    # ---------------------------
    ae_urban_out = os.path.join(ae_dir, "urban")
    ae_rural_out = os.path.join(ae_dir, "rural")
    ensure_dir(ae_urban_out)
    ensure_dir(ae_rural_out)

    ae_urban_csv = os.path.join(ae_urban_out, "urban_autoencoder.csv")
    ae_rural_csv = os.path.join(ae_rural_out, "rural_autoencoder.csv")

    run([
        sys.executable, "autoencoder.py",
        "--preset", "none",
        "--input", p1y_urban,
        "--output", ae_urban_csv,
        "--save-model", os.path.join(w_ae, "autoencoder_urban.pt"),
    ], env=env)
    run([
        sys.executable, "autoencoder.py",
        "--preset", "none",
        "--input", p1y_rural,
        "--output", ae_rural_csv,
        "--save-model", os.path.join(w_ae, "autoencoder_rural.pt"),
    ], env=env)

    # ---------------------------
    # 4) Synthetic data generation
    # ---------------------------
    synth_urban_root = os.path.join(synth_dir, "urban")
    synth_rural_root = os.path.join(synth_dir, "rural")
    ensure_dir(synth_urban_root)
    ensure_dir(synth_rural_root)

    run([
        sys.executable, "synthetic_urban.py",
        "--x-path", urban_csv,
        "--ae-path", ae_urban_csv,
        "--y-path", y_urban,
        "--output-root", synth_urban_root,
        "--route-unsafe-keep-prob", str(args.route_unsafe_keep_prob),
        "--route-unsafe-penalty", str(args.route_unsafe_penalty),
    ], env=env)
    run([
        sys.executable, "synthetic_rural.py",
        "--x-path", rural_csv,
        "--ae-path", ae_rural_csv,
        "--y-path", y_rural,
        "--output-root", synth_rural_root,
        "--route-unsafe-keep-prob", str(args.route_unsafe_keep_prob),
        "--route-unsafe-penalty", str(args.route_unsafe_penalty),
    ], env=env)

    # ---------------------------
    # 5) Phase 2 training
    # ---------------------------
    phase2_urban_out = os.path.join(phase2_dir, "urban")
    phase2_rural_out = os.path.join(phase2_dir, "rural")
    ensure_dir(phase2_urban_out)
    ensure_dir(phase2_rural_out)

    p2x_urban = os.path.join(synth_urban_root, "inputs", "p2x_urban.csv")
    p2y_urban = os.path.join(synth_urban_root, "inputs", "p2y_urban.csv")
    p2x_rural = os.path.join(synth_rural_root, "inputs", "p2x_rural.csv")
    p2y_rural = os.path.join(synth_rural_root, "inputs", "p2y_rural.csv")

    require_file(p2x_urban, "p2x_urban.csv")
    require_file(p2y_urban, "p2y_urban.csv")
    require_file(p2x_rural, "p2x_rural.csv")
    require_file(p2y_rural, "p2y_rural.csv")

    run([
        sys.executable, "phase2_urban.py",
        "--xfile", p2x_urban,
        "--yfile", p2y_urban,
        "--outdir", phase2_urban_out,
    ], env=env)
    run([
        sys.executable, "phase2_rural.py",
        "--xfile", p2x_rural,
        "--yfile", p2y_rural,
        "--outdir", phase2_rural_out,
    ], env=env)

    # Copy Phase 2 inference bundles to main folder
    shutil.copy2(
        os.path.join(phase2_urban_out, "stack_inference_bundle.joblib"),
        os.path.join(w_p2, "urban.joblib"),
    )
    shutil.copy2(
        os.path.join(phase2_rural_out, "stack_inference_bundle.joblib"),
        os.path.join(w_p2, "rural.joblib"),
    )

    print("\nTraining pipeline complete.")
    print(f"Outputs: {train_dir}/")
    print("Weights:")
    print(f"  Phase 1: {w_p1}/{{urban,rural}}.pt")
    print(f"  Autoencoder: {w_ae}/autoencoder_{{urban,rural}}.pt")
    print(f"  Phase 2: {w_p2}/{{urban,rural}}.joblib")


if __name__ == "__main__":
    main()
