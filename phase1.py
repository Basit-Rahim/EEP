import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser()

    # required selector
    parser.add_argument("--preset", type=str, required=True,
                        choices=["urban", "rural"],
                        help="Choose area type.")

    # pass-through args for either script
    parser.add_argument("--xfile", type=str, required=True)
    parser.add_argument("--yfile", type=str, required=True)
    parser.add_argument("--output_id", type=str, default="all")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--outdir", type=str, default=None)

    args = parser.parse_args()

    # Decide which backend script to execute
    if args.preset == "urban":
        script = "phase1_urban.py"
        default_outdir = "results_urban"
        default_epochs = 60
    else:
        script = "phase1_rural.py"
        default_outdir = "results_rural"
        default_epochs = 40

    # override defaults only when not provided manually
    outdir = args.outdir if args.outdir else default_outdir
    epochs = args.epochs if args.epochs is not None else default_epochs

    cmd = [
        sys.executable, script,
        "--xfile", args.xfile,
        "--yfile", args.yfile,
        "--output_id", args.output_id,
        "--epochs", str(epochs),
        "--lr", str(args.lr),
        "--batch", str(args.batch),
        "--outdir", outdir
    ]

    print(f"\nRunning: {script}\n")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()

