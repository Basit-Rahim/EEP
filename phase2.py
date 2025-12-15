import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--preset", type=str, required=True,
                        choices=["urban", "rural"],
                        help="Run Phase 2 for rural or urban dataset.")

    parser.add_argument("--xfile", type=str, required=True)
    parser.add_argument("--yfile", type=str, required=True)
    parser.add_argument("--outdir", type=str, default=None)

    args = parser.parse_args()

    # Select which script to execute
    if args.preset == "urban":
        script = "phase2_urban.py"
        default_outdir = "phase2_results_urban"
    else:
        script = "phase2_rural.py"
        default_outdir = "phase2_results_rural"

    outdir = args.outdir if args.outdir else default_outdir

    cmd = [
        sys.executable, script,
        "--xfile", args.xfile,
        "--yfile", args.yfile,
        "--outdir", outdir
    ]

    print(f"\nRunning {script}\n")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()

